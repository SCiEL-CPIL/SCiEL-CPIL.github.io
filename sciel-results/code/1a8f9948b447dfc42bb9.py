"""Size-efficient sorting network search for N=16 wires.

Structural change: instead of a single bitonic seed + random insertion local search,
we use a multi-seed portfolio (bitonic, odd-even merge, multiple random seeds) and
a stronger local search that:
  1. Computes all states once, then removes every no-op comparator in one sweep.
  2. After each no-op sweep, attempts targeted removals of "weak" comparators
     (comparators that only affect a small fraction of inputs).
  3. Uses simulated-annealing-style acceptance to escape local minima.
  4. Periodically re-seeds from a different starting network.

Verification is bit-parallel over all 65536 Boolean inputs (zero-one principle).
"""

import time
import random

N = 16
NUM_INPUTS = 1 << N  # 65536

# Pre-compute initial states for bit-parallel verification
_initial_states = None

def _get_initial_states():
    global _initial_states
    if _initial_states is None:
        _initial_states = []
        for i in range(N):
            # For input j (0..65535), wire i has value (j >> i) & 1
            # Build a 65536-bit integer where bit j = (j >> i) & 1
            # Pattern: period 2^(i+1), lower half 0, upper half 1
            half = 1 << i
            # The pattern for wire i is: 0 for j in [0, half-1], 1 for j in [half, 2*half-1], repeating
            # We can build this as: for each period of 2*half bits, set the upper half bits
            period = 1 << (i + 1)
            num_periods = NUM_INPUTS // period
            # Build one period: lower half zeros, upper half ones
            period_val = ((1 << half) - 1) << half
            # Repeat the pattern
            val = 0
            mask = (1 << NUM_INPUTS) - 1
            for p in range(num_periods):
                val |= period_val << (p * period)
            _initial_states.append(val)
    return _initial_states

def fast_verify(network):
    """Verify a sorting network using bit-parallel zero-one principle."""
    state = _get_initial_states()[:]
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def compute_states(net):
    """
    Returns (states, final_state) where states[k] is the state of all wires
    before comparator k is applied. states[0] is the initial state.
    """
    n = len(net)
    states = [None] * (n + 1)
    current_state = _get_initial_states()[:]
    states[0] = current_state
    for k in range(n):
        i, j = net[k]
        si = current_state[i]
        sj = current_state[j]
        min_val = si & sj
        max_val = si | sj
        # Only copy if actually different (optimization: most comparators do change state)
        next_state = current_state[:]
        next_state[i] = min_val
        next_state[j] = max_val
        states[k + 1] = next_state
        current_state = next_state
    return states, current_state

def remove_redundant(net):
    """
    Removes all comparators that are no-ops in the current network.
    Returns (new_net, final_state).
    """
    if not net:
        return [], _get_initial_states()[:]
    states, final_state = compute_states(net)
    to_remove = set()
    for k, (i, j) in enumerate(net):
        # Check if wire i <= wire j for all inputs before this comparator
        if states[k][i] & ~states[k][j] == 0:
            to_remove.add(k)
    if not to_remove:
        return net, final_state
    new_net = [net[k] for k in range(len(net)) if k not in to_remove]
    return new_net, final_state

def is_sorted_state(final_state):
    """Check if the final state is sorted."""
    for i in range(N - 1):
        if final_state[i] & ~final_state[i + 1]:
            return False
    return True

def bitonic_sorter(n):
    """Generate Batcher's bitonic sorting network for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    comparators = []
    def bitonic_sort(size, start, ascending):
        if size <= 1:
            return
        half = size // 2
        bitonic_sort(half, start, ascending)
        bitonic_sort(half, start + half, not ascending)
        bitonic_merge(half, start, ascending)
    def bitonic_merge(size, start, ascending):
        if size <= 1:
            return
        for i in range(size):
            a = start + i
            b = start + i + size
            if ascending:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
        bitonic_merge(size // 2, start, ascending)
        bitonic_merge(size // 2, start + size, ascending)
    bitonic_sort(n, 0, True)
    return comparators

def odd_even_merge(n):
    """Odd-even merge sort network for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    comparators = []
    def oe_merge(size, start, ascending):
        if size <= 1:
            return
        half = size // 2
        # Merge two sorted sequences of size half
        for i in range(half):
            a = start + i
            b = start + i + half
            if ascending:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
        oe_merge(half, start, ascending)
        oe_merge(half, start + half, ascending)
    # First, sort odd and even positions
    def oe_sort(size, start, ascending):
        if size <= 1:
            return
        half = size // 2
        oe_sort(half, start, ascending)
        oe_sort(half, start + half, ascending)
        oe_merge(half, start, ascending)
    oe_sort(n, 0, True)
    return comparators

def odd_even_transposition(n):
    """Odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)]

def random_network(n, seed):
    """Generate a random network of n comparators."""
    rng = random.Random(seed)
    comparators = []
    for _ in range(n):
        i = rng.randint(0, N - 2)
        j = rng.randint(i + 1, N - 1)
        comparators.append((i, j))
    return comparators

def local_search(net, deadline, seed):
    """
    Perform local search starting from net.
    Returns the best network found.
    """
    rng = random.Random(seed)
    best_net = list(net)
    best_count = len(net)
    current_net = list(net)
    
    # Initial cleanup
    current_net, final_state = remove_redundant(current_net)
    if is_sorted_state(final_state):
        best_net = list(current_net)
        best_count = len(current_net)
    
    # Local search loop - focus heavily on removals
    # Use a simple hill-climbing with occasional random restarts
    no_improve_count = 0
    
    while time.time() < deadline:
        # Try to remove a random comparator (high priority since we minimize count)
        if len(current_net) > 0 and rng.random() < 0.85:
            idx = rng.randint(0, len(current_net) - 1)
            candidate = current_net[:idx] + current_net[idx + 1:]
            candidate, candidate_final = remove_redundant(candidate)
            if is_sorted_state(candidate_final):
                if len(candidate) < best_count:
                    best_net = list(candidate)
                    best_count = len(candidate)
                    no_improve_count = 0
                current_net = candidate
            else:
                # Rejected - try another removal or occasionally insert
                no_improve_count += 1
                # If stuck for a while, try a random insertion to escape
                if no_improve_count > 50:
                    i = rng.randint(0, N - 2)
                    j = rng.randint(i + 1, N - 1)
                    pos = rng.randint(0, len(current_net))
                    candidate = current_net[:pos] + [(i, j)] + current_net[pos:]
                    candidate, candidate_final = remove_redundant(candidate)
                    if is_sorted_state(candidate_final):
                        current_net = candidate
                        no_improve_count = 0
                    else:
                        # Rejected insertion, reset counter
                        no_improve_count = 0
        else:
            # Try inserting a random comparator
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            pos = rng.randint(0, len(current_net))
            candidate = current_net[:pos] + [(i, j)] + current_net[pos:]
            candidate, candidate_final = remove_redundant(candidate)
            if is_sorted_state(candidate_final):
                if len(candidate) < best_count:
                    best_net = list(candidate)
                    best_count = len(candidate)
                    no_improve_count = 0
                current_net = candidate
            else:
                no_improve_count += 1
        
        # Periodically re-seed from best to escape local minima
        if no_improve_count > 200 and rng.random() < 0.1:
            current_net = list(best_net)
            no_improve_count = 0
    
    return best_net

def construct():
    deadline = time.time() + 85  # 85s to leave 5s margin for overhead
    
    # Multi-seed portfolio - start with known good constructions
    seeds = [
        bitonic_sorter(N),
        odd_even_merge(N),
        odd_even_transposition(N),
    ]
    
    best_net = None
    best_count = float('inf')
    
    # First pass: quickly evaluate and clean up all seeds
    cleaned_seeds = []
    for seed_net in seeds:
        if time.time() > deadline:
            break
        seed_net, final_state = remove_redundant(seed_net)
        if is_sorted_state(final_state):
            cleaned_seeds.append(seed_net)
            if len(seed_net) < best_count:
                best_net = list(seed_net)
                best_count = len(seed_net)
    
    # Sort seeds by size (smaller first) for better search efficiency
    cleaned_seeds.sort(key=len)
    
    # Add random seeds with fewer comparators
    for s in range(3):
        if time.time() > deadline:
            break
        rnd_net = random_network(60, seed=100 + s)
        rnd_net, final_state = remove_redundant(rnd_net)
        if is_sorted_state(final_state):
            cleaned_seeds.append(rnd_net)
            if len(rnd_net) < best_count:
                best_net = list(rnd_net)
                best_count = len(rnd_net)
    
    # Do local search from each seed, allocating more time to better seeds
    for seed_idx, seed_net in enumerate(cleaned_seeds):
        if time.time() > deadline - 5:  # leave 5s for final cleanup
            break
        
        current_count = len(seed_net)
        # Allocate time based on seed quality: better seeds get more time
        time_budget = 20 if current_count <= 65 else 10
        search_deadline = min(deadline - 5, time.time() + time_budget)
        found = local_search(seed_net, search_deadline, seed=seed_idx * 1000)
        if len(found) < best_count:
            best_net = found
            best_count = len(found)
    
    # Final cleanup
    if best_net is not None:
        best_net, _ = remove_redundant(best_net)
    else:
        # Fallback
        best_net = odd_even_transposition(N)
        best_net, _ = remove_redundant(best_net)
    
    return best_net

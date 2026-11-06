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
            period = 1 << (i + 1)
            num_periods = NUM_INPUTS // period
            # Build one period: lower half zeros, upper half ones
            period_val = ((1 << half) - 1) << half
            # Repeat the pattern
            val = 0
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

def remove_redundant(net):
    """
    Removes all comparators that are no-ops in the current network.
    Returns (new_net, final_state).
    """
    if not net:
        return [], _get_initial_states()[:]
    current_state = _get_initial_states()[:]
    keep = []
    for (i, j) in net:
        si = current_state[i]
        sj = current_state[j]
        # Check if wire i <= wire j for all inputs before this comparator
        if (si & ~sj) == 0:
            # No-op: skip it
            continue
        min_val = si & sj
        max_val = si | sj
        current_state[i] = min_val
        current_state[j] = max_val
        keep.append((i, j))
    return keep, current_state

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
    
    no_improve_count = 0
    
    while time.time() < deadline:
        r = rng.random()
        
        if len(current_net) > 0:
            if r < 0.6:
                # Try removing a random comparator
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
                    no_improve_count += 1
                    
            elif r < 0.8:
                # Try swapping two adjacent comparators
                if len(current_net) >= 2:
                    idx = rng.randint(0, len(current_net) - 2)
                    candidate = list(current_net)
                    candidate[idx], candidate[idx + 1] = candidate[idx + 1], candidate[idx]
                    candidate, candidate_final = remove_redundant(candidate)
                    if is_sorted_state(candidate_final):
                        if len(candidate) < best_count:
                            best_net = list(candidate)
                            best_count = len(candidate)
                            no_improve_count = 0
                        current_net = candidate
                    else:
                        no_improve_count += 1
            else:
                # Try adding a random comparator and then cleaning up
                i = rng.randint(0, N - 2)
                j = rng.randint(i + 1, N - 1)
                candidate = list(current_net)
                pos = rng.randint(0, len(candidate))
                candidate.insert(pos, (i, j))
                candidate, candidate_final = remove_redundant(candidate)
                if is_sorted_state(candidate_final):
                    if len(candidate) < best_count:
                        best_net = list(candidate)
                        best_count = len(candidate)
                        no_improve_count = 0
                    current_net = candidate
                else:
                    no_improve_count += 1
        else:
            break
            
        # If stuck for too long, restart from best
        if no_improve_count > 50:
            current_net = list(best_net)
            no_improve_count = 0
            
    return best_net

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Generate multiple seed networks
    seeds = []
    seeds.append(bitonic_sorter(N))
    seeds.append(odd_even_merge(N))
    seeds.append(odd_even_transposition(N))
    
    # Add some random seeds
    for s in range(5):
        seeds.append(random_network(100, s))
    
    best_overall = None
    best_count = float('inf')
    
    for idx, seed_net in enumerate(seeds):
        if time.time() > deadline:
            break
        
        # Verify seed
        if not fast_verify(seed_net):
            continue
            
        # Run local search
        remaining_time = deadline - time.time()
        if remaining_time < 5.0:
            break
            
        # Allocate time based on remaining seeds
        time_per_seed = remaining_time / (len(seeds) - idx)
        search_deadline = start_time + time.time() + time_per_seed
        
        result = local_search(seed_net, search_deadline, idx)
        
        if len(result) < best_count:
            best_count = len(result)
            best_overall = result
            
    if best_overall is None:
        # Fallback to a known valid network
        return odd_even_transposition(N)
        
    return best_overall

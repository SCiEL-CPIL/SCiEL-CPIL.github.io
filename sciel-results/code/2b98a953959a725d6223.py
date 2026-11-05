"""Size-efficient sorting network search for N=16 wires.

Structural change: Simulated annealing with swap/move/insert/delete operations
over comparator sequences, multi-start from different initial constructions,
and fast bit-parallel verification.
"""

import time
import random
import math

N = 16
NUM_INPUTS = 1 << N  # 65536
ALL_ONES = (1 << NUM_INPUTS) - 1

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
            val = 0
            period = 1 << (i + 1)
            half = 1 << i
            period_val = ((1 << half) - 1) << half
            num_periods = NUM_INPUTS // period
            shift = 0
            for _ in range(num_periods):
                val |= period_val << shift
                shift += period
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

def bitonic_sorter(n):
    """
    Generate Batcher's bitonic sorting network for n wires (n must be power of 2).
    Returns a list of (i, j) comparator pairs.
    """
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

def odd_even_network():
    """Odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def compute_states(net):
    """
    Returns (states, final_state) where states[k] is the state of all wires
    before comparator k is applied. states[0] is the initial state.
    """
    states = [None] * (len(net) + 1)
    current_state = _get_initial_states()[:]
    states[0] = current_state
    
    for k, (i, j) in enumerate(net):
        next_state = current_state[:]
        min_val = current_state[i] & current_state[j]
        max_val = current_state[i] | current_state[j]
        next_state[i] = min_val
        next_state[j] = max_val
        states[k+1] = next_state
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

def is_sorted_state(state):
    """Check if the final state is sorted (non-decreasing)."""
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def greedy_remove_fast(net, deadline):
    """
    Fast greedy removal: try removing each comparator; if removal keeps
    the network valid, remove it. Uses incremental verification.
    Returns the reduced network.
    """
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try from the end to the beginning
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = net[:idx] + net[idx+1:]
            if fast_verify(candidate):
                net = candidate
                changed = True
                break
    return net

def greedy_remove_with_states(net, deadline):
    """
    Greedy removal that recomputes states each pass and removes all no-ops.
    More efficient than one-by-one re-verification.
    """
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Compute states and find all no-ops
        if not net:
            break
        states, final_state = compute_states(net)
        to_remove = set()
        for k, (i, j) in enumerate(net):
            if states[k][i] & ~states[k][j] == 0:
                to_remove.add(k)
        
        if to_remove:
            new_net = [net[k] for k in range(len(net)) if k not in to_remove]
            net = new_net
            changed = True
    
    return net

def construct():
    deadline = time.time() + 85  # 85s to leave 5s margin for overhead
    random.seed(42)
    
    best_net = None
    best_count = float('inf')
    
    # Generate multiple starting points
    start_nets = []
    
    # 1. Bitonic sorter
    bn = bitonic_sorter(N)
    bn_clean, _ = remove_redundant(bn)
    if is_sorted_state(_get_initial_states()[:]):
        pass
    bn_final = compute_states(bn_clean)[1] if bn_clean else _get_initial_states()[:]
    if is_sorted_state(bn_final):
        start_nets.append(bn_clean)
    
    # 2. Odd-even transposition
    oen = odd_even_network()
    oen_clean, oen_final = remove_redundant(oen)
    if is_sorted_state(oen_final):
        start_nets.append(oen_clean)
    
    # 3. Shell sort variant (different gap sequences)
    def shell_sort_network():
        comparators = []
        gaps = [8, 4, 2, 1]
        for gap in gaps:
            for i in range(gap, N):
                comparators.append((i - gap, i))
        return comparators
    
    sn = shell_sort_network()
    sn_clean, sn_final = remove_redundant(sn)
    if is_sorted_state(sn_final):
        start_nets.append(sn_clean)
    
    # 4. Merge sort network
    def merge_sort_network(n):
        if n <= 1:
            return []
        comparators = []
        
        def merge_sort(size, start):
            if size <= 1:
                return
            half = size // 2
            merge_sort(half, start)
            merge_sort(half, start + half)
            merge(half, start)
        
        def merge(size, start):
            if size <= 1:
                return
            for i in range(size):
                a = start + i
                b = start + i + size
                comparators.append((a, b))
            merge(size // 2, start)
            merge(size // 2, start + size)
        
        merge_sort(n, 0)
        return comparators
    
    mn = merge_sort_network(N)
    mn_clean, mn_final = remove_redundant(mn)
    if is_sorted_state(mn_final):
        start_nets.append(mn_clean)
    
    if not start_nets:
        # Fallback
        start_nets = [odd_even_network()]
    
    # Simulated annealing from each starting point
    for start_idx, start_net in enumerate(start_nets):
        if time.time() > deadline:
            break
        
        net = list(start_net)
        net, _ = remove_redundant(net)
        current_len = len(net)
        
        if current_len < best_count:
            best_net = list(net)
            best_count = current_len
        
        # SA parameters
        T_start = 1.0
        T_end = 0.01
        sa_steps = 3000
        time_per_start = (deadline - time.time()) / max(1, len(start_nets) - start_idx)
        steps_per_second = 2000  # estimate
        max_steps = int(min(sa_steps, time_per_start * steps_per_second))
        
        if max_steps < 100:
            max_steps = 100
        
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / max_steps)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            T *= cooling
            
            # Choose a random operation
            op = random.random()
            candidate = list(net)
            
            if op < 0.3 and len(candidate) > 0:
                # Delete a random comparator
                idx = random.randint(0, len(candidate) - 1)
                candidate.pop(idx)
            elif op < 0.5 and len(candidate) >= 2:
                # Swap two comparators
                idx1 = random.randint(0, len(candidate) - 1)
                idx2 = random.randint(0, len(candidate) - 1)
                candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
            elif op < 0.7 and len(candidate) > 0:
                # Move a comparator to a new position
                idx = random.randint(0, len(candidate) - 1)
                comp = candidate.pop(idx)
                new_idx = random.randint(0, len(candidate))
                candidate.insert(new_idx, comp)
            elif op < 0.85:
                # Insert a random comparator
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                pos = random.randint(0, len(candidate))
                candidate.insert(pos, (i, j))
            else:
                # Replace a random comparator with a different one
                if len(candidate) > 0:
                    idx = random.randint(0, len(candidate) - 1)
                    i = random.randint(0, N - 2)
                    j = random.randint(i + 1, N - 1)
                    candidate[idx] = (i, j)
            
            # Clean the candidate
            candidate, cand_final = remove_redundant(candidate)
            
            if not is_sorted_state(cand_final):
                # Invalid - reject
                continue
            
            cand_len = len(candidate)
            curr_len = len(net)
            
            # Accept if better, or with probability exp(-delta/T)
            if cand_len < curr_len:
                net = candidate
                if cand_len < best_count:
                    best_net = list(net)
                    best_count = cand_len
            else:
                delta = cand_len - curr_len
                if delta == 0:
                    # Same length - accept with some probability to explore
                    if random.random() < 0.3:
                        net = candidate
                elif random.random() < math.exp(-delta / max(T, 1e-10)):
                    net = candidate
        
        # Final cleanup after SA
        net, _ = remove_redundant(net)
        if len(net) < best_count:
            best_net = list(net)
            best_count = len(net)
    
    # Final greedy removal on best
    if best_net is not None:
        best_net = greedy_remove_with_states(best_net, deadline)
        if len(best_net) < best_count:
            best_count = len(best_net)
        best_net, _ = remove_redundant(best_net)
    
    if best_net is None:
        # Ultimate fallback
        best_net = odd_even_network()
    
    return best_net

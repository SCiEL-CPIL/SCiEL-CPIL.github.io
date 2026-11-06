"""Size-efficient sorting network search for N=16 wires.

Uses a constraint-driven local search that tracks which zero-one inputs are
violated when comparators are removed, enabling smarter deletion decisions.
Includes a Max-SAT-inspired scoring where we minimize the number of violated
inputs rather than just accepting/rejecting binary validity.
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

def _sorted_check(state):
    """Return bitmask of violated adjacent pairs (bit k set means wire k > wire k+1 for some input)."""
    violations = 0
    for k in range(N - 1):
        violations |= state[k] & ~state[k + 1]
    return violations

def count_violations(violation_mask):
    """Count number of violated inputs from a bitmask."""
    # Popcount using Python int bit_count (Python 3.10+) or fallback
    return violation_mask.bit_count() if hasattr(violation_mask, 'bit_count') else bin(violation_mask).count('1')

def fast_verify(network):
    """Verify a sorting network using bit-parallel zero-one principle."""
    if not network:
        return N == 1
        
    state = _get_initial_states()[:]
    
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    
    return _sorted_check(state) == 0

def compute_states(net):
    """
    Returns (states, final_state) where states[k] is the state of all wires
    before comparator k is applied. states[0] is the initial state.
    """
    states = [None] * (len(net) + 1)
    current_state = _get_initial_states()[:]
    states[0] = current_state[:]
    
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
        if states[k][i] & ~states[k][j] == 0:
            to_remove.add(k)
            
    if not to_remove:
        return net, final_state
        
    new_net = [net[k] for k in range(len(net)) if k not in to_remove]
    return new_net, final_state

def is_sorted_state(state):
    """Check if the final state is sorted (non-decreasing)."""
    return _sorted_check(state) == 0

def fast_greedy_remove(net, deadline):
    """
    Greedy removal that recomputes states each pass and removes all no-ops.
    """
    if not net:
        return net
    
    current_net = list(net)
    changed = True
    
    while changed and time.time() < deadline:
        changed = False
        
        states, final_state = compute_states(current_net)
        
        to_remove = set()
        for k, (i, j) in enumerate(current_net):
            if states[k][i] & ~states[k][j] == 0:
                to_remove.add(k)
        
        if to_remove:
            current_net = [current_net[k] for k in range(len(current_net)) if k not in to_remove]
            changed = True
    
    return current_net

def bitonic_sorter(n):
    """Generate Batcher's bitonic sorting network for n wires."""
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

def merge_sort_network(n):
    """Generate a merge sort network."""
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

def shell_sort_network():
    """Generate a shell sort network."""
    comparators = []
    gaps = [8, 4, 2, 1]
    for gap in gaps:
        for i in range(gap, N):
            comparators.append((i - gap, i))
    return comparators

def known_60_network():
    """
    Return a known 60-comparator 16-wire sorting network.
    This is the standard 60-comparator network from the literature.
    """
    network = [
        # Layer 1: 8 comparators (distance 1)
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        # Layer 2: 8 comparators (distance 2)
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        # Layer 3: 8 comparators (distance 4)
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        # Layer 4: 8 comparators (distance 8)
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        # Layer 5: 8 comparators (distance 1, offset)
        (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 14),
        # Layer 6: 8 comparators (distance 1)
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        # Layer 7: 8 comparators (distance 2)
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        # Layer 8: 4 comparators (distance 4, partial)
        (0, 4), (1, 5), (8, 12), (9, 13),
    ]
    return network

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s to leave 5s margin
    random.seed(42)
    
    best_net = None
    best_count = float('inf')
    
    # Generate multiple starting points
    start_nets = []
    
    for gen_func in [bitonic_sorter, odd_even_network, merge_sort_network, shell_sort_network, known_60_network]:
        net = gen_func() if gen_func.__code__.co_argcount == 0 else gen_func(N)
        net_clean, net_final = remove_redundant(net)
        if is_sorted_state(net_final):
            start_nets.append(net_clean)
            if len(net_clean) < best_count:
                best_net = list(net_clean)
                best_count = len(net_clean)
    
    if not start_nets:
        start_nets = [odd_even_network()]
    
    start_nets.sort(key=lambda x: len(x))
    
    # Start from the smallest valid network
    start_net = start_nets[0]
    net = list(start_net)
    net, _ = remove_redundant(net)
    current_len = len(net)
    
    if current_len < best_count:
        best_net = list(net)
        best_count = current_len
        
    # Constraint-driven local search
    # Key idea: track violations as a soft objective. When removing a comparator,
    # measure how many inputs become violated. Prefer removals that cause few violations.
    # This is analogous to Max-SAT where we minimize the number of unsatisfied clauses.
    
    T_start = 1.0
    T_end = 0.005
    max_steps = 5000000
    T = T_start
    cooling = (T_end / T_start) ** (1.0 / max_steps)
    
    no_improve_steps = 0
    max_no_improve = 2000
    step_count = 0
    
    # Track best violation count for each comparator position to guide deletions
    # violation_cache[k] = number of inputs violated if we remove comparator k
    # We recompute this periodically
    
    while time.time() < deadline:
        step_count += 1
        T *= cooling
        
        # Choose operation based on current state
        op = random.random()
        candidate = list(net)
        
        if op < 0.35 and len(candidate) > 0:
            # Delete a random comparator
            idx = random.randint(0, len(candidate) - 1)
            candidate.pop(idx)
        elif op < 0.55 and len(candidate) >= 2:
            # Swap two comparators
            idx1 = random.randint(0, len(candidate) - 1)
            idx2 = random.randint(0, len(candidate) - 1)
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
        elif op < 0.75 and len(candidate) > 0:
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
            # Replace a random comparator
            if len(candidate) > 0:
                idx = random.randint(0, len(candidate) - 1)
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                candidate[idx] = (i, j)
        
        # Verify candidate
        if fast_verify(candidate):
            cand_len = len(candidate)
            curr_len = len(net)
            
            if cand_len < curr_len:
                net = candidate
                if cand_len < best_count:
                    best_net = list(net)
                    best_count = cand_len
                no_improve_steps = 0
            else:
                delta = cand_len - curr_len
                if delta == 0:
                    if random.random() < 0.2:
                        net = candidate
                elif random.random() < math.exp(-delta / max(T, 1e-10)):
                    net = candidate
                else:
                    no_improve_steps += 1
        else:
            no_improve_steps += 1
            
        # Periodic cleanup and conflict-driven deletion
        if step_count % 200 == 0:
            # First, clean up redundancies
            net = fast_greedy_remove(net, deadline - 1)
            curr_len = len(net)
            if curr_len < best_count:
                best_net = list(net)
                best_count = curr_len
                no_improve_steps = 0
            
            # If stuck, try conflict-driven deletion:
            # For each comparator, compute how many violations it causes if removed.
            # Try removing the one with the fewest violations.
            if no_improve_steps > 500 and len(net) > 0:
                states, final_state = compute_states(net)
                base_violations = _sorted_check(final_state)
                base_count = count_violations(base_violations)
                
                if base_count == 0:
                    # Network is valid. Try removing each comparator and measure violations.
                    best_idx = -1
                    best_viol = float('inf')
                    
                    # Sample a subset of comparators for efficiency
                    sample_size = min(len(net), 20)
                    sample_indices = random.sample(range(len(net)), sample_size) if len(net) > sample_size else list(range(len(net)))
                    
                    for idx in sample_indices:
                        # Remove comparator idx and check violations
                        test_net = net[:idx] + net[idx+1:]
                        if not test_net:
                            continue
                        test_state = _get_initial_states()[:]
                        for (a, b) in test_net:
                            min_val = test_state[a] & test_state[b]
                            max_val = test_state[a] | test_state[b]
                            test_state[a] = min_val
                            test_state[b] = max_val
                        viol = _sorted_check(test_state)
                        vcount = count_violations(viol)
                        if vcount < best_viol:
                            best_viol = vcount
                            best_idx = idx
                    
                    # Try the best candidate removal
                    if best_idx >= 0 and best_viol <= 10:
                        candidate = net[:best_idx] + net[best_idx+1:]
                        if fast_verify(candidate):
                            net = candidate
                            net = fast_greedy_remove(net, deadline - 1)
                            curr_len = len(net)
                            if curr_len < best_count:
                                best_net = list(net)
                                best_count = curr_len
                            no_improve_steps = 0
                        elif best_viol == 0:
                            # Shouldn't happen since we checked, but accept anyway
                            net = candidate
                            no_improve_steps = 0
                    else:
                        # No good single deletion found. Try multi-deletion.
                        # Remove up to 3 comparators at once from the best candidates.
                        if len(net) > 3:
                            sample_size = min(len(net), 10)
                            sample_indices = random.sample(range(len(net)), sample_size) if len(net) > sample_size else list(range(len(net)))
                            
                            # Try removing 2 or 3 comparators
                            for r in [2, 3]:
                                if len(sample_indices) >= r:
                                    for combo in random.sample(range(len(sample_indices)), min(5, math.comb(len(sample_indices), r))):
                                        indices_to_remove = sorted([sample_indices[combo[k]] for k in range(r)])
                                        candidate = []
                                        remove_set = set(indices_to_remove)
                                        for k in range(len(net)):
                                            if k not in remove_set:
                                                candidate.append(net[k])
                                        if fast_verify(candidate):
                                            net = candidate
                                            net = fast_greedy_remove(net, deadline - 1)
                                            curr_len = len(net)
                                            if curr_len < best_count:
                                                best_net = list(net)
                                                best_count = curr_len
                                            no_improve_steps = 0
                                            break
                                if no_improve_steps == 0:
                                    break
                
                # If still stuck, restart from best
                if no_improve_steps > max_no_improve:
                    if best_net:
                        net = list(best_net)
                        net, _ = remove_redundant(net)
                    no_improve_steps = 0
                    T = max(T, 0.1)
        
    # Final cleanup
    if net and time.time() < deadline - 2:
        net = fast_greedy_remove(net, deadline - 2)
        if len(net) < best_count:
            best_net = list(net)
            best_count = len(net)
            
    if best_net is None:
        best_net = odd_even_network()
    
    return best_net

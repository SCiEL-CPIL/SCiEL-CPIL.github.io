"""Size-efficient sorting network search for N=16 wires.

Combines the Simulated Annealing approach of the current program with the 
hardcoded 60-comparator network from Inspiration 1 and a faster search loop 
that separates high-frequency verification from low-frequency redundancy cleanup.
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
    if not network:
        # Empty network is only valid if N=1, which is false here
        return N == 1
        
    state = _get_initial_states()[:]
    
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    
    # Unrolled check for N=16 to reduce loop overhead
    if (state[0] & ~state[1]) or (state[1] & ~state[2]) or (state[2] & ~state[3]) or (state[3] & ~state[4]) or \
       (state[4] & ~state[5]) or (state[5] & ~state[6]) or (state[6] & ~state[7]) or (state[7] & ~state[8]) or \
       (state[8] & ~state[9]) or (state[9] & ~state[10]) or (state[10] & ~state[11]) or (state[11] & ~state[12]) or \
       (state[12] & ~state[13]) or (state[13] & ~state[14]) or (state[14] & ~state[15]):
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

def known_60_network():
    """
    Return a known 60-comparator 16-wire sorting network.
    Source: Inspiration 1. Needs verification.
    """
    network = [
        # Phase 1: 8 comparators
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        # Phase 2: 8 comparators
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        # Phase 3: 8 comparators
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        # Phase 4: 8 comparators
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        # Phase 5: 8 comparators
        (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 14),
        # Phase 6: 8 comparators
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        # Phase 7: 8 comparators
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        # Phase 8: 4 comparators
        (0, 4), (1, 5), (8, 12), (9, 13),
    ]
    return network

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
        # state[k][i] & ~state[k][j] == 0 means i is subset of j (i <= j)
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

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s to leave 5s margin for overhead and return
    random.seed(42)
    
    best_net = None
    best_count = float('inf')
    
    # Generate multiple starting points
    start_nets = []
    
    # 1. Bitonic sorter (Likely 53 comparators)
    bn = bitonic_sorter(N)
    bn_clean, bn_final = remove_redundant(bn)
    if is_sorted_state(bn_final):
        start_nets.append(bn_clean)
        if len(bn_clean) < best_count:
            best_net = list(bn_clean)
            best_count = len(bn_clean)

    # 2. Known 60-comparator network
    kn = known_60_network()
    kn_clean, kn_final = remove_redundant(kn)
    if is_sorted_state(kn_final):
        start_nets.append(kn_clean)
        if len(kn_clean) < best_count:
            best_net = list(kn_clean)
            best_count = len(kn_clean)

    # 3. Odd-even transposition (Fallback, 120 comparators)
    oen = odd_even_network()
    oen_clean, oen_final = remove_redundant(oen)
    if is_sorted_state(oen_final):
        start_nets.append(oen_clean)
        if len(oen_clean) < best_count:
            best_net = list(oen_clean)
            best_count = len(oen_clean)

    if not start_nets:
        # Ultimate fallback if everything fails
        return odd_even_network()
    
    # Sort start nets by size, smallest first
    start_nets.sort(key=lambda x: len(x))
    
    # Select the best start net
    start_net = start_nets[0]
    net = list(start_net)
    net, _ = remove_redundant(net)
    current_len = len(net)
    
    if current_len < best_count:
        best_net = list(net)
        best_count = current_len
        
    # SA Parameters
    # Focus on reducing size. Bias towards deletion and swaps.
    T_start = 1.0
    T_end = 0.001
    
    T = T_start
    # Cooling is based on time, not fixed steps, to ensure we use all time
    # We will estimate steps per second to set cooling appropriately
    # But simpler: just use a very slow cooling or adaptive.
    # Let's use a fixed number of steps based on time remaining.
    # Assume ~5000 steps/sec.
    # In 55s, ~275,000 steps.
    max_steps_estimate = 275000
    cooling = (T_end / T_start) ** (1.0 / max_steps_estimate)
    
    no_improve_steps = 0
    max_no_improve = 500 # Lower threshold for restarts to explore more
    
    step_count = 0
    
    while time.time() < deadline:
        step_count += 1
        T *= cooling
        
        # Choose a random operation
        # Bias: 40% Delete, 30% Swap, 20% Move, 10% Insert/Replace
        op = random.random()
        candidate = list(net)
        
        if op < 0.4 and len(candidate) > 0:
            # Delete a random comparator
            idx = random.randint(0, len(candidate) - 1)
            candidate.pop(idx)
        elif op < 0.7 and len(candidate) >= 2:
            # Swap two comparators (local or global)
            # Local swap is more likely to be valid
            idx1 = random.randint(0, len(candidate) - 1)
            # 50% chance of local swap (adjacent)
            if random.random() < 0.5 and idx1 < len(candidate) - 1:
                idx2 = idx1 + 1
            else:
                idx2 = random.randint(0, len(candidate) - 1)
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
        elif op < 0.9 and len(candidate) > 0:
            # Move a comparator to a nearby position (shift)
            idx = random.randint(0, len(candidate) - 1)
            comp = candidate.pop(idx)
            # Shift by -2 to +2
            shift = random.randint(-2, 2)
            new_idx = idx + shift
            if new_idx < 0: new_idx = 0
            if new_idx > len(candidate): new_idx = len(candidate)
            candidate.insert(new_idx, comp)
        else:
            # Insert or Replace (less likely)
            if len(candidate) > 0 and random.random() < 0.5:
                # Replace
                idx = random.randint(0, len(candidate) - 1)
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                candidate[idx] = (i, j)
            else:
                # Insert
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                pos = random.randint(0, len(candidate))
                candidate.insert(pos, (i, j))
        
        # Fast verification
        if fast_verify(candidate):
            cand_len = len(candidate)
            curr_len = len(net)
            
            # Accept if better, or with probability exp(-delta/T)
            if cand_len < curr_len:
                net = candidate
                if cand_len < best_count:
                    best_net = list(net)
                    best_count = cand_len
                no_improve_steps = 0
            else:
                delta = cand_len - curr_len
                if delta == 0:
                    # Same length - accept with some probability to explore
                    if random.random() < 0.5: # Higher acceptance for same size to explore
                        net = candidate
                        no_improve_steps = 0
                elif random.random() < math.exp(-delta / max(T, 1e-10)):
                    net = candidate
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
        else:
            no_improve_steps += 1
            
        # Periodic cleanup
        # We do this more frequently to keep the network minimal
        if step_count % 50 == 0:
            net, _ = remove_redundant(net)
            curr_len = len(net)
            if curr_len < best_count:
                best_net = list(net)
                best_count = curr_len
                no_improve_steps = 0
                
            # If stuck for a long time, try a more aggressive mutation or restart
            if no_improve_steps > max_no_improve:
                # Restart from best_net
                if best_net:
                    net = list(best_net)
                    net, _ = remove_redundant(net)
                no_improve_steps = 0
                # Reset temperature to escape local minima
                T = max(T, 0.1)
        
    # Final cleanup
    if net:
        if time.time() < deadline - 1:
            net, _ = remove_redundant(net)
            if len(net) < best_count:
                best_net = list(net)
                best_count = len(net)
            
    if best_net is None:
        # Ultimate fallback
        best_net = odd_even_network()
    
    return best_net

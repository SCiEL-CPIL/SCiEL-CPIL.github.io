"""Size-efficient sorting network search for N=16 wires.

Combined approach:
- Start from the known 64-comparator Batcher bitonic sorter.
- Use fast bit-parallel verification returning a failure mask (zero-one principle).
- Use a "Remove and Repair" local search strategy:
  1. Attempt to remove a comparator.
  2. If the network becomes invalid, use the failure mask to guide the insertion 
     of a new comparator that fixes the most unsorted inputs.
  3. If the network remains valid, the removal is accepted.
- This approach is more directed than random insertion and leverages the 
  specific structure of the failures to guide the search.
"""

import time
import random

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

def fast_verify_mask(network):
    """
    Verify a sorting network.
    Returns a bitmask where bit k is 1 if input k is NOT sorted.
    Returns 0 if the network is valid.
    """
    state = _get_initial_states()[:]
    
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    
    failure_mask = 0
    for i in range(N - 1):
        # If wire i has a 1 where wire i+1 has a 0, it's unsorted
        bad_bits = state[i] & ~state[i + 1]
        failure_mask |= bad_bits
        
    return failure_mask

def fast_verify(network):
    """Boolean wrapper for fast_verify_mask."""
    return fast_verify_mask(network) == 0

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

def compute_states_and_mask(net):
    """
    Computes the failure mask and the states before each comparator.
    Returns (failure_mask, states_list).
    states_list[k] is the state before comparator k.
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
        
    failure_mask = 0
    for i in range(N - 1):
        failure_mask |= current_state[i] & ~current_state[i + 1]
        
    return failure_mask, states

def remove_redundant(net):
    """
    Removes all comparators that are no-ops.
    Returns (new_net, final_failure_mask).
    """
    if not net:
        return [], fast_verify_mask(net)
        
    failure_mask, states = compute_states_and_mask(net)
    to_remove = set()
    
    for k, (i, j) in enumerate(net):
        # Check if wire i <= wire j for all inputs before this comparator
        # If state[i] has a 1 where state[j] has a 0, the comparator does something.
        # If state[i] & ~state[j] == 0, then state[i] <= state[j] everywhere, so it's a no-op.
        if states[k][i] & ~states[k][j] == 0:
            to_remove.add(k)
            
    if not to_remove:
        return net, failure_mask
        
    new_net = [net[k] for k in range(len(net)) if k not in to_remove]
    # Re-verify the new net to get the correct final mask (should be same if valid, but safe)
    final_mask = fast_verify_mask(new_net)
    return new_net, final_mask

def construct():
    deadline = time.time() + 50  # 50s to leave 40s margin for overhead
    
    # Start from bitonic sorter (64 comparators for n=16)
    net = bitonic_sorter(N)
    best_net = list(net)
    best_count = len(net)
    
    # Initial cleanup
    net, mask = remove_redundant(net)
    if mask == 0:
        best_net = list(net)
        best_count = len(net)
    else:
        # Fallback if bitonic generation is flawed (unlikely)
        best_net = bitonic_sorter(N)
        best_count = len(best_net)
        net = list(best_net)
        net, mask = remove_redundant(net)
        if mask == 0:
            best_net = list(net)
            best_count = len(net)

    random.seed(42)
    
    # Simulated Annealing / Local Search
    # We try to reduce the count.
    # Move: Remove a random comparator.
    # If invalid, try to repair by inserting a comparator that reduces the failure mask.
    
    current_mask = mask
    current_net = net
    
    # Temperature for Simulated Annealing (accepting worse moves temporarily?)
    # For now, strict hill climbing on (count, failure_mask)
    # Objective: Minimize count, subject to failure_mask == 0.
    # If failure_mask != 0, we prioritize reducing failure_mask.
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        
        # 1. Try to remove a random comparator
        if len(current_net) > 0:
            idx = random.randint(0, len(current_net) - 1)
            candidate_net = current_net[:idx] + current_net[idx+1:]
            candidate_mask = fast_verify_mask(candidate_net)
            
            if candidate_mask == 0:
                # Valid reduction!
                current_net = candidate_net
                current_mask = 0
                # Clean up redundant
                current_net, current_mask = remove_redundant(current_net)
                if current_mask == 0 and len(current_net) < best_count:
                    best_net = list(current_net)
                    best_count = len(current_net)
                continue
            
            # Invalid: Try to repair
            # Find a comparator that, if inserted, reduces the failure mask
            # We try a few random insertions
            repaired = False
            for _ in range(10):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                pos = random.randint(0, len(candidate_net))
                repair_net = candidate_net[:pos] + [(i, j)] + candidate_net[pos:]
                repair_mask = fast_verify_mask(repair_net)
                
                # Heuristic: Accept if it fixes more failures than it creates?
                # Or simply if it reduces the total failure count (popcount)
                # popcount is expensive on big ints, but we can use bit_length or bin().count('1')
                # Better: Check if repair_mask is a subset of candidate_mask (i.e., we fixed some, broke none)
                # If (repair_mask & ~candidate_mask) == 0, we only fixed things.
                if (repair_mask & ~candidate_mask) == 0 and repair_mask != candidate_mask:
                    current_net = repair_net
                    current_mask = repair_mask
                    repaired = True
                    break
            
            if not repaired:
                # Try to just revert to previous valid state if we were valid?
                # If current_mask was 0, we should revert to current_net (which was valid)
                # But we already modified current_net? No, we used candidate_net.
                # If current_mask was 0, and we failed to repair, we should stay with current_net.
                pass
        
        # 2. Occasional "Shuffle" or "Move" to escape local minima
        if steps % 100 == 0 and len(current_net) > 0:
            # Move a comparator to a different position
            idx1 = random.randint(0, len(current_net) - 1)
            idx2 = random.randint(0, len(current_net) - 1)
            if idx1 != idx2:
                comp = current_net.pop(idx1)
                current_net.insert(idx2, comp)
                new_mask = fast_verify_mask(current_net)
                # Only accept if it doesn't break validity if we were valid, or improves mask
                if current_mask == 0:
                    if new_mask != 0:
                        # Revert
                        current_net.pop(idx2)
                        current_net.insert(idx1, comp)
                else:
                    # Accept if mask improves (subset)
                    if (new_mask & ~current_mask) == 0 and new_mask != current_mask:
                        current_mask = new_mask
                    else:
                        # Revert
                        current_net.pop(idx2)
                        current_net.insert(idx1, comp)
                        
        # 3. Periodic cleanup
        if steps % 50 == 0:
            current_net, current_mask = remove_redundant(current_net)
            if current_mask == 0 and len(current_net) < best_count:
                best_net = list(current_net)
                best_count = len(current_net)

    # Final cleanup
    best_net, _ = remove_redundant(best_net)
    
    return best_net

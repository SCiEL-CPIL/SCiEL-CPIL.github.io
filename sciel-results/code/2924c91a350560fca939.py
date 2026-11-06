"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    N = 511
    # Weights: 1 / (512 - a)
    # Indices 0..510 correspond to a=1..511
    # Let's map a to index i = a - 1
    # Weight w[i] = 1 / (512 - (i+1)) = 1 / (511 - i)
    
    # Precompute conflicts
    # conflict[i] is a bitmask of j such that gcd(i+1, j+1) > 1
    conflicts = [0] * N
    for i in range(N):
        ai = i + 1
        mask = 0
        for j in range(N):
            aj = j + 1
            if gcd(ai, aj) > 1:
                mask |= (1 << j)
        conflicts[i] = mask

    # Sort indices by weight descending (which is a descending)
    # Since w[i] = 1/(511-i), larger i means larger weight.
    # So we process indices in order N-1, N-2, ..., 0
    # But B&B is easier if we iterate through a sorted list.
    # Let's just use the natural order from high a to low a.
    # Order: 510, 509, ..., 0
    
    best_sum = 0.0
    best_set = []
    
    # Global variables for B&B
    # We need to track the set of chosen indices to reconstruct the answer
    # Or we can just track the sum and reconstruct at the end? 
    # Reconstructing from B&B path is standard.
    
    # To speed up, we can use a greedy solution as the initial lower bound
    greedy_set = []
    greedy_mask = (1 << N) - 1 # All bits set
    for i in range(N - 1, -1, -1):
        if greedy_mask & (1 << i):
            greedy_set.append(i)
            # Update mask to remove i and its conflicts
            greedy_mask &= ~conflicts[i]
            greedy_mask &= ~(1 << i)
            
    for i in greedy_set:
        best_sum += 1.0 / (511 - i)
    best_set = list(greedy_set)

    # B&B Search
    # We represent the state by:
    # - current_index: the next index to consider in the sorted order (descending)
    # - current_sum: sum of weights of chosen nodes
    # - available_mask: bitmask of nodes >= current_index that are coprime to chosen nodes
    # - chosen_indices: list of chosen indices
    
    # Optimization: Precompute suffix sums of weights for the upper bound?
    # Upper bound = current_sum + sum(weights[j] for j in available_mask)
    # This sum is expensive to calculate every node. 
    # However, available_mask changes.
    # We can precompute weight_sum_mask? No, mask is 511 bits.
    # We can iterate through set bits of available_mask to sum weights.
    # Given N=511, this is fast enough if pruning is good.
    
    def popcount_sum(mask):
        # Sum weights of set bits in mask
        s = 0.0
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += 1.0 / (511 - idx)
            m ^= lsb
        return s

    def branch_and_bound(idx, current_sum, available_mask, chosen):
        nonlocal best_sum, best_set
        
        # Deadline check
        if time.time() > deadline:
            return

        if idx < 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = list(chosen)
            return

        # Calculate upper bound
        # Sum of weights of all available nodes
        # If we assume we can take all of them, that's the upper bound.
        ub = current_sum + popcount_sum(available_mask)
        
        if ub <= best_sum:
            return

        # Find the highest weight node in available_mask
        # Since weights decrease with index, the highest index in available_mask has the highest weight.
        # Wait, index i corresponds to a=i+1. Weight is 1/(511-i).
        # Larger i -> smaller denominator -> larger weight.
        # So we want the largest index i in available_mask.
        
        if available_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = list(chosen)
            return

        # Find largest index in available_mask
        # We can use bit_length - 1
        next_idx = available_mask.bit_length() - 1
        
        # Branch 1: Include next_idx
        # Check if next_idx is actually in available_mask (it is by definition)
        new_mask = available_mask & ~(1 << next_idx)
        new_mask &= ~conflicts[next_idx]
        
        chosen.append(next_idx)
        branch_and_bound(next_idx - 1, current_sum + 1.0 / (511 - next_idx), new_mask, chosen)
        chosen.pop()
        
        # Branch 2: Exclude next_idx
        # We just clear the bit and move to next_idx - 1
        new_mask_excl = available_mask & ~(1 << next_idx)
        branch_and_bound(next_idx - 1, current_sum, new_mask_excl, chosen)

    # Initialize B&B
    # Start with all nodes available
    initial_mask = (1 << N) - 1
    # We process from highest index to lowest.
    # The recursive function handles the "next_idx" logic by picking the max available.
    # But the `idx` parameter in my draft was meant to restrict the search space to indices <= idx?
    # Actually, the `available_mask` already restricts to valid nodes.
    # The `idx` parameter is redundant if we always pick the max available.
    # Let's simplify:
    # State: available_mask, current_sum, chosen
    # Next node to consider is max(available_mask).
    
    # Re-implementing B&B without idx parameter, relying on mask
    
    def bnb(available_mask, current_sum, chosen):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return

        if available_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = list(chosen)
            return

        # Upper bound
        # Sum of weights in available_mask
        ub = current_sum + popcount_sum(available_mask)
        if ub <= best_sum:
            return

        # Pick the node with highest weight (highest index)
        next_idx = available_mask.bit_length() - 1
        
        # Branch 1: Include
        new_mask_inc = available_mask & ~(1 << next_idx)
        new_mask_inc &= ~conflicts[next_idx]
        
        chosen.append(next_idx)
        bnb(new_mask_inc, current_sum + 1.0 / (511 - next_idx), chosen)
        chosen.pop()
        
        # Branch 2: Exclude
        new_mask_excl = available_mask & ~(1 << next_idx)
        bnb(new_mask_excl, current_sum, chosen)

    bnb(initial_mask, 0.0, [])
    
    # Convert indices back to values a
    result = [i + 1 for i in best_set]
    
    return result

# EVOLVE-BLOCK-END

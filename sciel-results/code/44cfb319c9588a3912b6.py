import numpy as np
import time
import random
from collections import defaultdict

N = 16
MAX_INPUTS = 2 ** N
DEADLINE = 85.0  # Safety margin for 90s limit

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    
    # 1. Symbolic Interval-Based Representation
    # Instead of tracking 65536 individual inputs, we track "types" of inputs.
    # An input is defined by the positions of its 1s. 
    # However, for sorting networks, the relative order of 1s and 0s is what matters.
    # We can represent the state of the network for a specific "pattern" of 1s.
    # Actually, a more powerful symbolic representation is to track the set of
    # possible "ranks" or "intervals" where a 1 can be.
    
    # Let's use a "Bitmask of Bitmasks" approach or simply a set of "states".
    # A state is a bitmask of length N representing which wires have a 1.
    # There are 2^16 = 65536 possible states. This is the same as the original.
    
    # The "Interval" insight:
    # Two inputs are "equivalent" with respect to a specific comparator sequence 
    # if they produce the same output pattern.
    # More importantly, for a *partial* network, we can group inputs by their 
    # current "signature" (the bitmask of 1s).
    # If we have a set of inputs that currently have the same signature, 
    # they will evolve identically through the rest of the network.
    # We can simulate one representative for the whole group.
    
    # But initially, all 65536 inputs have unique signatures.
    # As we apply comparators, signatures might merge? 
    # No, comparators are bijective on the set of signatures.
    # (a, b) -> (min, max) is a permutation of the set of bitmasks.
    # So the number of unique signatures remains 65536.
    
    # So where is the speedup?
    # The speedup comes from *early termination* or *reduced verification*.
    # If we are building a network, we don't need to verify against all 65536 
    # if we can prove that a subset is sufficient.
    # The Zero-One Principle says we must check all 2^N.
    
    # HOWEVER, the "Interval" strategy often refers to *symbolic simulation* 
    # where we don't store the full history, but rather compute the "effect" 
    # of a comparator on a *set* of possible inputs.
    
    # Let's try a different structural change: **Greedy Layered Construction**.
    # Instead of starting with a bad network and removing, we start with an empty 
    # network and add comparators that "do the most good".
    # "Most good" is defined as reducing the number of "inversions" in the 
    # symbolic representation.
    
    # Representation:
    # We maintain a set of "active" input bitmasks.
    # Initially, all 2^16 bitmasks.
    # We want to reach the state where all bitmasks are "sorted" 
    # (i.e., all 1s are in the lower indices? No, sorted ascending means 
    # 0s then 1s. So bitmask 0b0000...0001...1 is sorted? 
    # Wait, wire 0 is the smallest index.
    # Sorted ascending: W[0] <= W[1] ...
    # If W[0]=1, W[1]=0, that's an inversion.
    # A bitmask is sorted if it is of the form 0...01...1.
    # There are N+1 such sorted masks: 0, 1, 3, 7, ..., 2^N-1.
    
    # So the goal is to map all 65536 masks to one of these 17 sorted masks.
    # Specifically, mask with k ones must map to the sorted mask with k ones 
    # (low k bits set).
    
    # We can represent the "progress" of the network by the set of masks 
    # that are *not yet* in their correct sorted form.
    
    # To make this efficient, we can use the fact that many masks are "close" 
    # to sorted.
    
    # Let's implement a Greedy Search:
    # 1. Start with empty network.
    # 2. State: A dictionary mapping each input mask to its current mask.
    #    Initially, identity.
    # 3. Goal: All masks are in the set of SortedMasks.
    # 4. Step: Try adding a comparator (i, j) to the end of the network.
    #    Calculate the new state.
    #    Count how many masks are now "sorted" (i.e., in SortedMasks).
    #    Choose the comparator that maximizes the number of newly sorted masks.
    #    If multiple, choose one that minimizes "potential" (distance to sorted).
    
    # Optimization:
    # We don't need to track all 65536 masks if we group them.
    # But since comparators are permutations, the set of masks is always a 
    # permutation of the initial set.
    # So we can just track the "error" for each mask.
    
    # Let's use a NumPy array of shape (65536,) holding the current mask value.
    # `current_masks`[k] is the current bitmask of input k.
    # Initially, current_masks = arange(65536).
    # A mask m is sorted if m == (1 << popcount(m)) - 1.
    
    # Precompute popcount and target sorted mask for each input.
    inputs = np.arange(MAX_INPUTS, dtype=np.uint32)
    
    # Popcount
    # Using bit operations or lookup.
    # Since we are in Python, we can use a precomputed array or a fast loop.
    # NumPy doesn't have popcount for uint32 directly in older versions, 
    # but we can use a trick or just precompute it once.
    
    # Precompute popcount for all 65536 values.
    # This takes < 1ms.
    popcounts = np.zeros(MAX_INPUTS, dtype=np.uint8)
    for i in range(1, MAX_INPUTS):
        popcounts[i] = popcounts[i >> 1] + (i & 1)
        
    # Target sorted mask for each input
    # target[k] = (1 << popcounts[k]) - 1
    # Note: 1 << 16 is out of uint32 range? No, 1<<16 is 65536, fits in uint32.
    # But we are using uint32 for masks.
    # (1 << 16) - 1 = 65535.
    # (1 << 15) - 1 = 32767.
    
    targets = np.zeros(MAX_INPUTS, dtype=np.uint32)
    for i in range(MAX_INPUTS):
        targets[i] = (1 << popcounts[i]) - 1
        
    # Initial state
    current_masks = inputs.copy()
    
    # Helper to apply comparator (i, j) to a mask
    # mask & (1 << i), mask & (1 << j)
    # min goes to i, max to j.
    # If bit i is 1 and bit j is 0: swap.
    # If bit i is 0 and bit j is 1: swap.
    # If both 0 or both 1: no change.
    # So: if (mask >> i) != (mask >> j), swap bits i and j.
    # Specifically, if bit i > bit j (1 > 0), swap.
    
    def apply_comp_vectorized(masks, i, j):
        # masks is an array of uint32
        # bit_i = (masks >> i) & 1
        # bit_j = (masks >> j) & 1
        # swap_mask = (bit_i ^ bit_j)
        # If swap_mask is 1, we need to swap.
        # New bit i = min(bit_i, bit_j)
        # New bit j = max(bit_i, bit_j)
        
        bit_i = (masks >> i) & 1
        bit_j = (masks >> j) & 1
        
        # We want to clear bits i and j and set them correctly.
        # mask_clear = ~(1 << i) & ~(1 << j)
        # This is tricky with vectorized uint32.
        
        # Alternative:
        # diff = bit_i - bit_j
        # If diff == 1 (i=1, j=0), we need to set i=0, j=1.
        # If diff == -1 (i=0, j=1), we need to set i=1, j=0.
        # If diff == 0, no change.
        
        # Let's use logical ops.
        # new_i = bit_i & bit_j  (AND) -> min
        # new_j = bit_i | bit_j  (OR)  -> max
        
        new_i = bit_i & bit_j
        new_j = bit_i | bit_j
        
        # We need to replace bits i and j in masks with new_i and new_j.
        # mask_without_ij = masks & ~(1 << i) & ~(1 << j)
        # But ~(1 << i) creates negative numbers in Python if not careful with types.
        # In NumPy uint32, ~ works as bitwise NOT.
        
        mask_clear_i = ~(1 << i)
        mask_clear_j = ~(1 << j)
        
        # Ensure these are uint32
        mask_clear_i = np.uint32(mask_clear_i)
        mask_clear_j = np.uint32(mask_clear_j)
        
        cleared = masks & mask_clear_i & mask_clear_j
        
        # Set new bits
        # new_i is 0 or 1. We need to shift it to position i.
        # new_j shifted to position j.
        
        # But new_i and new_j are arrays of 0/1.
        # We need to broadcast them back to the mask position.
        
        # This is getting complex to vectorize efficiently for arbitrary i, j.
        # Let's try a different approach:
        # Since we only care about the *number* of sorted masks, 
        # and the operation is a permutation, maybe we can just simulate 
        # a smaller set of "representative" inputs?
        
        # NO, the Zero-One principle requires all.
        
        # Let's stick to the vectorized simulation but optimize the bit manipulation.
        
        # Extract bits
        bi = (masks >> i) & 1
        bj = (masks >> j) & 1
        
        # Calculate new bits
        ni = bi & bj
        nj = bi | bj
        
        # Clear old bits
        # We can do this by:
        # masks = (masks & ~(1<<i)) & ~(1<<j)
        # Then OR with (ni << i) | (nj << j)
        
        # In NumPy:
        # clear_mask = ~(np.uint32(1) << i) & ~(np.uint32(1) << j)
        # This creates a scalar uint32.
        
        clear_mask = np.uint32(~((1 << i) | (1 << j)))
        # Note: ~ on Python int gives negative. 
        # np.uint32(~x) works correctly as 2's complement wrap.
        
        # However, (1 << i) might exceed uint32 if i >= 32? No, N=16.
        
        new_masks = (masks & clear_mask) | ((ni << i) | (nj << j))
        return new_masks.astype(np.uint32)

    # Check if a mask is sorted
    # mask == targets[index] ?
    # Wait, `current_masks` holds the *current* mask for each *original* input index.
    # We want to check if `current_masks[k] == targets[k]`.
    
    def count_sorted(masks):
        return np.sum(masks == targets)
    
    # Initial count
    current_count = count_sorted(current_masks)
    total_sorted = N + 1 # 17 sorted masks, but each input maps to a specific one.
    # Actually, the number of inputs that are *already* sorted is the number of 
    # inputs k such that k == targets[k].
    # k == (1 << popcount(k)) - 1.
    # This is true for k = 0, 1, 3, 7, ...
    
    # Greedy Search
    network = []
    
    # To speed up, we can precompute the effect of each comparator on the 
    # "potential" or just simulate.
    # There are C(16, 2) = 120 possible comparators.
    # Simulating 120 comparators on 65536 masks is fast.
    # 120 * (few numpy ops on 65k array) is very fast.
    
    # We will iterate until all are sorted or time runs out.
    
    while current_count < MAX_INPUTS:
        if time.time() > DEADLINE:
            break
            
        best_gain = -1
        best_comp = None
        best_new_masks = None
        
        # Try all possible comparators
        for i in range(N):
            for j in range(i + 1, N):
                # Simulate
                new_masks = apply_comp_vectorized(current_masks, i, j)
                new_count = count_sorted(new_masks)
                gain = new_count - current_count
                
                if gain > best_gain:
                    best_gain = gain
                    best_comp = (i, j)
                    best_new_masks = new_masks
                elif gain == best_gain and gain > 0:
                    # Tie-breaking: maybe choose one that minimizes "disorder"?
                    # For now, keep the first one found.
                    pass
        
        if best_gain <= 0:
            # No comparator improves the count of sorted masks.
            # We might be stuck in a local minimum where we need to "unsort" 
            # some to sort others?
            # Or we need a different metric.
            # If no gain, we can try to minimize the "distance" to sorted.
            # Distance = sum(popcount(mask XOR target))
            
            # Let's calculate distance for all comps and pick the one that 
            # minimizes total distance, even if it doesn't increase sorted count.
            min_dist = float('inf')
            best_comp_dist = None
            best_new_masks_dist = None
            
            for i in range(N):
                for j in range(i + 1, N):
                    new_masks = apply_comp_vectorized(current_masks, i, j)
                    # Calculate distance
                    # dist = sum(popcount(new_masks[k] XOR targets[k]))
                    # We can precompute a lookup table for popcount of XOR?
                    # Or just use the existing popcounts array on the XOR result.
                    xor_vals = new_masks ^ targets
                    # popcount of xor_vals
                    # We can't directly index popcounts with an array of uint32 
                    # if the values are > 65535? No, masks are 16-bit.
                    # So xor_vals are 16-bit.
                    dist = np.sum(popcounts[xor_vals])
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_comp_dist = (i, j)
                        best_new_masks_dist = new_masks
            
            if best_comp_dist is not None:
                best_comp = best_comp_dist
                best_new_masks = best_new_masks_dist
                # Accept even if gain is 0, hoping to escape local min
                # But if gain < 0, we are moving away. 
                # Let's only accept if gain >= 0 or if we are stuck.
                if best_gain < 0:
                    # We are forced to take a step that might not increase sorted count
                    # but reduces overall distance.
                    current_masks = best_new_masks
                    network.append(best_comp)
                    current_count = count_sorted(current_masks)
                    continue
                else:
                    # If we had a gain 0 option before, we didn't take it.
                    # Now we are stuck. Take the distance-minimizing one.
                    current_masks = best_new_masks
                    network.append(best_comp)
                    current_count = count_sorted(current_masks)
                    continue
            else:
                # No comparators to try?
                break
        
        # Accept the best comparator
        current_masks = best_new_masks
        network.append(best_comp)
        current_count = count_sorted(current_masks)
        
        # Optional: Local Search / Backtracking?
        # If we get stuck (gain 0 for a while), we might want to backtrack.
        # But for now, the greedy distance-minimization should handle it.
        
    # If we didn't finish, we have a partial network.
    # We must ensure it's a valid sorting network?
    # The task requires it to sort ALL inputs.
    # If we didn't reach MAX_INPUTS sorted, the network is INVALID.
    # We must fall back to a known valid network if we fail.
    
    if current_count < MAX_INPUTS:
        # Fallback to Odd-Even Transposition
        network = []
        for phase in range(N):
            if phase % 2 == 0:
                for i in range(0, N - 1, 2):
                    network.append((i, i + 1))
            else:
                for i in range(1, N - 1, 2):
                    network.append((i, i + 1))
        
        # Verify it just in case
        # (We know it's valid)
        
    return network

# EVOLVE-BLOCK-END

import time
import random
from math import gcd

def construct():
    """
    Find a nonempty subset of [1, 512] with lcm(a,b) <= 512 for every pair.
    Maximize cardinality.
    """
    start_time = time.time()
    # Safety margin: Evaluator kills at 90s. We aim to finish by 85s.
    deadline = start_time + 85.0
    
    N = 512
    
    # 1. Precompute compatibility bitmasks
    # compat_mask[i] is a bitmask where bit j-1 is 1 if lcm(i, j) <= N
    # We use 0-indexed bits: bit (j-1) corresponds to number j
    compat_mask = [0] * (N + 1)
    
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i == j:
                mask |= (1 << (j - 1))
                continue
            g = gcd(i, j)
            if (i * j) // g <= N:
                mask |= (1 << (j - 1))
        compat_mask[i] = mask
        
    # Helper to get the list of numbers from a bitmask
    def mask_to_list(mask):
        res = []
        # Iterate through set bits
        m = mask
        while m:
            # Get index of lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            res.append(idx + 1)
            m ^= lsb
        return res
    
    # Helper to check if candidate x is compatible with set S (represented by mask)
    # x is compatible with S if all elements in S are compatible with x
    # This means (compat_mask[x] & S_mask) == S_mask
    def is_compatible(x, S_mask):
        return (compat_mask[x] & S_mask) == S_mask

    def local_search(S_list):
        """
        Perform local search on the set S_list.
        Returns the improved list.
        """
        S_mask = 0
        for x in S_list:
            S_mask |= (1 << (x - 1))
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # 1. Try to add any compatible element
            # We can find all addable elements by checking each x not in S
            # Optimization: If we maintain a list of "blocked" elements, we could be faster, 
            # but N=512 is small enough to scan.
            addable = []
            for x in range(1, N + 1):
                if not (S_mask & (1 << (x - 1))):
                    if is_compatible(x, S_mask):
                        addable.append(x)
            
            if addable:
                # Add the first one found (or random one)
                # Adding one at a time allows us to update mask incrementally
                # But adding all compatible ones at once is also valid since they are mutually compatible?
                # Wait, if x and y are both compatible with S, are they compatible with each other?
                # Yes, because S is a clique. But x and y might not be compatible with each other.
                # Actually, if x is compatible with S, and y is compatible with S, 
                # it does NOT imply x and y are compatible.
                # So we must add one by one.
                x = addable[0]
                S_mask |= (1 << (x - 1))
                S_list.append(x)
                improved = True
                continue
            
            # 2. If no additions possible, try swaps
            # Strategy: For each y in S, try removing it.
            # Then check if we can add 1 or 2 elements back.
            
            # To speed up, we can precompute for each y in S, the set of elements x outside S 
            # that are compatible with S \ {y} but NOT compatible with S.
            # These are the "unblocked" elements.
            
            best_swap_gain = 0
            best_swap_y = None
            best_swap_adds = []
            
            for y in S_list:
                y_bit = 1 << (y - 1)
                S_mask_without_y = S_mask ^ y_bit
                
                # Find elements x not in S that are compatible with S_mask_without_y
                # and were NOT compatible with S_mask (otherwise they would have been added in step 1)
                # Condition: is_compatible(x, S_mask_without_y) is True
                # Note: If x was compatible with S_mask, it would have been added.
                # So we only look at x where is_compatible(x, S_mask) is False.
                
                candidates = []
                for x in range(1, N + 1):
                    x_bit = 1 << (x - 1)
                    if S_mask & x_bit:
                        continue # x is in S
                    # Check if x is compatible with S \ {y}
                    if (compat_mask[x] & S_mask_without_y) == S_mask_without_y:
                        # x is compatible with S \ {y}
                        # We don't strictly need to check if it was incompatible with S,
                        # because if it was compatible with S, it would have been added in step 1.
                        # However, due to the "add one by one" logic in step 1, 
                        # it's possible x was compatible with S but we just didn't pick it?
                        # No, step 1 iterates all x. If x was compatible, it would be in addable.
                        # So if we are here, x is NOT compatible with S.
                        # Thus, removing y unblocks x.
                        candidates.append(x)
                
                if not candidates:
                    continue
                
                # Try to add 1 candidate
                # Gain = 1
                # We want to maximize gain. 
                # If we can add 2, gain = 2.
                
                # Check for pairs in candidates that are mutually compatible
                # If we find a pair (c1, c2) compatible with each other, we can add both.
                # Note: c1 and c2 are both compatible with S \ {y}.
                # We need lcm(c1, c2) <= N.
                
                found_pair = False
                # To speed up, limit candidate check if list is huge? 
                # Max candidates is ~500. Pairs is ~125k. 
                # 512 nodes * 125k checks * 512 nodes? No.
                # Loop is over y in S (size ~20-30).
                # Inside, loop over candidates (size ~10-50).
                # Pair check is O(C^2).
                # This should be fast.
                
                for i in range(len(candidates)):
                    if found_pair:
                        break
                    c1 = candidates[i]
                    for j in range(i + 1, len(candidates)):
                        c2 = candidates[j]
                        # Check if c1 and c2 are compatible
                        if (compat_mask[c1] & (1 << (c2 - 1))) != 0:
                            # They are compatible
                            # Gain is 2
                            if 2 > best_swap_gain:
                                best_swap_gain = 2
                                best_swap_y = y
                                best_swap_adds = [c1, c2]
                                found_pair = True
                            break
                            
                # If no pair found, check if we can add 1
                if not found_pair:
                    if len(candidates) > 0:
                        if 1 > best_swap_gain:
                            best_swap_gain = 1
                            best_swap_y = y
                            best_swap_adds = [candidates[0]]
            
            if best_swap_gain > 0:
                # Perform the swap
                S_list.remove(best_swap_y)
                y_bit = 1 << (best_swap_y - 1)
                S_mask ^= y_bit
                
                for x in best_swap_adds:
                    S_mask |= (1 << (x - 1))
                    S_list.append(x)
                
                improved = True
                continue
            
            # No improvements found
            break
            
        return S_list

    best_set = []
    best_size = 0
    
    # Strategy 1: Randomized Greedy Restarts
    # This explores the space of cliques by varying the order of insertion.
    num_restarts = 0
    max_restarts = 5000
    
    # Pre-shuffled list for random restarts
    numbers = list(range(1, N + 1))
    
    # First, try some structured seeds (divisors of highly composite numbers)
    # These are likely to be close to optimal
    seed_numbers = [360, 420, 480, 240, 120, 72, 60, 48, 36, 24, 180, 210, 280, 336, 360]
    for M in seed_numbers:
        if time.time() > deadline - 10.0:
            break
        divs = [i for i in range(1, N + 1) if M % i == 0]
        # Ensure we only include numbers that are compatible with each other?
        # Divisors of M form a clique?
        # lcm(a,b) divides M. So lcm(a,b) <= M.
        # If M <= 512, then lcm(a,b) <= 512.
        # So divisors of M are always a valid clique.
        S = local_search(divs)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    # Then, do random restarts
    for _ in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        random.shuffle(numbers)
        
        S_mask = 0
        S_list = []
        
        for x in numbers:
            if is_compatible(x, S_mask):
                S_mask |= (1 << (x - 1))
                S_list.append(x)
        
        # Local search to refine
        S_refined = local_search(S_list)
        
        if len(S_refined) > best_size:
            best_size = len(S_refined)
            best_set = S_refined[:]
            # If we hit a very high score, we might want to stop early?
            # Max possible size is bounded. 
            # Let's say if we get > 30, it's good.
            
    # Final fallback
    if not best_set:
        best_set = [1]
        
    return best_set

# EVOLVE-BLOCK-END

import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 12
    m = 40
    
    # 1. Pre-compute subset relations for all possible subsets of 0..11
    # There are 2^12 = 4096 possible subsets.
    # We represent each subset as an integer bitmask.
    all_masks = range(1 << n)
    
    # is_subset[i][j] is True if mask i is a strict subset of mask j
    # To save memory and time, we can compute this on the fly or use a matrix.
    # A 4096x4096 boolean matrix is ~16MB, which is fine.
    # However, constructing it might take a second. Let's do it efficiently.
    
    # Actually, we only need to evaluate sets we pick. 
    # But for local search, we need to evaluate swaps quickly.
    # Let's precompute a list of all masks and their properties.
    
    # Optimization: 
    # We want to maximize pairs (A, B) where A < B (strict subset).
    # This is equivalent to counting comparabilities in the induced subgraph.
    
    # Strategy:
    # 1. Generate a good initial solution using a greedy approach but faster.
    # 2. Perform local search: try swapping one set in the family with one set outside.
    #    If the score increases, accept.
    
    # Precompute subset relation table for all 4096 masks?
    # 4096 * 4096 is 16M operations. In Python, this is too slow to build a full matrix explicitly in a list of lists if done naively.
    # Better: Use bit operations.
    # A < B (strict subset) iff (A & B) == A and A != B.
    
    # Let's represent the current family as a list of 40 integers (bitmasks).
    # Score calculation:
    # For each pair (i, j), check if (masks[i] & masks[j]) == masks[i] and masks[i] != masks[j] OR vice versa.
    
    # To speed up local search, we need to calculate the change in score when swapping mask X with mask Y.
    # Delta = Gain(Y) - Loss(X)
    # Gain(Y) = number of sets in current_family that are comparable to Y (excluding X if X is in family, but we are swapping so X is removed).
    # Actually, if we swap X (in family) with Y (not in family):
    # New Score = Old Score - (comparabilities of X with others) + (comparabilities of Y with others)
    # Note: "others" are the same set of m-1 sets in both cases.
    
    # Let's precompute a helper function to count comparabilities of a mask with a list of masks.
    # def count_comp(mask, family_list):
    #     count = 0
    #     for m in family_list:
    #         if m == mask: continue # distinct sets
    #         if (mask & m) == mask: count += 1
    #         elif (mask & m) == m: count += 1
    #     return count
    
    # This is O(m) per evaluation.
    # In local search, we try many swaps.
    # If we try all 4096 candidates for each of the 40 slots, that's 40 * 4096 * 40 ops ~ 6.5M ops per pass.
    # 6.5M ops in Python might take ~0.1-0.2s.
    # We can do many passes in 50s.
    
    # Initialization:
    # Greedy construction to get a decent start.
    current_family = []
    current_masks_set = set()
    
    # Start with empty, singletons, full, near-full
    # These have high connectivity.
    initial_candidates = []
    initial_candidates.append(0) # empty
    for i in range(n):
        initial_candidates.append(1 << i) # singletons
    initial_candidates.append((1 << n) - 1) # full
    for i in range(n):
        initial_candidates.append(((1 << n) - 1) ^ (1 << i)) # size 11
    
    for mask in initial_candidates:
        if len(current_family) >= m:
            break
        if mask not in current_masks_set:
            current_family.append(mask)
            current_masks_set.add(mask)
            
    # Fill remaining with greedy max gain
    # We need a pool of candidates. Let's use all masks not in current_family.
    # To be fast, we don't iterate all 4096 every time for the initial fill if we can be smarter, 
    # but m is small (40). 40 * 4096 is 160k checks. Very fast.
    
    all_possible_masks = list(range(1 << n))
    
    while len(current_family) < m:
        best_gain = -1
        best_mask = None
        # We can optimize by only checking masks that are likely good, but let's just check all for now.
        # Actually, checking all 4096 for each of the remaining ~10 slots is fine.
        for mask in all_possible_masks:
            if mask in current_masks_set:
                continue
            gain = 0
            for existing in current_family:
                if (mask & existing) == mask:
                    gain += 1
                elif (mask & existing) == existing:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
        if best_mask is not None:
            current_family.append(best_mask)
            current_masks_set.add(best_mask)
        else:
            # Fallback: add any unused mask
            for mask in all_possible_masks:
                if mask not in current_masks_set:
                    current_family.append(mask)
                    current_masks_set.add(mask)
                    break
    
    # Current score
    def calc_score(family):
        score = 0
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                a = family[i]
                b = family[j]
                if a == b: continue
                if (a & b) == a or (a & b) == b:
                    score += 1
        return score

    current_score = calc_score(current_family)
    best_score = current_score
    best_family = current_family[:]
    
    # Local Search
    # Random restarts? Or just one long hill climb?
    # Let's do hill climbing with random restarts if stuck or time allows.
    
    # To make local search faster, we can precompute the "comparability count" of each mask with the current family?
    # No, that changes as we swap.
    
    # Optimized Local Search Step:
    # For each slot i (0..39):
    #   Calculate loss_i = comparabilities of current_family[i] with all other 39 sets.
    #   For each candidate mask Y not in family:
    #      Calculate gain_Y = comparabilities of Y with all other 39 sets.
    #      If gain_Y > loss_i, swap.
    
    # This is O(m * 4096 * m) per pass = 40 * 4096 * 40 ~ 6.5M inner loop iterations.
    # In Python, 6.5M simple ops is ~0.1s.
    # We can do ~500 passes in 50s.
    
    # Let's implement a single pass function that finds the best swap.
    
    def find_best_swap(family, family_set):
        best_delta = 0
        best_swap = None # (index, new_mask)
        
        # Precompute losses for all current members
        # loss[i] = number of comparabilities between family[i] and family[j] for j!=i
        losses = [0] * len(family)
        for i in range(len(family)):
            mi = family[i]
            for j in range(len(family)):
                if i == j: continue
                mj = family[j]
                if (mi & mj) == mi or (mi & mj) == mj:
                    losses[i] += 1
        
        # Now for each slot i, find best Y
        # We can cache gain_Y for each Y? 
        # gain_Y depends on the rest of the family (family excluding i).
        # Since we only swap one at a time, "rest of family" is almost the same.
        # But it's easier to just calculate gain_Y for each Y against the current family minus family[i].
        
        # To speed up, let's iterate candidates Y.
        # For a fixed Y, gain_Y against family excluding i is:
        # (comparabilities of Y with all family) - (comparability of Y with family[i] if any)
        
        # Let's precompute comp_count[Y] = number of comparabilities of Y with ALL family members.
        # This is expensive: 4096 * 40 * 40 ~ 6.5M ops. 
        # Doing this every pass is too slow.
        
        # Alternative:
        # Just do the O(m^2 * 4096) per pass.
        
        for i in range(len(family)):
            mi = family[i]
            loss_i = losses[i]
            
            # For each candidate Y
            # We need to check if Y is in family.
            # If Y == mi, skip (no swap).
            # If Y in family, skip (would create duplicate).
            
            # To avoid iterating 4096 for each i (40*4096=163k), 
            # and inside that iterating 39 others (163k * 39 = 6.3M), 
            # it's feasible per pass.
            
            # Optimization: Only check Y if it has potential.
            # We can skip Y if it's "similar" to mi? No, hard to define.
            
            # Let's just iterate.
            for Y in all_possible_masks:
                if Y in family_set:
                    continue
                
                # Calculate gain of Y against family excluding i
                gain_y = 0
                for j in range(len(family)):
                    if i == j: continue
                    mj = family[j]
                    if (Y & mj) == Y or (Y & mj) == mj:
                        gain_y += 1
                
                delta = gain_y - loss_i
                if delta > best_delta:
                    best_delta = delta
                    best_swap = (i, Y)
                    
        return best_swap, best_delta

    # Run local search
    while time.time() < deadline:
        swap, delta = find_best_swap(current_family, current_masks_set)
        
        if swap is None or delta <= 0:
            # Stuck. Try a random perturbation?
            # Pick a random index i and a random mask Y not in family.
            # If time is low, just break.
            if random.random() < 0.5:
                # Random restart?
                # Or just accept a neutral swap? No, delta <= 0.
                # Let's try a random swap to escape local minimum.
                i = random.randint(0, m - 1)
                while True:
                    Y = random.choice(all_possible_masks)
                    if Y not in current_masks_set:
                        break
                # Perform swap regardless of score? 
                # This is simulated annealing style.
                # Let's just do it if we have time.
                current_family[i] = Y
                current_masks_set.remove(current_family[i]) # Wait, need to remove old first
                # Correction:
                old_mi = current_family[i]
                current_masks_set.remove(old_mi)
                current_family[i] = Y
                current_masks_set.add(Y)
                current_score = calc_score(current_family)
                if current_score > best_score:
                    best_score = current_score
                    best_family = current_family[:]
            else:
                break
        
        # Perform the best swap found
        i, Y = swap
        old_mi = current_family[i]
        current_masks_set.remove(old_mi)
        current_family[i] = Y
        current_masks_set.add(Y)
        current_score += delta
        
        if current_score > best_score:
            best_score = current_score
            best_family = current_family[:]
            
        # Check time
        if time.time() > deadline:
            break

    # Convert best_family (list of ints) to list of lists of ints
    result = []
    for mask in best_family:
        s = [j for j in range(n) if (mask >> j) & 1]
        result.append(s)
        
    # Verify validity
    # 1. Exactly 40 distinct subsets
    if len(result) != 40:
        # Fallback to a valid simple set if something went wrong
        result = []
        for i in range(40):
            result.append([i % 12]) # This might have duplicates, bad fallback
        # Better fallback:
        result = []
        for i in range(40):
             # Generate distinct sets
             # Just use the first 40 masks
             mask = i
             s = [j for j in range(n) if (mask >> j) & 1]
             result.append(s)
             # Ensure distinct? The masks are distinct, so sets are distinct.
    
    # Ensure types are correct
    # "integers must be Python int"
    # "sequences must be list or tuple"
    
    return result


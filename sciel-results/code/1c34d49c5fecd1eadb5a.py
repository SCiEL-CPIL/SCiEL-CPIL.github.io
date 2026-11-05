import random
import time
import math

def construct():
    N = 256
    
    # Precompute 3-APs and 4-APs as bitmasks
    # Bit i corresponds to element i+1 (1-indexed)
    # Mask for element x is 1 << (x-1)
    
    three_aps = []
    four_aps = []
    
    # 3-APs: a, a+d, a+2d
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            three_aps.append(mask)
            
    # 4-APs: a, a+d, a+2d, a+3d
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
            four_aps.append(mask)
            
    # Precompute for each element, the masks of 3-APs and 4-APs it belongs to
    elem_to_threes = [[] for _ in range(N + 1)]
    elem_to_fours = [[] for _ in range(N + 1)]
    
    for mask in three_aps:
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length()
            elem_to_threes[idx].append(mask)
            m ^= lsb
            
    for mask in four_aps:
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length()
            elem_to_fours[idx].append(mask)
            m ^= lsb

    # Function to check if adding element x to mask is valid (no 4-AP created)
    def can_add(mask, x):
        x_mask = 1 << (x - 1)
        # Check all 4-APs containing x
        for ap_mask in elem_to_fours[x]:
            # The other 3 elements
            others = ap_mask & ~x_mask
            if (mask & others) == others:
                return False
        return True

    # Function to calculate the gain in 3-APs if x is added to mask
    # Assumes x is not in mask and can_add is True
    def delta_add(mask, x):
        x_mask = 1 << (x - 1)
        gain = 0
        for ap_mask in elem_to_threes[x]:
            # Check if the other two elements are in mask
            others = ap_mask & ~x_mask
            if (mask & others) == others:
                gain += 1
        return gain

    # Function to calculate the loss in 3-APs if x is removed from mask
    # Assumes x is in mask
    def delta_remove(mask, x):
        x_mask = 1 << (x - 1)
        loss = 0
        for ap_mask in elem_to_threes[x]:
            # Check if the other two elements are in mask
            others = ap_mask & ~x_mask
            if (mask & others) == others:
                loss += 1
        return loss

    # Constructive Greedy: Build a valid set by adding elements with highest marginal gain
    def constructive_greedy():
        mask = 0
        candidates = list(range(1, N + 1))
        
        while candidates:
            best_gain = -1
            best_candidates = []
            
            # Evaluate all candidates
            # Optimization: We can't just pick the global best every time if it's slow.
            # N=256, candidates ~256. delta_add is ~1000 ops. 256*1000 = 256k ops per step.
            # 256 steps = 65M ops. Might be too slow in Python for 60s if not optimized.
            # Let's sample a subset of candidates or limit the search.
            
            # Alternative: Randomized Greedy. Pick top K candidates, choose randomly.
            # Or just pick the best among a random sample of size 10?
            
            # Let's try evaluating all, but we have a time limit.
            # We will do a "lazy" evaluation.
            
            # Actually, let's just do a simple randomized greedy:
            # Pick a random element from candidates.
            # If it's valid, add it.
            # This is O(N^2) validity checks. 256^2 * 1000 = 65M. Too slow.
            
            # Better: 
            # 1. Filter candidates to those that are valid.
            # 2. Among valid ones, pick one with high delta.
            
            valid_candidates = []
            for x in candidates:
                if can_add(mask, x):
                    # Calculate gain
                    g = delta_add(mask, x)
                    valid_candidates.append((g, x))
            
            if not valid_candidates:
                break
                
            # Sort by gain descending
            valid_candidates.sort(key=lambda item: item[0], reverse=True)
            
            # Pick from top 5 to add diversity
            k = min(5, len(valid_candidates))
            chosen = random.choice(valid_candidates[:k])
            x = chosen[1]
            
            mask |= (1 << (x - 1))
            candidates.remove(x)
            
        return mask

    # Local Search with Simulated Annealing using Delta moves
    def local_search(initial_mask, time_limit):
        current_mask = initial_mask
        current_score = sum(1 for ap in three_aps if (current_mask & ap) == ap) # Initial calc
        best_mask = current_mask
        best_score = current_score
        
        start_time = time.time()
        T = 50.0
        T_min = 1.0
        cooling = 0.9995
        
        elements = list(range(1, N + 1))
        
        while T > T_min and time.time() - start_time < time_limit:
            x = random.choice(elements)
            x_mask = 1 << (x - 1)
            
            if current_mask & x_mask:
                # Remove x
                # Check if removing x breaks any 4-AP? No, removing always valid for 4-AP constraint.
                # But it reduces score.
                new_mask = current_mask & ~x_mask
                # Delta is negative loss
                loss = delta_remove(current_mask, x)
                delta = -loss
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
            else:
                # Add x
                if can_add(current_mask, x):
                    new_mask = current_mask | x_mask
                    gain = delta_add(current_mask, x)
                    delta = gain
                    
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_mask = new_mask
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
                
            T *= cooling
            
        return best_mask, best_score

    # Run multiple constructive greedy starts and refine with SA
    best_final_mask = 0
    best_final_score = -1
    
    time_start = time.time()
    time_limit_sa = 55 # Leave 5s for overhead
    
    # We can run multiple SA instances or one long one?
    # One long SA from a good start is usually better.
    # Let's run 3 constructive greedy, pick the best, then SA.
    
    starts = []
    for i in range(3):
        if time.time() - time_start > 5:
            break
        m = constructive_greedy()
        s = sum(1 for ap in three_aps if (m & ap) == ap)
        starts.append((s, m))
        
    if not starts:
        # Fallback
        m = 0
        for x in range(1, N+1):
            if can_add(m, x):
                m |= (1 << (x-1))
        starts.append((0, m))
        
    starts.sort(key=lambda x: x[0], reverse=True)
    best_start_score, best_start_mask = starts[0]
    
    final_mask, final_score = local_search(best_start_mask, time_limit_sa)
    
    # If we have time, try another SA from the best found?
    # Or just return.
    
    # Convert mask to list
    result = []
    for i in range(1, N + 1):
        if final_mask & (1 << (i - 1)):
            result.append(i)
            
    return sorted(result)

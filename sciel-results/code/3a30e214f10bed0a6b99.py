"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import sys

def _count_comparable_pairs_masks(masks):
    """Count unordered pairs where one mask is a strict subset of the other."""
    m = len(masks)
    cnt = 0
    for i in range(m):
        mi = masks[i]
        for j in range(i + 1, m):
            mj = masks[j]
            if (mi & mj) == mi or (mi & mj) == mj:
                cnt += 1
    return cnt

def _initial_solution_greedy(n=12, m=40, time_limit=2.0):
    """
    Greedy construction: start with empty and full set, then add the set
    that maximizes new comparable pairs.
    """
    start = time.time()
    deadline = start + time_limit
    N = n
    NUM_SUBSETS = 1 << N
    all_masks = list(range(NUM_SUBSETS))
    
    F_masks = []
    used = [False] * NUM_SUBSETS
    
    # Start with empty set (0) and full set (NUM_SUBSETS - 1)
    if m >= 1:
        F_masks.append(0)
        used[0] = True
    if m >= 2:
        F_masks.append(NUM_SUBSETS - 1)
        used[NUM_SUBSETS - 1] = True
    
    # Greedily add remaining sets
    for _ in range(m - len(F_masks)):
        if time.time() > deadline:
            break
        
        best_score = -1
        best_mask = -1
        
        # To speed up, we can sample or iterate all. 4096 is small enough to iterate all.
        for msk in all_masks:
            if used[msk]:
                continue
            
            score = 0
            for f in F_masks:
                if (msk & f) == msk:  # msk subset f
                    score += 1
                elif (msk & f) == f:  # f subset msk
                    score += 1
            
            if score > best_score:
                best_score = score
                best_mask = msk
        
        if best_mask == -1:
            break
            
        F_masks.append(best_mask)
        used[best_mask] = True
        
        if time.time() > deadline:
            break
    
    # Pad if needed
    while len(F_masks) < m:
        for msk in all_masks:
            if not used[msk]:
                F_masks.append(msk)
                used[msk] = True
                break
    
    return F_masks[:m]

def _simulated_annealing(masks, time_limit=80.0):
    """
    Simulated Annealing to maximize comparable pairs.
    """
    start = time.time()
    deadline = start + time_limit
    n = 12
    m = 40
    NUM_SUBSETS = 1 << n
    
    current = masks[:]
    current_set = set(current)
    current_score = _count_comparable_pairs_masks(current)
    
    best = current[:]
    best_score = current_score
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.99995
    
    # Precompute comparability? 
    # (a & b) == a or (a & b) == b is fast.
    
    while (time.time() - start) < deadline:
        # Pick a random index to replace
        idx = random.randint(0, m - 1)
        old_mask = current[idx]
        
        # Pick a new mask
        # We want a mask not in current_set
        while True:
            new_mask = random.randint(0, NUM_SUBSETS - 1)
            if new_mask not in current_set:
                break
        
        # Calculate delta
        # Delta = (comparables of new_mask with others) - (comparables of old_mask with others)
        
        old_comp = 0
        new_comp = 0
        for i in range(m):
            if i == idx:
                continue
            other = current[i]
            
            # Check old_mask vs other
            if (old_mask & other) == old_mask or (old_mask & other) == other:
                old_comp += 1
                
            # Check new_mask vs other
            if (new_mask & other) == new_mask or (new_mask & other) == other:
                new_comp += 1
        
        delta = new_comp - old_comp
        
        # Acceptance criterion
        if delta > 0 or random.random() < math.exp(delta / T):
            # Accept
            current_set.remove(old_mask)
            current[idx] = new_mask
            current_set.add(new_mask)
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best = current[:]
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    return best, best_score

def construct():
    start = time.time()
    total_deadline = 85.0  # Safety margin before 90s kill
    
    # Step 1: Greedy initial solution
    t0 = time.time()
    init_masks = _initial_solution_greedy(n=12, m=40, time_limit=2.0)
    t1 = time.time()
    
    # Step 2: Simulated Annealing
    remaining_time = total_deadline - (time.time() - start) - 1.0  # 1s buffer for conversion
    if remaining_time > 1.0:
        final_masks, final_score = _simulated_annealing(init_masks, time_limit=remaining_time)
    else:
        final_masks = init_masks
        final_score = _count_comparable_pairs_masks(final_masks)
    
    # Convert masks to lists of ints
    result = []
    for msk in final_masks:
        s = []
        for i in range(12):
            if msk & (1 << i):
                s.append(i)
        result.append(s)
    
    return result

# EVOLVE-BLOCK-END

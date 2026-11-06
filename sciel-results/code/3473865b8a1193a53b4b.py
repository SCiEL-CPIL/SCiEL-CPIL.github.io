import time
import random
import math
import numpy as np

def _score_mask(M):
    """
    Calculate the number of distinct d > 0 such that there exists a
    with a, a+d, a+2d in the set represented by bitmask M.
    M is an integer where bit i is 1 if i is in the set.
    Indices 1..512 are used.
    """
    count = 0
    for d in range(1, 256):
        if (M & (M >> d) & (M >> 2 * d)) != 0:
            count += 1
    return count

def _score_np(elems):
    """
    Vectorized score using numpy. elems is a list/array of 32 ints.
    Returns the number of distinct d > 0 with a 3-term AP in the set.
    """
    arr = np.array(elems, dtype=np.int32)
    # For each pair (i,j) with i<j, d = arr[j] - arr[i]
    # We need a third element arr[k] = arr[j] + d = 2*arr[j] - arr[i]
    # So for each pair, check if 2*arr[j] - arr[i] is in the set.
    # Use a set for O(1) lookup
    s = set(elems)
    count = 0
    # Collect all valid d values
    found_ds = set()
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            d = arr[j] - arr[i]
            if d <= 0:
                continue
            # Check if arr[j] + d is in the set
            third = arr[j] + d
            if third in s:
                found_ds.add(d)
    return len(found_ds)

def _score_np_fast(elems):
    """
    Faster vectorized score using numpy boolean operations.
    """
    arr = np.array(elems, dtype=np.int32)
    n = len(arr)
    s = set(elems)
    found_ds = set()
    # For each d from 1 to 255, check if there exists a such that
    # a, a+d, a+2d are all in the set.
    # Equivalent: for each pair (a, a+d) in the set, check if a+2d is in the set.
    # But we want distinct d values.
    for i in range(n):
        for j in range(i + 1, n):
            d = arr[j] - arr[i]
            if d <= 0:
                continue
            third = arr[j] + d
            if third in s:
                found_ds.add(d)
    return len(found_ds)

def _score_bitset_fast(M):
    """
    Fast bitset-based score. M is an integer bitmask.
    """
    count = 0
    for d in range(1, 256):
        if (M & (M >> d) & (M >> 2 * d)) != 0:
            count += 1
    return count

def _greedy_construct():
    """
    Greedy construction: start with a random seed, then greedily add
    elements that maximize the number of new progression differences.
    """
    N = 32
    MAXV = 512
    elems = []
    mask = 0
    
    # Start with a random element
    first = random.randint(1, MAXV)
    elems.append(first)
    mask |= (1 << first)
    
    for _ in range(N - 1):
        best_elem = -1
        best_score = -1
        # Try a random sample of candidates to speed up
        candidates = random.sample(range(1, MAXV + 1), min(100, MAXV - len(elems)))
        for c in candidates:
            if (mask >> c) & 1:
                continue
            new_mask = mask | (1 << c)
            sc = _score_bitset_fast(new_mask)
            if sc > best_score:
                best_score = sc
                best_elem = c
        if best_elem == -1:
            # Fallback: pick random
            while True:
                c = random.randint(1, MAXV)
                if not ((mask >> c) & 1):
                    best_elem = c
                    break
        elems.append(best_elem)
        mask |= (1 << best_elem)
    
    return elems, mask

def _simulated_annealing(elems, mask, start_time, deadline, T_init=10.0, T_min=0.1, cooling=0.9995):
    """
    Run simulated annealing on the given state.
    Returns (best_elems, best_mask, best_score).
    """
    N = 32
    MAXV = 512
    cur_elems = list(elems)
    cur_mask = mask
    cur_score = _score_bitset_fast(cur_mask)
    best_elems = list(elems)
    best_mask = mask
    best_score = cur_score
    
    T = T_init
    rng = random.Random()
    
    while time.time() < deadline and T > T_min:
        # Pick random element to remove
        pos_remove = cur_elems[rng.randint(0, N - 1)]
        
        # Pick random element to add
        while True:
            pos_add = rng.randint(1, MAXV)
            if not ((cur_mask >> pos_add) & 1):
                break
        
        if pos_remove == pos_add:
            continue
        
        # Apply move
        new_mask = cur_mask & ~(1 << pos_remove)
        new_mask |= (1 << pos_add)
        
        new_score = _score_bitset_fast(new_mask)
        delta = new_score - cur_score
        
        if delta > 0 or rng.random() < math.exp(delta / T):
            cur_mask = new_mask
            cur_elems = list(cur_elems)
            cur_elems.remove(pos_remove)
            cur_elems.append(pos_add)
            cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                best_mask = cur_mask
                best_elems = list(cur_elems)
        
        T *= cooling
    
    return best_elems, best_mask, best_score

def _local_search_polish(elems, mask, score, deadline):
    """
    Local search polish: try all single-element swaps and keep the best improvement.
    """
    N = 32
    MAXV = 512
    cur_elems = list(elems)
    cur_mask = mask
    cur_score = score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_new_score = cur_score
        best_new_mask = cur_mask
        best_new_elems = cur_elems
        
        for i in range(N):
            if time.time() > deadline:
                break
            rem = cur_elems[i]
            rem_bit = 1 << rem
            base_mask = cur_mask & ~rem_bit
            
            for add in range(1, MAXV + 1):
                if (cur_mask >> add) & 1:
                    continue
                cand_mask = base_mask | (1 << add)
                cand_score = _score_bitset_fast(cand_mask)
                
                if cand_score > best_new_score:
                    best_new_score = cand_score
                    best_new_mask = cand_mask
                    best_new_elems = list(cur_elems)
                    best_new_elems[i] = add
                    improved = True
        
        if improved:
            cur_mask = best_new_mask
            cur_score = best_new_score
            cur_elems = best_new_elems
    
    return cur_elems, cur_mask, cur_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    N = 32
    MAXV = 512
    
    # Phase 1: Greedy construction for a good starting point
    greedy_elems, greedy_mask = _greedy_construct()
    greedy_score = _score_bitset_fast(greedy_mask)
    
    # Phase 2: Multiple SA restarts
    # Use greedy result as initial best
    best_elems = greedy_elems
    best_mask = greedy_mask
    best_score = greedy_score
    
    # Time budget: ~100s for SA, ~35s for LS
    sa_deadline = start_time + 100.0
    ls_deadline = deadline - 5.0
    
    # Do multiple SA runs with different random seeds
    num_restarts = 5
    sa_time_per_restart = (sa_deadline - start_time) / num_restarts
    
    for r in range(num_restarts):
        restart_deadline = start_time + (r + 1) * sa_time_per_restart
        if restart_deadline > sa_deadline:
            restart_deadline = sa_deadline
        
        if r == 0:
            # Use greedy result
            elems_init = greedy_elems
            mask_init = greedy_mask
        else:
            # Random restart
            elems_init = random.sample(range(1, MAXV + 1), N)
            mask_init = 0
            for e in elems_init:
                mask_init |= (1 << e)
        
        res_elems, res_mask, res_score = _simulated_annealing(
            elems_init, mask_init, start_time, restart_deadline,
            T_init=10.0, T_min=0.1, cooling=0.9995
        )
        
        if res_score > best_score:
            best_score = res_score
            best_mask = res_mask
            best_elems = res_elems
    
    # Phase 3: Local search polish on the best solution
    best_elems, best_mask, best_score = _local_search_polish(
        best_elems, best_mask, best_score, ls_deadline
    )
    
    result = list(best_elems)
    return result

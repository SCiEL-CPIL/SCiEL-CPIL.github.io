import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _fast_score(mask):
    """
    Count the number of distinct d > 0 such that there exists a, a+d, a+2d all in the set.
    Uses bit manipulation for speed.
    """
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _score_set(s):
    """Score a set by converting to mask first."""
    mask = 0
    for x in s:
        mask |= (1 << x)
    return _fast_score(mask)

def _greedy_init(seed_val):
    """
    Greedy construction with multiple strategies.
    """
    random.seed(seed_val)
    s = set()
    mask = 0
    best_score = 0
    candidates = list(range(1, 513))
    random.shuffle(candidates)
    
    for _ in range(32):
        best_elem = None
        best_new_score = best_score
        for c in candidates:
            if c in s:
                continue
            new_mask = mask | (1 << c)
            new_score = _fast_score(new_mask)
            if new_score > best_new_score:
                best_new_score = new_score
                best_elem = c
        if best_elem is not None:
            s.add(best_elem)
            mask |= (1 << best_elem)
            best_score = best_new_score
        else:
            if candidates:
                c = candidates.pop()
                s.add(c)
                mask |= (1 << c)
                best_score = _fast_score(mask)
    return s, best_score

def _local_search(s, initial_score, deadline):
    """
    First-improvement local search with single-element swaps.
    Repeatedly scans all possible swaps and accepts the first improvement found.
    """
    current_s = set(s)
    current_mask = 0
    for x in current_s:
        current_mask |= (1 << x)
    current_score = initial_score
    
    best_s = set(current_s)
    best_score = current_score
    
    # Precompute list of non-members for faster iteration
    all_nums = set(range(1, 513))
    
    while time.time() < deadline:
        improved = False
        elems = list(current_s)
        random.shuffle(elems)
        
        for i in elems:
            if time.time() > deadline:
                break
            
            # Try swapping i with each non-member j
            non_members = list(all_nums - current_s)
            random.shuffle(non_members)
            
            for j in non_members:
                if time.time() > deadline:
                    break
                
                # Simulate swap: remove i, add j
                new_mask = (current_mask & ~(1 << i)) | (1 << j)
                new_score = _fast_score(new_mask)
                
                if new_score > current_score:
                    # Accept improvement
                    current_s.remove(i)
                    current_s.add(j)
                    current_mask = new_mask
                    current_score = new_score
                    improved = True
                    
                    if current_score > best_score:
                        best_s = set(current_s)
                        best_score = current_score
                    
                    # Restart scan after improvement
                    break
            
            if improved:
                break
        
        if not improved:
            # No improvement found, break
            break
    
    return best_s, best_score

def _simulated_annealing(s, initial_score, deadline):
    """
    Simulated Annealing with single-element swap moves.
    """
    current_s = set(s)
    current_mask = 0
    for x in current_s:
        current_mask |= (1 << x)
    current_score = initial_score
    
    best_s = set(current_s)
    best_score = current_score
    
    T_start = 5.0
    T_end = 0.001
    alpha = 0.9995
    
    T = T_start
    all_nums = set(range(1, 513))
    
    while T > T_end and time.time() < deadline:
        # Pick random i in S
        i = random.choice(list(current_s))
        
        # Pick random j not in S
        non_members = list(all_nums - current_s)
        j = random.choice(non_members)
        
        # Perform swap
        new_mask = (current_mask & ~(1 << i)) | (1 << j)
        new_score = _fast_score(new_mask)
        
        delta = new_score - current_score
        
        if delta > 0:
            current_s.remove(i)
            current_s.add(j)
            current_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_s = set(current_s)
                best_score = current_score
        else:
            # Accept with probability exp(delta/T)
            if delta != 0:  # avoid log(0)
                prob = np.exp(delta / T)
            else:
                prob = 1.0
            if random.random() < prob:
                current_s.remove(i)
                current_s.add(j)
                current_mask = new_mask
                current_score = new_score
        
        T *= alpha
    
    return best_s, best_score

def _multi_start_search(deadline):
    """
    Run multiple independent searches with different random seeds.
    """
    best_s = None
    best_score = -1
    
    # Phase 1: Multiple greedy initializations
    start_time = time.time()
    num_greedy = 5
    time_per_greedy = (deadline - start_time - 10) / (num_greedy + 5)  # reserve time for SA
    
    for seed in range(num_greedy):
        if time.time() > deadline - 10:
            break
        s, score = _greedy_init(seed * 1000 + 42)
        if score > best_score:
            best_s = set(s)
            best_score = score
    
    # Phase 2: Local search on best initial
    if best_s is not None and time.time() < deadline - 5:
        s, score = _local_search(best_s, best_score, deadline - 5)
        if score > best_score:
            best_s, best_score = s, score
    
    # Phase 3: Simulated annealing on best result
    if best_s is not None and time.time() < deadline - 2:
        s, score = _simulated_annealing(best_s, best_score, deadline - 2)
        if score > best_score:
            best_s, best_score = s, score
    
    return sorted(list(best_s)) if best_s else list(range(1, 33))

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    return _multi_start_search(deadline)

# EVOLVE-BLOCK-END

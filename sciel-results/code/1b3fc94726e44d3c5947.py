import time
import random
from math import log
import itertools

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def greedy_construct(target_size, max_val_limit, deadline, seed_elements):
    """Greedy construction: start from seed, add elements that maximize score."""
    U = set(seed_elements)
    current_score = score_set(list(U))
    
    # Build a candidate pool
    candidates = set()
    for i in range(1, min(max_val_limit + 1, 2000)):
        candidates.add(i)
    
    # Add structured candidates
    for base in [2, 3, 4, 5, 7, 10, 12, 16]:
        val = 1
        while val <= max_val_limit:
            candidates.add(val)
            val *= base
    
    # Fibonacci-like
    a, b = 1, 2
    while a <= max_val_limit:
        candidates.add(a)
        a, b = b, a + b
    
    while len(U) < target_size and time.time() < deadline:
        best_candidate = None
        best_new_score = current_score
        
        # Get valid candidates not already in U
        valid_cands = [c for c in candidates if c not in U and 0 < c <= max_val_limit]
        if not valid_cands:
            break
        
        # Sample for efficiency
        sample_size = min(len(valid_cands), 200)
        sampled = random.sample(valid_cands, sample_size)
        
        for c in sampled:
            new_U = U | {c}
            new_score = score_set(list(new_U))
            if new_score > best_new_score:
                best_new_score = new_score
                best_candidate = c
        
        if best_candidate is not None:
            U.add(best_candidate)
            current_score = best_new_score
            # Expand candidate pool based on new element
            for c in list(U):
                if best_candidate + c <= max_val_limit:
                    candidates.add(best_candidate + c)
                if abs(best_candidate - c) <= max_val_limit:
                    candidates.add(abs(best_candidate - c))
        else:
            break
    
    return sorted(U), current_score

def local_search_refine(U, deadline, max_val_limit, iterations=300):
    """Local search: try to improve by swapping or adding/removing elements."""
    U = sorted(U)
    n = len(U)
    current_score = score_set(U)
    
    for _ in range(iterations):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try adding a new element
        candidates = set()
        for a in U:
            for b in U:
                candidates.add(a + b)
                candidates.add(abs(a - b))
        
        valid_cands = [c for c in candidates if c not in U and 0 < c <= max_val_limit]
        if valid_cands:
            sample_size = min(len(valid_cands), 50)
            sampled = random.sample(valid_cands, sample_size)
            for c in sampled:
                new_U = U + [c]
                new_score = score_set(new_U)
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    improved = True
                    break
        
        # Try removing an element (if set is large enough)
        if len(U) > 3 and not improved:
            elements = [e for e in U if e != 0]
            random.shuffle(elements)
            for c in elements[:15]:
                new_U = [e for e in U if e != c]
                new_score = score_set(new_U)
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    improved = True
                    break
        
        # Try swapping an element
        if not improved and len(U) > 3:
            idx = random.randint(1, len(U) - 1)
            old_val = U[idx]
            new_val = random.randint(1, max_val_limit)
            if new_val != old_val and new_val not in U:
                new_U = U.copy()
                new_U[idx] = new_val
                new_U.sort()
                new_score = score_set(new_U)
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    improved = True
        
        if not improved:
            break
    
    return U, current_score

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    best_score = -1
    best_set = [0, 1, 4, 9, 11]
    
    # Update best if initial is good
    s = score_set(best_set)
    if s > best_score:
        best_score = s
        best_set = list(best_set)
        
    # Try some structured sets
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
        [0, 1, 2, 4, 8, 16, 32, 64],
        [0, 1, 3, 7, 15, 31, 63],
        [0, 1, 4, 10, 20, 35, 56],
        [0, 1, 2, 5, 10, 20, 40, 80],
        [0, 1, 3, 6, 10, 15, 21, 28],
    ]
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Greedy construction with different seeds and target sizes
    seed_options = [
        [0, 1],
        [0, 1, 2],
        [0, 1, 3],
        [0, 1, 4],
        [0, 1, 2, 4],
        [0, 1, 3, 5],
    ]
    
    for seed in seed_options:
        if time.time() > deadline - 20:
            break
        
        for target_size in [8, 12, 16, 20]:
            if time.time() > deadline - 10:
                break
            
            U_greedy, score_greedy = greedy_construct(
                target_size, 500, deadline - 5, seed
            )
            
            if score_greedy > best_score:
                best_score = score_greedy
                best_set = list(U_greedy)
                
            # Refine with local search
            if time.time() < deadline - 5:
                U_refined, score_refined = local_search_refine(
                    U_greedy, deadline - 3, 500, iterations=200
                )
                if score_refined > best_score:
                    best_score = score_refined
                    best_set = list(U_refined)
    
    # Final refinement pass on the best set found
    if time.time() < deadline - 5:
        U_final, score_final = local_search_refine(
            best_set, deadline - 2, 500, iterations=300
        )
        if score_final > best_score:
            best_score = score_final
            best_set = list(U_final)
    
    return best_set

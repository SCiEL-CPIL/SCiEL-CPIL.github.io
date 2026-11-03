import time
from math import log
import itertools
import random

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

def local_search(U, deadline, max_iter=500):
    """Hill-climbing local search: try adding/removing/swapping elements"""
    U = sorted(set(U))
    if 0 not in U:
        return U
    best_score = score_set(U)
    best_set = list(U)
    max_val = max(U) + 20
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        
        # Try adding a new element
        candidates_to_add = [x for x in range(1, max_val + 1) if x not in U]
        if candidates_to_add:
            # Try a few random additions
            for _ in range(min(5, len(candidates_to_add))):
                if time.time() > deadline:
                    break
                new_elem = random.choice(candidates_to_add)
                new_U = sorted(set(U) | {new_elem})
                s = score_set(new_U)
                if s > best_score:
                    best_score = s
                    best_set = new_U
                    U = new_U
                    max_val = max(max_val, new_elem + 10)
                    improved = True
                    break
        
        # Try removing an element (not 0)
        if len(U) > 2:
            for _ in range(min(3, len(U) - 1)):
                if time.time() > deadline:
                    break
                idx = random.randint(1, len(U) - 1)
                new_U = sorted(set(U) - {U[idx]})
                s = score_set(new_U)
                if s > best_score:
                    best_score = s
                    best_set = new_U
                    U = new_U
                    improved = True
                    break
        
        # Try swapping an element
        if len(U) > 2:
            for _ in range(min(3, len(U) - 1)):
                if time.time() > deadline:
                    break
                idx = random.randint(1, len(U) - 1)
                old_elem = U[idx]
                new_elem = random.randint(1, max_val)
                if new_elem == old_elem or new_elem in U:
                    continue
                new_U = sorted(set(U) - {old_elem} | {new_elem})
                s = score_set(new_U)
                if s > best_score:
                    best_score = s
                    best_set = new_U
                    U = new_U
                    max_val = max(max_val, new_elem + 10)
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_set

def construct():
    deadline = time.time() + 110  # Safety margin
    
    best_score = -1
    best_set = [0, 1, 4, 9, 11]
    
    # Try some known good constructions first
    initial_candidates = [
        [0, 1, 2, 4, 8],
        [0, 1, 3, 9, 27],
        [0, 1, 4, 9, 16],
        [0, 1, 4, 16, 64],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 2, 4, 8, 16, 32],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 2, 5, 10, 20],
        [0, 1, 4, 10, 28],
        [0, 1, 3, 10, 30],
        [0, 1, 5, 15, 45],
        [0, 1, 2, 6, 12, 24],
    ]
    
    for cand in initial_candidates:
        if time.time() > deadline:
            break
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Exhaustive search for small sets (sizes 3-5)
    for n in range(3, 6):
        if time.time() > deadline:
            break
        
        max_val = 40 if n <= 4 else 30
        
        for combo in itertools.combinations(range(1, max_val + 1), n - 1):
            if time.time() > deadline:
                break
            
            U = (0,) + combo
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    # Local search to improve the best found set
    if time.time() < deadline:
        improved = local_search(best_set, deadline)
        s = score_set(improved)
        if s > best_score:
            best_score = s
            best_set = improved
    
    # Try a few more random starting points with local search
    for _ in range(5):
        if time.time() > deadline:
            break
        n = random.randint(4, 8)
        max_val = random.randint(20, 60)
        rand_set = [0] + sorted(random.sample(range(1, max_val + 1), n - 1))
        improved = local_search(rand_set, deadline, max_iter=200)
        s = score_set(improved)
        if s > best_score:
            best_score = s
            best_set = improved
    
    return best_set

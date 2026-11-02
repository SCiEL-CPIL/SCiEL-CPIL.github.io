import time
import math
import random
from itertools import combinations

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    # Compute sumset and difference set
    sumset = set()
    diffset = set()
    
    U_list = list(U_set)
    for a in U_list:
        for b in U_list:
            sumset.add(a + b)
            diffset.add(a - b)
            
    if not sumset:
        return 0.0
        
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _greedy_golomb(n):
    """Generate a Golomb ruler of order n using a greedy approach."""
    if n <= 0:
        return []
    U = [0]
    for _ in range(1, n):
        current_max = max(U)
        candidate = current_max + 1
        while True:
            # Check if adding candidate creates duplicate differences
            diffs = set()
            valid = True
            for u in U:
                d = candidate - u
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
            
            if valid:
                U.append(candidate)
                break
            candidate += 1
    return sorted(U)

def _local_search(U, max_iters=500, deadline=None, upper_bound=None):
    """Improve set U via local search."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    
    if upper_bound is None:
        upper_bound = max(best_U) + 20
        
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        
        # 1. Try moving one element
        for i in range(len(best_U)):
            if improved:
                break
            if best_U[i] == 0:
                continue
            old_val = best_U[i]
            # Search neighborhood
            candidates = set()
            for delta in range(-5, 6):
                v = old_val + delta
                if 1 <= v <= upper_bound and v != old_val:
                    candidates.add(v)
            # Random global
            for _ in range(3):
                v = random.randint(1, upper_bound)
                if v != old_val:
                    candidates.add(v)
            
            for new_val in candidates:
                if new_val in best_U:
                    continue
                new_U = best_U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-10:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
        
        if improved:
            continue
            
        # 2. Try adding an element
        add_candidates = set()
        for v in best_U:
            for delta in range(-2, 3):
                nv = v + delta
                if 1 <= nv <= upper_bound and nv not in best_U:
                    add_candidates.add(nv)
        for _ in range(5):
            nv = random.randint(1, upper_bound)
            if nv not in best_U:
                add_candidates.add(nv)
                
        for new_val in add_candidates:
            new_U = best_U[:] + [new_val]
            new_U.sort()
            s = _score_set(new_U)
            if s > best_score + 1e-10:
                best_score = s
                best_U = new_U[:]
                improved = True
                break
                
        if improved:
            continue
            
        # 3. Try removing an element
        if len(best_U) > 3:
            for i in range(1, len(best_U)):
                new_U = best_U[:i] + best_U[i+1:]
                s = _score_set(new_U)
                if s > best_score + 1e-10:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
                    
        if not improved:
            break
            
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_U = [0, 1, 4]
    best_score = _score_set(best_U)
    
    # 1. Try small exhaustive sets
    # n=3, max_val=20
    for a in range(1, 20):
        for b in range(a+1, 20):
            U = [0, a, b]
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U
                
    # 2. Generate Golomb rulers
    for n in range(3, 12):
        if time.time() > deadline - 10:
            break
        G = _greedy_golomb(n)
        s = _score_set(G)
        if s > best_score:
            best_score = s
            best_U = G[:]
            
        # Local search from Golomb ruler
        if time.time() < deadline - 5:
            U_sol, s_sol = _local_search(G, max_iters=100, deadline=deadline)
            if s_sol > best_score:
                best_score = s_sol
                best_U = U_sol
                
    # 3. Final local search on best found
    if time.time() < deadline - 5:
        best_U, best_score = _local_search(best_U, max_iters=200, deadline=deadline)
        
    return sorted(set(best_U))

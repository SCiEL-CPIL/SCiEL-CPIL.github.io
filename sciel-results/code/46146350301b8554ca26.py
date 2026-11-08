import time
import math
import numpy as np

def _score_set(U):
    """Compute the score for a finite set U of nonnegative integers containing 0."""
    if not U or 0 not in U:
        return 0.0
    U_sorted = sorted(set(U))
    n = len(U_sorted)
    max_u = U_sorted[-1]
    
    # Compute U+U (sumset)
    sum_set = set()
    for a in U_sorted:
        for b in U_sorted:
            sum_set.add(a + b)
    p = len(sum_set)
    
    # Compute U-U (difference set)
    diff_set = set()
    for a in U_sorted:
        for b in U_sorted:
            diff_set.add(a - b)
    d = len(diff_set)
    
    if p == 0 or max_u < 0:
        return 0.0
    
    denom = 2 * max_u + 1
    if denom <= 1:
        return 1.0
    
    ratio = d / p
    if ratio <= 0:
        return 0.0
    
    score = 1.0 + math.log(ratio) / math.log(denom)
    return score

def _try_add_element(U, candidate):
    """Try adding a candidate element to U and return the new score."""
    new_U = list(U) + [candidate]
    return _score_set(new_U)

def _try_remove_element(U, idx):
    """Try removing element at index idx from U and return the new score."""
    new_U = [U[i] for i in range(len(U)) if i != idx]
    if not new_U or 0 not in new_U:
        return 0.0
    return _score_set(new_U)

def _local_search(U, deadline, max_iter=500):
    """Perform local search by adding/removing elements."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    
    current_score = _score_set(U)
    max_u = U[-1] if U else 0
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        best_gain = 0
        best_move = None
        best_new_U = None
        
        # Try adding elements near existing ones or at strategic positions
        candidates = set()
        for u in U:
            for delta in [1, 2, 3, -1, -2, -3]:
                c = u + delta
                if c >= 0 and c not in U and c <= max_u + 10:
                    candidates.add(c)
        
        # Also try some larger jumps
        for u in U:
            for mult in [2, 3]:
                c = u * mult
                if c >= 0 and c not in U and c <= max_u * 2 + 5:
                    candidates.add(c)
        
        for cand in sorted(candidates):
            if time.time() > deadline:
                break
            new_score = _score_set(U + [cand])
            gain = new_score - current_score
            if gain > best_gain:
                best_gain = gain
                best_move = ('add', cand)
                best_new_U = U + [cand]
        
        # Try removing elements (except 0)
        for i in range(len(U)):
            if i == 0 and U[0] == 0:
                continue  # Don't remove 0
            if time.time() > deadline:
                break
            new_score = _score_set([U[j] for j in range(len(U)) if j != i])
            gain = new_score - current_score
            if gain > best_gain:
                best_gain = gain
                best_move = ('remove', i)
                best_new_U = [U[j] for j in range(len(U)) if j != i]
        
        if best_move is not None and best_gain > 1e-9:
            U = sorted(best_new_U)
            current_score = _score_set(U)
            max_u = U[-1] if U else 0
        else:
            break
    
    return U, current_score

def _construct_greedy(deadline):
    """Build a set greedily, trying to maximize the score."""
    U = [0]
    current_score = _score_set(U)
    
    max_u = 0
    for step in range(200):
        if time.time() > deadline:
            break
        
        best_gain = 0
        best_cand = None
        
        # Try candidates in a range
        upper_bound = max_u + 50
        for cand in range(1, upper_bound + 1):
            if cand in U:
                continue
            new_score = _score_set(U + [cand])
            gain = new_score - current_score
            if gain > best_gain:
                best_gain = gain
                best_cand = cand
        
        if best_cand is not None and best_gain > 1e-9:
            U.append(best_cand)
            U = sorted(U)
            current_score = _score_set(U)
            max_u = U[-1]
        else:
            break
    
    return U, current_score

def _random_restart(deadline, base_set):
    """Do a random restart with perturbation."""
    import random
    random.seed(42)
    
    U = list(base_set)
    current_score = _score_set(U)
    
    # Perturb: add a few random elements
    max_u = max(U) if U else 0
    for _ in range(3):
        if time.time() > deadline:
            break
        cand = random.randint(1, max_u + 20)
        if cand not in U:
            U.append(cand)
            U = sorted(U)
    
    U, current_score = _local_search(U, deadline, max_iter=100)
    return U, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with a reasonable initial set
    # [0, 1, 3] is the seed, but let's start with something slightly better
    initial_sets = [
        [0, 1, 3],
        [0, 1, 2, 4],
        [0, 1, 3, 4],
        [0, 1, 2, 3, 5],
    ]
    
    best_U = None
    best_score = 0.0
    
    # Evaluate initial sets
    for init in initial_sets:
        if time.time() > deadline:
            break
        s = _score_set(init)
        if s > best_score:
            best_score = s
            best_U = list(init)
    
    if best_U is None:
        return [0, 1, 3]
    
    # Phase 1: Greedy construction
    greedy_U, greedy_score = _construct_greedy(deadline - 30.0)
    if greedy_score > best_score:
        best_score = greedy_score
        best_U = greedy_U
    
    # Phase 2: Local search from best
    if time.time() < deadline - 20.0:
        ls_U, ls_score = _local_search(best_U, deadline - 20.0, max_iter=300)
        if ls_score > best_score:
            best_score = ls_score
            best_U = ls_U
    
    # Phase 3: A few random restarts
    restart_deadline = deadline - 10.0
    for _ in range(5):
        if time.time() > restart_deadline:
            break
        rr_U, rr_score = _random_restart(restart_deadline, best_U)
        if rr_score > best_score:
            best_score = rr_score
            best_U = rr_U
    
    # Final polish: one more local search
    if time.time() < deadline - 5.0:
        final_U, final_score = _local_search(best_U, deadline - 5.0, max_iter=200)
        if final_score > best_score:
            best_score = final_score
            best_U = final_U
    
    # Ensure 0 is in the set and return sorted
    best_U = sorted(set(best_U))
    if 0 not in best_U:
        best_U.append(0)
        best_U = sorted(best_U)
    
    return best_U

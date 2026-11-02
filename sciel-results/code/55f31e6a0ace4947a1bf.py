"""Construct a finite set U of nonnegative integers containing 0 to maximize
the sumset/difference-set exponent lower bound.

Score = 1 + log(|U-U| / |U+U|) / log(2*max(U) + 1)

Strategy:
  Use symmetric core representation U = {0} ∪ {x, M-x | x ∈ S} for chosen M and S.
  This forces U-U to span [-M, M] while U+U concentrates around 0, M, 2M.
  Then apply local search to refine.
"""
import time
import math
import random

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill


def _score(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    U_set = set(U)
    n = len(U_set)
    if n < 2:
        return 0.0
    maxU = max(U_set)
    if maxU == 0:
        return 0.0
    diff_set = set()
    sum_set = set()
    for a in U_set:
        for b in U_set:
            diff_set.add(a - b)
            sum_set.add(a + b)
    log_arg = len(diff_set) / len(sum_set)
    if log_arg <= 0:
        return 0.0
    denom = math.log(2 * maxU + 1)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(log_arg) / denom


def _score_fast(U_set):
    """Faster score computation given a set directly."""
    n = len(U_set)
    if n < 2:
        return 0.0
    maxU = max(U_set)
    if maxU == 0:
        return 0.0
    diff_set = set()
    sum_set = set()
    for a in U_set:
        for b in U_set:
            diff_set.add(a - b)
            sum_set.add(a + b)
    log_arg = len(diff_set) / len(sum_set)
    if log_arg <= 0:
        return 0.0
    denom = math.log(2 * maxU + 1)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(log_arg) / denom


def _build_symmetric(M, S):
    """Build U = {0} ∪ {x, M-x | x in S}, ensuring 0 <= x <= M-x."""
    U = {0}
    for x in S:
        if 0 <= x <= M:
            U.add(x)
            U.add(M - x)
    return U


def _greedy_symmetric(M, k, deadline):
    """Greedily construct S of size k to minimize |U+U| for given M."""
    S = []
    U = {0}
    for _ in range(k):
        best_x = None
        best_sum_size = float('inf')
        # Try candidates in order of smallness
        for x in range(1, M // 2 + 1):
            if x in S or (M - x) in S or x in U or (M - x) in U:
                continue
            new_U = U | {x, M - x}
            # Compute |U+U|
            sum_set = set()
            for a in new_U:
                for b in new_U:
                    sum_set.add(a + b)
            if len(sum_set) < best_sum_size:
                best_sum_size = len(sum_set)
                best_x = x
            if time.time() > deadline:
                break
            if best_x is not None and x > 50:
                break
        if best_x is None:
            break
        S.append(best_x)
        U = U | {best_x, M - best_x}
    return S, U


def _local_search(U, deadline, max_iter=3000):
    """Local search: try removing, adding, or swapping elements."""
    U = sorted(set(U))
    U_set = set(U)
    best_score = _score_fast(U_set)
    
    iters = 0
    while time.time() < deadline and iters < max_iter:
        iters += 1
        improved = False
        current_max = max(U_set) if U_set else 0
        
        # Try removing each element
        for i in range(len(U)):
            if time.time() > deadline:
                return U, best_score
            new_U_set = U_set - {U[i]}
            if len(new_U_set) < 2:
                continue
            s = _score_fast(new_U_set)
            if s > best_score + 1e-10:
                U_set = new_U_set
                U = sorted(U_set)
                best_score = s
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping each element with a candidate
        candidates = set()
        for x in range(1, min(200, current_max + 50)):
            if x not in U_set:
                candidates.add(x)
        for _ in range(30):
            x = random.randint(1, int(current_max * 2))
            if x not in U_set:
                candidates.add(x)
        
        for i in range(len(U)):
            if time.time() > deadline:
                return U, best_score
            for x in candidates:
                if x in U_set:
                    continue
                new_U_set = U_set - {U[i]} | {x}
                s = _score_fast(new_U_set)
                if s > best_score + 1e-10:
                    U_set = new_U_set
                    U = sorted(U_set)
                    best_score = s
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try adding a new element
        candidates_add = set()
        for x in range(1, min(300, current_max * 2 + 50)):
            if x not in U_set:
                candidates_add.add(x)
        for _ in range(50):
            x = random.randint(1, int(current_max * 3))
            if x not in U_set:
                candidates_add.add(x)
        
        best_x = None
        best_s = best_score
        for x in candidates_add:
            if time.time() > deadline:
                break
            new_U_set = U_set | {x}
            s = _score_fast(new_U_set)
            if s > best_s + 1e-10:
                best_s = s
                best_x = x
        
        if best_x is not None:
            U_set.add(best_x)
            U = sorted(U_set)
            best_score = best_s
        else:
            break
    
    return U, best_score


def construct():
    """Construct a finite set U of nonnegative integers containing 0."""
    start = time.time()
    deadline = start + 135.0  # 135s deadline for safety
    
    best_U = [0, 1, 3, 7, 12, 18]
    best_score = _score_fast(set(best_U))
    
    # Phase 1: Search over symmetric cores with various M and k
    for M in [10, 20, 30, 50, 80, 120, 200, 350, 600, 1000, 1500, 2500, 4000, 6000, 10000]:
        if time.time() > deadline - 30:
            break
        for k in range(1, 6):
            if time.time() > deadline - 30:
                break
            S, U_sym = _greedy_symmetric(M, k, deadline - 30)
            if len(U_sym) >= 2:
                s = _score_fast(U_sym)
                if s > best_score:
                    best_score = s
                    best_U = sorted(U_sym)
    
    # Phase 2: Local search refinement
    if time.time() < deadline:
        best_U, best_score = _local_search(best_U, deadline, max_iter=5000)
    
    # Ensure 0 is in the set
    best_U = sorted(set(best_U))
    if 0 not in best_U:
        best_U.insert(0, 0)
    
    return best_U

"""Construct a finite set U of nonnegative integers containing 0 to maximize
the sumset/difference-set exponent lower bound.

Score = 1 + log(|U-U| / |U+U|) / log(2*max(U) + 1)

Strategy:
  Use Golomb rulers (sets with distinct pairwise differences) as the core structure,
  then apply local search with add/remove/swap operations to optimize the score.
  Golomb rulers maximize |U-U| for a given size and max element.
  Local search then fine-tunes to also minimize |U+U| relative to |U-U|.
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

def _golomb_ruler(n, max_val=None):
    """Construct a Golomb ruler with n marks starting at 0."""
    if max_val is None:
        # Estimate max length needed for n-mark Golomb ruler
        # Known: optimal length ~ n^2 for small n
        max_val = n * n * 4
    
    marks = [0]
    used_diffs = set()
    
    # Greedy construction
    for _ in range(n - 1):
        best_x = None
        best_min_diff = -1
        # Try candidates in order
        for x in range(marks[-1] + 1, max_val + 1):
            valid = True
            min_d = float('inf')
            for m in marks:
                d = abs(x - m)
                if d in used_diffs:
                    valid = False
                    break
                min_d = min(min_d, d)
            if valid:
                # Prefer smaller x to keep max element small
                if min_d > best_min_diff or (min_d == best_min_diff and (best_x is None or x < best_x)):
                    best_min_diff = min_d
                    best_x = x
                if best_x is not None and x > marks[-1] + 50:
                    break
        if best_x is not None:
            marks.append(best_x)
            for m in marks[:-1]:
                used_diffs.add(best_x - m)
        else:
            break
    
    return marks

def _local_search(U, deadline, max_iter=2000):
    """Local search: try removing, adding, or swapping elements."""
    U = sorted(set(U))
    U_set = set(U)
    best_score = _score_fast(U_set)
    
    iters = 0
    while time.time() < deadline and iters < max_iter:
        iters += 1
        improved = False
        
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
        current_max = max(U_set) if U_set else 0
        candidates = set()
        # Small candidates
        for x in range(1, min(300, current_max + 100)):
            if x not in U_set:
                candidates.add(x)
        # Medium candidates
        for _ in range(50):
            x = random.randint(1, int(current_max * 1.5))
            if x not in U_set:
                candidates.add(x)
        # Some larger candidates
        for _ in range(20):
            x = random.randint(1, int(current_max * 3))
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
        for x in range(1, min(500, current_max * 2 + 100)):
            if x not in U_set:
                candidates_add.add(x)
        for _ in range(100):
            x = random.randint(1, int(current_max * 4))
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
            # No improvement found, break
            break
    
    return U, best_score

def _exhaustive_small_search(deadline):
    """Search over small sets (subsets of 0..30) to find good configurations."""
    best_U = [0]
    best_score = 0.0
    
    # Try all subsets of size 3-8 from 0..25
    from itertools import combinations
    
    elements = list(range(26))
    
    for size in range(3, 9):
        if time.time() > deadline:
            break
        for combo in combinations(elements, size):
            if 0 not in combo:
                continue
            U_set = set(combo)
            s = _score_fast(U_set)
            if s > best_score:
                best_score = s
                best_U = list(U_set)
    
    return best_U, best_score

def construct():
    """Construct a finite set U of nonnegative integers containing 0."""
    start = time.time()
    deadline = start + 135.0  # 135s deadline for safety
    
    best_U = [0, 1, 3, 7, 12, 18]  # A known Golomb ruler
    best_score = _score_fast(set(best_U))
    
    # Phase 1: Exhaustive search over small sets
    if time.time() < deadline - 60:
        U_small, s_small = _exhaustive_small_search(deadline - 60)
        if s_small > best_score:
            best_U = U_small
            best_score = s_small
    
    # Phase 2: Golomb ruler construction with different sizes
    for n in range(4, 20):
        if time.time() > deadline - 40:
            break
        ruler = _golomb_ruler(n)
        if len(ruler) >= 2:
            s = _score_fast(set(ruler))
            if s > best_score:
                best_score = s
                best_U = list(ruler)
    
    # Phase 3: Local search refinement
    if time.time() < deadline:
        best_U, best_score = _local_search(best_U, deadline, max_iter=5000)
    
    # Ensure 0 is in the set
    best_U = sorted(set(best_U))
    if 0 not in best_U:
        best_U.insert(0, 0)
    
    return best_U

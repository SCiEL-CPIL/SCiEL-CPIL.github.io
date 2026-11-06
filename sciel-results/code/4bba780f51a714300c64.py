import time
import random
import numpy as np
from itertools import combinations

N = 16
DEADLINE = 325.0
N_TOTAL = N * N

# Precompute all points and pairwise squared distances
_ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]
_POINT_INDEX = {p: i for i, p in enumerate(_ALL_POINTS)}
_D2 = np.zeros((N_TOTAL, N_TOTAL), dtype=np.int32)
for i, p1 in enumerate(_ALL_POINTS):
    for j, p2 in enumerate(_ALL_POINTS):
        _D2[i, j] = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def _is_isosceles_triple(a, b, c):
    """Check if points a, b, c form an isosceles triangle."""
    d1 = _D2[a, b]
    d2 = _D2[a, c]
    d3 = _D2[b, c]
    return d1 == d2 or d1 == d3 or d2 == d3

def _can_add(idx_p, chosen_set):
    """Check if point idx_p can be added to chosen without creating isosceles triangle."""
    if len(chosen_set) < 2:
        return True
    ch = list(chosen_set)
    for i in range(len(ch)):
        for j in range(i + 1, len(ch)):
            a, b = ch[i], ch[j]
            if _is_isosceles_triple(idx_p, a, b):
                return False
    return True

def _is_no_isosceles(indices):
    """Check if set of point indices forms no-isosceles set."""
    n = len(indices)
    if n < 3:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _is_isosceles_triple(indices[i], indices[j], indices[k]):
                    return False
    return True

def _is_maximal(chosen_indices):
    """Check if chosen set is maximal (no point can be added)."""
    chosen_set = set(chosen_indices)
    for i in range(N_TOTAL):
        if i in chosen_set:
            continue
        if _can_add(i, chosen_set):
            return False
    return True

def _reduce_to_minimal(chosen_indices):
    """Iteratively remove points if remaining set is still maximal."""
    S = list(chosen_indices)
    changed = True
    while changed:
        changed = False
        for i in range(len(S)):
            if len(S) - 1 < 2:
                continue
            S_test = S[:i] + S[i + 1:]
            if _is_maximal(S_test):
                S = S_test
                changed = True
                break
    return S

def _greedy_hitting_set(deadline_margin=15):
    """
    Model as minimum hitting set on hypergraph.
    
    Key insight: A maximal no-isosceles set S must "hit" every isosceles triple
    that could be formed with any two points in S. More precisely, for S to be maximal,
    every point p outside S must form an isosceles triangle with some pair (a,b) in S.
    
    We use a greedy set-cover approach:
    - Universe: all grid points that need to be "blocked"
    - Sets: for each candidate point p, the set of points it can block (i.e., points q
      such that there exist a, b in current S with (p, q, a) or (p, q, b) isosceles)
    
    Actually, let's think differently. For maximality:
    For each point p NOT in S, there must exist a, b in S such that (p, a, b) is isosceles.
    
    So we need to choose S such that:
    1. S is no-isosceles (no three points in S form isosceles)
    2. For every p not in S, exists a, b in S with (p,a,b) isosceles
    
    Greedy approach: Start with empty set, repeatedly add the point that covers
    the most uncovered "requirements" while maintaining no-isosceles property.
    """
    start_time = time.time()
    
    chosen = []
    chosen_set = set()
    
    # Track which points are currently "covered" (blocked)
    # A point q is covered if there exist a, b in chosen such that (q, a, b) is isosceles
    covered = [False] * N_TOTAL
    
    while True:
        # Check deadline
        if time.time() - start_time > DEADLINE - deadline_margin:
            return None
        
        # Find all addable points
        addable = []
        for i in range(N_TOTAL):
            if i in chosen_set:
                continue
            if _can_add(i, chosen_set):
                addable.append(i)
        
        if not addable:
            break
        
        # Check if already maximal
        if _is_maximal(chosen):
            return chosen
        
        # For each addable point, compute how many uncovered points it would cover
        best_p = None
        best_gain = -1
        
        for p in addable:
            gain = 0
            for q in range(N_TOTAL):
                if q in chosen_set:
                    continue
                if covered[q]:
                    continue
                # Check if adding p would cover q: exists a in chosen such that (q, p, a) is isosceles
                for a in chosen:
                    if _is_isosceles_triple(q, p, a):
                        gain += 1
                        break
            if gain > best_gain:
                best_gain = gain
                best_p = p
        
        if best_p is None:
            break
        
        chosen.append(best_p)
        chosen_set.add(best_p)
        
        # Update covered array
        for q in range(N_TOTAL):
            if q in chosen_set:
                continue
            if not covered[q]:
                for a in chosen:
                    if _is_isosceles_triple(q, best_p, a):
                        covered[q] = True
                        break
    
    return chosen

def _local_search_improve(S, deadline_margin=10):
    """Try to improve set S by removing one point and adding another."""
    start_time = time.time()
    S_list = list(S)
    n_S = len(S_list)
    
    for i in range(n_S):
        if time.time() - start_time > DEADLINE - deadline_margin:
            break
        
        S_without = S_list[:i] + S_list[i + 1:]
        S_without_set = set(S_without)
        
        # Try adding each possible point
        best_new = None
        best_size = n_S  # We want to find size < n_S
        
        for p_add in range(N_TOTAL):
            if p_add in S_without_set:
                continue
            
            if not _can_add(p_add, S_without_set):
                continue
            
            S_new = S_without + [p_add]
            
            if not _is_maximal(S_new):
                continue
            
            # Try to reduce
            reduced = _reduce_to_minimal(S_new)
            
            if len(reduced) < best_size:
                best_size = len(reduced)
                best_new = reduced
        
        if best_new is not None:
            return best_new
    
    return S_list

def construct():
    start_time = time.time()
    
    best_set = None
    best_size = N_TOTAL + 1
    
    rng = random.Random(42)
    
    # Strategy 1: Greedy hitting set from empty
    result = _greedy_hitting_set(deadline_margin=20)
    if result is not None:
        reduced = _reduce_to_minimal(result)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
    
    # Strategy 2: Random cores + greedy build
    for core_size in [2, 3, 4, 5]:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        n_attempts = 100
        for attempt in range(n_attempts):
            if time.time() - start_time > DEADLINE - 15:
                break
            
            core_indices = rng.sample(range(N_TOTAL), core_size)
            
            if not _is_no_isosceles(core_indices):
                continue
            
            # Greedy build from core
            chosen = list(core_indices)
            chosen_set = set(chosen)
            
            while True:
                if time.time() - start_time > DEADLINE - 10:
                    break
                
                addable = []
                for i in range(N_TOTAL):
                    if i in chosen_set:
                        continue
                    if _can_add(i, chosen_set):
                        addable.append(i)
                
                if not addable:
                    break
                
                if _is_maximal(chosen):
                    break
                
                # Pick addable point that covers most uncovered
                best_p = None
                best_gain = -1
                for p in addable:
                    gain = 0
                    for q in range(N_TOTAL):
                        if q in chosen_set:
                            continue
                        for a in chosen:
                            if _is_isosceles_triple(q, p, a):
                                gain += 1
                                break
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                
                if best_p is None:
                    break
                
                chosen.append(best_p)
                chosen_set.add(best_p)
            
            if _is_maximal(chosen):
                reduced = _reduce_to_minimal(chosen)
                if len(reduced) < best_size:
                    best_size = len(reduced)
                    best_set = reduced[:]
            
            if best_size <= 8:
                break
    
    # Strategy 3: Local search improvement
    if best_set is not None and best_size > 8:
        for _ in range(20):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            improved = _local_search_improve(best_set, deadline_margin=10)
            if improved is not None and len(improved) < best_size:
                best_size = len(improved)
                best_set = improved[:]
            else:
                break
    
    # Fallback
    if best_set is None:
        chosen = []
        for i in range(N_TOTAL):
            if _can_add(i, set(chosen)):
                chosen.append(i)
        best_set = _reduce_to_minimal(chosen)
    
    return [_ALL_POINTS[i] for i in best_set]

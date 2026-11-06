"""Combined approach: Greedy construction + Minimality Reduction + Coverage Heuristic + MILP refinement."""
import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

N = 16
DEADLINE = 320.0

def _dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def _isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _is_maximal(S, all_points):
    """
    Check if S is a maximal no-isosceles set.
    """
    S_set = set(S)
    S_list = list(S)
    n_S = len(S_list)
    
    if n_S < 2:
        return False
        
    for p in all_points:
        if p in S_set:
            continue
            
        blocked = False
        for i in range(n_S):
            if blocked:
                break
            a = S_list[i]
            da = _dist2(p, a)
            for j in range(i + 1, n_S):
                b = S_list[j]
                d_pb = _dist2(p, b)
                d_ab = _dist2(a, b)
                
                if da == d_pb or da == d_ab or d_pb == d_ab:
                    blocked = True
                    break
                    
        if not blocked:
            return False
            
    return True

def _reduce_to_minimal(S, all_points):
    """
    Iteratively remove points from S if the remaining set is still maximal.
    """
    S_list = list(S)
    changed = True
    while changed:
        changed = False
        for i in range(len(S_list)):
            p = S_list[i]
            S_test = S_list[:i] + S_list[i+1:]
            
            if len(S_test) < 2:
                continue
                
            if _is_maximal(S_test, all_points):
                S_list = S_test
                changed = True
                break
                
    return S_list

def _can_add(p, chosen):
    """Check if point p can be added to chosen without creating an isosceles triangle."""
    for i in range(len(chosen)):
        a = chosen[i]
        da = _dist2(p, a)
        for j in range(i + 1, len(chosen)):
            b = chosen[j]
            d_pb = _dist2(p, b)
            d_ab = _dist2(a, b)
            if da == d_pb or da == d_ab or d_pb == d_ab:
                return False
    return True

def _coverage_score(p, chosen, all_points, S_set):
    """
    Score how many points p would block if added to chosen.
    """
    if not _can_add(p, chosen):
        return 0
    
    new_blocked = 0
    for q in all_points:
        if q in S_set or q == p:
            continue
        already_blocked = False
        for i in range(len(chosen)):
            a = chosen[i]
            dq_a = _dist2(q, a)
            for j in range(i + 1, len(chosen)):
                b = chosen[j]
                dq_b = _dist2(q, b)
                d_ab = _dist2(a, b)
                if dq_a == dq_b or dq_a == d_ab or dq_b == d_ab:
                    already_blocked = True
                    break
            if already_blocked:
                break
        if already_blocked:
            continue
        
        for a in chosen:
            dp_a = _dist2(p, a)
            dq_a = _dist2(q, a)
            dp_q = _dist2(p, q)
            if dp_a == dq_a or dp_a == dp_q or dq_a == dp_q:
                new_blocked += 1
                break
                
    return new_blocked

def _greedy_coverage(all_points, rng):
    """
    Greedy construction using coverage heuristic.
    """
    remaining = set(all_points)
    chosen = []
    
    start = rng.choice(list(remaining))
    remaining.remove(start)
    chosen.append(start)
    S_set = {start}
    
    while remaining:
        best_p = None
        best_score = -1
        
        candidates = list(remaining)
        if len(candidates) > 200:
            candidates = rng.sample(candidates, 200)
        
        for p in candidates:
            score = _coverage_score(p, chosen, all_points, S_set)
            if score > best_score:
                best_score = score
                best_p = p
        
        if best_p is None or best_score <= 0:
            break
            
        remaining.remove(best_p)
        chosen.append(best_p)
        S_set.add(best_p)
        
    return chosen

def _milp_refinement(S, all_points, time_limit=30.0):
    """
    Try to reduce the size of S using MILP.
    Formulate as: minimize sum(x_i) subject to maximality constraints.
    """
    start_time = time.time()
    all_points_list = list(all_points)
    point_to_idx = {p: i for i, p in enumerate(all_points_list)}
    n_points = len(all_points_list)
    
    # Build constraints for maximality
    # For each point q not in S, we need at least one pair (a, b) in S such that {q, a, b} is isosceles
    # This is a disjunctive constraint, which is hard for MILP directly.
    # Instead, we use a heuristic: try to remove points one by one and check if the set remains maximal.
    # If MILP is too complex, fall back to local search.
    
    # For now, use a simpler approach: try to swap points to reduce size
    S_set = set(S)
    S_list = list(S)
    n_S = len(S_list)
    
    best_size = n_S
    best_set = S_list[:]
    
    # Try removing each point and see if we can still make it maximal by adding a different point
    for i in range(n_S):
        if time.time() - start_time > time_limit:
            break
            
        p_remove = S_list[i]
        S_without = S_list[:i] + S_list[i+1:]
        S_without_set = set(S_without)
        
        # Try to add a point that blocks the most
        best_add = None
        best_score = -1
        
        for p_add in all_points:
            if p_add in S_without_set:
                continue
            if p_add == p_remove:
                continue
            
            if not _can_add(p_add, S_without):
                continue
            
            # Score: how many points does this new set block?
            # Approximation: just check if it's maximal
            S_new = S_without + [p_add]
            
            if _is_maximal(S_new, all_points):
                # Try to reduce further
                reduced = _reduce_to_minimal(S_new, all_points)
                if len(reduced) < best_size:
                    best_size = len(reduced)
                    best_set = reduced[:]
                    break
                    
    return best_set if best_size < n_S else None

def _local_search(S, all_points, time_limit=30.0):
    """
    Try to improve S by swapping one point for another.
    """
    start_time = time.time()
    S_set = set(S)
    S_list = list(S)
    n_S = len(S_list)
    
    for i in range(n_S):
        if time.time() - start_time > time_limit:
            break
            
        p_remove = S_list[i]
        S_without = S_list[:i] + S_list[i+1:]
        S_without_set = set(S_without)
        
        for p_add in all_points:
            if p_add in S_without_set:
                continue
            if p_add == p_remove:
                continue
            
            if not _can_add(p_add, S_without):
                continue
            
            S_new = S_without + [p_add]
            
            if _is_maximal(S_new, all_points):
                reduced = _reduce_to_minimal(S_new, all_points)
                if len(reduced) < n_S:
                    return reduced
                    
    return None

def construct():
    start_time = time.time()
    
    all_points = [(x, y) for y in range(N) for x in range(N)]
    n_total = len(all_points)
    
    best_set = None
    best_size = n_total + 1
    
    rng = random.Random(42)
    
    # Strategy 1: Random greedy + reduction
    for attempt in range(200):
        if time.time() - start_time > DEADLINE:
            break
            
        order = all_points[:]
        rng.shuffle(order)
        
        chosen = []
        for p in order:
            if _can_add(p, chosen):
                chosen.append(p)
        
        reduced = _reduce_to_minimal(chosen, all_points)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
            
        if best_size <= 6:
            break
    
    # Strategy 2: Coverage greedy + reduction
    for attempt in range(100):
        if time.time() - start_time > DEADLINE:
            break
            
        chosen = _greedy_coverage(all_points, rng)
        
        if not chosen:
            continue
            
        S_set = set(chosen)
        for p in all_points:
            if p in S_set:
                continue
            if _can_add(p, chosen):
                chosen.append(p)
                S_set.add(p)
        
        reduced = _reduce_to_minimal(chosen, all_points)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
            
        if best_size <= 6:
            break
    
    # Strategy 3: Local search on best found so far
    if best_set is not None:
        improved = _local_search(best_set, all_points)
        if improved is not None and len(improved) < best_size:
            best_size = len(improved)
            best_set = improved[:]
    
    if best_set is None:
        chosen = []
        for p in all_points:
            if all(not _isosceles(p, a, b) for a, b in itertools.combinations(chosen, 2)):
                chosen.append(p)
        return chosen
    
    return best_set

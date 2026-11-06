import time
import random
import sys
from itertools import combinations

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Precompute squared distances
def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

# Precompute isosceles triangles
# For each point i, store the set of pairs (j, k) such that {i, j, k} is isosceles
# This allows fast checking: can we add point i to set S?
# Yes iff for all pairs (j,k) in iso_pairs[i], NOT (j in S and k in S)

iso_pairs = [set() for _ in range(NUM_POINTS)]

def _is_isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

# Precompute
for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        for k in range(j + 1, NUM_POINTS):
            if _is_isosceles(POINTS[i], POINTS[j], POINTS[k]):
                iso_pairs[i].add((j, k))
                iso_pairs[j].add((i, k))
                iso_pairs[k].add((i, j))

# For each point, also precompute the list of points it conflicts with
# Two points p, q conflict if there exists r such that {p,q,r} is isosceles
# This gives us a pairwise conflict graph
conflict = [set() for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    for (j, k) in iso_pairs[i]:
        conflict[i].add(j)
        conflict[i].add(k)
        conflict[j].add(i)
        conflict[j].add(k)
        conflict[k].add(i)
        conflict[k].add(j)

def _can_add(p, S_set):
    """Check if point p can be added to set S_set without creating an isosceles triangle."""
    for (j, k) in iso_pairs[p]:
        if j in S_set and k in S_set:
            return False
    return True

def _is_maximal(S_set):
    """Check if S_set is maximal: no point can be added."""
    for p in range(NUM_POINTS):
        if p not in S_set and _can_add(p, S_set):
            return False
    return True

def _is_valid(S_set):
    """Check if S_set is a valid no-isosceles set."""
    for p in S_set:
        for (j, k) in iso_pairs[p]:
            if j in S_set and k in S_set:
                return False
    return True

def _branch_and_bound(S_set, C_set, best_size, deadline):
    """
    Branch and bound to find minimum maximal no-isosceles set.
    S_set: current selected points
    C_set: candidate points that can still be added
    Returns the best (smallest) maximal set found, or None.
    """
    if time.time() > deadline:
        return None
    
    current_size = len(S_set)
    
    # If we can't add any more points, check if it's maximal
    if not C_set:
        if _is_maximal(S_set):
            if current_size < best_size:
                return set(S_set)
        return None
    
    # Lower bound: we need at least this many more points to achieve maximality
    # A point in S can "block" at most len(conflict[p]) points from being added
    # So to make the set maximal, we need to cover all non-selected points
    # This is a set cover problem, which is hard, but we can use a simple bound
    
    # Simple bound: if current_size + len(C_set) < best_size, we might still find something
    # But we need maximality, so let's be more careful
    
    # Try adding points from C_set, ordered by how many candidates they block
    # Points that block more candidates are more "efficient" for achieving maximality
    
    candidates = list(C_set)
    # Sort by number of conflicts (points they block)
    candidates.sort(key=lambda p: len(conflict[p] & C_set), reverse=True)
    
    # Also try the option of NOT adding any more points (if already maximal)
    if _is_maximal(S_set):
        if current_size < best_size:
            return set(S_set)
    
    for p in candidates:
        if time.time() > deadline:
            return None
        
        # Try adding p
        new_S = S_set | {p}
        
        # Compute new candidate set: remove p and all points that conflict with p given current S
        new_C = set(C_set)
        new_C.discard(p)
        
        # Remove points that would form isosceles triangle with p and some point in new_S
        # Actually, we need to remove points q such that q cannot be added to new_S
        # q cannot be added if there exist a, b in new_S such that {q, a, b} is isosceles
        # But we already know p can be added. For other points, we need to check.
        
        # Simpler: just filter C_set to keep only points that can be added to new_S
        new_C = set()
        for q in C_set:
            if q == p:
                continue
            if _can_add(q, new_S):
                new_C.add(q)
        
        result = _branch_and_bound(new_S, new_C, best_size, deadline)
        if result is not None:
            return result
        
        # Backtrack: try not adding p
        # We continue to next candidate
    
    return None

def _greedy_with_local_search(rng, deadline):
    """
    Greedy construction followed by local search to minimize size.
    """
    S = []
    S_set = set()
    
    # Greedy: add points in order of how many candidates they block
    candidates = list(range(NUM_POINTS))
    rng.shuffle(candidates)
    
    for p in candidates:
        if _can_add(p, S_set):
            S.append(p)
            S_set.add(p)
    
    # Local search: try to remove points and see if set remains maximal
    # If removing a point still leaves a maximal set, do it
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for p in list(S_set):
            # Try removing p
            trial_S = S_set - {p}
            if _is_maximal(trial_S):
                S_set = trial_S
                S = list(S_set)
                improved = True
                break
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    best_set = None
    best_len = float('inf')
    
    # Phase 1: Multiple greedy restarts with local search
    rng = random.Random(42)
    restart_count = 0
    while time.time() < deadline:
        current_rng = random.Random(restart_count)
        S = _greedy_with_local_search(current_rng, deadline)
        
        if len(S) < best_len:
            best_len = len(S)
            best_set = S
        
        restart_count += 1
        
        # If we've found a good solution, try branch-and-bound to improve
        if best_len <= 20 and time.time() < deadline - 60:
            # Try branch-and-bound starting from best_set
            S_set = set(best_set)
            # Compute candidates that can be added
            C_set = set()
            for p in range(NUM_POINTS):
                if p not in S_set and _can_add(p, S_set):
                    C_set.add(p)
            
            # Branch and bound to try to find smaller maximal set
            result = _branch_and_bound(S_set, C_set, best_len, deadline)
            if result is not None:
                best_set = list(result)
                best_len = len(best_set)
    
    # Convert indices back to coordinates
    return [POINTS[i] for i in best_set] if best_set else []

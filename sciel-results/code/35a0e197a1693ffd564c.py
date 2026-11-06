"""Improved program for minimum maximal no-isosceles set."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time

# Precompute isosceles conflicts for speed
# conflict[a][b] is a set of points c such that (a,b,c) is isosceles
# Note: a,b,c are indices 0..N^2-1 or tuples? Let's use tuples for clarity, 
# but for speed in local search, we'll use a dict of sets.

def _build_conflict_graph():
    points = list(itertools.product(range(N), repeat=2))
    # Map point to index
    point_to_idx = {p: i for i, p in enumerate(points)}
    # conflict[i][j] -> set of k such that points[i], points[j], points[k] is isosceles
    # We only need to check if a new point k is in conflict with existing pairs.
    # For a set S, k is valid if for all a,b in S, k is NOT in conflict[a][b].
    
    conflict = {}
    for i, a in enumerate(points):
        for j, b in enumerate(points):
            if i == j: continue
            confs = set()
            for k, c in enumerate(points):
                if k == i or k == j: continue
                # Check isosceles
                d1 = (a[0] - b[0])**2 + (a[1] - b[1])**2
                d2 = (a[0] - c[0])**2 + (a[1] - c[1])**2
                d3 = (b[0] - c[0])**2 + (b[1] - c[1])**2
                if d1 == d2 or d1 == d3 or d2 == d3:
                    confs.add(k)
            conflict[(i, j)] = confs
    return points, conflict

POINTS, CONFLICTS = _build_conflict_graph()
NUM_POINTS = len(POINTS)

def _is_independent(S_idx):
    """Check if set of indices is independent (no isosceles triples)."""
    S_list = list(S_idx)
    for i in range(len(S_list)):
        for j in range(i+1, len(S_list)):
            # If there is any k in S_list that is in conflict with i,j, it's not independent
            # Actually, we just need to check if any pair in S has a conflict *within* S.
            # conflict[(i,j)] contains k. If k is in S, then it's not independent.
            # We can just check if S intersects with conflict[(i,j)].
            # But we need to be careful: conflict[(i,j)] includes i and j? No, we excluded them in build.
            # So if S has size > 2, and any k in S is in conflict[(i,j)], it's bad.
            # Since we check all pairs, if S is {i,j,k} and k in conflict[(i,j)], it's bad.
            # Optimization: Just check if len(S & conflict[(i,j)]) > 0.
            # But S is a set of indices.
            if S_idx & CONFLICTS[(S_list[i], S_list[j])]:
                return False
    return True

def _is_maximal(S_idx):
    """Check if S_idx is maximal."""
    S_set = set(S_idx)
    for k in range(NUM_POINTS):
        if k in S_set:
            continue
        # Check if k is blocked by any pair in S
        blocked = False
        S_list = list(S_set)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                if k in CONFLICTS[(S_list[i], S_list[j])]:
                    blocked = True
                    break
            if blocked:
                break
        if not blocked:
            return False
    return True

def _greedy_init():
    """Generate a random independent set using greedy."""
    S = set()
    order = list(range(NUM_POINTS))
    random.shuffle(order)
    for k in order:
        # Check if k can be added
        can_add = True
        for i in S:
            for j in S:
                if i == j: continue
                if k in CONFLICTS[(i, j)]:
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            S.add(k)
    return S

def _local_search_reduce(S):
    """Try to remove points from S while maintaining maximality and independence."""
    S = set(S)
    # First, ensure independence (greedy ensures it, but let's be safe)
    # Then try to remove points
    changed = True
    while changed:
        changed = False
        for p in list(S):
            S_candidate = S - {p}
            if not S_candidate:
                continue
            # Check maximality of S_candidate
            if _is_maximal(S_candidate):
                S = S_candidate
                changed = True
                break # Restart loop as S changed
    return S

def construct():
    best_set = None
    best_size = float('inf')
    
    start_time = time.time()
    time_limit = 250 # Leave 50s buffer
    
    # 1. Get initial solutions via greedy
    candidates = []
    for _ in range(50):
        if time.time() - start_time > time_limit:
            break
        S = _greedy_init()
        S = _local_search_reduce(S)
        candidates.append(S)
        
    # 2. Iterated Local Search / Simulated Annealing
    # We will try to find a smaller set by swapping points.
    for _ in range(500):
        if time.time() - start_time > time_limit:
            break
        # Pick a random candidate from the best few
        candidates.sort(key=len)
        S = set(candidates[0]) # Start from best
        
        # Perform a random walk
        for step in range(20):
            if time.time() - start_time > time_limit:
                break
            # Pick a random point to remove and a random point to add
            if not S:
                break
            p_remove = random.choice(list(S))
            # Find a candidate to add
            # Try to add a point that is not in S
            S_without_p = S - {p_remove}
            # Check if S_without_p is maximal? If yes, we found a smaller set!
            if _is_maximal(S_without_p):
                S = S_without_p
                break # Exit walk, we reduced size
            
            # Otherwise, try to swap
            # Pick a random point not in S
            all_points = set(range(NUM_POINTS))
            candidates_to_add = list(all_points - S)
            if not candidates_to_add:
                break
            p_add = random.choice(candidates_to_add)
            
            S_new = S - {p_remove} | {p_add}
            
            # Check independence
            if not _is_independent(S_new):
                continue
                
            # Check maximality
            if _is_maximal(S_new):
                # Accept if same size (to explore) 
                # We don't accept if larger
                if len(S_new) <= len(S):
                    S = S_new
                    # Try to reduce again
                    S = _local_search_reduce(S)
                    break
        
        # Add result to candidates
        if S:
            candidates.append(S)
            
    # Find best
    for S in candidates:
        if len(S) < best_size:
            best_size = len(S)
            best_set = S

    if best_set is None:
        # Fallback to a simple greedy
        S = _greedy_init()
        S = _local_search_reduce(S)
        best_set = S
        
    return [POINTS[i] for i in best_set]
# EVOLVE-BLOCK-END

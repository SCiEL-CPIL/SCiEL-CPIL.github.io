import itertools
import random
import time
import numpy as np

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
POINT_TO_ID = {p: i for i, p in enumerate(POINTS)}
ID_TO_POINT = {i: p for i, p in enumerate(POINTS)}
NUM_POINTS = len(POINTS)

# Precompute squared distances
DIST = np.zeros((NUM_POINTS, NUM_POINTS), dtype=int)
for i in range(NUM_POINTS):
    x1, y1 = POINTS[i]
    for j in range(i+1, NUM_POINTS):
        x2, y2 = POINTS[j]
        d = (x1-x2)**2 + (y1-y2)**2
        DIST[i][j] = d
        DIST[j][i] = d

# Precompute conflict relations: point i conflicts with point j if they form an isosceles triangle with some third point
# Actually, we need: for a set S to be no-isosceles, no three points in S form an isosceles triangle.
# Two points can coexist unless there's a third point that would create an isosceles triangle with both.
# But the constraint is on triples, so we need to track which pairs are "compatible" given the rest of the set.

# Better approach: define for each point p, which other points q are such that {p, q} cannot both be in S
# because there exists some r such that p,q,r form an isosceles triangle AND r is also in S.
# This is tricky because it depends on the full set.

# Alternative: use the direct approach. A set S is no-isosceles iff for all distinct a,b,c in S,
# d(a,b) != d(a,c) and d(a,b) != d(b,c) and d(a,c) != d(b,c).

# For branch and bound on minimum maximal independent set:
# We decide for each point: include or exclude.
# If we include point p, then any point q that forms an isosceles triangle with p and some other included point must be excluded.
# If we exclude point p, then p must conflict with at least one included point (for maximality).

# Let's precompute: for each pair (a,b), find all c such that {a,b,c} is isosceles.
# iso_conflicts[a][b] = set of c such that triangle abc is isosceles

ISO_CONFLICTS = [set() for _ in range(NUM_POINTS)]
for a in range(NUM_POINTS):
    for b in range(a+1, NUM_POINTS):
        for c in range(b+1, NUM_POINTS):
            da = DIST[a][b]
            db = DIST[a][c]
            dc = DIST[b][c]
            if da == db or da == dc or db == dc:
                ISO_CONFLICTS[a].add(b)
                ISO_CONFLICTS[a].add(c)
                ISO_CONFLICTS[b].add(a)
                ISO_CONFLICTS[b].add(c)
                ISO_CONFLICTS[c].add(a)
                ISO_CONFLICTS[c].add(b)

# For branch and bound:
# State: (decision for points 0..k-1, current included set, current excluded set)
# When we decide to include point k:
#   - Add k to included
#   - For each already-included point i, if k is in ISO_CONFLICTS[i], we have a conflict -> invalid
#   - Actually, we need to check: does including k create an isosceles triangle with any two already-included points?
#   - For each pair (i,j) in included, check if {i,j,k} is isosceles. If yes, cannot include k.
#   - Also, any excluded point e that only conflicted with k (and no other included point) now needs another conflict.

# When we decide to exclude point k:
#   - k must be "covered" by at least one included point i such that k is in ISO_CONFLICTS[i]
#   - But wait, the conflict is about triples, not pairs. Let me reconsider.

# Actually, the maximality condition: a point p not in S cannot be added to S without creating an isosceles triangle.
# This means: for every p not in S, there exist two points a,b in S such that {p,a,b} is isosceles.

# So for branch and bound:
# - We build S incrementally.
# - When we include a point, we check it doesn't form isosceles with any two existing points.
# - When we exclude a point, we need to ensure it will be "blocked" by some future included points.
# - At the end, check all excluded points are blocked.

# This is complex. Let me use a simpler heuristic-based approach with local search improvements.

def can_add(p_id, current_ids):
    """
    Check if p_id can be added to current_ids without forming an isosceles triangle.
    """
    n = len(current_ids)
    if n < 2:
        return True
    
    for i in range(n):
        a = current_ids[i]
        d_pa = DIST[p_id][a]
        for j in range(i + 1, n):
            b = current_ids[j]
            d_pb = DIST[p_id][b]
            d_ab = DIST[a][b]
            
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return False
    return True

def is_maximal(current_ids):
    """
    Check if current_ids is a maximal no-isosceles set.
    A set is maximal if no additional point can be added.
    """
    current_set = set(current_ids)
    for p_id in range(NUM_POINTS):
        if p_id in current_set:
            continue
        if can_add(p_id, current_ids):
            return False
    return True

def greedy_fill(initial_ids, order):
    """
    Greedily add points from 'order' to 'initial_ids' to make it maximal.
    """
    current_set = list(initial_ids)
    current_set_set = set(current_set)
    
    for p_id in order:
        if p_id in current_set_set:
            continue
        if can_add(p_id, current_set):
            current_set.append(p_id)
            current_set_set.add(p_id)
            
    return current_set

def local_search(initial_ids, max_iters=20):
    """
    Perform local search by removing a random point and re-filling.
    """
    current_set = list(initial_ids)
    current_size = len(current_set)
    
    for _ in range(max_iters):
        if current_size == 0:
            break
            
        remove_idx = random.randint(0, current_size - 1)
        p_to_remove = current_set[remove_idx]
        
        temp_set = [p for i, p in enumerate(current_set) if i != remove_idx]
        
        all_points = list(range(NUM_POINTS))
        random.shuffle(all_points)
        
        new_set = greedy_fill(temp_set, all_points)
        
        if len(new_set) < current_size:
            current_set = new_set
            current_size = len(current_set)
        elif len(new_set) == current_size and random.random() < 0.2:
            current_set = new_set
            
    return current_set

def branch_and_bound_min_maximal(deadline):
    """
    Branch and bound to find minimum maximal no-isosceles set.
    """
    best_size = float('inf')
    best_set = None
    
    # Order points to try including first
    # Heuristic: points that are "more constraining" should be decided first
    # Use degree in conflict graph as heuristic
    degrees = [len(ISO_CONFLICTS[i]) for i in range(NUM_POINTS)]
    order = sorted(range(NUM_POINTS), key=lambda i: -degrees[i])
    
    # Reverse order for decisions: we decide in this order
    decision_order = order
    
    included = set()
    excluded = set()
    
    def backtrack(idx):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        if idx == NUM_POINTS:
            # Check maximality
            if is_maximal(list(included)):
                if len(included) < best_size:
                    best_size = len(included)
                    best_set = set(included)
            return
        
        p = decision_order[idx]
        
        # Option 1: Include p
        # Check if p can be added
        can_include = True
        for a in included:
            for b in included:
                if a == b:
                    continue
                d_pa = DIST[p][a]
                d_pb = DIST[p][b]
                d_ab = DIST[a][b]
                if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                    can_include = False
                    break
            if not can_include:
                break
        
        if can_include:
            included.add(p)
            backtrack(idx + 1)
            included.remove(p)
        
        # Option 2: Exclude p
        # For now, we'll check maximality at the end
        excluded.add(p)
        backtrack(idx + 1)
        excluded.remove(p)
    
    backtrack(0)
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    best_set = []
    best_size = float('inf')
    
    # First, try branch and bound with a time limit
    if time.time() < deadline - 100.0:
        bb_set = branch_and_bound_min_maximal(deadline - 100.0)
        if bb_set is not None:
            best_set = list(bb_set)
            best_size = len(best_set)
    
    # Then, do local search from random restarts to potentially improve
    while time.time() < deadline:
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        
        init_set = greedy_fill([], order)
        improved_set = local_search(init_set, max_iters=5)
        
        if len(improved_set) < best_size:
            best_size = len(improved_set)
            best_set = improved_set
    
    return [ID_TO_POINT[i] for i in best_set]

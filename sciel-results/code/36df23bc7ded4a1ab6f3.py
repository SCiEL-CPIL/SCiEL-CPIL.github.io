"""Rewritten program for minimum maximal no-isosceles set using branch-and-bound."""
import time
import itertools
import numpy as np
from collections import defaultdict

N = 16
DEADLINE = 300.0  # seconds

def _time_remaining(start_time):
    return DEADLINE - (time.time() - start_time)

def _build_conflict_graph():
    """
    Precompute: for each point p, which pairs (a,b) of other points 
    would form an isosceles triangle with p.
    
    Returns:
        conflicts[p] = set of (i,j) pairs where i,j < p in ordering,
                       such that {p, points[i], points[j]} is isosceles
    """
    points = list(itertools.product(range(N), repeat=2))
    n = len(points)
    
    # For each point, store which other points are at each squared distance
    # conflicts[p] will be a set of frozenset({i,j}) for pairs that conflict with p
    conflicts = [set() for _ in range(n)]
    
    # Precompute squared distances
    dist = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            d = (xi-xj)**2 + (yi-yj)**2
            dist[i, j] = d
            dist[j, i] = d
    
    # For each triple, check if isosceles and record conflicts
    # A triple (a,b,c) is isosceles if any two distances are equal
    # For point c, it conflicts with pair (a,b) if dist[a,c]==dist[b,c] or 
    # dist[a,c]==dist[a,b] or dist[b,c]==dist[a,b]
    
    # More efficient: for each point c, for each pair (a,b), check isosceles
    # But that's O(n^3). Let's use distance-based grouping.
    
    # For each point c, group other points by distance from c
    # If two points a,b are at the same distance from c, then (a,b,c) is isosceles
    # Also if dist[a,b] equals dist[a,c] or dist[b,c]
    
    for c in range(n):
        # Group points by distance from c
        dist_groups = defaultdict(list)
        for a in range(n):
            if a == c:
                continue
            d = dist[a, c]
            dist_groups[d].append(a)
        
        # Case 1: two points at same distance from c
        for d, pts in dist_groups.items():
            if len(pts) >= 2:
                for i in range(len(pts)):
                    for j in range(i+1, len(pts)):
                        a, b = pts[i], pts[j]
                        # Only record if a < b and (a,b) is a valid pair
                        # We'll store the pair as a frozenset
                        pair = frozenset((a, b))
                        conflicts[c].add(pair)
        
        # Case 2: dist[a,c] == dist[a,b] for some b
        # This is trickier. For each a, we need b such that dist[a,b] == dist[a,c]
        # But we already handle case 1. Let's check: 
        # Isosceles means d1==d2, d1==d3, or d2==d3
        # d1 = dist[a,b], d2 = dist[a,c], d3 = dist[b,c]
        # Case 1 above: d2==d3 (same distance from c)
        # Case 2: d1==d2 (dist[a,b]==dist[a,c])
        # Case 3: d1==d3 (dist[a,b]==dist[b,c])
        
        # For case 2: for each a, find b where dist[a,b] == dist[a,c]
        # For case 3: for each b, find a where dist[a,b] == dist[b,c]
        
        # Actually, let's just iterate over all pairs for correctness, but use numpy for speed
        # Since n=256, n^3 = 16M which is too slow. Let's be smarter.
        
        # Alternative: for each c, iterate over all pairs (a,b) with a<b, a!=c, b!=c
        # and check if isosceles. That's O(n^3) total.
        # 256^3 / 6 ≈ 2.8M checks. Each check is O(1). This should be fast enough.
    
    # Let's do it the straightforward way with numpy vectorization where possible
    # Reset conflicts
    conflicts = [set() for _ in range(n)]
    
    # For each point c, check all pairs (a,b) where a<b, a!=c, b!=c
    for c in range(n):
        # Get all points except c
        others = [i for i in range(n) if i != c]
        for idx_a in range(len(others)):
            a = others[idx_a]
            for idx_b in range(idx_a+1, len(others)):
                b = others[idx_b]
                d_ab = dist[a, b]
                d_ac = dist[a, c]
                d_bc = dist[b, c]
                if d_ab == d_ac or d_ab == d_bc or d_ac == d_bc:
                    pair = frozenset((a, b))
                    conflicts[c].add(pair)
    
    return points, conflicts, dist

def _point_to_idx(points):
    """Create mapping from point to index."""
    return {p: i for i, p in enumerate(points)}

def construct():
    start_time = time.time()
    
    # Precompute
    points, conflicts, dist = _build_conflict_graph()
    n = len(points)
    idx_map = _point_to_idx(points)
    
    # We want to find a minimal maximal no-isosceles set.
    # 
    # Approach: Branch and bound
    # - We build a set S of points
    # - S is valid if no three points in S form an isosceles triangle
    # - S is maximal if for every point p not in S, there exist a,b in S 
    #   such that {p,a,b} is isosceles
    #
    # We want to minimize |S| subject to maximality.
    #
    # This is equivalent to: find a minimal set S such that:
    # 1. S is an independent set in the "isosceles triangle" 3-uniform hypergraph
    # 2. S is a dominating set in the sense that every external point is "covered"
    #
    # Alternative formulation: 
    # A point p is "blocked" by S if there exist a,b in S such that {p,a,b} is isosceles.
    # We need S to be isosceles-free and to block all points not in S.
    #
    # Let's use a different approach:
    # 1. First, find a valid (isosceles-free) set
    # 2. Check if it's maximal
    # 3. If not, add points to make it maximal
    # 4. Try to minimize the total size
    #
    # Better: use iterative deepening or branch-and-bound directly.
    #
    # Let's try a greedy approach with local search:
    # 1. Start with a greedy maximal set
    # 2. Try to remove points while maintaining maximality
    # 3. Use simulated annealing or local search to find smaller sets
    
    # First, let's get a baseline greedy solution
    def greedy_maximal(order=None):
        if order is None:
            order = list(range(n))
        chosen = []
        chosen_set = set()
        
        for p_idx in order:
            p = points[p_idx]
            # Check if adding p creates an isosceles triangle with any pair in chosen
            valid = True
            for i in range(len(chosen)):
                for j in range(i+1, len(chosen)):
                    a_idx = chosen[i]
                    b_idx = chosen[j]
                    d_ab = dist[a_idx, b_idx]
                    d_ap = dist[a_idx, p_idx]
                    d_bp = dist[b_idx, p_idx]
                    if d_ab == d_ap or d_ab == d_bp or d_ap == d_bp:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                chosen.append(p_idx)
                chosen_set.add(p_idx)
        
        # Check maximality and add points if needed
        # A point q is blocked if there exist a,b in chosen such that {q,a,b} is isosceles
        def is_blocked(q_idx, chosen_list):
            for i in range(len(chosen_list)):
                a_idx = chosen_list[i]
                for j in range(i+1, len(chosen_list)):
                    b_idx = chosen_list[j]
                    d_ab = dist[a_idx, b_idx]
                    d_aq = dist[a_idx, q_idx]
                    d_bq = dist[b_idx, q_idx]
                    if d_ab == d_aq or d_ab == d_bq or d_aq == d_bq:
                        return True
            return False
        
        # Check if all unchosen points are blocked
        not_maximal = False
        for q_idx in range(n):
            if q_idx not in chosen_set:
                if not is_blocked(q_idx, chosen):
                    not_maximal = True
                    break
        
        if not not_maximal:
            return chosen
        
        # If not maximal, add points greedily until maximal
        # But this might create isosceles triangles... 
        # Actually, we need to be careful. Let's just return the greedy set
        # and rely on the local search to fix it.
        return chosen
    
    # Get initial greedy solution
    best_set = greedy_maximal()
    best_size = len(best_set)
    
    # Local search: try to remove points while maintaining isosceles-free property
    # and check if the set remains maximal (or can be made maximal with fewer additions)
    
    def is_isosceles_free(chosen_list):
        for i in range(len(chosen_list)):
            for j in range(i+1, len(chosen_list)):
                for k in range(j+1, len(chosen_list)):
                    a, b, c = chosen_list[i], chosen_list[j], chosen_list[k]
                    d_ab = dist[a, b]
                    d_ac = dist[a, c]
                    d_bc = dist[b, c]
                    if d_ab == d_ac or d_ab == d_bc or d_ac == d_bc:
                        return False
        return True
    
    def is_maximal(chosen_list):
        chosen_set = set(chosen_list)
        for q_idx in range(n):
            if q_idx not in chosen_set:
                # Check if q is blocked
                blocked = False
                for i in range(len(chosen_list)):
                    a_idx = chosen_list[i]
                    for j in range(i+1, len(chosen_list)):
                        b_idx = chosen_list[j]
                        d_ab = dist[a_idx, b_idx]
                        d_aq = dist[a_idx, q_idx]
                        d_bq = dist[b_idx, q_idx]
                        if d_ab == d_aq or d_ab == d_bq or d_aq == d_bq:
                            blocked = True
                            break
                    if blocked:
                        break
                if not blocked:
                    return False
        return True
    
    # Try to improve by removing points
    improved = True
    while improved and _time_remaining(start_time) > 10:
        improved = False
        for i in range(len(best_set)):
            # Try removing best_set[i]
            new_set = best_set[:i] + best_set[i+1:]
            if is_isosceles_free(new_set):
                # Check if we can make it maximal with the same or fewer points
                # For now, just check if it's already maximal
                if is_maximal(new_set):
                    best_set = new_set
                    best_size = len(best_set)
                    improved = True
                    break
                # If not maximal, we'd need to add points, which might not help
                # So skip for now
    
    # Convert back to points
    return [points[i] for i in best_set]

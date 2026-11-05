import math
import time
import itertools
from fractions import Fraction

# EVOLVE-BLOCK-START

def _is_square_frac(f):
    """Check if Fraction f is a perfect square of a rational."""
    if f < 0:
        return False
    n, d = f.numerator, f.denominator
    if n < 0:
        return False
    rn = math.isqrt(n)
    rd = math.isqrt(d)
    return rn * rn == n and rd * rd == d

def _sqrt_frac(f):
    """Compute sqrt of a Fraction that is known to be a perfect square."""
    n, d = f.numerator, f.denominator
    rn = math.isqrt(n)
    rd = math.isqrt(d)
    return Fraction(rn, rd)

def _dist_sq(p1, p2, D):
    """Squared distance between two points (x,y) in the (x, y*sqrt(D)) metric."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + Fraction(D) * dy * dy

def _is_rational_dist(p1, p2, D):
    d2 = _dist_sq(p1, p2, D)
    return _is_square_frac(d2)

def _no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                if cross == 0:
                    return False
    return True

def _det4(M):
    """Compute 4x4 determinant."""
    a = M[0]; b = M[1]; c = M[2]; d = M[3]
    return (a[0]*(b[1]*(c[2]*d[3]-c[3]*d[2]) - b[2]*(c[1]*d[3]-c[3]*d[1]) + b[3]*(c[1]*d[2]-c[2]*d[1]))
         - a[1]*(b[0]*(c[2]*d[3]-c[3]*d[2]) - b[2]*(c[0]*d[3]-c[3]*d[0]) + b[3]*(c[0]*d[2]-c[2]*d[0]))
         + a[2]*(b[0]*(c[1]*d[3]-c[3]*d[1]) - b[1]*(c[0]*d[3]-c[3]*d[0]) + b[3]*(c[0]*d[1]-c[1]*d[0]))
         - a[3]*(b[0]*(c[1]*d[2]-c[2]*d[1]) - b[1]*(c[0]*d[2]-c[2]*d[0]) + b[2]*(c[0]*d[1]-c[1]*d[0])))

def _no_four_concyclic(points):
    n = len(points)
    one = Fraction(1)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    M = [
                        [p1[0], p1[1], p1[0]**2 + p1[1]**2, one],
                        [p2[0], p2[1], p2[0]**2 + p2[1]**2, one],
                        [p3[0], p3[1], p3[0]**2 + p3[1]**2, one],
                        [p4[0], p4[1], p4[0]**2 + p4[1]**2, one],
                    ]
                    if _det4(M) == 0:
                        return False
    return True

def _no_four_concyclic_with_new(P, Q):
    """Check that no four points among P + [Q] are concyclic, assuming P has no 4 concyclic."""
    n = len(P)
    q0, q1 = Q[0], Q[1]
    q2 = q0 * q0 + q1 * q1
    one = Fraction(1)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = P[i], P[j], P[k]
                M = [
                    [p1[0], p1[1], p1[0]**2 + p1[1]**2, one],
                    [p2[0], p2[1], p2[0]**2 + p2[1]**2, one],
                    [p3[0], p3[1], p3[0]**2 + p3[1]**2, one],
                    [q0, q1, q2, one],
                ]
                if _det4(M) == 0:
                    return False
    return True

def _no_three_collinear_with_new(P, Q):
    """Check that no three points among P + [Q] are collinear, assuming P has no 3 collinear."""
    n = len(P)
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2, p3 = P[i], P[j], Q
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            if cross == 0:
                return False
    return True

def _validate(points, D):
    if len(points) < 3:
        return False
    if len(set(points)) != len(points):
        return False
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if not _is_rational_dist(points[i], points[j], D):
                return False
    if not _no_three_collinear(points):
        return False
    if not _no_four_concyclic(points):
        return False
    return True

def _generate_pythagorean_offsets(max_k):
    """Generate all integer offsets (dx, dy) such that dx^2 + dy^2 is a perfect square <= max_k^2."""
    offsets = set()
    for k in range(1, max_k + 1):
        k2 = k * k
        for dx in range(-k, k + 1):
            dy2 = k2 - dx * dx
            if dy2 < 0:
                continue
            dy = math.isqrt(dy2)
            if dy * dy == dy2:
                offsets.add((dx, dy))
                offsets.add((dx, -dy))
    return list(offsets)

def _build_compatibility_graph(points, D, max_dist_sq_num):
    """
    Build a graph where vertices are indices of points, and edges connect
    points with rational distance.
    """
    n = len(points)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _is_rational_dist(points[i], points[j], D):
                adj[i].add(j)
                adj[j].add(i)
    return adj

def _search_clique(adj, target_size, deadline):
    """
    Backtracking search for a clique of size target_size in the graph adj.
    Returns the list of indices if found, else None.
    """
    n = len(adj)
    # Order vertices by degree descending to improve pruning
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    # Map ordered index to original index
    # We will search in the order of 'order'
    # But we need to maintain the set of available candidates
    
    best_clique = []
    
    def backtrack(current_clique, candidates):
        if len(current_clique) == target_size:
            return current_clique
        if not candidates:
            return None
        if time.time() > deadline:
            return None
            
        # Try to add the first candidate
        # Actually, standard Bron-Kerbosch or simple backtracking
        # We pick a vertex from candidates, add it, and recurse with neighbors
        
        # Heuristic: pick vertex with highest degree in the subgraph induced by candidates
        # For simplicity, just iterate
        for i in range(len(candidates)):
            v = candidates[i]
            new_clique = current_clique + [v]
            
            # Check geometric constraints
            pts = [points_idx_to_point[v] for v in new_clique]
            if not _no_three_collinear(pts):
                continue
            if len(pts) >= 4 and not _no_four_concyclic(pts):
                continue
                
            # New candidates are neighbors of v that are in candidates and after i?
            # No, we just need to intersect candidates with neighbors of v
            new_candidates = [c for c in candidates[i+1:] if c in adj[v]]
            
            res = backtrack(new_clique, new_candidates)
            if res:
                return res
        return None

    # We need access to points for geometric checks
    # Let's pass points as a closure or global
    return backtrack([], order)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    best_points = None
    best_D = 1
    best_count = 0
    
    # Known 5-point set
    known_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    points_known = [(Fraction(x), Fraction(y)) for x, y in known_5]
    if _validate(points_known, D):
        best_points = points_known
        best_count = len(points_known)

    # Strategy: Search for 6-point sets in a grid of integer points with D=1
    # We will generate a set of candidate points and build a compatibility graph.
    
    # Define grid size. Larger grid means more candidates, but more edges to check.
    # We want points that are part of Pythagorean triples.
    # Let's try a grid of [-50, 50] first.
    
    grid_size = 60
    candidates = []
    candidate_set = set()
    
    for x in range(-grid_size, grid_size + 1):
        for y in range(-grid_size, grid_size + 1):
            candidates.append((Fraction(x), Fraction(y)))
            candidate_set.add((x, y))
            
    # To make search feasible, we should filter candidates.
    # A point must have at least 5 other points at rational distance.
    # This is expensive to check for all pairs.
    # Instead, let's just build the graph and run the clique search.
    # Number of points ~ 121^2 = 14641.
    # Pairwise checks ~ 100 million. Too slow in Python.
    
    # Optimization: Only consider points that are "rich" in Pythagorean neighbors.
    # Or, restrict the grid size.
    # Let's try a smaller grid, e.g., [-20, 20].
    # 41^2 = 1681 points.
    # Pairs ~ 1.4 million. This is feasible.
    
    grid_size_small = 25
    candidates_small = []
    for x in range(-grid_size_small, grid_size_small + 1):
        for y in range(-grid_size_small, grid_size_small + 1):
            candidates_small.append((Fraction(x), Fraction(y)))
            
    n_cand = len(candidates_small)
    adj = [set() for _ in range(n_cand)]
    
    # Build graph
    for i in range(n_cand):
        if time.time() > deadline - 10:
            break
        for j in range(i + 1, n_cand):
            if _is_rational_dist(candidates_small[i], candidates_small[j], 1):
                adj[i].add(j)
                adj[j].add(i)
                
    # Find Max Clique
    # We want to find a clique of size 6, 7, ...
    # We'll try to find size 6 first.
    
    # To speed up clique search, we can use a greedy coloring bound (Tomita algorithm)
    # But a simple backtracking with degree ordering might suffice for small N.
    
    # Let's implement a simple backtracking for max clique
    # We'll sort vertices by degree.
    
    degrees = [len(adj[i]) for i in range(n_cand)]
    # Only keep vertices with degree >= 5 (need at least 5 neighbors to form a 6-clique)
    active_vertices = [i for i in range(n_cand) if degrees[i] >= 5]
    
    # Map active vertices to a smaller index space?
    # No, just use the original indices but filter in the search.
    
    # Sort active vertices by degree descending
    active_vertices.sort(key=lambda i: -degrees[i])
    
    best_clique_found = None
    
    def search_clique(current_clique, available):
        nonlocal best_clique_found
        if len(current_clique) > best_count:
            # Check if this is a valid geometric set
            pts = [candidates_small[i] for i in current_clique]
            if _validate(pts, 1):
                best_clique_found = list(current_clique)
                # We can stop if we find 8, but let's see if we can find more
                # For now, just record it.
        
        if not available:
            return
            
        # Heuristic: if max possible size (len(current) + len(available)) <= best_count, prune
        # This is a weak bound.
        
        for i in range(len(available)):
            v = available[i]
            # Check if v is connected to all in current_clique
            # Since we are picking from 'available' which is intersected with neighbors, this should be true
            # But we need to ensure 'available' is maintained correctly.
            
            new_clique = current_clique + [v]
            
            # Geometric check early
            pts = [candidates_small[idx] for idx in new_clique]
            if not _no_three_collinear(pts):
                continue
            if len(pts) >= 4 and not _no_four_concyclic(pts):
                continue
                
            # New available: intersection of available[i+1:] and neighbors of v
            new_available = [u for u in available[i+1:] if u in adj[v]]
            
            search_clique(new_clique, new_available)
            
            if time.time() > deadline - 5:
                return
                
    search_clique([], active_vertices)
    
    if best_clique_found:
        new_points = [candidates_small[i] for i in best_clique_found]
        if len(new_points) > best_count:
            best_points = new_points
            best_count = len(new_points)
            
    # If we didn't find a better set in the small grid, try the known 5-point set extension
    # Or just return the best we have.
    
    if best_points is None:
        best_points = points_known
        best_D = 1
        
    result_points = []
    for x, y in best_points:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
        
    return {'D': best_D, 'points': result_points}

# EVOLVE-BLOCK-END

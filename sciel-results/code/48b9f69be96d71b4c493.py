"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd, isqrt
from itertools import combinations

def construct():
    """Search for integral point sets using graph clique approach."""
    t_start = time.time()
    DEADLINE = t_start + 130  # 20s safety margin before 150s kill

    def time_ok():
        return time.time() < DEADLINE

    def is_square(n):
        if n < 0:
            return False
        r = isqrt(n)
        return r * r == n

    def check_collinear(p1, p2, p3):
        (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def check_concyclic(p1, p2, p3, p4):
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p1, p2, p3, p4
        a1, b1, c1 = x1, y1, x1*x1 + y1*y1
        a2, b2, c2 = x2, y2, x2*x2 + y2*y2
        a3, b3, c3 = x3, y3, x3*x3 + y3*y3
        a4, b4, c4 = x4, y4, x4*x4 + y4*y4
        det = (a1 * (b2 * (c3 - c4) - c2 * (b3 - b4) + (b3 * c4 - c3 * b4))
             - b1 * (a2 * (c3 - c4) - c2 * (a3 - a4) + (a3 * c4 - c3 * a4))
             + c1 * (a2 * (b3 - b4) - b2 * (a3 - a4) + (a3 * b4 - b3 * a4))
             - (a2 * (b3 * c4 - c3 * b4) - b2 * (a3 * c4 - c3 * a4) + c2 * (a3 * b4 - b3 * a4)))
        return det == 0

    def validate_full(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                if not is_square(dx*dx + dy*dy):
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_collinear(points[i], points[j], points[k]):
                        return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if check_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    def validate_incremental(points, new_point):
        """Check if new_point can be added to points."""
        for p in points:
            dx = new_point[0] - p[0]
            dy = new_point[1] - p[1]
            if not is_square(dx*dx + dy*dy):
                return False
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if check_collinear(points[i], points[j], new_point):
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_concyclic(points[i], points[j], points[k], new_point):
                        return False
        return True

    def build_graph(points_list):
        """Build adjacency: edge between i,j if distance is integer."""
        n = len(points_list)
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                dx = points_list[i][0] - points_list[j][0]
                dy = points_list[i][1] - points_list[j][1]
                if is_square(dx*dx + dy*dy):
                    adj[i].add(j)
                    adj[j].add(i)
        return adj

    def find_maximal_cliques(adj, n, min_size=5):
        """Bron–Kerbosch to find all maximal cliques of size >= min_size."""
        cliques = []
        
        def bronk(R, P, X):
            if not P and not X:
                if len(R) >= min_size:
                    cliques.append(frozenset(R))
                return
            # Choose pivot from P ∪ X to minimize branching
            pivot_set = P | X
            if pivot_set:
                pivot = max(pivot_set, key=lambda v: len(adj[v] & P))
                candidates = P - adj[pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                R2 = R | {v}
                P2 = P & adj[v]
                X2 = X & adj[v]
                bronk(R2, P2, X2)
                P = P - {v}
                X = X | {v}
        
        bronk(frozenset(), frozenset(range(n)), frozenset())
        return cliques

    def clique_to_points(clique, points_list):
        return [points_list[i] for i in sorted(clique)]

    # Known good 5-point set as fallback
    fallback = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_points = list(fallback)
    best_size = 5

    random.seed(123)

    # Strategy 1: Build graph over bounded lattice region, find cliques
    # The known 5-point set spans x in [-56, 16], y in [0, 63]
    # Expand the search region to find larger cliques
    
    for radius in [80, 120, 160, 200]:
        if not time_ok():
            break
        
        # Generate candidate lattice points in the region
        # Focus on the bounding box of the known set plus margin
        x_min, x_max = -radius, radius
        y_min, y_max = 0, radius
        
        # To keep graph manageable, we use a smarter candidate generation:
        # Points must be at integer distance from at least one known point
        # This dramatically reduces the candidate set
        
        known = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        candidates = set()
        
        for kx, ky in known:
            for r in range(1, radius + 1):
                for dx in range(-r, r + 1):
                    dy2 = r*r - dx*dx
                    if dy2 < 0:
                        continue
                    dy = isqrt(dy2)
                    if dy*dy != dy2:
                        continue
                    for s in ([1, -1] if dy != 0 else [0]):
                        cand = (kx + dx, ky + s*dy)
                        if x_min <= cand[0] <= x_max and y_min <= cand[1] <= y_max:
                            candidates.add(cand)
        
        # Also add the known points
        for p in known:
            candidates.add(p)
        
        # Remove duplicates and sort for determinism
        candidates_list = sorted(candidates)
        n_cands = len(candidates_list)
        
        if n_cands > 5000:
            # Too many, subsample
            random.shuffle(candidates_list)
            candidates_list = candidates_list[:5000]
            n_cands = len(candidates_list)
        
        if n_cands < 5:
            continue
        
        # Build the graph
        adj = build_graph(candidates_list, n_cands)
        
        # Find maximal cliques
        cliques = find_maximal_cliques(adj, n_cands, min_size=5)
        
        # Validate each clique
        for clique in cliques:
            if len(clique) <= best_size:
                continue
            pts = clique_to_points(clique, candidates_list)
            if validate_full(pts):
                best_points = pts
                best_size = len(pts)
                if best_size >= 8:
                    break
        if best_size >= 8:
            break
    
    # Strategy 2: Incremental growth from best found set
    if best_size < 8 and time_ok():
        current = list(best_points)
        for attempt in range(20):
            if not time_ok() or len(current) >= 8:
                break
            # Generate candidates at integer distance from all current points
            multi_cands = set()
            for p in current:
                for r in range(1, 150):
                    for dx in range(-r, r + 1):
                        dy2 = r*r - dx*dx
                        if dy2 < 0:
                            continue
                        dy = isqrt(dy2)
                        if dy*dy != dy2:
                            continue
                        for s in ([1, -1] if dy != 0 else [0]):
                            cand = (p[0] + dx, p[1] + s*dy)
                            multi_cands.add(cand)
            
            # Intersect: must be at integer distance from ALL current points
            valid_cands = []
            for cand in multi_cands:
                if cand in current:
                    continue
                ok = True
                for p in current:
                    dx = cand[0] - p[0]
                    dy = cand[1] - p[1]
                    if not is_square(dx*dx + dy*dy):
                        ok = False
                        break
                if ok:
                    valid_cands.append(cand)
            
            random.shuffle(valid_cands)
            improved = False
            for cand in valid_cands:
                if not time_ok():
                    break
                if validate_incremental(current, cand):
                    current.append(cand)
                    improved = True
                    break
            
            if not improved:
                # Try removing a random point and re-adding
                if len(current) > 3:
                    idx = random.randint(1, len(current) - 1)
                    removed = current.pop(idx)
                    # Try to find a different point
                    for cand in valid_cands:
                        if cand == removed:
                            continue
                        if validate_incremental(current, cand):
                            current.append(cand)
                            break
                    else:
                        current.append(removed)
                else:
                    break
        
        if len(current) > best_size:
            best_points = current
            best_size = len(best_points)
    
    # Final validation
    if not validate_full(best_points):
        best_points = list(fallback)
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END

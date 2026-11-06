"""Rewritten program with stronger algorithm for integral point sets."""
import random
import math
import time
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_square(n):
    """Check if n is a perfect square (n is a non-negative int)."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _dist_sq(p1, p2, D):
    """Squared distance between two points (x1,y1),(x2,y2) with y*sqrt(D)."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + D * dy * dy

def _is_collinear(p1, p2, p3):
    """Check if three points are collinear."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    return det == 0

def _is_concyclic(p1, p2, p3, p4, D):
    """Check if four points are concyclic using exact rational arithmetic."""
    # Circle equation: x^2 + D*y^2 + a*x + b*y + c = 0
    # For 4 points, the 4x4 determinant must be zero
    def det4(m):
        return (m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
            - m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
            + m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
            - m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])))
    
    vals = []
    for p in [p1, p2, p3, p4]:
        x, y = p
        vals.append([x * x + D * y * y, x, y, 1])
    return det4(vals) == 0

def _check_valid_4tuple(pts, D):
    """Check if 4 points form a valid configuration (all pairwise distances rational, no 3 collinear, no 4 concyclic)."""
    n = len(pts)
    # Check all pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            ds = _dist_sq(pts[i], pts[j], D)
            if not _is_square(ds):
                return False
    # Check no 3 collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _is_collinear(pts[i], pts[j], pts[k]):
                    return False
    # Check no 4 concyclic
    if n >= 4:
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        if _is_concyclic(pts[i], pts[j], pts[k], pts[l], D):
                            return False
    return True

def _check_valid_set(points, D):
    """Check if a set of points is valid."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            ds = _dist_sq(points[i], points[j], D)
            if not _is_square(ds):
                return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _is_collinear(points[i], points[j], points[k]):
                    return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _is_concyclic(points[i], points[j], points[k], points[l], D):
                        return False
    return True

def _find_large_clique(adj, max_size, deadline):
    """Find a large clique in the graph using greedy + backtracking."""
    n = len(adj)
    if n == 0:
        return []
    
    # Greedy: pick vertex with highest degree, then find clique among neighbors
    best = []
    best_size = 0
    
    # Try greedy from each vertex
    for start in range(min(n, 50)):
        if time.time() > deadline:
            break
        clique = [start]
        candidates = [v for v in range(n) if v > start and adj[start][v]]
        
        while candidates:
            # Pick candidate with most connections to current clique
            best_cand = None
            best_score = -1
            for c in candidates:
                score = sum(1 for v in clique if adj[c][v])
                if score > best_score:
                    best_score = score
                    best_cand = c
            if best_cand is None:
                break
            clique.append(best_cand)
            candidates = [v for v in candidates if v > best_cand and adj[best_cand][v]]
        
        if len(clique) > best_size:
            best_size = len(clique)
            best = list(clique)
    
    return best

def _build_constraint_graph(valid_4tuples, D, deadline):
    """Build a constraint graph where nodes are valid 4-tuples and edges indicate compatibility."""
    n = len(valid_4tuples)
    # adj[i][j] = True if 4-tuple i and j can coexist (share at most 1 point, and combined set is valid)
    adj = [[False] * n for _ in range(n)]
    
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            # Check if they share at most 1 point
            set_i = set(valid_4tuples[i])
            set_j = set(valid_4tuples[j])
            shared = set_i & set_j
            if len(shared) > 1:
                continue
            
            # Check if combined set is valid
            combined = list(set_i | set_j)
            if len(combined) > 6:  # Too large to check quickly
                continue
            if _check_valid_set(combined, D):
                adj[i][j] = True
                adj[j][i] = True
    
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    D = 1
    random.seed(42)
    
    # Known valid 5-point set
    initial = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verify initial set
    assert _check_valid_set(initial, D), "Initial set is invalid"
    
    best_points = list(initial)
    
    # Strategy 1: Try to extend the known set by adding points one at a time
    # Generate candidate points in a larger range
    max_coord = 200
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            if (x, y) not in initial:
                candidates.append((x, y))
    
    random.shuffle(candidates)
    
    # Greedily add points
    current = list(initial)
    for cand in candidates:
        if time.time() > deadline:
            break
        new_set = current + [cand]
        if _check_valid_set(new_set, D):
            current.append(cand)
            if len(current) > len(best_points):
                best_points = list(current)
            if len(current) >= 12:
                break
    
    # Strategy 2: Find valid 4-tuples and build constraint graph
    if time.time() < deadline - 30:
        # Find all valid 4-tuples from a smaller candidate pool
        max_coord2 = 80
        pool = []
        for x in range(-max_coord2, max_coord2 + 1):
            for y in range(-max_coord2, max_coord2 + 1):
                pool.append((x, y))
        
        # Sample a subset of 4-tuples to keep computation manageable
        random.shuffle(pool)
        pool = pool[:300]  # Limit pool size
        
        valid_4tuples = []
        for combo in combinations(range(len(pool)), 4):
            if time.time() > deadline - 10:
                break
            pts = [pool[i] for i in combo]
            if _check_valid_4tuple(pts, D):
                valid_4tuples.append(tuple(pts))
        
        if len(valid_4tuples) >= 4:
            # Build constraint graph
            adj = _build_constraint_graph(valid_4tuples, D, deadline - 5)
            
            # Find large clique
            clique = _find_large_clique(adj, 20, deadline - 2)
            
            # Extract points from clique
            clique_points = set()
            for idx in clique:
                for p in valid_4tuples[idx]:
                    clique_points.add(p)
            
            if len(clique_points) > len(best_points):
                if _check_valid_set(list(clique_points), D):
                    best_points = list(clique_points)
    
    # Strategy 3: Local search - try to swap points to improve
    if time.time() < deadline - 10:
        current = list(best_points)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to add a new point
            for cand in candidates[:100]:
                if time.time() > deadline - 5:
                    break
                if cand in current:
                    continue
                new_set = current + [cand]
                if _check_valid_set(new_set, D):
                    current = new_set
                    improved = True
                    break
    
    if len(current) > len(best_points):
        best_points = current
    
    # Convert to required format
    result_points = []
    for x, y in best_points:
        result_points.append([[x, 1], [y, 1]])
    
    return {'D': D, 'points': result_points}
# EVOLVE-BLOCK-END

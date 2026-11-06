import time
import itertools
import math

def _is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def _dist_sq(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _check_general_position(points):
    n = len(points)
    # Check no three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Cross product (p2-p1) x (p3-p1)
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                if cross == 0:
                    return False
    # Check no four concyclic
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    # Determinant of the circle equation
                    # x^2 + y^2 + Dx + Ey + F = 0
                    # | x1 y1 x1^2+y1^2 1 |
                    # | x2 y2 x2^2+y2^2 1 | = 0
                    # | x3 y3 x3^2+y3^2 1 |
                    # | x4 y4 x4^2+y4^2 1 |
                    def det4(M):
                        # M is 4x4
                        return (
                            M[0][0]*(M[1][1]*(M[2][2]*M[3][3]-M[2][3]*M[3][2]) - M[1][2]*(M[2][1]*M[3][3]-M[2][3]*M[3][1]) + M[1][3]*(M[2][1]*M[3][2]-M[2][2]*M[3][1])) -
                            M[0][1]*(M[1][0]*(M[2][2]*M[3][3]-M[2][3]*M[3][2]) - M[1][2]*(M[2][0]*M[3][3]-M[2][3]*M[3][0]) + M[1][3]*(M[2][0]*M[3][2]-M[2][2]*M[3][0])) +
                            M[0][2]*(M[1][0]*(M[2][1]*M[3][3]-M[2][3]*M[3][1]) - M[1][1]*(M[2][0]*M[3][3]-M[2][3]*M[3][0]) + M[1][3]*(M[2][0]*M[3][1]-M[2][1]*M[3][0])) -
                            M[0][3]*(M[1][0]*(M[2][1]*M[3][2]-M[2][2]*M[3][1]) - M[1][1]*(M[2][0]*M[3][2]-M[2][2]*M[3][0]) + M[1][2]*(M[2][0]*M[3][1]-M[2][1]*M[3][0]))
                        )
                    
                    M = [
                        [p1[0], p1[1], p1[0]**2 + p1[1]**2, 1],
                        [p2[0], p2[1], p2[0]**2 + p2[1]**2, 1],
                        [p3[0], p3[1], p3[0]**2 + p3[1]**2, 1],
                        [p4[0], p4[1], p4[0]**2 + p4[1]**2, 1]
                    ]
                    if det4(M) == 0:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130.0 # 20s margin
    
    # Known 5-point set as fallback
    fallback_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to find 6 points
    # Search space: grid [-20, 20]
    # 41x41 = 1681 points. Pairs ~ 1.4M. Fast.
    # Let's try [-30, 30] -> 61x61 = 3721 points. Pairs ~ 6.9M. Still fast.
    
    # We will use a backtracking clique search.
    # To make it faster, we can restrict the search or use heuristics.
    
    # First, try to extend the known 5-point set
    # Check if any point in a larger grid is at integer distance from all 5
    # and maintains general position.
    
    # Grid for candidate 6th point
    max_coord = 100
    candidates_6th = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            if (x, y) in fallback_points:
                continue
            # Check distance to all 5
            valid = True
            for p in fallback_points:
                d2 = _dist_sq(p, (x, y))
                if not _is_square(d2):
                    valid = False
                    break
            if valid:
                candidates_6th.append((x, y))
    
    # Try adding each candidate
    for cand in candidates_6th:
        if time.time() > deadline:
            break
        test_set = fallback_points + [cand]
        if _check_general_position(test_set):
            return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in test_set]}
            
    # If extension failed, search for any 6-point clique in a grid
    # Grid size 30x30 centered at 0
    grid_limit = 30
    grid = [(x, y) for x in range(-grid_limit, grid_limit + 1) for y in range(-grid_limit, grid_limit + 1)]
    n = len(grid)
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d2 = _dist_sq(grid[i], grid[j])
            if _is_square(d2):
                adj[i].append(j)
                adj[j].append(i)
                
    # Sort nodes by degree
    nodes = list(range(n))
    nodes.sort(key=lambda i: len(adj[i]), reverse=True)
    
    found_6 = None
    
    def find_clique(start_idx, current_clique, size):
        nonlocal found_6
        if time.time() > deadline:
            return False
        if len(current_clique) == size:
            found_6 = [grid[i] for i in current_clique]
            return True
        if start_idx >= n:
            return False
        if (n - start_idx) < (size - len(current_clique)):
            return False
            
        for i in range(start_idx, n):
            node = nodes[i]
            compatible = True
            for c in current_clique:
                if node not in adj[c]:
                    compatible = False
                    break
            if compatible:
                current_clique.append(node)
                if find_clique(i + 1, current_clique, size):
                    return True
                current_clique.pop()
                
    if find_clique(0, [], 6):
        if found_6 and _check_general_position(found_6):
            return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in found_6]}
            
    # If 6 not found, return fallback
    points = fallback_points
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}

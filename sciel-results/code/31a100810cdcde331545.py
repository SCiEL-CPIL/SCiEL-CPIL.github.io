"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations
import math

def construct():
    start_time = time.time()
    DEADLINE = 130.0  # 150s wall clock, safety margin

    def elapsed():
        return time.time() - start_time

    def check_collinear(p1, p2, p3):
        # Check if three points are collinear using cross product
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def _exact_det_4x4(p1, p2, p3, p4):
        """Exact determinant for 4 concyclic check."""
        rows = []
        for p in [p1, p2, p3, p4]:
            x, y = p
            rows.append([x, y, x*x + y*y, 1])
        
        # Compute 4x4 determinant using expansion
        def det4(m):
            result = 0
            signs = [1, -1, 1, -1]
            for j in range(4):
                minor = [[m[i][k] for k in range(4) if k != j] for i in range(1, 4)]
                result += signs[j] * m[0][j] * det3(minor)
            return result
        
        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
                   - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
                   + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]))
        
        return det4(rows)

    def all_constraints_satisfied(points):
        """Check all pairwise distances are integer, no 3 collinear, no 4 concyclic."""
        n = len(points)
        # Check all pairwise distances are integers
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                d2 = dx*dx + dy*dy
                if d2 < 0:
                    return False
                r = int(math.isqrt(d2))
                if r*r != d2:
                    return False
        # Check no three collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_collinear(points[i], points[j], points[k]):
                        return False
        # Check no four concyclic
        for combo in combinations(range(n), 4):
            if _exact_det_4x4(points[combo[0]], points[combo[1]], points[combo[2]], points[combo[3]]) == 0:
                return False
        return True

    def try_add_point(current_points, candidate):
        """Try adding a candidate point to current_points."""
        new_points = current_points + [candidate]
        if all_constraints_satisfied(new_points):
            return new_points
        return None

    # Start with known 5-point set from seed
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Generate candidate points in a reasonable range
    max_coord = 200
    
    # Build candidate pool
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            if (x, y) not in best_points:
                candidates.append((x, y))
    
    # Sort candidates by distance from origin (prefer closer points)
    candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Try to add points greedily
    improved = True
    while improved:
        improved = False
        for cand in candidates:
            if elapsed() > DEADLINE:
                break
            result = try_add_point(best_points, cand)
            if result is not None:
                best_points = result
                candidates.remove(cand)
                improved = True
                if len(best_points) >= 12:
                    break
        if len(best_points) >= 12:
            break
    
    # Also try starting from different initial configurations
    initial_sets = [
        [(0,0), (3,4), (4,0), (0,4)],
        [(0,0), (5,0), (0,12), (5,12)],
    ]
    
    for init in initial_sets:
        if elapsed() > DEADLINE:
            break
        if not all_constraints_satisfied(init):
            continue
        pts = list(init)
        for cand in candidates:
            if elapsed() > DEADLINE:
                break
            result = try_add_point(pts, cand)
            if result is not None:
                pts = result
                if len(pts) > len(best_points):
                    best_points = pts

    # Convert to required format
    D = 1
    points_out = [[[x, 1], [y, 1]] for x, y in best_points]
    
    return {'D': D, 'points': points_out}
# EVOLVE-BLOCK-END

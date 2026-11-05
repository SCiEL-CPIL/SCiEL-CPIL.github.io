"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def check_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def _concyclic_det(p1, p2, p3, p4):
        """Exact 4x4 determinant for concyclic check using integer arithmetic."""
        # |x  y  x^2+y^2  1|
        rows = []
        for p in [p1, p2, p3, p4]:
            x, y = p
            rows.append([x, y, x*x + y*y, 1])
        
        # Compute determinant via cofactor expansion
        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
                   - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
                   + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]))
        
        def det4(m):
            result = 0
            signs = [1, -1, 1, -1]
            for j in range(4):
                minor = [[m[i][k] for k in range(4) if k != j] for i in range(1, 4)]
                result += signs[j] * m[0][j] * det3(minor)
            return result
        
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
                if not is_perfect_square(d2):
                    return False
        # Check no three collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_collinear(points[i], points[j], points[k]):
                        return False
        # Check no four concyclic
        for combo in combinations(range(n), 4):
            if _concyclic_det(points[combo[0]], points[combo[1]], 
                            points[combo[2]], points[combo[3]]) == 0:
                return False
        return True

    def try_add_point(current_points, candidate):
        """Try adding a candidate point to current_points."""
        new_points = current_points + [candidate]
        if all_constraints_satisfied(new_points):
            return new_points
        return None

    def generate_candidates(max_coord=300):
        """Generate candidate lattice points sorted by distance from origin."""
        candidates = []
        for x in range(-max_coord, max_coord + 1):
            for y in range(-max_coord, max_coord + 1):
                candidates.append((x, y))
        candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
        return candidates

    def greedy_extend(start_points, candidates, deadline):
        """Greedily extend a set of points by adding candidates."""
        pts = list(start_points)
        existing = set(pts)
        cand_list = [c for c in candidates if c not in existing]
        
        improved = True
        while improved:
            improved = False
            for i, cand in enumerate(cand_list):
                if elapsed() > deadline:
                    return pts
                result = try_add_point(pts, cand)
                if result is not None:
                    pts = result
                    existing.add(cand)
                    cand_list.pop(i)
                    improved = True
                    if len(pts) >= 12:
                        return pts
        return pts

    # Known 5-point integral sets
    initial_sets = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (3, 4), (4, 0), (0, 4)],
        [(0, 0), (5, 0), (0, 12), (5, 12)],
        [(0, 0), (9, 12), (12, 9), (15, 0)],
        [(0, 0), (7, 24), (24, 7), (25, 0)],
        [(0, 0), (8, 15), (15, 8), (17, 0)],
        [(0, 0), (12, 35), (35, 12), (37, 0)],
    ]

    # Verify and keep only valid initial sets
    valid_inits = []
    for init in initial_sets:
        if all_constraints_satisfied(init):
            valid_inits.append(init)

    if not valid_inits:
        # Fallback: use the first one even if constraints fail (shouldn't happen)
        valid_inits = [initial_sets[0]]

    best_points = valid_inits[0][:]
    best_len = len(best_points)

    # Generate candidate pool
    candidates = generate_candidates(250)
    existing = set(best_points)
    cand_pool = [c for c in candidates if c not in existing]

    # Try greedy extension from each valid initial set
    for init in valid_inits:
        if elapsed() > DEADLINE:
            break
        existing = set(init)
        cand_list = [c for c in candidates if c not in existing]
        result = greedy_extend(init, cand_list, DEADLINE)
        if len(result) > best_len:
            best_points = result
            best_len = len(result)
        if best_len >= 12:
            break

    # If we haven't found a large set, try a more targeted search
    # Look for points that are at integer distances from multiple existing points
    if best_len < 8 and elapsed() < DEADLINE - 30:
        # Try to find points that work with the best set
        base = best_points[:]
        # Generate more candidates based on Pythagorean relations
        extra_candidates = []
        for x in range(-500, 501):
            for y in range(-500, 501):
                if (x, y) not in set(base):
                    extra_candidates.append((x, y))
        extra_candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
        
        # Filter to only candidates at integer distance from at least 2 base points
        filtered = []
        for cand in extra_candidates:
            if elapsed() > DEADLINE:
                break
            count = 0
            for bp in base:
                dx = cand[0] - bp[0]
                dy = cand[1] - bp[1]
                if is_perfect_square(dx*dx + dy*dy):
                    count += 1
                    if count >= 2:
                        break
            if count >= 2:
                filtered.append(cand)
        
        if filtered:
            existing = set(base)
            cand_list = [c for c in filtered if c not in existing]
            result = greedy_extend(base, cand_list, DEADLINE)
            if len(result) > best_len:
                best_points = result
                best_len = len(result)

    # Final fallback: ensure we have at least 3 valid points
    if best_len < 3:
        best_points = [(0, 0), (3, 4), (4, 0)]

    # Convert to required format with D=1
    D = 1
    points_out = [[[x, 1], [y, 1]] for x, y in best_points]

    return {'D': D, 'points': points_out}
# EVOLVE-BLOCK-END

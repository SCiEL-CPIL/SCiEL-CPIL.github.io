"""Combined approach: fast evaluation + lattice candidates + two-line insight + targeted local search."""
import time
import numpy as np
from collections import Counter
import itertools

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 50.0  # Safety margin for 90s limit

    # Precompute triple indices for n=30
    triple_indices = np.array([(i, j, k) for i in range(30) for j in range(i+1, 30) for k in range(j+1, 30)], dtype=np.int64)
    I, J, K = triple_indices[:, 0], triple_indices[:, 1], triple_indices[:, 2]
    N_TRIPLES = len(I)

    def evaluate_fast(points):
        """
        Vectorized evaluation of max multiplicity of positive doubled areas.
        points: list of 30 (x, y) tuples.
        """
        pts = np.array(points, dtype=np.int64)
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Compute doubled area (determinant) for all triples
        # det = (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i)
        det = (x[J] - x[I]) * (y[K] - y[I]) - (y[J] - y[I]) * (x[K] - x[I])
        
        # Take absolute value and filter positive
        det = np.abs(det)
        det = det[det > 0]
        
        if len(det) == 0:
            return 0
        
        vals, cnts = np.unique(det, return_counts=True)
        return int(cnts.max())

    def valid(points):
        if len(points) != 30:
            return False
        if len(set(points)) != 30:
            return False
        for px, py in points:
            if not (-30 <= px <= 30 and -30 <= py <= 30):
                return False
        return True

    def make_grid(a, b, ox=0, oy=0, scale=1):
        """Create a x b grid of points scaled by 'scale' and offset by (ox, oy)."""
        pts = []
        for i in range(a):
            for j in range(b):
                x = ox + i * scale
                y = oy + j * scale
                pts.append((x, y))
        return pts

    def make_lattice(a, b, u1, u2, v1, v2, ox=0, oy=0):
        """Create a x b lattice with basis vectors (u1,u2) and (v1,v2)."""
        pts = []
        for i in range(a):
            for j in range(b):
                x = ox + i * u1 + j * v1
                y = oy + i * u2 + j * v2
                pts.append((x, y))
        return pts

    # Generate candidates
    candidates = []

    # Standard axis-aligned grids
    for a, b in [(5,6), (6,5), (3,10), (10,3), (2,15), (15,2), (4,8), (8,4)]:
        for ox in [-12, -10, -8, -5, 0, 5]:
            for oy in [-12, -10, -8, -5, 0, 5]:
                pts = make_grid(a, b, ox=ox, oy=oy)
                if valid(pts):
                    candidates.append(pts)

    # Sheared lattices
    for a, b in [(5,6), (6,5), (4,8), (8,4), (3,10)]:
        for s in [-3, -2, -1, 0, 1, 2, 3]:
            for ox in [-12, -8, -4, 0, 4]:
                for oy in [-12, -8, -4, 0, 4]:
                    pts = make_lattice(a, b, 1, 0, s, 1, ox=ox, oy=oy)
                    if valid(pts):
                        candidates.append(pts)

    # Rotated lattices with unimodular basis
    basis_vectors = [
        (1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (2, -1), (1, 2), (1, -2),
        (3, 1), (3, -1), (1, 3), (1, -3), (2, 3), (2, -3), (3, 2), (3, -2),
        (4, 1), (1, 4), (4, -1), (-1, 4)
    ]
    for a, b in [(5,6), (6,5), (4,8), (8,4), (3,10), (10,3)]:
        for i, (u1, u2) in enumerate(basis_vectors):
            for j, (v1, v2) in enumerate(basis_vectors[i+1:], i+1):
                det_val = u1 * v2 - u2 * v1
                if abs(det_val) != 1:
                    continue
                for ox in [-15, -10, -5, 0, 5]:
                    for oy in [-15, -10, -5, 0, 5]:
                        pts = make_lattice(a, b, u1, u2, v1, v2, ox=ox, oy=oy)
                        if valid(pts):
                            candidates.append(pts)

    # Two-parallel-lines construction (from inspiration 2)
    # Points on line y=0 and y=d
    for a in range(10, 21):
        b = 30 - a
        for d in [1, 2, 3, 4, 5, 6]:
            pts = []
            # Spread points on each line to maximize area variety
            # Use consecutive integers centered
            for i in range(a):
                pts.append((i - a//2, 0))
            for i in range(b):
                pts.append((i - b//2, d))
            # Center
            max_x = max(p[0] for p in pts)
            min_x = min(p[0] for p in pts)
            max_y = max(p[1] for p in pts)
            min_y = min(p[1] for p in pts)
            cx = (max_x + min_x) // 2
            cy = (max_y + min_y) // 2
            pts = [(x - cx, y - cy) for x, y in pts]
            if valid(pts):
                candidates.append(pts)

    # Three-parallel-lines construction
    for a, b, c in [(10,10,10), (9,11,10), (11,9,10), (8,12,10), (12,8,10), (7,11,12), (12,11,7)]:
        for d1 in [1, 2, 3]:
            for d2 in [d1+1, d1+2, d1+3, d1+4]:
                pts = []
                for i in range(a):
                    pts.append((i - a//2, 0))
                for i in range(b):
                    pts.append((i - b//2, d1))
                for i in range(c):
                    pts.append((i - c//2, d2))
                max_x = max(p[0] for p in pts)
                min_x = min(p[0] for p in pts)
                max_y = max(p[1] for p in pts)
                min_y = min(p[1] for p in pts)
                cx = (max_x + min_x) // 2
                cy = (max_y + min_y) // 2
                pts = [(x - cx, y - cy) for x, y in pts]
                if valid(pts):
                    candidates.append(pts)

    # Number-theoretic: generalized arithmetic progression with non-unimodular basis
    # For basis vectors u, v with |det(u,v)| = D > 1, the doubled area of triangle
    # (i1*u + j1*v, i2*u + j2*v, i3*u + j3*v) = D * |det of index triangle|
    # So all positive areas are multiples of D. We want to maximize the count of
    # triples whose index-triangle area equals some fixed value k, giving area D*k.
    # For a 5x6 grid of indices (i in 0..4, j in 0..5), the number of index-triples
    # with area = 1 (i.e., minimal non-zero) is large.
    # Let's try various (a, b, u, v) combinations with |det| = 2, 3, etc.
    for a, b in [(5,6), (6,5), (4,8), (8,4), (3,10), (10,3), (5,5), (6,5)]:
        # Try basis with determinant 2
        # u=(1,0), v=(0,2): det=2
        for (u1,u2), (v1,v2) in [((1,0),(0,2)), ((1,0),(1,2)), ((2,0),(0,1)), ((2,1),(0,1))]:
            det_val = u1*v2 - u2*v1
            if det_val <= 0:
                continue
            for ox in [-15, -10, -5, 0, 5]:
                for oy in [-15, -10, -5, 0, 5]:
                    pts = make_lattice(a, b, u1, u2, v1, v2, ox=ox, oy=oy)
                    if valid(pts):
                        candidates.append(pts)
        # Try basis with determinant 3
        for (u1,u2), (v1,v2) in [((1,0),(0,3)), ((1,0),(1,3)), ((3,0),(0,1)), ((1,1),(0,3)), ((3,1),(0,1))]:
            det_val = u1*v2 - u2*v1
            if det_val <= 0:
                continue
            for ox in [-15, -10, -5, 0, 5]:
                for oy in [-15, -10, -5, 0, 5]:
                    pts = make_lattice(a, b, u1, u2, v1, v2, ox=ox, oy=oy)
                    if valid(pts):
                        candidates.append(pts)
        # Try basis with determinant 4
        for (u1,u2), (v1,v2) in [((1,0),(0,4)), ((2,0),(0,2)), ((1,0),(2,4)), ((2,1),(0,2))]:
            det_val = u1*v2 - u2*v1
            if det_val <= 0:
                continue
            for ox in [-12, -8, -4, 0, 4]:
                for oy in [-12, -8, -4, 0, 4]:
                    pts = make_lattice(a, b, u1, u2, v1, v2, ox=ox, oy=oy)
                    if valid(pts):
                        candidates.append(pts)

    # Two-parallel-lines with variable spacing (not just unit spacing)
    for a in range(10, 21):
        b = 30 - a
        for d in [1, 2, 3, 4, 5, 6, 7, 8]:
            for spacing in [1, 2]:
                pts = []
                for i in range(a):
                    pts.append((i * spacing - (a-1)*spacing//2, 0))
                for i in range(b):
                    pts.append((i * spacing - (b-1)*spacing//2, d))
                max_x = max(p[0] for p in pts)
                min_x = min(p[0] for p in pts)
                max_y = max(p[1] for p in pts)
                min_y = min(p[1] for p in pts)
                cx = (max_x + min_x) // 2
                cy = (max_y + min_y) // 2
                pts = [(x - cx, y - cy) for x, y in pts]
                if valid(pts):
                    candidates.append(pts)

    # Remove duplicates
    seen = set()
    unique_candidates = []
    for pts in candidates:
        key = tuple(sorted(pts))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(pts)
    candidates = unique_candidates

    # Evaluate all candidates
    best_points = None
    best_score = 0

    for pts in candidates:
        if time.time() > deadline - 5:
            break
        score = evaluate_fast(pts)
        if score > best_score:
            best_score = score
            best_points = pts

    if best_points is None:
        best_points = make_grid(5, 6, ox=0, oy=0)
        best_score = evaluate_fast(best_points)

    # Local search with targeted moves
    if time.time() < deadline - 10:
        current = [list(p) for p in best_points]
        current_score = best_score

        def local_search(points, start_score, time_limit):
            pts = [list(p) for p in points]
            score = start_score
            directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1),
                          (2,0), (-2,0), (0,2), (0,-2), (2,1), (-2,1), (2,-1), (-2,-1)]

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                for idx in range(30):
                    if time.time() > time_limit:
                        break
                    ox, oy = pts[idx]
                    for dx, dy in directions:
                        nx, ny = ox + dx, oy + dy
                        if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                            continue
                        new_pts = [tuple(p) for p in pts]
                        new_pts[idx] = (nx, ny)
                        if len(set(new_pts)) != 30:
                            continue
                        s = evaluate_fast(new_pts)
                        if s > score:
                            pts[idx] = [nx, ny]
                            score = s
                            improved = True
                            break
                    if improved:
                        break
            return pts, score

        current, current_score = local_search(best_points, best_score, deadline - 5)
        if current_score > best_score:
            best_points = [tuple(p) for p in current]
            best_score = current_score

        # Second pass with different directions if time remains
        if time.time() < deadline - 10:
            def local_search_v2(points, start_score, time_limit):
                pts = [list(p) for p in points]
                score = start_score
                # Try larger jumps
                directions = [(3,0), (-3,0), (0,3), (0,-3), (3,1), (-3,1), (3,-1), (-3,-1),
                              (1,3), (-1,3), (1,-3), (-1,-3), (2,2), (-2,-2), (2,-2), (-2,2)]

                improved = True
                while improved and time.time() < time_limit:
                    improved = False
                    for idx in range(30):
                        if time.time() > time_limit:
                            break
                        ox, oy = pts[idx]
                        for dx, dy in directions:
                            nx, ny = ox + dx, oy + dy
                            if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                                continue
                            new_pts = [tuple(p) for p in pts]
                            new_pts[idx] = (nx, ny)
                            if len(set(new_pts)) != 30:
                                continue
                            s = evaluate_fast(new_pts)
                            if s > score:
                                pts[idx] = [nx, ny]
                                score = s
                                improved = True
                                break
                        if improved:
                            break
                return pts, score

            current2, current_score2 = local_search_v2(best_points, best_score, deadline - 5)
            if current_score2 > best_score:
                best_points = [tuple(p) for p in current2]
                best_score = current_score2

    return best_points
# EVOLVE-BLOCK-END

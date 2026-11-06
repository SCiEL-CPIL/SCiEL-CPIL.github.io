"""Rewritten program: search for 30 points maximizing multiplicity of a doubled area."""
import time
import random
import numpy as np
from collections import Counter

# Precompute indices for all triples (i, j, k) with 0 <= i < j < k < 30
_TRIPLES_I = []
_TRIPLES_J = []
_TRIPLES_K = []
for i in range(30):
    for j in range(i+1, 30):
        for k in range(j+1, 30):
            _TRIPLES_I.append(i)
            _TRIPLES_J.append(j)
            _TRIPLES_K.append(k)
_TRIPLES_I = np.array(_TRIPLES_I, dtype=np.int32)
_TRIPLES_J = np.array(_TRIPLES_J, dtype=np.int32)
_TRIPLES_K = np.array(_TRIPLES_K, dtype=np.int32)

def _score(points):
    """Compute the largest multiplicity of a positive doubled area among all triples using NumPy."""
    if len(points) < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    
    # Compute areas for all precomputed triples
    x1, y1 = pts[_TRIPLES_I, 0], pts[_TRIPLES_I, 1]
    x2, y2 = pts[_TRIPLES_J, 0], pts[_TRIPLES_J, 1]
    x3, y3 = pts[_TRIPLES_K, 0], pts[_TRIPLES_K, 1]
    
    areas = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    areas = areas[areas > 0]
    
    if len(areas) == 0:
        return 0
        
    # Find max multiplicity using bincount (faster than unique for large arrays)
    max_area = int(areas.max())
    if max_area > 0:
        counts = np.bincount(areas.astype(np.int64), minlength=max_area+1)
        return int(counts.max())
    return 0

def _local_search(points, max_iters=200, deadline=None):
    """Improve a set of points via local search: move one point to a better position."""
    points = list(points)
    n = len(points)
    best_score = _score(points)
    point_set = set(points)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        # Try moving each point to a nearby random position
        for idx in range(n):
            if deadline and time.time() > deadline:
                break
            orig = points[idx]
            # Try several candidate positions with adaptive step size
            step_size = 3 if it < max_iters // 2 else 5
            for _ in range(6):
                nx = orig[0] + random.randint(-step_size, step_size)
                ny = orig[1] + random.randint(-step_size, step_size)
                if -30 <= nx <= 30 and -30 <= ny <= 30:
                    new_pt = (nx, ny)
                    if new_pt not in point_set:
                        points[idx] = new_pt
                        point_set.discard(orig)
                        point_set.add(new_pt)
                        s = _score(points)
                        if s > best_score:
                            best_score = s
                            improved = True
                            break
                        else:
                            points[idx] = orig
                            point_set.discard(new_pt)
                            point_set.add(orig)
            if improved:
                break
        if not improved:
            # Try larger jumps
            for _ in range(3):
                if deadline and time.time() > deadline:
                    break
                idx = random.randint(0, n-1)
                orig = points[idx]
                for _ in range(10):
                    nx = random.randint(-30, 30)
                    ny = random.randint(-30, 30)
                    new_pt = (nx, ny)
                    if new_pt not in point_set:
                        points[idx] = new_pt
                        point_set.discard(orig)
                        point_set.add(new_pt)
                        s = _score(points)
                        if s > best_score:
                            best_score = s
                            improved = True
                            break
                        else:
                            points[idx] = orig
                            point_set.discard(new_pt)
                            point_set.add(orig)
                if improved:
                    break
            if not improved:
                break
    return points, best_score

def _generate_arithmetic_progression():
    """
    Generate 30 points forming a generalized arithmetic progression (parallelogram grid).
    Points are of the form p0 + i*u + j*v for i in [0, m), j in [0, n).
    m*n >= 30.
    """
    # Try multiple grid dimensions
    dimensions = [(5, 6), (6, 5), (4, 8), (8, 4), (3, 10), (10, 3)]
    
    # More diverse lattice bases
    candidates = [
        ((1, 0), (0, 1)),
        ((1, 0), (1, 1)),
        ((1, 1), (0, 1)),
        ((2, 1), (1, 2)),
        ((1, 2), (2, 1)),
        ((3, 1), (1, 3)),
        ((1, 3), (3, 1)),
        ((2, 0), (0, 2)),
        ((1, 2), (2, 3)),
        ((2, 3), (1, 2)),
    ]
    
    best_pts = None
    best_s = 0
    
    for m, n in dimensions:
        for u, v in candidates:
            # Try different origins to fit in [-30, 30]
            for ox in range(-20, 21, 4):
                for oy in range(-20, 21, 4):
                    pts = []
                    seen = set()
                    valid = True
                    for i in range(m):
                        for j in range(n):
                            x = ox + i * u[0] + j * v[0]
                            y = oy + i * u[1] + j * v[1]
                            if -30 <= x <= 30 and -30 <= y <= 30:
                                p = (x, y)
                                if p not in seen:
                                    seen.add(p)
                                    pts.append(p)
                            else:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if len(pts) == 30 and valid:
                        s = _score(pts)
                        if s > best_s:
                            best_s = s
                            best_pts = pts[:]
                            
    if best_pts is None:
        # Fallback to simple grid
        return [(x - 2, y - 2) for x in range(6) for y in range(5)]
        
    return best_pts

def construct():
    # Strict deadline: 40 seconds to be safe within 60s limit
    deadline = time.time() + 40
    
    # Initialize with number-theoretic construction (Arithmetic Progression)
    best_points = _generate_arithmetic_progression()
    best_score = _score(best_points)
    
    random.seed(42)
    
    # Also try some random initial configurations
    for _ in range(5):
        if time.time() > deadline:
            break
        random_pts = []
        seen = set()
        while len(random_pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in seen:
                seen.add((x, y))
                random_pts.append((x, y))
        s = _score(random_pts)
        if s > best_score:
            best_score = s
            best_points = random_pts[:]
    
    # Local search loop
    # We will run multiple rounds of local search, restarting from best or slightly perturbed best
    for restart in range(25): 
        if time.time() > deadline:
            break
        
        if restart == 0:
            pts = best_points[:]
        elif restart % 5 == 0:
            # Every 5th restart, try a fresh random start
            random_pts = []
            seen = set()
            while len(random_pts) < 30:
                x = random.randint(-30, 30)
                y = random.randint(-30, 30)
                if (x, y) not in seen:
                    seen.add((x, y))
                    random_pts.append((x, y))
            pts = random_pts
        else:
            # Perturb the best solution
            pts = best_points[:]
            # Move a few points randomly
            for _ in range(4):
                idx = random.randint(0, 29)
                nx = random.randint(-30, 30)
                ny = random.randint(-30, 30)
                if (nx, ny) not in pts:
                    pts[idx] = (nx, ny)
        
        # Local search
        pts, s = _local_search(pts, max_iters=180, deadline=deadline)
        if s > best_score:
            best_score = s
            best_points = pts[:]
            
    # Final polish with more iterations
    if time.time() < deadline - 1:
        pts, s = _local_search(best_points[:], max_iters=400, deadline=deadline)
        if s > best_score:
            best_score = s
            best_points = pts[:]
    
    return best_points

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
        
    # Find max multiplicity
    vals, counts = np.unique(areas, return_counts=True)
    return int(np.max(counts))

def _local_search(points, max_iters=200, deadline=None):
    """Improve a set of points via local search: move one point to a better position."""
    points = list(points)
    n = len(points)
    best_score = _score(points)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        # Try moving each point to a nearby random position
        for idx in range(n):
            if deadline and time.time() > deadline:
                break
            orig = points[idx]
            # Try several candidate positions
            for _ in range(5): # Reduced from 8
                nx = orig[0] + random.randint(-3, 3)
                ny = orig[1] + random.randint(-3, 3)
                if -30 <= nx <= 30 and -30 <= ny <= 30:
                    new_pt = (nx, ny)
                    if new_pt not in points:
                        points[idx] = new_pt
                        s = _score(points)
                        if s > best_score:
                            best_score = s
                            improved = True
                            break
                        else:
                            points[idx] = orig
            if improved:
                break
        if not improved:
            # Try larger jumps
            for _ in range(2): # Reduced from 3
                if deadline and time.time() > deadline:
                    break
                idx = random.randint(0, n-1)
                orig = points[idx]
                for _ in range(8): # Reduced from 12
                    nx = random.randint(-30, 30)
                    ny = random.randint(-30, 30)
                    new_pt = (nx, ny)
                    if new_pt not in points:
                        points[idx] = new_pt
                        s = _score(points)
                        if s > best_score:
                            best_score = s
                            improved = True
                            break
                        else:
                            points[idx] = orig
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
    # Choose dimensions m, n such that m*n >= 30
    # Common choices: 5x6, 6x5, 5x7 (drop some), etc.
    # Let's pick 5x6 = 30 exactly.
    m, n = 5, 6
    
    # Choose primitive vectors u, v and origin p0
    # To maximize specific area, we might want u and v to be "nice"
    # e.g., u=(1,0), v=(0,1) gives areas 1, 2, 3...
    # But we want to maximize multiplicity of a SINGLE area.
    # In a grid, areas are multiples of det(u,v).
    # If u=(a,b), v=(c,d), det = ad-bc.
    # The areas of triangles formed by grid points are multiples of det/2.
    # The doubled areas are multiples of det.
    # If we want high multiplicity, we want many triples to have the same area.
    # For a grid, the area of triangle (i1,j1), (i2,j2), (i3,j3) is
    # 0.5 * det( (i2-i1, j2-j1), (i3-i1, j3-j1) ) * det(u,v).
    # The "grid area" is 0.5 * det( di, dj ).
    # We want many triples to have the same grid area.
    # For a standard grid, the number of triples with grid area 0.5 (doubled area 1) is high?
    # Actually, for a k x l grid, the number of unit area triangles is related to the number of edges.
    
    # Let's try a few configurations of u, v
    candidates = [
        ((1, 0), (0, 1)), # Standard grid
        ((1, 0), (1, 1)), # Skewed
        ((2, 1), (1, 2)), # Larger steps
        ((1, 2), (2, 1)),
        ((3, 1), (1, 3)),
    ]
    
    best_pts = None
    best_s = 0
    
    for u, v in candidates:
        # Try different origins to fit in [-30, 30]
        for ox in range(-20, 21, 5):
            for oy in range(-20, 21, 5):
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
                            # If out of bounds, we might still try to fill with random or skip
                            # For now, if it goes out of bounds, we mark invalid for this specific origin
                            pass
                
                if len(pts) == 30:
                    s = _score(pts)
                    if s > best_s:
                        best_s = s
                        best_pts = pts[:]
                        
    if best_pts is None:
        # Fallback to simple grid
        return [(x - 2, y - 2) for x in range(6) for y in range(5)]
        
    return best_pts

def construct():
    # Strict deadline: 45 seconds to be safe within 60s limit
    deadline = time.time() + 45
    
    # Initialize with number-theoretic construction (Arithmetic Progression)
    best_points = _generate_arithmetic_progression()
    best_score = _score(best_points)
    
    random.seed(42)
    
    # Local search loop
    # We will run multiple rounds of local search, restarting from best or slightly perturbed best
    for restart in range(20): 
        if time.time() > deadline:
            break
        
        if restart == 0:
            pts = best_points[:]
        else:
            # Perturb the best solution
            pts = best_points[:]
            # Move a few points randomly
            for _ in range(3):
                idx = random.randint(0, 29)
                nx = random.randint(-30, 30)
                ny = random.randint(-30, 30)
                if (nx, ny) not in pts:
                    pts[idx] = (nx, ny)
        
        # Local search
        pts, s = _local_search(pts, max_iters=150, deadline=deadline)
        if s > best_score:
            best_score = s
            best_points = pts[:]
            
    # Final polish
    if time.time() < deadline - 2:
        pts, s = _local_search(best_points[:], max_iters=300, deadline=deadline)
        if s > best_score:
            best_score = s
            best_points = pts[:]
    
    return best_points

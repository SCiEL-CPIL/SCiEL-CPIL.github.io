"""Optimized program for maximizing equal-area triangle multiplicity."""
import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

DEADLINE = 55.0  # Safety margin before 60s limit

def _doubled_areas(points):
    """Compute all doubled areas (absolute determinants) for all triples."""
    n = len(points)
    areas = []
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            for k in range(j+1, n):
                xk, yk = points[k]
                det = abs(xi*(yj-yk) - yi*(xj-xk) + xj*yk - yj*xk)
                if det > 0:
                    areas.append(det)
    return areas

def _max_multiplicity(areas):
    """Find the maximum multiplicity of any single area value."""
    if not areas:
        return 0
    c = Counter(areas)
    return max(c.values())

def _score(points):
    """Score: max multiplicity of a single positive doubled area."""
    areas = _doubled_areas(points)
    return _max_multiplicity(areas)

def _valid_points(points):
    """Check that points are 30 distinct integer pairs in [-30,30]^2."""
    if len(points) != 30:
        return False
    seen = set()
    for p in points:
        x, y = p
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if not (-30 <= x <= 30 and -30 <= y <= 30):
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    return True

def _hexagonal_grid_15(cx, cy, scale):
    """Generate 15 points on a hexagonal/triangular lattice centered at (cx,cy)."""
    # Triangular lattice: points at (i*s, j*s*sqrt(3)/2) approximately
    # Use integer-friendly basis: (2,0) and (1, 1) scaled
    # For scale s, basis vectors are (2s, 0) and (s, s)
    points = []
    s = scale
    # Generate points in a hexagonal pattern of 15 points
    # Center + ring of 6 + ring of 8 (or similar)
    # Use axial coordinates for triangular lattice
    # Point = cx + i*(2s) + j*(s), cy + j*(s)
    # Let's use a simpler approach: 5x3 grid rotated
    for i in range(-2, 3):  # -2 to 2
        for j in range(-1, 2):  # -1 to 1
            # Triangular lattice basis: (2,0) and (1,1) times scale
            x = cx + i * (2 * s) + j * (s)
            y = cy + j * (s)
            x, y = int(round(x)), int(round(y))
            if -30 <= x <= 30 and -30 <= y <= 30:
                points.append((x, y))
    # Take first 15 unique points
    unique = list(dict.fromkeys(points))
    return unique[:15]

def _initial_config():
    """Create initial 30-point configuration from two shifted triangular lattices."""
    random.seed(42)
    best_points = None
    best_score = 0
    
    for trial in range(50):
        # Two 15-point triangular lattices
        s1 = random.randint(2, 5)
        s2 = random.randint(2, 5)
        cx1 = random.randint(-10, 10)
        cy1 = random.randint(-10, 10)
        cx2 = random.randint(-10, 10)
        cy2 = random.randint(-10, 10)
        
        pts1 = _hexagonal_grid_15(cx1, cy1, s1)
        pts2 = _hexagonal_grid_15(cx2, cy2, s2)
        
        points = pts1 + pts2
        
        # Remove duplicates
        unique = list(dict.fromkeys(points))
        
        # If we have too many, trim; if too few, add random points
        while len(unique) > 30:
            idx = random.randint(0, len(unique)-1)
            unique.pop(idx)
        while len(unique) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in unique:
                unique.append((x, y))
        
        if len(unique) == 30 and _valid_points(unique):
            sc = _score(unique)
            if sc > best_score:
                best_score = sc
                best_points = unique[:]
    
    if best_points is None:
        # Fallback: simple grid
        best_points = [(x, y) for x in range(6) for y in range(5)]
    
    return best_points, best_score

def construct():
    start_time = time.time()
    
    # Get initial configuration
    points, current_score = _initial_config()
    best_points = points[:]
    best_score = current_score
    
    # Simulated annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    # Precompute indices for speed
    n = 30
    
    while time.time() - start_time < DEADLINE:
        # Check deadline more frequently
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Propose a move: change one point
        idx = random.randint(0, n - 1)
        old_x, old_y = points[idx]
        
        # Random new position near current or far
        if random.random() < 0.7:
            new_x = max(-30, min(30, old_x + random.randint(-5, 5)))
            new_y = max(-30, min(30, old_y + random.randint(-5, 5)))
        else:
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
        
        # Check uniqueness
        new_point = (new_x, new_y)
        if new_point in points[:idx] + points[idx+1:]:
            continue
        
        points[idx] = new_point
        new_score = _score(points)
        
        # Accept or reject
        if new_score >= current_score or random.random() < np.exp((new_score - current_score) / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points[:]
        else:
            points[idx] = (old_x, old_y)
        
        T = max(T_min, T * cooling)
    
    # Final verification
    if _valid_points(best_points):
        return best_points
    else:
        return [(x, y) for x in range(6) for y in range(5)]

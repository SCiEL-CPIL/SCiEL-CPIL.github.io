"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def _triangle_area_vec(pts):
    """Compute areas of all triangles from a set of points. Returns array of areas."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                areas.append(area)
    return np.array(areas)

def _min_area(pts):
    """Compute the minimum area of any triangle formed by three of the points."""
    n = len(pts)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_a:
                    min_a = area
                    if min_a < 1e-12:
                        return 0.0
    return min_a

def _point_in_triangle(p, v0, v1, v2, eps=1e-9):
    """Check if point p is inside or on the boundary of triangle v0-v1-v2."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, v0, v1)
    d2 = sign(p, v1, v2)
    d3 = sign(p, v2, v0)
    
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    
    return not (has_neg and has_pos)

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle."""
    x, y = p
    # Barycentric coordinates
    # p = a*v0 + b*v1 + c*v2 where a+b+c=1
    # v0=(0,0), v1=(1,0), v2=(0.5, sqrt(3)/2)
    # y = c * sqrt(3)/2 => c = 2*y/sqrt(3)
    # x = b + 0.5*c => b = x - 0.5*c
    # a = 1 - b - c
    
    c = 2 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1 - b - c
    
    # Clamp to [0, 1] with small margin
    eps = 1e-10
    a = max(eps, min(1 - 2*eps, a))
    b = max(eps, min(1 - 2*eps, b))
    c = max(eps, min(1 - 2*eps, c))
    
    # Normalize
    s = a + b + c
    a /= s
    b /= s
    c /= s
    
    return a * V0 + b * V1 + c * V2

def _objective_flat(flat_pts):
    """Objective function to minimize (negative of min area)."""
    pts = flat_pts.reshape(-1, 2)
    # Clamp to triangle
    pts_clamped = np.array([_clamp_to_triangle(p) for p in pts])
    min_a = _min_area(pts_clamped)
    return -min_a

def _objective_with_penalty(flat_pts):
    """Objective with penalty for points outside triangle."""
    pts = flat_pts.reshape(-1, 2)
    penalty = 0.0
    for p in pts:
        if not _point_in_triangle(p, V0, V1, V2, eps=1e-6):
            penalty += 1.0
    pts_clamped = np.array([_clamp_to_triangle(p) for p in pts])
    min_a = _min_area(pts_clamped)
    return -min_a + 10.0 * penalty

def _initial_placement():
    """Create a good initial placement using a grid-like pattern."""
    # Use a structured approach: place points at vertices, edge midpoints, and interior
    # For 11 points in an equilateral triangle, use a triangular lattice pattern
    
    # Level 0: 1 point (center of triangle)
    # Level 1: 3 points (near vertices)
    # Level 2: 4 points (near edge midpoints)
    # Level 3: 3 points (additional interior)
    
    # Actually, let's use a more systematic approach
    # Divide the triangle into smaller triangles using a grid
    
    # Use barycentric coordinates for a 3x3 grid (9 points) plus 2 more
    # Or use a hexagonal packing pattern
    
    # Let's use a combination: vertices, edge midpoints, centroid, and some interior points
    points = []
    
    # Vertices (slightly inside)
    margin = 0.05
    v0_in = V0 + margin * (V1 - V0) * 0.5 + margin * (V2 - V0) * 0.5
    v1_in = V1 + margin * (V0 - V1) * 0.5 + margin * (V2 - V1) * 0.5
    v2_in = V2 + margin * (V0 - V2) * 0.5 + margin * (V1 - V2) * 0.5
    
    # Actually, let's just use a simple grid in barycentric coordinates
    # For n=11, use rows of points
    
    # Row 0 (bottom): 4 points
    # Row 1 (middle): 4 points  
    # Row 2 (top): 3 points
    
    pts = []
    
    # Bottom row (y = small)
    y0 = 0.05
    for i in range(4):
        x = 0.05 + i * (0.9 / 3)
        pts.append([x, y0])
    
    # Middle row
    y1 = 0.35
    for i in range(4):
        x = 0.15 + i * (0.7 / 3)
        pts.append([x, y1])
    
    # Top row
    y2 = 0.65
    for i in range(3):
        x = 0.25 + i * (0.5 / 2)
        pts.append([x, y2])
    
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = 130.0  # Leave 20s margin for 150s limit
    
    # Start with a good initial placement
    best_pts = _initial_placement()
    best_score = _min_area(best_pts)
    
    # Try multiple optimization strategies
    rng = np.random.default_rng(42)
    
    # Strategy 1: Local optimization from the initial placement
    flat_init = best_pts.flatten()
    
    try:
        res = minimize(
            _objective_flat,
            flat_init,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-10}
        )
        if time.time() - start_time > deadline - 10:
            return best_pts
        opt_pts = np.array([_clamp_to_triangle(p) for p in res.x.reshape(-1, 2)])
        opt_score = _min_area(opt_pts)
        if opt_score > best_score:
            best_pts = opt_pts
            best_score = opt_score
    except Exception:
        pass
    
    # Strategy 2: Try different initial placements and optimize
    for trial in range(20):
        if time.time() - start_time > deadline - 5:
            break
        
        # Create a perturbed initial placement
        trial_pts = best_pts + rng.normal(0, 0.05, size=(N, 2))
        trial_pts = np.array([_clamp_to_triangle(p) for p in trial_pts])
        
        flat_trial = trial_pts.flatten()
        try:
            res = minimize(
                _objective_flat,
                flat_trial,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            opt_pts = np.array([_clamp_to_triangle(p) for p in res.x.reshape(-1, 2)])
            opt_score = _min_area(opt_pts)
            if opt_score > best_score:
                best_pts = opt_pts
                best_score = opt_score
        except Exception:
            pass
    
    # Strategy 3: Simulated annealing-like refinement
    if time.time() - start_time < deadline - 10:
        current_pts = best_pts.copy()
        current_score = best_score
        T = 0.01
        for iteration in range(2000):
            if time.time() - start_time > deadline - 2:
                break
            
            # Pick a random point and perturb it
            idx = rng.integers(0, N)
            delta = rng.normal(0, T, size=2)
            new_pts = current_pts.copy()
            new_pts[idx] = _clamp_to_triangle(current_pts[idx] + delta)
            
            new_score = _min_area(new_pts)
            
            if new_score > current_score or rng.random() < np.exp((new_score - current_score) / max(T, 1e-10)):
                current_pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_pts = current_pts.copy()
                    best_score = current_score
            
            T *= 0.999
    
    # Final check and return
    best_pts = np.array([_clamp_to_triangle(p) for p in best_pts])
    return best_pts
# EVOLVE-BLOCK-END

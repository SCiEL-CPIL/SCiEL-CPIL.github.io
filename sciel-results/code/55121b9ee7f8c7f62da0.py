"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import time
import itertools
import random

N = 11
H = np.sqrt(3) / 2.0
AREA_EQUI = np.sqrt(3) / 4.0

# Pre-compute all C(11,3) = 165 triangle index combinations
_idx = list(itertools.combinations(range(N), 3))
_IDX_I = np.array([c[0] for c in _idx], dtype=np.int32)
_IDX_J = np.array([c[1] for c in _idx], dtype=np.int32)
_IDX_K = np.array([c[2] for c in _idx], dtype=np.int32)


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the N points.
    pts is shape (N, 2). Returns the minimum absolute area."""
    pi = pts[_IDX_I]
    pj = pts[_IDX_J]
    pk = pts[_IDX_K]
    # Area = 0.5 * |cross(pj - pi, pk - pi)|
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.min(np.abs(cross))


def _is_inside_triangle(pts):
    """Check if all points are inside or on the boundary of the equilateral triangle
    with vertices (0,0), (1,0), (0.5, sqrt(3)/2)."""
    # Triangle inequalities:
    # 1. y >= 0
    # 2. y <= sqrt(3) * x  (below the left edge from (0,0) to (0.5, H))
    # 3. y <= sqrt(3) * (1 - x)  (below the right edge from (1,0) to (0.5, H))
    x = pts[:, 0]
    y = pts[:, 1]
    tol = 1e-9
    return np.all(y >= -tol) and np.all(y <= np.sqrt(3) * x + tol) and np.all(y <= np.sqrt(3) * (1 - x) + tol)


def _project_to_triangle(pts):
    """Project points that are outside the triangle back inside."""
    pts = pts.copy()
    x = pts[:, 0]
    y = pts[:, 1]
    # Clip to y >= 0
    y = np.maximum(y, 0.0)
    # Clip to y <= sqrt(3) * x
    y = np.minimum(y, np.sqrt(3) * x)
    # Clip to y <= sqrt(3) * (1 - x)
    y = np.minimum(y, np.sqrt(3) * (1 - x))
    # Also clip x to [0, 1]
    x = np.clip(x, 0.0, 1.0)
    pts[:, 0] = x
    pts[:, 1] = y
    return pts


def _local_optimize(pts, max_iter=5000, step_init=0.1, step_decay=0.999, min_step=1e-6, deadline=None):
    """Local optimization via gradient-free perturbation (coordinate descent style)."""
    pts = pts.copy()
    best_score = _min_triangle_area(pts)
    step = step_init
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        if step < min_step:
            break
        
        # Try moving each point in small random directions
        improved = False
        for i in range(N):
            for _ in range(4):  # 4 random directions per point
                if deadline is not None and time.time() > deadline:
                    break
                # Random direction
                angle = random.random() * 2 * np.pi
                dx = step * np.cos(angle)
                dy = step * np.sin(angle)
                new_pts = pts.copy()
                new_pts[i, 0] += dx
                new_pts[i, 1] += dy
                new_pts = _project_to_triangle(new_pts)
                
                score = _min_triangle_area(new_pts)
                if score > best_score + 1e-12:
                    pts = new_pts
                    best_score = score
                    improved = True
                    break  # move to next point
        
        if not improved:
            step *= step_decay
    
    return pts, best_score


def _structured_init():
    """Create a structured triangular lattice initialization for 11 points."""
    # The optimal configuration for Heilbronn's problem often has symmetry.
    # For 11 points, a common approach is to use a triangular grid with some points
    # at the vertices and centroid, plus interior points.
    
    # Let's try a configuration based on a 3x3 grid in barycentric coordinates,
    # but we need exactly 11 points.
    
    # Approach: Start with a regular triangular arrangement.
    # Place points at:
    # - 3 vertices of the triangle
    # - Points along edges
    # - Interior points
    
    # A known good starting point for 11 points:
    # Use barycentric coordinates. The triangle has vertices A=(0,0), B=(1,0), C=(0.5, H)
    # Barycentric (a, b, c) with a+b+c=1 maps to:
    # x = a*0 + b*1 + c*0.5 = b + 0.5*c
    # y = a*0 + b*0 + c*H = c*H
    
    # Let's try a 4x4 grid in barycentric space, selecting 11 points.
    # Actually, let's try a simpler structured approach:
    # Divide the triangle into smaller triangles.
    
    # Try: 3 vertices + 3 edge midpoints + 5 interior points in a pentagonal arrangement
    pts = []
    
    # Vertices
    pts.append([0.0, 0.0])
    pts.append([1.0, 0.0])
    pts.append([0.5, H])
    
    # Edge midpoints
    pts.append([0.5, 0.0])  # midpoint of AB
    pts.append([0.75, H/2])  # midpoint of BC
    pts.append([0.25, H/2])  # midpoint of CA
    
    # Interior: centroid and surrounding
    cx, cy = 1/3, H/3
    pts.append([cx, cy])  # centroid
    
    # 4 more interior points forming a small square/diamond around centroid
    r = 0.15
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        pts.append([cx + r*np.cos(angle), cy + r*np.sin(angle)])
    
    return np.array(pts)


def _random_init(rng):
    """Generate a random initial configuration of 11 points inside the triangle."""
    pts = []
    for _ in range(N):
        # Random point in triangle using barycentric coordinates
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        # Barycentric to Cartesian
        x = u * 0 + v * 1 + (1 - u - v) * 0.5
        y = u * 0 + v * 0 + (1 - u - v) * H
        pts.append([x, y])
    return np.array(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    best_pts = None
    best_score = 0.0
    
    # Try multiple initializations
    inits = []
    
    # 1. Structured lattice initialization
    inits.append(_structured_init())
    
    # 2. Random initializations with different seeds
    rng = random.Random(42)
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        inits.append(_random_init(rng))
    
    # 3. A few more structured variants
    # Shift the structured init slightly
    base = _structured_init()
    for shift in [0.02, -0.02, 0.05]:
        variant = base.copy()
        variant[6:, 0] += shift  # shift interior points
        variant = _project_to_triangle(variant)
        inits.append(variant)
    
    # Optimize each initialization
    for init_pts in inits:
        if time.time() > deadline - 5:
            break
        
        init_pts = _project_to_triangle(init_pts)
        init_score = _min_triangle_area(init_pts)
        
        if init_score > best_score:
            best_score = init_score
            best_pts = init_pts.copy()
        
        # Local optimization
        opt_pts, opt_score = _local_optimize(
            init_pts, 
            max_iter=3000,
            step_init=0.05,
            step_decay=0.9995,
            min_step=1e-7,
            deadline=deadline - 5
        )
        
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts.copy()
    
    # If we still have time, do a few more random restarts from the best solution
    if best_pts is not None and time.time() < deadline - 10:
        rng2 = random.Random(123)
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            # Perturb the best solution
            perturbed = best_pts.copy()
            for i in range(N):
                angle = rng2.random() * 2 * np.pi
                r = rng2.random() * 0.1
                perturbed[i, 0] += r * np.cos(angle)
                perturbed[i, 1] += r * np.sin(angle)
            perturbed = _project_to_triangle(perturbed)
            
            opt_pts, opt_score = _local_optimize(
                perturbed,
                max_iter=2000,
                step_init=0.03,
                step_decay=0.9995,
                min_step=1e-7,
                deadline=deadline - 5
            )
            
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
    
    if best_pts is None:
        # Fallback: return the structured init
        best_pts = _structured_init()
    
    # Final safety check: ensure points are inside the triangle
    best_pts = _project_to_triangle(best_pts)
    
    return best_pts

"""Heilbronn triangles: place 11 points to maximize the smallest triangle area."""
N = 11
import numpy as np
from scipy.optimize import minimize
import itertools
import time

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
SQRT3 = np.sqrt(3.0)

def bary_to_cart(bary):
    """Convert barycentric coords (a,b,c) with a+b+c=1 to Cartesian."""
    return bary[:, 0] * V0 + bary[:, 1] * V1 + bary[:, 2] * V2

def cart_to_bary(pts):
    """Convert Cartesian points to barycentric coords (a,b,c)."""
    # px = b + 0.5*c, py = (sqrt(3)/2)*c
    c = 2.0 * pts[:, 1] / SQRT3
    b = pts[:, 0] - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])

def min_triangle_area(pts):
    """Compute minimum triangle area among all triples of points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) -
                                  (pts[k, 0] - pts[i, 0]) * (pts[j, 1] - pts[i, 1]))
                if area < min_area:
                    min_area = area
    return min_area

def min_triangle_area_vec(pts):
    """Vectorized minimum triangle area computation."""
    n = len(pts)
    min_area = float('inf')
    # Precompute
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) -
                                  (pts[k, 0] - pts[i, 0]) * (pts[j, 1] - pts[i, 1]))
                if area < min_area:
                    min_area = area
    return min_area

def negative_min_area_bary(bary_flat):
    """Objective in barycentric coordinates with penalty for out-of-triangle."""
    bary = bary_flat.reshape(-1, 3)
    # Normalize barycentric coords
    s = bary.sum(axis=1, keepdims=True)
    s = np.where(s < 1e-15, 1.0, s)
    bary = bary / s
    # Penalty for negative barycentric coords
    neg = np.maximum(0, -bary)
    penalty = 100.0 * np.sum(neg)
    pts = bary_to_cart(bary)
    area = min_triangle_area(pts)
    return -area + penalty

def coordinate_descent_bary(bary, max_iter=500, tol=1e-12, deadline=None):
    """Coordinate descent on barycentric coordinates."""
    bary = bary.copy()
    n = len(bary)
    # Normalize
    for i in range(n):
        s = bary[i].sum()
        if s < 1e-15:
            bary[i] = np.array([1.0, 0.0, 0.0])
        else:
            bary[i] /= s
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            for coord in range(3):
                # Try to adjust bary[i, coord] while keeping sum=1
                # We can move mass from this coord to others
                old_val = bary[i, coord]
                
                # Try small perturbations
                best_val = old_val
                best_area = -min_triangle_area(bary_to_cart(bary))
                
                # Binary search for optimal value
                lo = 0.0
                hi = 1.0
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    bary[i, coord] = mid
                    # Renormalize: distribute remaining mass proportionally
                    remaining = 1.0 - mid
                    other_coords = [c for c in range(3) if c != coord]
                    old_other_sum = bary[i, other_coords].sum()
                    if old_other_sum > 1e-15:
                        for oc in other_coords:
                            bary[i, oc] = remaining * (bary[i, oc] / old_other_sum)
                    else:
                        # Distribute evenly
                        for oc in other_coords:
                            bary[i, oc] = remaining / 2.0
                    
                    area = -min_triangle_area(bary_to_cart(bary))
                    if area < best_area:
                        best_area = area
                        best_val = mid
                    
                    # Check gradient direction
                    if best_val > mid:
                        lo = mid
                    else:
                        hi = mid
                
                bary[i, coord] = best_val
                # Renormalize
                remaining = 1.0 - best_val
                other_coords = [c for c in range(3) if c != coord]
                old_other_sum = bary[i, other_coords].sum()
                if old_other_sum > 1e-15:
                    for oc in other_coords:
                        bary[i, oc] = remaining * (bary[i, oc] / old_other_sum)
                else:
                    for oc in other_coords:
                        bary[i, oc] = remaining / 2.0
                
                if abs(bary[i, coord] - old_val) > tol:
                    improved = True
        
        if not improved:
            break
    
    # Final normalization
    for i in range(n):
        s = bary[i].sum()
        if s < 1e-15:
            bary[i] = np.array([1.0, 0.0, 0.0])
        else:
            bary[i] /= s
    
    return bary

def nelder_mead_bary(bary_flat, deadline=None):
    """Nelder-Mead optimization in barycentric coordinates."""
    try:
        result = minimize(
            negative_min_area_bary,
            bary_flat,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10,
                     'adaptive': True}
        )
        return result.x
    except Exception:
        return bary_flat

def generate_structured_initialization(seed_offset=0):
    """Generate diverse structured initial points in barycentric coords."""
    rng = np.random.RandomState(42 + seed_offset)
    bary = np.zeros((11, 3))
    
    # Pattern 1: Vertex-clustered
    # 3 points near vertices
    bary[0] = [0.85, 0.075, 0.075]
    bary[1] = [0.075, 0.85, 0.075]
    bary[2] = [0.075, 0.075, 0.85]
    
    # 3 points near edge midpoints
    bary[3] = [0.35, 0.35, 0.30]
    bary[4] = [0.30, 0.35, 0.35]
    bary[5] = [0.35, 0.30, 0.35]
    
    # 3 points in interior
    bary[6] = [0.33, 0.33, 0.34]
    bary[7] = [0.28, 0.36, 0.36]
    bary[8] = [0.36, 0.28, 0.36]
    
    # 2 more interior points
    bary[9] = [0.40, 0.30, 0.30]
    bary[10] = [0.30, 0.40, 0.30]
    
    # Add jitter
    jitter = rng.random((11, 3)) * 0.08
    bary += jitter
    # Normalize
    for i in range(11):
        s = bary[i].sum()
        bary[i] /= s
    
    return bary

def generate_grid_initialization(seed_offset=0):
    """Generate grid-based initial points in barycentric coords."""
    rng = np.random.RandomState(42 + seed_offset)
    bary = np.zeros((11, 3))
    
    # Create a 3x3 grid in barycentric space + center
    # Barycentric grid: (a, b, c) where a+b+c=1
    grid_vals = [0.1, 0.4, 0.5]
    
    idx = 0
    for a in [0.15, 0.35, 0.5]:
        for b in [0.15, 0.35, 0.5]:
            c = 1.0 - a - b
            if c > 0.05 and c < 0.95:
                bary[idx] = [a, b, c]
                idx += 1
                if idx >= 11:
                    break
        if idx >= 11:
            break
    
    # Fill remaining with random
    while idx < 11:
        u = rng.random()
        v = rng.random()
        a = u * 0.7 + 0.05
        b = v * (1 - a) * 0.7 + 0.05
        c = 1.0 - a - b
        if c > 0.05:
            bary[idx] = [a, b, c]
        idx += 1
    
    # Add jitter
    jitter = rng.random((11, 3)) * 0.06
    bary += jitter
    for i in range(11):
        s = bary[i].sum()
        bary[i] /= s
    
    return bary

def generate_random_initialization(seed_offset=0):
    """Generate random initial points in barycentric coords."""
    rng = np.random.RandomState(42 + seed_offset)
    bary = np.zeros((11, 3))
    for i in range(11):
        u = rng.random()
        v = rng.random()
        a = u * 0.85 + 0.05
        b = v * (1 - a) * 0.85 + 0.05
        c = 1.0 - a - b
        bary[i] = [a, b, c]
    return bary

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    best_bary = None
    best_score = -float('inf')
    
    # Generate diverse initializations
    initializations = []
    for s in range(30):
        if s % 3 == 0:
            initializations.append(generate_structured_initialization(s))
        elif s % 3 == 1:
            initializations.append(generate_grid_initialization(s))
        else:
            initializations.append(generate_random_initialization(s))
    
    # Multi-start optimization
    for init_idx, bary_init in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        bary = bary_init.copy()
        
        # Phase 1: Coordinate descent
        bary = coordinate_descent_bary(bary, max_iter=200, deadline=deadline - 15.0)
        
        if time.time() > deadline - 15.0:
            break
        
        # Phase 2: Nelder-Mead polish
        bary_flat = bary.flatten()
        bary_flat = nelder_mead_bary(bary_flat, deadline=deadline - 10.0)
        bary = bary_flat.reshape(-1, 3)
        
        # Normalize and clip
        for i in range(11):
            bary[i] = np.maximum(0, bary[i])
            s = bary[i].sum()
            if s < 1e-15:
                bary[i] = np.array([1.0/3, 1.0/3, 1.0/3])
            else:
                bary[i] /= s
        
        # Evaluate
        pts = bary_to_cart(bary)
        score = min_triangle_area(pts)
        
        if score > best_score:
            best_score = score
            best_bary = bary.copy()
    
    # Final refinement on best result
    if best_bary is not None and time.time() < deadline - 10.0:
        # More coordinate descent iterations
        best_bary = coordinate_descent_bary(best_bary, max_iter=500, deadline=deadline - 5.0)
        for i in range(11):
            best_bary[i] = np.maximum(0, best_bary[i])
            s = best_bary[i].sum()
            if s < 1e-15:
                best_bary[i] = np.array([1.0/3, 1.0/3, 1.0/3])
            else:
                best_bary[i] /= s
        
        # Nelder-Mead polish
        bary_flat = best_bary.flatten()
        bary_flat = nelder_mead_bary(bary_flat, deadline=deadline - 3.0)
        best_bary = bary_flat.reshape(-1, 3)
        for i in range(11):
            best_bary[i] = np.maximum(0, best_bary[i])
            s = best_bary[i].sum()
            if s < 1e-15:
                best_bary[i] = np.array([1.0/3, 1.0/3, 1.0/3])
            else:
                best_bary[i] /= s
    
    if best_bary is None:
        # Fallback
        best_bary = generate_structured_initialization(0)
    
    pts = bary_to_cart(best_bary)
    
    # Final validation: ensure all points are in triangle
    for i in range(len(pts)):
        if pts[i, 0] < -1e-9 or pts[i, 0] > 1 + 1e-9 or pts[i, 1] < -1e-9 or pts[i, 1] > SQRT3/2 + 1e-9:
            # Project to triangle
            px, py = pts[i]
            c = max(0, min(1, 2 * py / SQRT3))
            b = max(0, min(1, px - 0.5 * c))
            a = max(0, 1 - b - c)
            s = a + b + c
            if s < 1e-15:
                pts[i] = V0
            else:
                pts[i] = (a/s) * V0 + (b/s) * V1 + (c/s) * V2
    
    return pts

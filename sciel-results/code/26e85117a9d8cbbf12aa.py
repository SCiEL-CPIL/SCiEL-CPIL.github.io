import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
AREA_EQUI = 0.5 * H

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def get_min_area_info(bary):
    """
    Returns min_area and the list of triangle indices that achieve this min area.
    """
    areas = calc_areas_bary(bary)
    min_val = np.min(areas)
    # Find indices where area is close to min
    threshold = min_val + 1e-8
    active_idx = np.where(areas <= threshold)[0]
    return min_val, active_idx

def optimize_point_local(bary, point_idx, active_triangles, max_iter=50):
    """
    Optimize a single point to maximize the minimum area among active triangles involving that point.
    We use a small local optimization (Nelder-Mead) on just 2 variables (u, v) of the point.
    """
    current_u = bary[point_idx, 0]
    current_v = bary[point_idx, 1]
    
    # Identify which active triangles involve this point
    # active_triangles is an array of indices into tri_idx
    # tri_idx[i] = (i, j, k)
    # We need to find which of these have point_idx in them
    
    # For efficiency, we can just evaluate the objective function which checks all active triangles.
    # But to be faster, we can filter.
    
    def obj(x):
        u, v = x
        new_bary = bary.copy()
        new_bary[point_idx, 0] = u
        new_bary[point_idx, 1] = v
        
        # Clamp to ensure validity
        # We only clamp this point to stay inside, assuming others are fixed and valid
        # u+v <= 1, u>=0, v>=0
        if u < 0 or v < 0 or u + v > 1:
            return -1e10 # Penalty
            
        # Calculate areas for only the active triangles? 
        # No, moving one point changes areas of triangles NOT involving it? No.
        # Triangles NOT involving point_idx are unchanged.
        # So we only need to check triangles involving point_idx.
        
        # However, the "active" set might change. 
        # To be robust, let's calculate all areas, but we can shortcut if we only care about the min.
        # Actually, if we only maximize the min of the *currently* active triangles, we might break others.
        # So we must ensure the global min doesn't drop.
        # Let's just calculate global min for safety, it's fast enough (165 triangles).
        
        areas = calc_areas_bary(new_bary)
        return -np.min(areas)
        
    # Initial guess
    x0 = np.array([current_u, current_v])
    
    # Bounds for Nelder-Mead are handled by penalty or careful initialization
    # We can use constrained optimization if needed, but NM with penalty is often enough for small steps
    # Let's use a very tight NM
    
    try:
        res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.fun < obj(x0): # If improved
            new_u, new_v = res.x
            # Ensure validity
            if new_u >= 0 and new_v >= 0 and new_u + new_v <= 1:
                bary[point_idx, 0] = new_u
                bary[point_idx, 1] = new_v
    except Exception:
        pass
        
    return bary

def coordinated_ascent(bary, time_limit=10.0):
    """
    Perform coordinated ascent: iterate over points, optimizing each one with respect to the current active set.
    """
    start_time = time.time()
    prev_min = -1e10
    
    for _ in range(1000): # Max iterations
        if time.time() - start_time > time_limit:
            break
            
        min_val, active_idx = get_min_area_info(bary)
        
        # If no improvement in a full cycle, stop
        if abs(min_val - prev_min) < 1e-12:
            break
        prev_min = min_val
        
        # Determine which points are involved in the active triangles
        # A point is "active" if it is part of any triangle in active_idx
        active_points = set()
        for t_idx in active_idx:
            i, j, k = tri_idx[t_idx]
            active_points.add(i)
            active_points.add(j)
            active_points.add(k)
            
        # Optimize each active point
        # Sort points by some heuristic? Or just iterate?
        # Iterating in order is fine.
        for p in sorted(active_points):
            bary = optimize_point_local(bary, p, active_idx)
            
    return bary

def generate_structured_inits():
    """
    Generate diverse structured initializations.
    """
    inits = []
    
    # 1. Triangular Lattice (10 pts) + Center
    # Grid 3x3 (10 pts)
    pts = []
    for i in range(4):
        for j in range(4 - i):
            pts.append([i/3.0, j/3.0])
    # Add centroid
    pts.append([1/3.0, 1/3.0])
    inits.append(clamp_bary(np.array(pts)))
    
    # 2. Perturbed Lattice
    rng = np.random.default_rng(42)
    pert = np.array(pts) + rng.normal(0, 0.02, (11, 2))
    inits.append(clamp_bary(pert))
    
    # 3. Hexagonal-ish packing
    # 6 around center, 1 center, 4 corners?
    # Center (1/3, 1/3)
    # Distance 0.15
    center = np.array([1/3.0, 1/3.0])
    hex_pts = [center]
    for angle in np.linspace(0, 2*np.pi, 7)[:-1]:
        # In barycentric, moving in cartesian is complex. 
        # Let's just use random points for the other inits.
        pass
    
    # 4. Random
    u = rng.random(11)
    v = rng.random(11) * (1 - u)
    inits.append(clamp_bary(np.column_stack([u, v])))
    
    # 5. Random 2
    u = rng.random(11)
    v = rng.random(11) * (1 - u)
    inits.append(clamp_bary(np.column_stack([u, v])))
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = -np.inf
    best_bary = None
    
    inits = generate_structured_inits()
    
    # Evaluate initial inits
    for init in inits:
        val = -np.min(calc_areas_bary(init))
        if val > best_val:
            best_val = val
            best_bary = init.copy()
    
    # Phase 1: Coordinated Ascent on the best init
    if time.time() < deadline - 20.0:
        polished = coordinated_ascent(best_bary, time_limit=20.0)
        val = -np.min(calc_areas_bary(polished))
        if val > best_val:
            best_val = val
            best_bary = polished
    
    # Phase 2: Try other inits with Coordinated Ascent
    for init in inits[1:]:
        if time.time() > deadline - 15.0:
            break
            
        # Quick polish
        polished = coordinated_ascent(init.copy(), time_limit=10.0)
        val = -np.min(calc_areas_bary(polished))
        if val > best_val:
            best_val = val
            best_bary = polished
    
    # Phase 3: Random Restarts with Coordinated Ascent
    rng = np.random.default_rng(123)
    while time.time() < deadline - 10.0:
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        rand_bary = clamp_bary(np.column_stack([u, v]))
        
        # Quick polish
        polished = coordinated_ascent(rand_bary, time_limit=5.0)
        val = -np.min(calc_areas_bary(polished))
        if val > best_val:
            best_val = val
            best_bary = polished
    
    # Final safety check
    if best_bary is None:
        # Fallback to first init
        best_bary = inits[0]
        
    cart = bary_to_cart(best_bary)
    return cart

# EVOLVE-BLOCK-END

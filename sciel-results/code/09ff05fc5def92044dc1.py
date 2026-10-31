import numpy as np
from scipy.optimize import minimize
import time
import math
from itertools import combinations

N = 11
TIME_LIMIT = 145.0

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

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

def clamp_bary(bary, eps=1e-12):
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

def epigraph_obj(x):
    """
    Objective function for SLSQP: minimize -t subject to t <= Area_i.
    x = [bary_0_u, bary_0_v, ..., bary_10_u, bary_10_v, t]
    """
    bary = x[:N*2].reshape(N, 2)
    t = x[N*2]
    
    # Ensure barycentric coords are valid for area calculation
    # SLSQP might push slightly out, but constraints handle it. 
    # For safety in area calc, we use the raw values but constraints keep them in.
    areas = calc_areas_bary(bary)
    
    return -t

def epigraph_grad(x):
    """
    Gradient of the objective function.
    Only t has a gradient (-1). Barycentric coords have 0 gradient in the objective.
    """
    grad = np.zeros_like(x)
    grad[N*2] = -1.0
    return grad

def epigraph_constraints(x):
    """
    Constraints: t <= Area_i for all i.
    Formulated as Area_i - t >= 0.
    """
    bary = x[:N*2].reshape(N, 2)
    t = x[N*2]
    
    areas = calc_areas_bary(bary)
    
    # Constraints: areas - t >= 0
    return areas - t

def epigraph_con(x):
    """
    Combined constraints for SLSQP.
    1. t <= Area_i (165 constraints)
    2. Barycentric bounds: u >= 0, v >= 0, u+v <= 1 (3*N constraints)
    """
    bary = x[:N*2].reshape(N, 2)
    t = x[N*2]
    
    areas = calc_areas_bary(bary)
    
    cons = []
    
    # Area constraints
    for i in range(len(areas)):
        cons.append(areas[i] - t)
        
    # Barycentric constraints for each point
    for i in range(N):
        u = bary[i, 0]
        v = bary[i, 1]
        cons.append(u)       # u >= 0
        cons.append(v)       # v >= 0
        cons.append(1 - u - v) # u+v <= 1
        
    return cons

def generate_symmetric_init(seed=0):
    """
    Generate initial configurations based on symmetry templates.
    We exploit D3 symmetry. 
    Points can be classified into orbits under D3.
    Orbits sizes: 1 (center), 3 (vertices/edge midpoints/axis points), 6 (generic).
    
    For N=11:
    Possible partitions of 11 into orbits of size 1, 3, 6:
    1. 1 (center) + 3 (axis) + 3 (axis) + 3 (axis) + 1 (center? no, only 1 center) -> Invalid
    2. 1 (center) + 2*3 (axis) + 1*6 (generic) = 1 + 6 + 6 = 13 (Too many)
    3. 1 (center) + 1*3 (axis) + 1*6 (generic) = 10 (Need 1 more? No, N=11)
    
    Let's look at known good configurations for 11 points.
    Often, points are placed near vertices, midpoints, and center.
    
    Template A: 
    1 Center
    3 Near Vertices (one per vertex)
    3 Near Edge Midpoints (one per edge)
    3 Remaining? 
    Total 10. Need 1 more.
    Maybe 2 Centers? No, symmetry breaks.
    
    Let's try:
    1 Center
    2*3 Axis points (6 pts)
    1*6 Generic points (6 pts)
    Total 13. Too many.
    
    1 Center
    1*3 Axis points (3 pts)
    1*6 Generic points (6 pts)
    Total 10.
    Add 1 more? Breaks symmetry.
    
    Actually, for N=11, a common pattern is:
    1 Center
    3 Points on axes (one on each axis, symmetric) -> 3 pts
    6 Points in pairs (symmetric under 120 deg rotation) -> 6 pts
    Total 10.
    Where is the 11th?
    Maybe 2 points on one axis? No.
    
    Let's just use a "Lattice" initialization which is robust, and a "Perturbed Grid".
    """
    rng = np.random.default_rng(seed)
    
    # Lattice Init:
    # Place points on a triangular lattice inside the triangle.
    # For N=11, a 4x4 grid has 10 points. A 5x5 has 15.
    # Let's use a 4x4 grid (10 pts) + 1 center.
    
    # 4x4 Triangular Lattice
    # Rows of points.
    # Row 0 (top): 1 pt
    # Row 1: 2 pts
    # Row 2: 3 pts
    # Row 3: 4 pts
    # Total 10.
    
    pts = []
    # Barycentric coordinates for lattice
    # u = col * step, v = row * step?
    # In barycentric, a lattice can be defined by u, v being multiples of 1/4.
    # Constraints: u>=0, v>=0, u+v<=1.
    
    step = 0.25
    for i in range(4):
        for j in range(4):
            u = i * step
            v = j * step
            if u + v <= 1.0 + 1e-9:
                pts.append([u, v])
                
    # We have 10 points. Add center [0.5, 0.5]?
    # Center in barycentric is [1/3, 1/3].
    pts.append([1.0/3, 1.0/3])
    
    # We have 11 points.
    bary = np.array(pts)
    
    # Add small random perturbation to break exact symmetry if needed, 
    # but SLSQP can handle symmetry.
    # Let's add a tiny perturbation to help escape flat regions.
    noise = rng.normal(0, 0.01, bary.shape)
    bary += noise
    bary = clamp_bary(bary)
    
    return bary

def generate_random_init(seed):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_vertex_edge_center_init(seed):
    """
    Init with points near vertices, edge midpoints, and center.
    """
    rng = np.random.default_rng(seed)
    bary = np.zeros((N, 2))
    
    # 3 Vertices (perturbed slightly inward)
    bary[0] = [0.1, 0.1] # Near V0 (u=0, v=0) -> wait, barycentric V0 is (0,0) in (u,v) where w=1.
    # V0: u=0, v=0
    # V1: u=0, v=1
    # V2: u=1, v=0
    
    bary[0] = [0.05, 0.05]
    bary[1] = [0.05, 0.95]
    bary[2] = [0.95, 0.05]
    
    # 3 Edge Midpoints
    # Mid V0-V1: u=0, v=0.5
    # Mid V0-V2: u=0.5, v=0
    # Mid V1-V2: u=0.5, v=0.5
    
    bary[3] = [0.0, 0.5] + rng.normal(0, 0.05, 2)
    bary[4] = [0.5, 0.0] + rng.normal(0, 0.05, 2)
    bary[5] = [0.5, 0.5] + rng.normal(0, 0.05, 2)
    
    # 1 Center
    bary[6] = [1/3, 1/3] + rng.normal(0, 0.05, 2)
    
    # 3 More points? 
    # Place them near the "inner" lattice points.
    bary[7] = [0.25, 0.25] + rng.normal(0, 0.05, 2)
    bary[8] = [0.25, 0.5] + rng.normal(0, 0.05, 2)
    bary[9] = [0.5, 0.25] + rng.normal(0, 0.05, 2)
    
    # 1 Last point?
    bary[10] = [0.33, 0.33] + rng.normal(0, 0.1, 2)
    
    bary = clamp_bary(bary)
    return bary

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        if bary is None: return
        bary = clamp_bary(bary)
        areas = calc_areas_bary(bary)
        val = np.min(areas)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Generate multiple initial candidates
    inits = []
    
    # 1. Lattice based
    inits.append(generate_symmetric_init(0))
    
    # 2. Vertex/Edge/Center based
    inits.append(generate_vertex_edge_center_init(1))
    
    # 3. Random starts
    for i in range(5):
        inits.append(generate_random_init(100 + i))
    
    # Run SLSQP on each init
    for i, init_bary in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        # Flatten: [u0, v0, ..., u10, v10, t]
        # Initial t: small value, e.g., 0.01
        x0 = np.concatenate([init_bary.flatten(), [0.01]])
        
        # Bounds for SLSQP
        # u, v in [0, 1]
        # t in [0, 1] (area max is 1/2 in barycentric? No, max area of triangle in unit area triangle is 1/2? 
        # Actually, the area of the big triangle is 1. 
        # In barycentric coords, the determinant gives the area relative to the big triangle.
        # So t is in [0, 1].
        bounds = []
        for _ in range(N):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0)) # t
        
        # Constraints
        # We use the function form for constraints to allow SLSQP to use them efficiently
        # Note: SLSQP supports 'fun' for constraints.
        # But passing 165+33 constraints as a list of scalars is slow.
        # Let's try to pass them as a single array if possible?
        # SLSQP accepts a list of dicts or a single dict with 'fun' returning an array.
        
        # Let's define a single constraint function returning an array.
        def con_fun(x):
            bary = x[:N*2].reshape(N, 2)
            t = x[N*2]
            areas = calc_areas_bary(bary)
            cons = areas - t
            
            # Barycentric constraints
            bary_cons = np.zeros(3*N)
            bary_cons[0::3] = bary[:, 0]       # u
            bary_cons[1::3] = bary[:, 1]       # v
            bary_cons[2::3] = 1.0 - bary[:, 0] - bary[:, 1] # 1-u-v
            
            return np.concatenate([cons, bary_cons])
        
        constraints = {'type': 'ineq', 'fun': con_fun}
        
        try:
            res = minimize(
                epigraph_obj,
                x0,
                method='SLSQP',
                jac=epigraph_grad,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            
            if res.success or (res.fun < 0 and not np.isnan(res.fun)):
                # Extract bary and t
                res_bary = res.x[:N*2].reshape(N, 2)
                res_t = res.x[N*2]
                
                # Verify
                res_bary = clamp_bary(res_bary)
                areas = calc_areas_bary(res_bary)
                min_area = np.min(areas)
                
                # Use the actual min area, not t (t might be slightly off or conservative)
                if min_area > best_val:
                    best_val = min_area
                    best_bary = res_bary.copy()
                    
        except Exception as e:
            pass

    # Fallback
    if best_bary is None:
        best_bary = generate_random_init(0)
        
    cart = bary_to_cart(best_bary)
    return cart

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize

# Helper function to calculate the minimum area ratio for a set of points in barycentric coords
def _min_area_bary(points_bary):
    """
    Calculate the minimum area ratio of triangles formed by N points 
    given in barycentric coordinates (u, v) where w = 1 - u - v.
    The area of a triangle formed by points i, j, k is proportional to 
    the absolute value of the determinant of their barycentric coordinates.
    
    Returns the minimum determinant value (which is proportional to area).
    """
    n = len(points_bary)
    if n < 3:
        return 0.0
        
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    min_det = float('inf')
    
    # Iterate over all unique triplets
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Determinant of the 3x3 matrix with rows (u, v, w)
                # | u_i v_i w_i |
                # | u_j v_j w_j |
                # | u_k v_k w_k |
                # Using Sarrus rule or direct expansion for speed
                det = (u[i] * (v[j] * w[k] - v[k] * w[j]) -
                       v[i] * (u[j] * w[k] - u[k] * w[j]) +
                       w[i] * (u[j] * v[k] - u[k] * v[j]))
                abs_det = abs(det)
                if abs_det < min_det:
                    min_det = abs_det
    return min_det

def _objective(x):
    """
    Objective function for minimization.
    We want to maximize the min area, so we minimize the negative min area.
    Input x is a flat array of 2*N elements (u0, v0, u1, v1, ...).
    """
    points_bary = x.reshape(-1, 2)
    
    # Constraint: points must be inside the triangle
    # u >= 0, v >= 0, u + v <= 1
    # If points go outside, we penalize heavily to keep the optimizer in bounds.
    # However, Nelder-Mead can drift. A better approach is to project or use penalty.
    # Simple penalty: if any point is outside, return a large value (bad score).
    # But for max-min, "bad" means small area. If points are outside, the "area" calculation
    # might still yield a number, but it's geometrically invalid for the domain.
    # We will use a hard penalty for validity.
    
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    
    # Check validity
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(u + v > 1 + 1e-9):
        # Return a very small "min area" (negative) to indicate badness? 
        # No, we are minimizing -area. So we want to return a large positive value 
        # to make the optimizer move away? 
        # Actually, standard Nelder-Mead minimizes. 
        # If we return a large positive value, it's considered "bad" (high cost).
        # We want to maximize area, so we minimize -area.
        # If invalid, we return a large positive number (high cost).
        return 1e6
    
    min_det = _min_area_bary(points_bary)
    
    # Return negative min_det so that minimizing this maximizes min_det
    return -min_det

def _generate_grid_init():
    """Generate a structured grid-based initialization."""
    # A simple 3x3 grid in barycentric coords, plus center
    # Coordinates (u, v)
    coords = []
    for i in range(3):
        for j in range(3):
            # Map 0..2 to 0.1..0.8 to stay away from edges initially
            u = 0.1 + 0.4 * i
            v = 0.1 + 0.4 * j
            if u + v <= 1.0:
                coords.append([u, v])
    
    # If we don't have enough, add random or specific points
    # We need exactly 11 points.
    # The grid above generates:
    # i=0: j=0 (0.1,0.1), j=1 (0.1,0.5), j=2 (0.1,0.9) -> sum 1.0 ok
    # i=1: j=0 (0.5,0.1), j=1 (0.5,0.5) -> sum 1.0 ok, j=2 (0.5,0.9) -> 1.4 no
    # i=2: j=0 (0.9,0.1) -> 1.0 ok, j=1 (0.9,0.5) -> 1.4 no
    # Total 6 points.
    
    # Let's try a denser or different grid.
    # Standard Heilbronn configurations often have symmetry.
    # Let's use a specific known good structure for 11 points:
    # Vertices, Midpoints, Centroid, and points slightly inward.
    
    init = np.array([
        [0.0, 0.0],       # Vertex 1
        [1.0, 0.0],       # Vertex 2
        [0.0, 1.0],       # Vertex 3
        [0.5, 0.0],       # Mid 1-2
        [0.0, 0.5],       # Mid 1-3
        [0.5, 0.5],       # Mid 2-3
        [1.0/3, 1.0/3],   # Centroid
        [0.2, 0.2],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.5, 0.25]
    ])
    
    # Ensure they are valid
    valid = []
    for p in init:
        if p[0] >= 0 and p[1] >= 0 and p[0] + p[1] <= 1.0:
            valid.append(p)
    
    # If not enough, add random
    rng = np.random.RandomState(42)
    while len(valid) < N:
        u, v = rng.random(2)
        if u + v <= 1.0:
            valid.append([u, v])
            
    return np.array(valid[:N])

def _generate_random_init(rng):
    """Generate a random initialization inside the triangle."""
    pts = []
    while len(pts) < N:
        u, v = rng.random(2)
        if u + v <= 1.0:
            pts.append([u, v])
    return np.array(pts)

def construct():
    """
    Main entry point. Returns an array of shape (N, 2) with Cartesian coordinates.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety
    
    best_score = -float('inf')
    best_points_bary = None
    
    # Generate initial configurations
    inits = []
    
    # 1. Grid/Symmetric init
    inits.append(_generate_grid_init())
    
    # 2. Multiple random inits
    rng = np.random.RandomState(123)
    for _ in range(10):
        inits.append(_generate_random_init(rng))
    
    # 3. A slightly perturbed symmetric init
    sym_init = _generate_grid_init()
    noise = rng.normal(0, 0.05, size=sym_init.shape)
    # Project back to triangle if needed, but noise is small
    perturbed = sym_init + noise
    # Simple clipping
    u = perturbed[:, 0]
    v = perturbed[:, 1]
    # Normalize if out of bounds? For now, just check validity
    valid_mask = (u >= 0) & (v >= 0) & (u + v <= 1)
    if np.sum(valid_mask) == N:
        inits.append(perturbed)
    
    # Optimize each initial configuration
    for idx, init in enumerate(inits):
        current_time = time.time()
        if current_time > deadline - 10.0:
            # Not enough time for another full optimization cycle
            break
            
        x0 = init.flatten()
        
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'maxiter': 1000,
                    'maxfev': 1000
                }
            )
            
            # Check if result is valid
            res_points = result.x.reshape(-1, 2)
            u = res_points[:, 0]
            v = res_points[:, 1]
            
            if np.all(u >= -1e-9) and np.all(v >= -1e-9) and np.all(u + v <= 1 + 1e-9):
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_points_bary = res_points.copy()
        except Exception:
            continue

    # If no optimization succeeded, use the best initial
    if best_points_bary is None:
        best_points_bary = inits[0]
    
    # Convert barycentric to Cartesian
    # Reference triangle vertices in Cartesian:
    # A = (0, 0)
    # B = (1, 0)
    # C = (0.5, sqrt(3)/2)
    # Point P = u*A + v*B + w*C
    # x = u*0 + v*1 + w*0.5 = v + (1-u-v)*0.5 = 0.5 + 0.5v - 0.5u
    # y = u*0 + v*0 + w*(sqrt(3)/2) = (1-u-v)*sqrt(3)/2
    
    u = best_points_bary[:, 0]
    v = best_points_bary[:, 1]
    w = 1.0 - u - v
    
    cart_x = v + 0.5 * w
    cart_y = w * (3 ** 0.5) / 2.0
    
    return np.column_stack([cart_x, cart_y])

# EVOLVE-BLOCK-END

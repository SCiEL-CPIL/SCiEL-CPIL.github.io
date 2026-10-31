import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 16
D = 2

# Precompute upper triangle indices for speed
IU = np.triu_indices(N, k=1)

def pairwise_distances_sq(pts):
    """Compute all pairwise squared distances (faster, no sqrt)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    return dist_sq[IU]

def objective(x):
    """(max_dist / min_dist)^2"""
    pts = x.reshape(N, D)
    d_sq = pairwise_distances_sq(pts)
    if np.any(d_sq < 1e-24):
        return 1e10
    d = np.sqrt(d_sq)
    ratio = d.max() / d.min()
    return ratio**2

def objective_smooth(x, beta=50.0):
    """
    Smooth approximation using log-sum-exp on distances.
    Uses log-space for better numerical stability.
    """
    pts = x.reshape(N, D)
    d_sq = pairwise_distances_sq(pts)
    if np.any(d_sq < 1e-24):
        return 1e10
    d = np.sqrt(d_sq)
    
    # Soft max of d
    max_d = np.max(d)
    soft_max_d = max_d + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d - max_d))))
    
    # Soft min of d
    min_d = np.min(d)
    soft_min_d = min_d - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d - min_d))))
    
    if soft_min_d < 1e-12:
        return 1e10
        
    ratio = soft_max_d / soft_min_d
    return ratio**2

def make_seeds():
    """Generate multiple diverse initial configurations."""
    seeds = []
    
    # 1. 4x4 grid
    grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    seeds.append(grid.flatten())
    
    # 2. Hexagonal lattice subset
    hex_pts = []
    for row in range(4):
        x_offset = (row % 2) * 0.5
        for col in range(4):
            hex_pts.append([col + x_offset, row * np.sqrt(3)/2])
    hex_pts = np.array(hex_pts[:16])
    seeds.append(hex_pts.flatten())
    
    # 3. Concentric circles: 1 center + 5 + 10
    pts = [np.array([0.0, 0.0])]
    for i in range(5):
        angle = 2 * np.pi * i / 5
        pts.append(np.array([1.0 * np.cos(angle), 1.0 * np.sin(angle)]))
    for i in range(10):
        angle = 2 * np.pi * i / 10 + np.pi/10
        pts.append(np.array([2.2 * np.cos(angle), 2.2 * np.sin(angle)]))
    seeds.append(np.array(pts).flatten())
    
    # 4. Two concentric octagons
    pts = []
    for i in range(8):
        angle = 2 * np.pi * i / 8
        pts.append(np.array([1.0 * np.cos(angle), 1.0 * np.sin(angle)]))
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi/8
        pts.append(np.array([2.0 * np.cos(angle), 2.0 * np.sin(angle)]))
    seeds.append(np.array(pts).flatten())
    
    # 5. 1 center + 6 + 9 (hexagon + outer)
    pts = [np.array([0.0, 0.0])]
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append(np.array([1.0 * np.cos(angle), 1.0 * np.sin(angle)]))
    for i in range(9):
        angle = 2 * np.pi * i / 9 + np.pi/9
        pts.append(np.array([2.0 * np.cos(angle), 2.0 * np.sin(angle)]))
    seeds.append(np.array(pts).flatten())
    
    # 6. 1 center + 7 + 8
    pts = [np.array([0.0, 0.0])]
    for i in range(7):
        angle = 2 * np.pi * i / 7
        pts.append(np.array([1.0 * np.cos(angle), 1.0 * np.sin(angle)]))
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi/8
        pts.append(np.array([2.2 * np.cos(angle), 2.2 * np.sin(angle)]))
    seeds.append(np.array(pts).flatten())
    
    # 7. 1 center + 8 + 7
    pts = [np.array([0.0, 0.0])]
    for i in range(8):
        angle = 2 * np.pi * i / 8
        pts.append(np.array([1.0 * np.cos(angle), 1.0 * np.sin(angle)]))
    for i in range(7):
        angle = 2 * np.pi * i / 7 + np.pi/7
        pts.append(np.array([2.0 * np.cos(angle), 2.0 * np.sin(angle)]))
    seeds.append(np.array(pts).flatten())
    
    # 8. 2 center + 6 + 8
    pts = [np.array([-0.5, 0.0]), np.array([0.5, 0.0])]
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append(np.array([1.5 * np.cos(angle), 1.5 * np.sin(angle)]))
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi/8
        pts.append(np.array([2.8 * np.cos(angle), 2.8 * np.sin(angle)]))
    seeds.append(np.array(pts).flatten())
    
    # 9. Regular 16-gon
    pts = []
    for i in range(16):
        angle = 2 * np.pi * i / 16
        pts.append(np.array([np.cos(angle), np.sin(angle)]))
    seeds.append(np.array(pts).flatten())
    
    # 10. 4x4 grid with slight perturbation
    grid2 = np.array([(i + 0.1*(i%2), j + 0.1*(j%2)) for i in range(4) for j in range(4)], dtype=float)
    seeds.append(grid2.flatten())
    
    return seeds

def make_deformed_grid(params):
    """
    Create a 4x4 grid with low-rank sinusoidal deformation.
    params: [a1, a2, a3, a4, b1, b2, b3, b4, cx, cy, sx, sy, rot]
    """
    a1, a2, a3, a4, b1, b2, b3, b4, cx, cy, sx, sy, rot = params
    
    i = np.arange(4)
    j = np.arange(4)
    I, J = np.meshgrid(i - 1.5, j - 1.5)
    base_x = I
    base_y = J
    
    r = np.sqrt(base_x**2 + base_y**2)
    theta = np.arctan2(base_y, base_x)
    
    deform_x = a1 * np.sin(theta) * np.cos(r) + a2 * np.cos(2*theta) * np.sin(r)
    deform_y = b1 * np.cos(theta) * np.cos(r) + b2 * np.sin(2*theta) * np.sin(r)
    
    deform_x += a3 * np.sin(3*theta) * r**0.5 + a4 * np.cos(3*theta) * r**0.5
    deform_y += b3 * np.cos(3*theta) * r**0.5 + b4 * np.sin(3*theta) * r**0.5
    
    x = base_x + deform_x
    y = base_y + deform_y
    
    x_scaled = sx * x
    y_scaled = sy * y
    
    cos_r = np.cos(rot)
    sin_r = np.sin(rot)
    x_rot = x_scaled * cos_r - y_scaled * sin_r + cx
    y_rot = x_scaled * sin_r + y_scaled * cos_r + cy
    
    pts = np.column_stack([x_rot.ravel(), y_rot.ravel()])
    return pts

def objective_deformed(params):
    """Objective for deformed grid parameters."""
    pts = make_deformed_grid(params)
    d_sq = pairwise_distances_sq(pts)
    if np.any(d_sq < 1e-10):
        return 1e10
    d = np.sqrt(d_sq)
    ratio = d.max() / d.min()
    return ratio**2

def local_optimize(x, deadline):
    """
    Multi-stage local optimization:
    1. Nelder-Mead on exact objective (robust).
    2. Powell on exact objective.
    3. L-BFGS-B on smooth objective with annealing (exploits gradients).
    """
    best_x = x.copy()
    best_val = objective(x)
    
    # Stage 1: Nelder-Mead
    if time.time() < deadline - 2:
        try:
            res = minimize(
                objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            val = objective(res.x)
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Stage 2: Powell
    if time.time() < deadline - 2:
        try:
            res = minimize(
                objective,
                best_x,
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            val = objective(res.x)
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Stage 3: L-BFGS-B with annealing on smooth objective
    betas = [10.0, 50.0, 200.0]
    for beta in betas:
        if time.time() > deadline - 1:
            break
        try:
            res = minimize(
                lambda p: objective_smooth(p, beta=beta),
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            val = objective(res.x)
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
        except Exception:
            pass
            
    return best_x, best_val

def construct():
    start_time = time.time()
    # Safety margin: 120s limit, use 110s for safety
    deadline = start_time + 110.0
    
    best_x = None
    best_val = np.inf
    
    def update_best(x, val):
        nonlocal best_x, best_val
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
    # Phase 1: Global Search with Differential Evolution on deformed grid
    # This explores a structured family of shapes efficiently.
    if time.time() < deadline - 40:
        try:
            bounds = [(-0.3, 0.3)] * 8 + [(-1.0, 1.0), (-1.0, 1.0), (0.5, 2.0), (0.5, 2.0), (-0.5, 0.5)]
            de_res = differential_evolution(
                objective_deformed,
                bounds,
                maxiter=50,
                popsize=10,
                tol=1e-5,
                seed=42,
                polish=True,
                init='latinhypercube'
            )
            if de_res.success or de_res.fun < best_val:
                de_pts = make_deformed_grid(de_res.x)
                val = objective(de_pts.flatten())
                update_best(de_pts.flatten(), val)
        except Exception:
            pass

    # Phase 2: Multi-start Local Optimization from diverse seeds
    seeds = make_seeds()
    
    # Also add the DE result as a seed if found
    if best_x is not None:
        seeds.append(best_x)
        
    for seed_idx, seed in enumerate(seeds):
        if time.time() > deadline - 10:
            break
        
        # Ensure seed is valid shape
        if seed.shape != (N*D,):
            continue
            
        # Run local optimization
        try:
            opt_x, opt_val = local_optimize(seed, deadline)
            update_best(opt_x, opt_val)
        except Exception:
            pass

    # Phase 3: Final Polish with tight tolerances on the best solution
    if best_x is not None and time.time() < deadline - 1:
        try:
            res = minimize(
                objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            val = objective(res.x)
            update_best(res.x, val)
        except Exception:
            pass

    if best_x is None:
        # Fallback: 4x4 grid
        grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        return grid
    
    return best_x.reshape(N, D)

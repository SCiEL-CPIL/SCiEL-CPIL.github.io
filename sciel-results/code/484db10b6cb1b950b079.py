import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 16
D = 2

def pairwise_distances(pts):
    """Compute all pairwise distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    iu = np.triu_indices(N, k=1)
    return dists[iu]

def pairwise_distances_sq(pts):
    """Compute all pairwise squared distances (faster, no sqrt)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    iu = np.triu_indices(N, k=1)
    return dist_sq[iu]

def objective(x):
    """(max_dist / min_dist)^2"""
    pts = x.reshape(N, D)
    d_sq = pairwise_distances_sq(pts)
    if np.any(d_sq < 1e-24):
        return 1e10
    d = np.sqrt(d_sq)
    ratio = d.max() / d.min()
    return ratio**2

def objective_smooth(x, beta=200.0):
    """Smooth approximation using log-sum-exp on distances."""
    pts = x.reshape(N, D)
    d_sq = pairwise_distances_sq(pts)
    if np.any(d_sq < 1e-24):
        return 1e10
    d = np.sqrt(d_sq)
    # log-sum-exp for max
    max_d = np.log(np.sum(np.exp(beta * d))) / beta
    # For min, use negative log-sum-exp
    min_d = -np.log(np.sum(np.exp(-beta * d))) / beta
    if min_d < 1e-12:
        return 1e10
    ratio = max_d / min_d
    return ratio**2

def make_ring_topology(n_inner=6, n_outer=9, r_inner=1.0, r_outer=2.0, angle_offset=0.0):
    """Create ring topology: center + inner ring + outer ring."""
    pts = [np.array([0.0, 0.0])]
    for i in range(n_inner):
        angle = 2 * np.pi * i / n_inner + angle_offset
        pts.append(np.array([r_inner * np.cos(angle), r_inner * np.sin(angle)]))
    for i in range(n_outer):
        angle = 2 * np.pi * i / n_outer + angle_offset + np.pi / n_outer
        pts.append(np.array([r_outer * np.cos(angle), r_outer * np.sin(angle)]))
    return np.array(pts)

def make_hexagonal_cluster(params):
    """
    Generate 16 points as a hexagonal lattice cluster with 3 parameters:
    params[0] = aspect ratio (y-scale relative to x)
    params[1] = shear angle (radians)
    params[2] = radial scaling factor for outer ring
    """
    aspect = params[0]
    shear = params[1]
    outer_scale = params[2]
    
    base_points = []
    # Row 0 (top): 4 points
    for i in range(4):
        x = (i - 1.5) * 1.0
        y = 2.0
        base_points.append([x, y])
    # Row 1: 3 points, offset by 0.5
    for i in range(3):
        x = (i - 1.0) * 1.0 + 0.5
        y = 2.0 - np.sqrt(3)/2
        base_points.append([x, y])
    # Row 2: 4 points
    for i in range(4):
        x = (i - 1.5) * 1.0
        y = 2.0 - np.sqrt(3)
        base_points.append([x, y])
    # Row 3: 3 points, offset by 0.5
    for i in range(3):
        x = (i - 1.0) * 1.0 + 0.5
        y = 2.0 - 1.5*np.sqrt(3)
        base_points.append([x, y])
    # Row 4 (bottom): 2 points
    for i in range(2):
        x = (i - 0.5) * 1.0
        y = 2.0 - 2.0*np.sqrt(3)
        base_points.append([x, y])
    
    base_points = np.array(base_points, dtype=float)
    
    # Center the points
    centroid = base_points.mean(axis=0)
    base_points -= centroid
    
    # Apply aspect ratio (scale y)
    base_points[:, 1] *= aspect
    
    # Apply shear transformation
    shear_matrix = np.array([[1, np.tan(shear)], [0, 1]])
    base_points = base_points @ shear_matrix.T
    
    # Identify outer ring points (those farther from center)
    dists_from_center = np.linalg.norm(base_points, axis=1)
    median_dist = np.median(dists_from_center)
    outer_mask = dists_from_center > median_dist
    
    # Apply radial scaling to outer ring only
    if np.any(outer_mask):
        outer_points = base_points[outer_mask]
        outer_dists = np.linalg.norm(outer_points, axis=1)
        outer_dists[outer_dists < 1e-12] = 1e-12
        scale_factors = outer_scale / outer_dists
        base_points[outer_mask] *= scale_factors[:, np.newaxis]
    
    return base_points

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
    
    # 5. 1 center + 6 + 9 (hexagon + outer) - known good topology
    seeds.append(make_ring_topology(6, 9, 1.0, 2.0).flatten())
    
    # 6. 1 center + 7 + 8
    seeds.append(make_ring_topology(7, 8, 1.0, 2.2).flatten())
    
    # 7. 1 center + 8 + 7
    seeds.append(make_ring_topology(8, 7, 1.0, 2.0).flatten())
    
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
    
    # 11-14. Hexagonal cluster variants
    hex_params_list = [
        [1.0, 0.0, 1.0],
        [1.1, 0.1, 1.05],
        [0.9, -0.1, 0.95],
        [1.2, 0.2, 1.1],
    ]
    for params in hex_params_list:
        pts = make_hexagonal_cluster(np.array(params))
        seeds.append(pts.flatten())
    
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_x = None
    best_val = np.inf
    
    def update_best(x, val):
        nonlocal best_x, best_val
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    # Phase 1: Optimize hexagonal cluster parameters (3 DOF) - fast
    hex_param_seeds = [
        [1.0, 0.0, 1.0],
        [1.1, 0.1, 1.05],
        [0.9, -0.1, 0.95],
        [1.2, 0.2, 1.1],
        [0.8, 0.0, 1.2],
    ]
    
    for hex_seed in hex_param_seeds:
        if time.time() > deadline - 10:
            break
        try:
            def hex_objective(params):
                pts = make_hexagonal_cluster(params)
                return objective_smooth(pts.flatten(), beta=200.0)
            
            res = minimize(
                hex_objective,
                hex_seed,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.fun < best_val:
                hex_pts = make_hexagonal_cluster(res.x)
                update_best(hex_pts.flatten(), res.fun)
        except Exception:
            pass
    
    # Phase 2: Differential Evolution on deformed grid parameters
    bounds = [(-0.3, 0.3)] * 8 + [(-1.0, 1.0), (-1.0, 1.0), (0.5, 2.0), (0.5, 2.0), (-0.5, 0.5)]
    
    if time.time() < deadline - 50:
        try:
            de_res = differential_evolution(
                objective_deformed,
                bounds,
                maxiter=80,
                popsize=12,
                tol=1e-6,
                seed=42,
                polish=True,
                init='latinhypercube'
            )
            if de_res.success or de_res.fun < best_val:
                de_pts = make_deformed_grid(de_res.x)
                update_best(de_pts.flatten(), de_res.fun)
        except Exception:
            pass
    
    # Phase 3: Optimize from multiple seeds using Nelder-Mead and Powell
    seeds = make_seeds()
    
    for seed_idx, seed in enumerate(seeds):
        if time.time() > deadline - 15:
            break
        
        # Try Nelder-Mead
        if time.time() < deadline - 10:
            try:
                res = minimize(
                    objective,
                    seed,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                val = objective(res.x)
                update_best(res.x, val)
            except Exception:
                pass
        
        # Try Powell
        if time.time() < deadline - 5:
            try:
                res = minimize(
                    objective,
                    seed,
                    method='Powell',
                    options={'maxiter': 1500, 'xtol': 1e-8, 'ftol': 1e-8}
                )
                val = objective(res.x)
                update_best(res.x, val)
            except Exception:
                pass
    
    # Phase 4: Final polish with tight tolerances
    if best_x is not None and time.time() < deadline - 3:
        try:
            res = minimize(
                objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            val = objective(res.x)
            update_best(res.x, val)
        except Exception:
            pass
    
    if best_x is None:
        grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        return grid
    
    return best_x.reshape(N, D)

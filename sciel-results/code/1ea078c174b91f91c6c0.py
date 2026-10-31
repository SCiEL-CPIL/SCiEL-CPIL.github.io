import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 16
D = 2
TIME_LIMIT = 150.0

def pairwise_distances(pts):
    """Compute all pairwise distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    # Exclude self-distances
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
    # Handle degenerate cases
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
    # log-sum-exp for max (numerically stable)
    d_max_val = d.max()
    max_d = d_max_val + np.log(np.sum(np.exp(beta * (d - d_max_val)))) / beta
    # For min, use negative log-sum-exp: min = -logsumexp(-beta*d)/beta
    d_min_val = d.min()
    min_d = d_min_val + np.log(np.sum(np.exp(-beta * (d - d_min_val)))) / beta
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

def make_deformed_grid(params):
    """
    Create a 4x4 grid with low-rank sinusoidal deformation.
    params: [a1, a2, a3, a4, b1, b2, b3, b4, cx, cy, sx, sy, rot]
    - a1..a4: amplitudes for x-deformation modes
    - b1..b4: amplitudes for y-deformation modes  
    - cx, cy: center offset
    - sx, sy: scale
    - rot: rotation angle
    """
    a1, a2, a3, a4, b1, b2, b3, b4, cx, cy, sx, sy, rot = params
    
    # Base 4x4 grid centered at origin
    i = np.arange(4)
    j = np.arange(4)
    I, J = np.meshgrid(i - 1.5, j - 1.5)
    base_x = I
    base_y = J
    
    # Sinusoidal deformation modes
    # Mode 1: radial breathing
    r = np.sqrt(base_x**2 + base_y**2)
    theta = np.arctan2(base_y, base_x)
    
    # Low-rank deformations
    deform_x = a1 * np.sin(theta) * np.cos(r) + a2 * np.cos(2*theta) * np.sin(r)
    deform_y = b1 * np.cos(theta) * np.cos(r) + b2 * np.sin(2*theta) * np.sin(r)
    
    # Higher order modes
    deform_x += a3 * np.sin(3*theta) * r**0.5 + a4 * np.cos(3*theta) * r**0.5
    deform_y += b3 * np.cos(3*theta) * r**0.5 + b4 * np.sin(3*theta) * r**0.5
    
    x = base_x + deform_x
    y = base_y + deform_y
    
    # Apply scaling and rotation
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
    # Check for distinct points
    d_sq = pairwise_distances_sq(pts)
    if np.any(d_sq < 1e-10):
        return 1e10
    d = np.sqrt(d_sq)
    ratio = d.max() / d.min()
    return ratio**2

def make_fourier_polygon(c0, c2, c4, c6, r_base=1.0):
    """
    Create 16 points on a star-shaped convex polygon using even-harmonic Fourier series.
    r(θ) = r_base * (c0 + c2*cos(2θ) + c4*cos(4θ) + c6*cos(6θ))
    Only even harmonics → central symmetry (r(θ) = r(θ+π)).
    Points are placed at equal angular intervals.
    """
    thetas = 2 * np.pi * np.arange(N) / N
    r = r_base * (c0 + c2 * np.cos(2 * thetas) + c4 * np.cos(4 * thetas) + c6 * np.cos(6 * thetas))
    # Ensure positive radii
    r = np.maximum(r, 0.1)
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    return np.column_stack([x, y])

def objective_fourier(params):
    """Objective for Fourier polygon parameters: [c0, c2, c4, c6, r_base]."""
    c0, c2, c4, c6, r_base = params
    pts = make_fourier_polygon(c0, c2, c4, c6, r_base)
    d_sq = pairwise_distances_sq(pts)
    if np.any(d_sq < 1e-10):
        return 1e10
    d = np.sqrt(d_sq)
    ratio = d.max() / d.min()
    return ratio**2

def objective_fourier_smooth(params, beta=100.0):
    """Smooth objective for Fourier polygon parameters."""
    c0, c2, c4, c6, r_base = params
    pts = make_fourier_polygon(c0, c2, c4, c6, r_base)
    d_sq = pairwise_distances_sq(pts)
    if np.any(d_sq < 1e-24):
        return 1e10
    d = np.sqrt(d_sq)
    d_max_val = d.max()
    max_d = d_max_val + np.log(np.sum(np.exp(beta * (d - d_max_val)))) / beta
    d_min_val = d.min()
    min_d = d_min_val + np.log(np.sum(np.exp(-beta * (d - d_min_val)))) / beta
    if min_d < 1e-12:
        return 1e10
    ratio = max_d / min_d
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
    
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    best_x = None
    best_val = np.inf
    
    def update_best(x, val):
        nonlocal best_x, best_val
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    # Phase 1a: Optimize Fourier polygon (low-dim manifold, convex star-shaped)
    if time.time() < deadline - 50:
        # Fourier params: [c0, c2, c4, c6, r_base]
        fourier_bounds = [(0.5, 1.5), (-0.4, 0.4), (-0.3, 0.3), (-0.2, 0.2), (0.5, 2.0)]
        try:
            # First try with smooth objective for better landscape
            de_res_f = differential_evolution(
                objective_fourier_smooth,
                fourier_bounds,
                maxiter=80,
                popsize=12,
                tol=1e-6,
                seed=42,
                polish=True,
                init='latinhypercube'
            )
            if de_res_f.fun < best_val:
                f_pts = make_fourier_polygon(*de_res_f.x)
                exact_val = objective(f_pts.flatten())
                update_best(f_pts.flatten(), exact_val)
        except Exception:
            pass

        # Refine with exact objective
        if time.time() < deadline - 45:
            try:
                res_f = minimize(
                    objective_fourier,
                    de_res_f.x,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
                )
                f_pts = make_fourier_polygon(*res_f.x)
                val = objective(f_pts.flatten())
                update_best(f_pts.flatten(), val)
            except Exception:
                pass

    # Phase 1b: Differential Evolution on deformed grid parameters
    # Parameters: [a1,a2,a3,a4, b1,b2,b3,b4, cx,cy, sx,sy, rot]
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
    
    # Phase 2: Optimize from multiple seeds using Nelder-Mead and Powell
    seeds = make_seeds()
    
    for seed_idx, seed in enumerate(seeds):
        if time.time() > deadline - 25:
            break
        
        # Try Nelder-Mead
        if time.time() < deadline - 20:
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
        if time.time() < deadline - 15:
            try:
                res = minimize(
                    objective,
                    seed,
                    method='Powell',
                    options={'maxiter': 1000, 'xtol': 1e-8, 'ftol': 1e-8}
                )
                val = objective(res.x)
                update_best(res.x, val)
            except Exception:
                pass
        
        # Try L-BFGS with smooth objective for better gradient-based refinement
        if time.time() < deadline - 12:
            try:
                res_smooth = minimize(
                    lambda x: objective_smooth(x, beta=150.0),
                    seed,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-10}
                )
                # Then refine with exact objective
                res_exact = minimize(
                    objective,
                    res_smooth.x,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
                )
                val = objective(res_exact.x)
                update_best(res_exact.x, val)
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish
    if best_x is not None and time.time() < deadline - 5:
        # First: smooth L-BFGS to navigate out of any rough local minimum
        if time.time() < deadline - 8:
            try:
                res_smooth = minimize(
                    lambda x: objective_smooth(x, beta=300.0),
                    best_x,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12}
                )
                val = objective(res_smooth.x)
                update_best(res_smooth.x, val)
            except Exception:
                pass
        
        # Then: tight Nelder-Mead polish
        if time.time() < deadline - 3:
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
        
        # Then: Powell for final axis-aligned refinement
        if time.time() < deadline - 1:
            try:
                res = minimize(
                    objective,
                    best_x,
                    method='Powell',
                    options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-10}
                )
                val = objective(res.x)
                update_best(res.x, val)
            except Exception:
                pass
    
    if best_x is None:
        grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        return grid
    
    return best_x.reshape(N, D)

"""Rewritten program for sphere packing in a unit cube."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100


def _fcc_lattice_init(n_points):
    """Generate initial points from an FCC lattice structure."""
    # FCC lattice: points at corners and face centers of a cube
    # Scale to fit in [-0.5, 0.5]^3
    points = []
    # Generate FCC points in a larger grid, then select closest to center
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ])
    
    # Generate lattice points in a range
    extent = 3
    lattice_points = []
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    lattice_points.append(p)
    
    lattice_points = np.array(lattice_points)
    
    # Scale to fit in [-0.5, 0.5]^3
    # The lattice spans roughly [-extent, extent+0.5] in each dimension
    span = 2 * extent + 0.5
    scale = 0.95 / span  # leave some margin
    lattice_points *= scale
    
    # Center at origin
    lattice_points -= np.mean(lattice_points, axis=0)
    
    # Select the n_points closest to origin
    if len(lattice_points) >= n_points:
        dists_to_origin = np.linalg.norm(lattice_points, axis=1)
        idx = np.argsort(dists_to_origin)[:n_points]
        points = lattice_points[idx]
    else:
        points = lattice_points
        # Pad with random points if needed
        while len(points) < n_points:
            points = np.vstack([points, np.random.randn(1, 3) * 0.1])
    
    # Clip to stay within bounds
    points = np.clip(points, -0.45, 0.45)
    
    return points


def _compute_radius(c):
    """Compute the exact minimum radius for configuration c."""
    c = c.reshape(N, 3)
    wall_dists = np.min(0.5 - np.abs(c), axis=1)
    dists = cdist(c, c)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(np.min(wall_dists), min_pair_dist / 2.0)


def _soft_min_radius(c, beta=10.0):
    """
    Smooth approximation of the radius using soft-min via log-sum-exp.
    """
    c = c.reshape(N, 3)
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    min_wall_per_point = np.min(wall_dists, axis=1)  # shape (N,)
    
    # Pairwise distances
    dists = cdist(c, c)
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    mask = ~np.eye(N, dtype=bool)
    pair_vals_flat = pair_vals[mask]
    
    # Soft-min of pairwise distances
    if len(pair_vals_flat) > 0:
        neg_vals = -pair_vals_flat / beta
        max_neg = np.max(neg_vals)
        soft_min_pair = -beta * (max_neg + np.log(np.sum(np.exp(neg_vals - max_neg))))
    else:
        soft_min_pair = np.inf
    
    # Soft-min of wall distances
    neg_wall = -min_wall_per_point / beta
    max_neg_wall = np.max(neg_wall)
    soft_min_wall = -beta * (max_neg_wall + np.log(np.sum(np.exp(neg_wall - max_neg_wall))))
    
    # Combine: soft-min of the two aggregate values
    vals = np.array([soft_min_wall, soft_min_pair])
    neg_comb = -vals / beta
    max_neg_comb = np.max(neg_comb)
    result = -beta * (max_neg_comb + np.log(np.sum(np.exp(neg_comb - max_neg_comb))))
    
    return result


def _soft_min_radius_grad(c, beta=10.0):
    """
    Gradient of -soft_min_radius with respect to c.
    Returns negative gradient (for maximization via minimization).
    """
    c = c.reshape(N, 3)
    grad = np.zeros_like(c)
    
    # Wall distances gradient
    wall_dists = 0.5 - np.abs(c)
    min_wall_idx = np.argmin(wall_dists, axis=1)  # (N,)
    min_wall_vals = wall_dists[np.arange(N), min_wall_idx]  # (N,)
    
    # Soft-min weights for wall distances
    neg_wall = -min_wall_vals / beta
    max_neg_wall = np.max(neg_wall)
    exp_wall = np.exp(neg_wall - max_neg_wall)
    weights_wall = exp_wall / np.sum(exp_wall)  # (N,)
    
    # For each point, the gradient of -min_wall is in the direction of the min coordinate
    for i in range(N):
        coord_idx = min_wall_idx[i]
        # d(0.5 - |c_ik|)/dc_ik = -sign(c_ik)
        sign_val = np.sign(c[i, coord_idx])
        if sign_val == 0:
            sign_val = 1.0
        grad[i, coord_idx] += weights_wall[i] * (-sign_val)
    
    # Pairwise distances gradient
    dists = cdist(c, c)
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    mask = ~np.eye(N, dtype=bool)
    pair_vals_flat = pair_vals[mask]
    
    if len(pair_vals_flat) > 0:
        neg_vals = -pair_vals_flat / beta
        max_neg = np.max(neg_vals)
        exp_vals = np.exp(neg_vals - max_neg)
        weights_pair = exp_vals / np.sum(exp_vals)  # (N*(N-1)/2,)
        
        # Get indices for the flattened upper triangle
        i_idx, j_idx = np.where(mask)
        
        for idx in range(len(i_idx)):
            i, j = i_idx[idx], j_idx[idx]
            w = weights_pair[idx]
            diff = c[i] - c[j]
            dist = dists[i, j]
            if dist > 1e-12:
                # d(dist_ij)/dc_i = diff/dist, d(dist_ij)/dc_j = -diff/dist
                # d(pair_val)/dc_i = diff/(2*dist), d(pair_val)/dc_j = -diff/(2*dist)
                grad[i] += w * (-diff / dist)
                grad[j] += w * (diff / dist)
    
    return grad


def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Initialize from FCC lattice
    x0 = _fcc_lattice_init(N)
    best_x = x0.copy()
    best_r = _compute_radius(x0)
    
    # Phase 1: Soft-min optimization with annealing beta
    max_restarts = 10
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb best solution
            noise_scale = 0.02 * (1.0 + 0.3 * restart)
            x_start = best_x + np.random.randn(N, 3) * noise_scale
            x_start = np.clip(x_start, -0.48, 0.48)
        
        # Anneal beta: start smooth, get sharper
        beta = max(3.0, 15.0 - 1.5 * restart)
        
        try:
            def objective(c, b=beta):
                return -_soft_min_radius(c, b)
            
            def gradient(c, b=beta):
                return -_soft_min_radius_grad(c, b)
            
            result = minimize(
                objective,
                x_start,
                jac=gradient,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'maxls': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            x_opt = result.x
            r_opt = _compute_radius(x_opt)
            
            if r_opt > best_r + 1e-12:
                best_r = r_opt
                best_x = x_opt.copy()
                
        except Exception:
            pass
        
        if time.time() > deadline - 10:
            break
    
    # Phase 2: Polish with exact objective
    if time.time() < deadline - 5:
        try:
            def neg_exact(c):
                return -_compute_radius(c)
            
            result = minimize(
                neg_exact,
                best_x,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 150, 'maxls': 40, 'ftol': 1e-12}
            )
            
            x_polished = result.x
            r_polished = _compute_radius(x_polished)
            
            if r_polished > best_r + 1e-12:
                best_r = r_polished
                best_x = x_polished.copy()
        except Exception:
            pass
    
    # Phase 3: Local refinement with small perturbations
    if time.time() < deadline - 3:
        for attempt in range(5):
            if time.time() > deadline - 3:
                break
            
            x_try = best_x + np.random.randn(N, 3) * 0.005
            x_try = np.clip(x_try, -0.49, 0.49)
            
            try:
                def neg_exact2(c):
                    return -_compute_radius(c)
                
                result = minimize(
                    neg_exact2,
                    x_try,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={'maxiter': 50, 'maxls': 20, 'ftol': 1e-12}
                )
                
                x_opt = result.x
                r_opt = _compute_radius(x_opt)
                
                if r_opt > best_r + 1e-12:
                    best_r = r_opt
                    best_x = x_opt.copy()
            except Exception:
                pass
    
    # Final validation
    best_x = np.clip(best_x, -0.5, 0.5)
    
    return best_x

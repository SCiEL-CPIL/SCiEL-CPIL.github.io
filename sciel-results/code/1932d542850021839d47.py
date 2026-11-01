"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _score(centers):
    """Calculate the score (radius) for a given set of centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Min pairwise distance
    min_pair_dist = np.min(dists[np.triu_indices(N, k=1)])
    
    # Min distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min(values, beta):
    """
    Smooth approximation of min(values) using log-sum-exp.
    As beta -> inf, soft_min -> min.
    """
    m = np.min(values)
    # Shift for numerical stability
    shifted = values - m
    # Use exp(-beta * shifted)
    exp_vals = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_vals))
    return m - log_sum / beta

def _compute_all_distances(centers):
    """Compute pairwise distances and wall distances."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    # Only upper triangle for pairs
    iu = np.triu_indices(N, k=1)
    pair_dists_sq = dists_sq[iu[0], iu[1]]
    pair_dists = np.sqrt(pair_dists_sq)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dists = np.min(wall_dists, axis=1)
    
    return pair_dists, min_wall_dists

def _smooth_objective(centers_flat, beta):
    """
    Smooth objective function to minimize.
    We want to maximize the minimum radius.
    Radius r = min( min_pair_dist/2, min_wall_dist )
    So we minimize -r, but with a smooth approximation.
    
    Actually, it's better to minimize a penalty for small distances.
    Let's use the soft-min of the "constraints" directly.
    We want to maximize min( pair_dists/2, wall_dists ).
    This is equivalent to minimizing -soft_min( pair_dists/2, wall_dists ).
    
    Let V = soft_min( [pair_dists/2, wall_dists], beta ).
    We minimize -V.
    """
    centers = centers_flat.reshape(-1, 3)
    pair_dists, min_wall_dists = _compute_all_distances(centers)
    
    # Combine all distances
    # We want the minimum of (pair_dists/2) and (wall_dists)
    # Let's create a vector of all relevant distances
    # To keep size manageable, we can just use the soft min of the two groups
    # Or concatenate. N=50, pairs=1225. Total ~1275 values.
    
    vals = np.concatenate([pair_dists / 2.0, min_wall_dists])
    
    r_smooth = _soft_min(vals, beta)
    
    return -r_smooth

def _smooth_gradient(centers_flat, beta):
    """
    Gradient of the smooth objective.
    Objective = -soft_min(vals, beta)
    soft_min(vals, beta) = min(vals) - (1/beta) * log(sum(exp(-beta*(vals - min))))
    
    Let M = min(vals).
    S = sum(exp(-beta*(vals - M)))
    soft_min = M - (1/beta) * log(S)
    
    d(soft_min)/d(val_k) = ?
    Let u_k = -beta * (val_k - M).
    S = sum(exp(u_j)).
    d(log S)/du_k = exp(u_k) / S.
    du_k/d(val_k) = -beta (if k != argmin) or -beta + beta * sum(dM/dval_j) ... 
    Actually, M depends on val_k if val_k is the minimum.
    This is tricky.
    
    Alternative:
    soft_min(vals, beta) = -1/beta * log( sum(exp(-beta * vals)) )
    This is the standard log-sum-exp form.
    Let's verify:
    If one value is much smaller, say val_1 << val_2...
    exp(-beta val_1) >> exp(-beta val_2).
    log(sum) ≈ log(exp(-beta val_1)) = -beta val_1.
    -1/beta * (-beta val_1) = val_1.
    Yes, this works and is differentiable everywhere.
    
    So, soft_min = -1/beta * log( sum_j exp(-beta * val_j) ).
    
    d(soft_min)/d(val_k) = -1/beta * ( 1 / sum(exp) ) * ( -beta * exp(-beta val_k) )
                         = exp(-beta val_k) / sum(exp(-beta val_j))
    
    Let w_k = exp(-beta val_k) / sum(exp(-beta val_j)).
    Then d(soft_min)/d(val_k) = w_k.
    Note that sum(w_k) = 1.
    
    Objective F = -soft_min.
    dF/d(val_k) = -w_k.
    
    Now, val_k depends on centers.
    For pair distances: val_k = |c_i - c_j| / 2.
    d(val_k)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
    d(val_k)/d(c_j) = (c_j - c_i) / (2 * |c_i - c_j|)
    
    For wall distances: val_k = min_m (0.5 - |c_im|).
    This is non-smooth. We approximate by taking the gradient of the active constraint.
    Or, since we are using a smooth objective for the *pair* part mostly,
    and the wall part is a min over 6 faces.
    Actually, the soft-min is over the *values* of the distances.
    The values themselves are:
    v_1..v_P: pair distances / 2
    v_{P+1}..v_{P+N}: wall distances (min over faces for each sphere)
    
    The wall distance v_{P+i} = min(0.5-c_ix, 0.5+c_ix, 0.5-c_iy, 0.5+c_iy, 0.5-c_iz, 0.5+c_iz).
    The gradient of v_{P+i} with respect to c_i is the gradient of the active face.
    
    So, total gradient:
    grad = sum_k w_k * (dF/dv_k * dv_k/dc)
    dF/dv_k = -w_k.
    So grad = - sum_k w_k * (dv_k/dc).
    
    Let's compute this.
    """
    centers = centers_flat.reshape(-1, 3)
    grad = np.zeros_like(centers)
    
    # 1. Compute weights w_k
    pair_dists, min_wall_dists = _compute_all_distances(centers)
    vals = np.concatenate([pair_dists / 2.0, min_wall_dists])
    
    # For numerical stability in exp
    m = np.min(vals)
    exp_vals = np.exp(-beta * (vals - m))
    sum_exp = np.sum(exp_vals)
    w = exp_vals / sum_exp
    
    w_pairs = w[:N*(N-1)//2]
    w_walls = w[N*(N-1)//2:]
    
    # 2. Gradient from pair distances
    # v_k = |c_i - c_j| / 2
    # dv_k/dc_i = (c_i - c_j) / (2 * |c_i - c_j|)
    # dv_k/dc_j = (c_j - c_i) / (2 * |c_i - c_j|)
    
    # We need to map w_pairs to indices (i, j)
    # iu = np.triu_indices(N, k=1)
    iu = np.triu_indices(N, k=1)
    i_idx = iu[0]
    j_idx = iu[1]
    
    # diff = c_i - c_j
    # dist = |c_i - c_j|
    # coeff = w_k / (2 * dist)
    # grad[c_i] += coeff * (c_i - c_j)
    # grad[c_j] += coeff * (c_j - c_i)
    # But remember the overall sign is negative.
    # grad = - sum w_k * dv_k/dc
    # So grad[c_i] -= w_k * (c_i - c_j) / (2 * dist)
    #    grad[c_j] -= w_k * (c_j - c_i) / (2 * dist)
    
    # Vectorized computation
    # diff[i, j] = c_i - c_j
    diff = centers[i_idx] - centers[j_idx]
    dists = pair_dists
    # Avoid division by zero
    dists[dists < 1e-12] = 1e-12
    
    coeff = w_pairs / (2.0 * dists)
    
    # grad_i += -coeff * diff
    # grad_j += +coeff * diff
    
    # Use np.add.at for accumulation
    np.add.at(grad, i_idx, -coeff[:, np.newaxis] * diff)
    np.add.at(grad, j_idx, coeff[:, np.newaxis] * diff)
    
    # 3. Gradient from wall distances
    # v_{P+i} = min_m (0.5 - |c_im|)
    # The gradient is -w_walls[i] * grad(v_{P+i})
    # grad(v_{P+i}) is the unit vector pointing towards the closest face (inwards).
    # If closest face is x=0.5 (c_ix > 0), grad is (-1, 0, 0).
    # If closest face is x=-0.5 (c_ix < 0), grad is (1, 0, 0).
    # Generally, grad(v) = -sign(c) for the active coordinate.
    
    # Find active face for each sphere
    # dists_to_faces = 0.5 - |c|
    # active_idx = argmin(dists_to_faces, axis=1)
    # We need to know which coordinate and which side.
    
    # Let's compute the signed distances to faces
    # x: 0.5 - c_x, 0.5 + c_x
    # y: 0.5 - c_y, 0.5 + c_y
    # z: 0.5 - c_z, 0.5 + c_z
    
    # We can just check the sign of c and the magnitude.
    # If |c_x| is the max of |c_x|, |c_y|, |c_z|, then x is the active axis.
    # And the side is determined by sign(c_x).
    
    abs_c = np.abs(centers)
    active_axis = np.argmax(abs_c, axis=1)
    active_sign = np.sign(centers)
    
    # grad_wall = -w_walls[:, np.newaxis] * active_sign
    # But only for the active axis component.
    
    grad_wall = np.zeros_like(centers)
    # For each sphere i:
    # grad_wall[i, active_axis[i]] = -w_walls[i] * active_sign[i, active_axis[i]]
    
    # Vectorized:
    # Create a mask or use advanced indexing
    # grad_wall[i, active_axis[i]] = ...
    
    # np.put_along_axis or simply:
    # Since N is small, a loop is fine, or we can use fancy indexing.
    # Let's use fancy indexing.
    # indices = (np.arange(N), active_axis)
    # grad_wall[indices] = -w_walls[:, np.newaxis] * active_sign[indices]
    # Wait, active_sign is (N,3). We need the component at active_axis.
    # active_sign_vals = active_sign[np.arange(N), active_axis]
    
    active_sign_vals = active_sign[np.arange(N), active_axis]
    grad_wall[np.arange(N), active_axis] = -w_walls * active_sign_vals
    
    grad += grad_wall
    
    return grad.flatten()

def _initial_fcc_like():
    """Generate an FCC-like initial configuration."""
    # FCC basis in unit cell [0,1]^3:
    # (0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
    # Shift to [-0.5, 0.5]^3
    # We want 50 points.
    # A 3x3x3 grid of FCC cells would be 27 * 4 = 108 points. Too many.
    # Let's try a 2x2x2 grid of FCC cells? 8 * 4 = 32 points. Not enough.
    # Let's try a 3x3x3 grid of simple cubic points and perturb them towards FCC?
    # Or just take the first 50 points from a dense FCC lattice.
    
    # Generate FCC lattice points
    # Basis vectors:
    # a1 = (0.5, 0.5, 0)
    # a2 = (0.5, 0, 0.5)
    # a3 = (0, 0.5, 0.5)
    # Actually, standard FCC primitive vectors for cubic lattice constant a=1:
    # v1 = (0, 0.5, 0.5)
    # v2 = (0.5, 0, 0.5)
    # v3 = (0.5, 0.5, 0)
    # Points: n1*v1 + n2*v2 + n3*v3
    
    points = []
    for n1 in range(-2, 3):
        for n2 in range(-2, 3):
            for n3 in range(-2, 3):
                p = n1 * np.array([0, 0.5, 0.5]) + \
                    n2 * np.array([0.5, 0, 0.5]) + \
                    n3 * np.array([0.5, 0.5, 0])
                # Check if inside [-0.5, 0.5]^3
                if np.all(p >= -0.5) and np.all(p <= 0.5):
                    points.append(p)
    
    points = np.array(points)
    
    # We might have more or fewer than 50.
    # If more, pick 50 closest to center? Or random 50?
    # If fewer, add random points?
    
    if len(points) >= N:
        # Pick 50 points. To break symmetry, maybe pick a random subset?
        # Or the ones closest to the center of the cube?
        # The center is (0,0,0).
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        return points[idx]
    else:
        # Add random points to fill up to N
        rng = np.random.default_rng(42)
        while len(points) < N:
            p = rng.uniform(-0.45, 0.45, 3)
            points = np.vstack([points, p])
        return points[:N]

def _initial_grid():
    """Generate a grid initial configuration."""
    # 4x4x4 grid = 64 points.
    ticks = np.array([-0.375, -0.125, 0.125, 0.375])
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Remove 14 points randomly?
    # Or just take the first 50?
    # Taking the first 50 from the list (which is ordered by z, y, x)
    # might be biased.
    # Let's shuffle and take 50.
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(grid))[:N]
    return grid[idx]

def _refine_lloyd(centers, num_iters=50, beta=50.0):
    """
    Lloyd-like iteration:
    1. Compute Voronoi cells (or approximate centers).
    2. Move centers to the centroid of their cell, clipped to the valid region.
    This tends to equalize distances.
    
    For spheres in a box, a simpler heuristic is:
    Move each sphere away from its nearest neighbor and towards the center if near a wall.
    
    Let's try a simple force-based relaxation instead, as Voronoi in 3D with boundaries is complex.
    """
    # We will use the smooth gradient to do a few steps of gradient descent.
    # This is effectively what the optimizer does, but we can do it manually for control.
    
    centers = centers.copy()
    for i in range(num_iters):
        grad = _smooth_gradient(centers.flatten(), beta)
        grad = grad.reshape(-1, 3)
        
        # Adaptive step size
        # Estimate Lipschitz constant? Or just use a small step.
        # The gradient magnitude can vary.
        step_size = 0.01 / (1 + np.linalg.norm(grad, axis=1).max())
        
        centers -= step_size * grad
        centers = np.clip(centers, -0.49, 0.49)
        
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_score = 0.0
    best_centers = None
    
    rng = np.random.default_rng(42)
    
    # Pre-compute indices for speed
    iu = np.triu_indices(N, k=1)
    
    # Strategy 1: Grid Pruning
    ticks = np.array([-0.375, -0.125, 0.125, 0.375])
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Strategy 2: FCC-like
    fcc_centers = _initial_fcc_like()
    
    # List of initial configurations
    inits = []
    
    # 1. Grid Pruning (multiple random removals)
    for r in range(5):
        idx = rng.permutation(len(grid))[:N]
        centers = grid[idx].copy()
        centers += rng.normal(0, 0.005, centers.shape)
        inits.append(centers)
        
    # 2. FCC-like
    inits.append(fcc_centers + rng.normal(0, 0.01, fcc_centers.shape))
    
    # 3. Random
    inits.append(rng.uniform(-0.45, 0.45, (N, 3)))
    
    # 4. Dense Grid (5x5x5)
    ticks_fine = np.linspace(-0.45, 0.45, 5)
    grid_fine = np.array([(x, y, z) for z in ticks_fine for y in ticks_fine for x in ticks_fine])
    d_center = np.linalg.norm(grid_fine, axis=1)
    idx_sorted = np.argsort(d_center)
    idx = idx_sorted[:N]
    centers = grid_fine[idx].copy()
    centers += rng.normal(0, 0.01, centers.shape)
    inits.append(centers)
    
    # Optimize each initial configuration
    for centers in inits:
        if time.time() > deadline:
            break
            
        centers = np.clip(centers, -0.49, 0.49)
        x0 = centers.flatten()
        
        # Multi-scale beta annealing
        betas = [10.0, 50.0, 100.0, 200.0]
        current_centers = centers.copy()
        
        for beta in betas:
            if time.time() > deadline:
                break
                
            try:
                res = minimize(_smooth_objective, current_centers.flatten(), 
                               method='L-BFGS-B', 
                               jac=lambda x, b=beta: _smooth_gradient(x, b),
                               bounds=[(-0.5, 0.5)] * (N * 3), 
                               options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                
                opt_centers = res.x.reshape(-1, 3)
                opt_centers = np.clip(opt_centers, -0.5, 0.5)
                current_centers = opt_centers
                
            except Exception as e:
                pass
        
        # Final polish with high beta
        if time.time() < deadline - 10:
            try:
                res = minimize(_smooth_objective, current_centers.flatten(), 
                               method='L-BFGS-B', 
                               jac=lambda x: _smooth_gradient(x, 500.0),
                               bounds=[(-0.5, 0.5)] * (N * 3), 
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
                
                opt_centers = res.x.reshape(-1, 3)
                opt_centers = np.clip(opt_centers, -0.5, 0.5)
                current_centers = opt_centers
            except Exception:
                pass
        
        # Local search / Nelder-Mead on the raw score
        if time.time() < deadline - 5:
            def neg_radius(c_flat):
                c = c_flat.reshape(-1, 3)
                if np.any(c < -0.5) or np.any(c > 0.5):
                    return 1000.0
                return -_score(c)
            
            try:
                res = minimize(neg_radius, current_centers.flatten(), method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                opt_centers = res.x.reshape(-1, 3)
                opt_centers = np.clip(opt_centers, -0.5, 0.5)
                current_centers = opt_centers
            except Exception:
                pass
        
        score = _score(current_centers)
        if score > best_score:
            best_score = score
            best_centers = current_centers.copy()
            
    # If time remains, do a final polish on the best candidate
    if best_centers is not None and (deadline - time.time()) > 10:
        # Run L-BFGS again on the best candidate with very tight tolerance
        x0 = best_centers.flatten()
        try:
            res = minimize(_smooth_objective, x0, method='L-BFGS-B', 
                           jac=lambda x: _smooth_gradient(x, 1000.0), 
                           bounds=[(-0.5, 0.5)] * (N * 3), 
                           options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            opt_centers = res.x.reshape(-1, 3)
            opt_centers = np.clip(opt_centers, -0.5, 0.5)
            score = _score(opt_centers)
            if score > best_score:
                best_score = score
                best_centers = opt_centers.copy()
        except Exception:
            pass
        
        # Final Nelder-Mead polish
        if (deadline - time.time()) > 5:
            def neg_radius(c_flat):
                c = c_flat.reshape(-1, 3)
                if np.any(c < -0.5) or np.any(c > 0.5):
                    return 1000.0
                return -_score(c)
            
            try:
                res = minimize(neg_radius, best_centers.flatten(), method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
                opt_centers = res.x.reshape(-1, 3)
                opt_centers = np.clip(opt_centers, -0.5, 0.5)
                score = _score(opt_centers)
                if score > best_score:
                    best_score = score
                    best_centers = opt_centers.copy()
            except Exception:
                pass

    if best_centers is None:
        # Fallback
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END

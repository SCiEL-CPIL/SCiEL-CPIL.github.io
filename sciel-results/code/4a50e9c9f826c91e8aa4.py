"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

HALF = 0.5
DEADLINE_MARGIN = 5.0  # seconds to leave before hard limit

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall constraints: distance from center to nearest wall
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to exclude self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # The radius is half the minimum pairwise distance, limited by wall distance
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _fcc_init():
    """
    Generate initial positions using an FCC lattice structure.
    FCC (Face-Centered Cubic) has a packing density of pi/sqrt(6) ~ 0.74,
    which is optimal for infinite space. We use this as a high-quality starting point.
    """
    # We need 100 points.
    # Let's create a grid of FCC unit cells.
    # Each FCC unit cell contains 4 atoms.
    # We can arrange them in a box.
    # Let's try a 5x5x2 grid of unit cells? 5*5*2 = 50 cells * 4 atoms/cell = 200 atoms. Too many.
    # 4x4x2 = 32 cells * 4 = 128 atoms. Close.
    # 4x3x3 = 36 cells * 4 = 144 atoms.
    # 3x3x3 = 27 cells * 4 = 108 atoms. Very close to 100.
    
    # Let's use a 3x3x3 grid of FCC unit cells, then prune to 100 best?
    # Or just generate 100 points from a larger FCC grid and pick the ones inside a central cube?
    
    # Alternative: Just generate points on an FCC lattice that fits in [-0.5, 0.5]^3.
    # FCC lattice vectors:
    # a1 = (a/2, a/2, 0)
    # a2 = (a/2, 0, a/2)
    # a3 = (0, a/2, a/2)
    # Points: n1*a1 + n2*a2 + n3*a3 + (0,0,0) and face centers etc.
    # Simpler: The FCC lattice points are all integer combinations of (1,1,0), (1,0,1), (0,1,1) with even sum of coordinates? No.
    # FCC points are (x,y,z) where x,y,z are integers and x+y+z is even, scaled by a/2?
    # Let's use the standard definition:
    # Lattice points at (i, j, k) * a/2 where i,j,k are integers and i+j+k is even.
    
    # We want to fit these in [-0.5, 0.5]^3.
    # Let's pick a lattice constant 'a' such that the points fit.
    # If we use a grid of indices i, j, k in range [-K, K], the extent is roughly K*a/2.
    # We want K*a/2 <= 0.5 => a <= 1/K.
    
    # Let's try K=3. Indices -3 to 3.
    # Total points in cube of side 7 (from -3 to 3) with parity constraint.
    # Number of points approx (2K+1)^3 / 2.
    # For K=3: 7^3 / 2 = 343/2 = 171.5. We have plenty.
    
    # Let's generate all FCC points in a large enough box, filter to [-0.45, 0.45]^3 (margin for optimization), and pick 100.
    # Actually, we want them to be spread out.
    # Let's just generate a 4x4x4 grid of points, but shifted to FCC?
    
    # Let's try a simpler approach:
    # Generate a simple cubic grid of 5x5x5 = 125 points.
    # Then perturb them towards FCC positions?
    
    # Let's stick to the FCC lattice generation properly.
    # We will generate points on the FCC lattice with a specific spacing.
    # Let's choose the spacing such that we get roughly 100 points in the cube.
    # Volume of cube = 1. Volume of one sphere ~ (4/3)pi r^3.
    # Max density 0.74.
    # 100 * (4/3)pi r^3 <= 0.74 * 1.
    # r <= (0.74 * 3 / (400 * pi))^(1/3) ~ (0.00176)^(1/3) ~ 0.12.
    # Diameter ~ 0.24.
    # So lattice spacing should be around 0.2-0.25.
    
    # Let's use a lattice constant a = 0.25.
    # FCC points: (i*a/2, j*a/2, k*a/2) where i+j+k is even.
    # Range of i,j,k: -4 to 4 (since 4*0.25/2 = 0.5).
    
    points = []
    a = 0.25
    half_a = a / 2.0
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                if (i + j + k) % 2 == 0:
                    x = i * half_a
                    y = j * half_a
                    z = k * half_a
                    # Check if inside cube with a small margin
                    if abs(x) < 0.48 and abs(y) < 0.48 and abs(z) < 0.48:
                        points.append([x, y, z])
    
    points = np.array(points)
    
    # We might have more than 100 points.
    if len(points) > N:
        # We want to select 100 points that are well-distributed.
        # A simple heuristic: pick points that are farthest from each other?
        # Or just pick the first 100? They are ordered.
        # Let's try to pick a subset that maximizes the minimum distance? That's hard.
        # Let's just shuffle and take 100? No, that breaks symmetry.
        
        # Better: Use a greedy farthest-point sampling from the set of FCC points.
        # Start with the center point (0,0,0) if it exists, or closest to center.
        # Then iteratively pick the point that is farthest from the current set.
        
        # Find point closest to origin
        dists_origin = np.sum(points**2, axis=1)
        start_idx = np.argmin(dists_origin)
        selected = [start_idx]
        points_set = points[start_idx]
        
        for _ in range(N - 1):
            # Compute distance from each unselected point to the nearest selected point
            # This is O(N_selected * N_total). N_total ~ 100-200. N_selected up to 99.
            # 200 * 100 = 20000 ops. Fast.
            
            # Vectorized:
            # dists[i, j] = |points[i] - points_set[j]|
            # We want min_j dists[i, j]
            
            # To avoid memory blowup if points is large, we can do it in chunks or just assume it's small.
            # Let's just do it.
            diff = points[:, np.newaxis, :] - points_set[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            min_dists = np.min(dists, axis=1)
            
            # Mask out already selected
            for s_idx in selected:
                min_dists[s_idx] = -1
                
            next_idx = np.argmax(min_dists)
            selected.append(next_idx)
            points_set = np.vstack([points_set, points[next_idx]])
            
    elif len(points) < N:
        # If we have too few, add random points?
        # This shouldn't happen with a=0.25 and range -4..4.
        # Let's just pad with random points if necessary.
        while len(points) < N:
            p = np.random.uniform(-0.4, 0.4, 3)
            points = np.vstack([points, p])
        
    return points[:N]

def _riesz_energy_and_grad(x, s=4.0, alpha=1.0):
    """
    Calculate the Riesz s-energy and its gradient.
    Energy = sum_{i<j} |x_i - x_j|^{-s} + sum_{i,k} wall_penalty(x_ik)
    
    We use a soft wall penalty to keep points inside.
    Wall penalty: sum (0.5 - |x_ik|)^{-p} ? No, that blows up.
    Let's use a quadratic penalty for being outside, or a repulsive force from walls.
    
    Actually, the prompt suggests: "repulsive Riesz s-energy with s > 3".
    This handles the pairwise spacing.
    For walls, we can add a term that penalizes proximity to walls.
    Let's use: E_wall = sum_{i,k} (1 / (0.5 - |x_ik| + eps)^2) ?
    Or simpler: just use L-BFGS-B bounds to enforce the cube, and let the Riesz energy handle the spacing.
    L-BFGS-B handles bounds well.
    
    So, Objective:
    E = sum_{i<j} |x_i - x_j|^{-s}
    
    Gradient:
    dE/dx_i = sum_{j != i} s * |x_i - x_j|^{-s-2} * (x_i - x_j)
    
    Note: This energy goes to infinity as points get close. Minimizing it pushes points apart.
    However, it doesn't explicitly maximize the *minimum* distance. It maximizes the *average* inverse distance (roughly).
    For uniform distributions, this often leads to good packings.
    
    To strictly enforce the minimum distance, we might need a different potential, like a "hard core" or a very high power s.
    With s=4 or higher, the energy is dominated by the closest pairs.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Avoid division by zero
    dists_sq = np.maximum(dists_sq, 1e-12)
    dists = np.sqrt(dists_sq)
    
    # Energy: sum_{i<j} d_ij^{-s}
    # We compute sum over all i != j, then divide by 2.
    # E = 0.5 * sum_{i != j} d_ij^{-s}
    
    # We need to be careful with the gradient.
    # dE/dx_i = 0.5 * sum_{j != i} d(d_ij^{-s})/dx_i
    # d(d_ij^{-s})/dx_i = -s * d_ij^{-s-1} * (d_ij / dx_i)
    # d(d_ij)/dx_i = (x_i - x_j) / d_ij
    # So dE/dx_i = 0.5 * sum_{j != i} -s * d_ij^{-s-1} * (x_i - x_j) / d_ij
    #             = -0.5 * s * sum_{j != i} d_ij^{-s-2} * (x_i - x_j)
    
    # Vectorized calculation:
    # We want to sum over j for each i.
    # coeff_ij = s * d_ij^{-s-2}
    # grad_i = -0.5 * sum_j coeff_ij * (x_i - x_j)
    
    # Let's compute the matrix of coefficients C where C[i,j] = s * d_ij^{-s-2}
    # C is symmetric.
    
    # d_ij^{-s-2} = (dists_sq)^{-(s+2)/2}
    # Note: d_ij^{-s-2} = d_ij^{-s} * d_ij^{-2} = d_ij^{-s} / dists_sq
    
    # Let's compute inv_dists_sq = 1 / dists_sq
    # inv_dists_pow = inv_dists_sq ** ((s+2)/2.0)
    
    # This might be numerically unstable for very small distances.
    # But we clamped dists_sq.
    
    inv_dists_sq = 1.0 / dists_sq
    # Power: (s+2)/2
    exp_val = (s + 2.0) / 2.0
    coeff_matrix = s * (inv_dists_sq ** exp_val)
    
    # Diagonal is 0 (since dists_sq was inf, inv is 0)
    np.fill_diagonal(coeff_matrix, 0.0)
    
    # grad_i = -0.5 * sum_j C[i,j] * (x_i - x_j)
    # x_i is broadcasted.
    # term = C[i,j] * (x_i - x_j)
    # Sum over j.
    
    # Vectorized:
    # diff is (N, N, 3).
    # coeff_matrix is (N, N).
    # We want to multiply coeff_matrix[i,j] with diff[i,j].
    # coeff_expanded = coeff_matrix[:, :, np.newaxis]
    # weighted_diff = coeff_expanded * diff
    # grad = -0.5 * np.sum(weighted_diff, axis=1)
    
    coeff_expanded = coeff_matrix[:, :, np.newaxis]
    weighted_diff = coeff_expanded * diff
    grad = -0.5 * np.sum(weighted_diff, axis=1)
    
    # Energy
    # E = 0.5 * sum_{i != j} d_ij^{-s}
    # d_ij^{-s} = (dists_sq)^{-s/2}
    inv_dists_pow_s = (inv_dists_sq ** (s / 2.0))
    np.fill_diagonal(inv_dists_pow_s, 0.0)
    energy = 0.5 * np.sum(inv_dists_pow_s)
    
    return energy, grad.flatten()

def _optimize_riesz(init_centers, s=4.0, max_iter=1000):
    """
    Minimize the Riesz s-energy using L-BFGS-B.
    """
    def obj_and_grad(x):
        val, g = _riesz_energy_and_grad(x, s=s)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            obj_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        return new_centers, res.success
    except Exception:
        return init_centers, False

def _local_refine_smooth(centers, s=4.0, max_iter=200):
    """
    Refine centers by minimizing a smooth energy that approximates the min-distance objective.
    We use a "soft min" of the distances.
    Objective: minimize -soft_min(distances)
    soft_min(dists) = -logsumexp(-beta * dists) / beta
    """
    beta = 50.0 # Sharpness of the soft min
    
    def obj_and_grad(x):
        pos = x.reshape(N, 3)
        
        # Pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        
        # Wall distances
        # d_wall_ik = 0.5 - |pos_ik|
        d_walls = 0.5 - np.abs(pos)
        wall_dists = d_walls.flatten()
        
        # Effective distances for radius
        # Pair dist d -> radius d/2
        # Wall dist d -> radius d
        # We want to maximize the min of these.
        # Let's form a vector of all effective radii.
        
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        eff_pair_rads = pair_dists / 2.0
        eff_wall_rads = wall_dists
        
        all_rads = np.concatenate([eff_pair_rads, eff_wall_rads])
        
        # Soft min
        scaled = -beta * all_rads
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min_val = -log_sum_exp / beta
        
        # We want to MAXIMIZE soft_min_val.
        # So we minimize -soft_min_val.
        
        # Gradient
        weights = np.exp(scaled - max_scaled)
        weights /= np.sum(weights)
        
        grad_pos = np.zeros_like(pos)
        
        # Pair gradient
        n_pairs = len(iu[0])
        pair_weights = weights[:n_pairs]
        
        i_idx = iu[0]
        j_idx = iu[1]
        d_vals = pair_dists
        
        valid = d_vals > 1e-14
        if np.any(valid):
            i_v = i_idx[valid]
            j_v = j_idx[valid]
            d_v = d_vals[valid]
            w_v = pair_weights[valid]
            
            vecs = pos[i_v] - pos[j_v]
            # d(soft_min)/d(pos_i) = sum_j w_j * d(d_ij/2)/d(pos_i) = sum_j w_j * (pos_i - pos_j) / (2 d_ij)
            coeffs = w_v / (2.0 * d_v)
            
            np.add.at(grad_pos, i_v, coeffs[:, np.newaxis] * vecs)
            np.add.at(grad_pos, j_v, -coeffs[:, np.newaxis] * vecs)
        
        # Wall gradient
        # d(soft_min)/d(pos_ik) = w_k * d(d_wall_ik)/d(pos_ik) = w_k * (-sign(pos_ik))
        wall_weights = weights[n_pairs:]
        signs = np.sign(pos)
        signs[signs == 0] = 1.0
        
        grad_wall = -wall_weights.reshape(N, 3) * signs
        grad_pos += grad_wall
        
        # Minimize -soft_min
        return -soft_min_val, -grad_pos.flatten()

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            obj_and_grad,
            centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        return res.x.reshape(N, 3)
    except Exception:
        return centers

def construct():
    start_time = time.time()
    deadline = start_time + 120.0 - DEADLINE_MARGIN
    
    # 1. Initialization: FCC
    centers = _fcc_init()
    best_centers = centers.copy()
    best_r = _compute_radius(best_centers)
    
    # 2. Optimization Strategy:
    # We will alternate between:
    # A) Minimizing Riesz energy (smooth, global-ish, good for spacing)
    # B) Minimizing Soft-Min objective (sharp, local, good for maximizing min dist)
    
    # Let's do a few rounds of Riesz to get a good spread, then Soft-Min to polish.
    
    # Round 1: Riesz Energy (s=4)
    if time.time() < deadline:
        centers, success = _optimize_riesz(centers, s=4.0, max_iter=500)
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # Round 2: Soft-Min Polish
    if time.time() < deadline:
        centers = _local_refine_smooth(centers, max_iter=200)
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # Round 3: Restart with Riesz s=6 (steeper potential, more rigid)
    if time.time() < deadline - 10:
        # Use best_centers as start? Or new FCC?
        # Using best_centers is better for continuity.
        centers, success = _optimize_riesz(best_centers, s=6.0, max_iter=300)
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
            
        # Polish again
        if time.time() < deadline:
            centers = _local_refine_smooth(centers, max_iter=100)
            r = _compute_radius(centers)
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
                
    # Round 4: If time permits, try a different initialization (e.g. random) and run the same pipeline
    if time.time() < deadline - 20:
        # Random init
        centers = np.random.uniform(-0.45, 0.45, (N, 3))
        
        # Riesz
        centers, success = _optimize_riesz(centers, s=4.0, max_iter=300)
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
            
        # Polish
        if time.time() < deadline:
            centers = _local_refine_smooth(centers, max_iter=100)
            r = _compute_radius(centers)
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
    
    return best_centers

# EVOLVE-BLOCK-END

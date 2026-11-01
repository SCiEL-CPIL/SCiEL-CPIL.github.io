"""Packing 50 equal circles in a unit circle using soft-min surrogate + gradient optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # Safety margin for 150s limit

# EVOLVE-BLOCK-START

def _compute_score(centers):
    """
    Compute the maximum radius r for which circles centered at 'centers' 
    are disjoint and inside the unit circle.
    r = min( min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|) )
    """
    if centers.size == 0:
        return 0.0
    n = centers.shape[0]
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    min_bound_dist = np.min(1.0 - norms)
    
    return min(min_pair_dist / 2.0, min_bound_dist)

def _soft_min_obj_and_grad(params, n, beta):
    """
    Objective: -soft_min(constraints)
    Constraints: d_ij / 2 for pairs, 1 - |c_i| for boundary.
    We want to maximize the minimum of these, so we minimize the negative soft-min.
    
    soft_min(vals) = min(vals) - (1/beta) * log(sum(exp(-beta*(vals - min(vals)))))
    d(soft_min)/d(vals_k) = -exp(-beta*(vals_k - min))/sum(exp(-beta*(vals - min)))
    d(-soft_min)/d(vals_k) = exp(-beta*(vals_k - min))/sum(exp(-beta*(vals - min)))
    """
    centers = params.reshape(n, 2)
    
    # 1. Compute pair distances (vectorized)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # 2. Compute boundary distances
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    bound_dists = 1.0 - norms
    
    # 3. Combine all constraints into a single vector
    n_pairs = n * (n - 1) // 2
    vals = np.concatenate([pair_dists / 2.0, bound_dists])
    
    # 4. Compute Soft Min with numerical stability
    m = np.min(vals)
    shifted = vals - m
    # Clip to avoid overflow in exp
    shifted_clipped = np.maximum(shifted, -50.0 / beta)
    exp_terms = np.exp(-beta * shifted_clipped)
    S = np.sum(exp_terms)
    
    soft_min_val = m - (1.0 / beta) * np.log(S)
    obj_val = -soft_min_val
    
    # 5. Compute Gradient (vectorized)
    # d(-soft_min)/d(vals_k) = w_k where w_k = exp(-beta*(vals_k - m)) / S
    weights = exp_terms / S
    
    grad_centers = np.zeros_like(centers)
    
    # Pair contributions: d(-soft_min)/d(c_i) = sum_k w_k * d(vals_k)/d(c_i)
    # For pair (i,j): vals_k = d_ij / 2
    # d(d_ij/2)/d(c_i) = 0.5 * (c_i - c_j) / d_ij
    # d(d_ij/2)/d(c_j) = 0.5 * (c_j - c_i) / d_ij = -0.5 * (c_i - c_j) / d_ij
    pair_weights = weights[:n_pairs]
    # Only include pairs with significant weight
    mask = pair_weights > 1e-14
    if np.any(mask):
        i_m = i_idx[mask]
        j_m = j_idx[mask]
        w_m = pair_weights[mask]
        d_m = dists[i_m, j_m]
        d_safe = np.where(d_m > 1e-10, d_m, 1e-10)
        vecs = (centers[i_m] - centers[j_m]) / d_safe[:, np.newaxis]
        # d(-soft_min)/d(c_i) += w_k * 0.5 * (c_i - c_j)/d_ij
        # d(-soft_min)/d(c_j) += w_k * (-0.5) * (c_i - c_j)/d_ij
        contrib_i = w_m[:, np.newaxis] * 0.5 * vecs
        contrib_j = -w_m[:, np.newaxis] * 0.5 * vecs
        # Scatter add for i indices
        np.add.at(grad_centers, i_m, contrib_i)
        # Scatter add for j indices
        np.add.at(grad_centers, j_m, contrib_j)

    # Boundary contributions: d(-soft_min)/d(c_i) = w_k * d(1-|c_i|)/d(c_i)
    # d(1-|c_i|)/d(c_i) = -c_i / |c_i|
    bound_weights = weights[n_pairs:]
    mask_b = bound_weights > 1e-14
    if np.any(mask_b):
        idx_b = np.where(mask_b)[0]
        w_b = bound_weights[idx_b]
        norm_b = norms[idx_b]
        norm_safe = np.where(norm_b > 1e-10, norm_b, 1e-10)
        vecs_b = centers[idx_b] / norm_safe[:, np.newaxis]
        # d(-soft_min)/d(c_i) += w_k * (-c_i / |c_i|)
        contrib_b = -w_b[:, np.newaxis] * vecs_b
        np.add.at(grad_centers, idx_b, contrib_b)
            
    return obj_val, grad_centers.flatten()

def _get_ring_initialization(n, ring_radii=None):
    """
    Generate a ring-based initialization.
    """
    centers = []
    centers.append([0.0, 0.0])
    
    if ring_radii is None:
        rings = [
            (6, 0.22),
            (12, 0.44),
            (12, 0.66),
            (9, 0.82),
            (10, 0.92)
        ]
    else:
        rings = ring_radii
    
    count = 1
    for count_in_ring, radius in rings:
        if count >= n:
            break
        angles = np.linspace(0, 2 * np.pi, count_in_ring, endpoint=False)
        for angle in angles:
            if count >= n:
                break
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
            count += 1
            
    return np.array(centers[:n])

def _get_hex_lattice_initialization(n):
    """
    Generate a hexagonal lattice initialization.
    """
    r_est = 0.12
    s = 2 * r_est
    points = []
    max_range = int(2.0 / s) + 3
    
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
                
    if len(points) < n:
        return _get_ring_initialization(n)
        
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    return points[sorted_idx[:n]]

def _get_random_initialization(n, seed=42):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(10000):
        if len(centers) >= n:
            break
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        if x*x + y*y > 0.95**2:
            continue
        if centers:
            pts = np.array(centers)
            dists = np.hypot(pts[:,0] - x, pts[:,1] - y)
            if np.min(dists) < 0.15:
                continue
        centers.append([x, y])
    
    while len(centers) < n:
        x = rng.uniform(-0.9, 0.9)
        y = rng.uniform(-0.9, 0.9)
        centers.append([x, y])
        
    return np.array(centers[:n])

def _force_directed_relax(centers, time_budget=5.0):
    """
    Pre-processing step: use force-directed relaxation to resolve overlaps
    and push points to a valid configuration before L-BFGS.
    Vectorized for speed.
    """
    centers = centers.copy()
    start_time = time.time()
    step = 0.01
    best_centers = centers.copy()
    best_score = _compute_score(centers)
    n = len(centers)
    
    while time.time() - start_time < time_budget:
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion (vectorized)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, 1e10)
        
        # Force magnitude: 1/(d^2 + eps), direction: diff/d
        force_mag = 1.0 / (dists * dists + 1e-6)
        # Zero out diagonal
        np.fill_diagonal(force_mag, 0.0)
        # Scale diff by force_mag/d to get force vector
        force_vec = diff * force_mag[:, :, np.newaxis] / np.maximum(dists, 1e-10)[:, :, np.newaxis]
        forces = np.sum(force_vec, axis=1)
        
        # Boundary constraint (vectorized)
        radii = np.sqrt(np.sum(centers * centers, axis=1))
        boundary_mask = radii > 0.95
        if np.any(boundary_mask):
            r_safe = np.maximum(radii, 1e-10)
            forces[boundary_mask] -= 0.5 * centers[boundary_mask] / r_safe[boundary_mask][:, np.newaxis]
        
        new_centers = centers + step * forces
        new_radii = np.sqrt(np.sum(new_centers * new_centers, axis=1))
        mask = new_radii > 1.0
        if np.any(mask):
            new_centers[mask] = new_centers[mask] / new_radii[mask][:, None] * 0.999
        
        new_score = _compute_score(new_centers)
        if new_score > best_score:
            best_score = new_score
            best_centers = new_centers.copy()
            centers = new_centers
            step *= 1.05
        else:
            step *= 0.9
            centers = best_centers.copy()
            
        if step < 1e-8:
            break
            
    return best_centers, best_score

def _optimize_from_start(centers, time_budget=20.0):
    """
    Optimize starting from a given configuration using L-BFGS with increasing beta.
    Includes a final high-precision polish pass.
    """
    start_time = time.time()
    current_centers = centers.copy()
    best_centers = centers.copy()
    best_score = _compute_score(centers)
    
    # Force-directed relaxation first
    current_centers, _ = _force_directed_relax(current_centers, time_budget=min(3.0, time_budget * 0.15))
    score = _compute_score(current_centers)
    if score > best_score:
        best_score = score
        best_centers = current_centers.copy()
        
    # L-BFGS optimization with smoother beta annealing
    betas = [20.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 25000.0, 50000.0, 100000.0, 250000.0]
    
    for beta in betas:
        if time.time() - start_time > time_budget - 5.0:
            break
            
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 2.0:
            break
        
        # Scale maxiter based on remaining time
        maxiter = max(500, int(min(5000, remaining * 300)))
        
        def fun_and_grad(x):
            val, grad = _soft_min_obj_and_grad(x, N, beta)
            return val, grad
            
        try:
            result = minimize(
                fun_and_grad,
                current_centers.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': maxiter * 5}
            )
            current_centers = result.x.reshape(N, 2)
        except Exception:
            break
            
        score = _compute_score(current_centers)
        if score > best_score:
            best_score = score
            best_centers = current_centers.copy()
    
    # Final high-precision polish passes with very high beta
    if time.time() - start_time < time_budget - 3.0:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining > 3.0:
            # Multiple polish passes with different high betas
            polish_betas = [500000.0, 1000000.0, 2000000.0]
            for polish_beta in polish_betas:
                if time.time() - start_time > time_budget - 2.0:
                    break
                try:
                    def fun_and_grad_polish(x):
                        val, grad = _soft_min_obj_and_grad(x, N, polish_beta)
                        return val, grad
                    
                    result = minimize(
                        fun_and_grad_polish,
                        best_centers.flatten(),
                        method='L-BFGS-B',
                        jac=True,
                        options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 15000}
                    )
                    polished_centers = result.x.reshape(N, 2)
                    # Ensure validity
                    radii = np.sqrt(np.sum(polished_centers * polished_centers, axis=1))
                    mask = radii > 1.0
                    if np.any(mask):
                        polished_centers[mask] = polished_centers[mask] / radii[mask][:, None] * 0.999999
                    polish_score = _compute_score(polished_centers)
                    if polish_score > best_score:
                        best_score = polish_score
                        best_centers = polished_centers.copy()
                except Exception:
                    pass
    
    return best_centers, best_score

def construct():
    """Main function to construct the packing."""
    start_time = time.time()
    
    # Generate multiple diverse initializations
    initializations = []
    
    # Ring structure
    initializations.append(_get_ring_initialization(N))
    
    # Alternate ring structures
    rings_v2 = [
        (6, 0.20),
        (12, 0.42),
        (12, 0.64),
        (9, 0.80),
        (10, 0.90)
    ]
    initializations.append(_get_ring_initialization(N, ring_radii=rings_v2))
    
    # Hex lattice
    initializations.append(_get_hex_lattice_initialization(N))
    
    # Random with different seeds
    for seed in [1, 2, 3, 4, 5]:
        initializations.append(_get_random_initialization(N, seed=seed))
    
    best_centers = None
    best_score = 0.0
    
    # First pass: quick optimization of all initializations
    n_inits = len(initializations)
    for i, init in enumerate(initializations):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15.0:
            break
            
        remaining = DEADLINE - elapsed - 10.0
        time_per_opt = max(1.0, min(remaining / (n_inits - i), 15.0))
        
        optimized, score = _optimize_from_start(init, time_budget=time_per_opt)
        
        if score > best_score:
            best_score = score
            best_centers = optimized.copy()
    
    # Second pass: refine the best solution with more time
    if best_centers is not None and time.time() - start_time < DEADLINE - 10.0:
        remaining = DEADLINE - time.time() - 5.0
        if remaining > 5.0:
            # Try multiple high-beta polish passes
            final_betas = [500000.0, 1000000.0, 2000000.0]
            for final_beta in final_betas:
                if time.time() - start_time > DEADLINE - 5.0:
                    break
                try:
                    def fun_and_grad_final(x):
                        val, grad = _soft_min_obj_and_grad(x, N, final_beta)
                        return val, grad
                    
                    maxiter = max(2000, int(min(10000, remaining * 500)))
                    result = minimize(
                        fun_and_grad_final,
                        best_centers.flatten(),
                        method='L-BFGS-B',
                        jac=True,
                        options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': maxiter * 5}
                    )
                    final_centers = result.x.reshape(N, 2)
                    # Ensure validity
                    radii = np.sqrt(np.sum(final_centers * final_centers, axis=1))
                    mask = radii > 1.0
                    if np.any(mask):
                        final_centers[mask] = final_centers[mask] / radii[mask][:, None] * 0.999999
                    final_score = _compute_score(final_centers)
                    if final_score > best_score:
                        best_score = final_score
                        best_centers = final_centers.copy()
                except Exception:
                    pass
    
    # Third pass: local perturbation and re-optimization with varying scales
    if best_centers is not None and time.time() - start_time < DEADLINE - 8.0:
        remaining = DEADLINE - time.time() - 5.0
        if remaining > 5.0:
            rng = np.random.default_rng(12345)
            perturbation_scales = [0.005, 0.01, 0.02, 0.002]
            for perturbation_scale in perturbation_scales:
                if time.time() - start_time > DEADLINE - 8.0:
                    break
                # Perturb the best solution
                perturbed = best_centers + rng.normal(0, perturbation_scale, size=best_centers.shape)
                # Clip to unit circle
                radii = np.sqrt(np.sum(perturbed * perturbed, axis=1))
                mask = radii > 1.0
                if np.any(mask):
                    perturbed[mask] = perturbed[mask] / radii[mask][:, None] * 0.999999
                
                # Optimize the perturbed solution
                opt_budget = min(remaining, 4.0)
                try:
                    optimized, score = _optimize_from_start(perturbed, time_budget=opt_budget)
                    if score > best_score:
                        best_score = score
                        best_centers = optimized.copy()
                except Exception:
                    pass
    
    if best_centers is None:
        best_centers = initializations[0]
    
    # Ensure validity
    radii = np.sqrt(np.sum(best_centers * best_centers, axis=1))
    mask = radii > 1.0
    if np.any(mask):
        best_centers[mask] = best_centers[mask] / radii[mask][:, None] * 0.999999
        
    return best_centers
# EVOLVE-BLOCK-END

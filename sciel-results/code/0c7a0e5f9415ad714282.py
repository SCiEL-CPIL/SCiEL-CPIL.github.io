"""Packing 50 equal circles in a unit circle using adaptive soft-min + gradient optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 112.0  # Safety margin for 150s limit

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

def _soft_min_obj_and_grad(params, n, beta, critical_mask=None):
    """
    Objective: -soft_min(constraints)
    If critical_mask is provided, only use those constraints (for focused optimization).
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
    all_vals = np.concatenate([pair_dists / 2.0, bound_dists])
    n_pairs = n * (n - 1) // 2
    
    # 4. If critical mask provided, filter to only critical constraints
    if critical_mask is not None:
        vals = all_vals[critical_mask]
        # Track which pairs and bounds are critical
        pair_crit = critical_mask[:n_pairs]
        bound_crit = critical_mask[n_pairs:]
    else:
        vals = all_vals
        pair_crit = np.ones(n_pairs, dtype=bool)
        bound_crit = np.ones(n, dtype=bool)
    
    if len(vals) == 0:
        return 0.0, np.zeros_like(params)
    
    # 5. Compute Soft Min
    m = np.min(vals)
    shifted = vals - m
    exp_terms = np.exp(-beta * shifted)
    S = np.sum(exp_terms)
    
    obj_val = -(m - (1.0 / beta) * np.log(S))
    
    # 6. Compute Gradient (vectorized)
    weights = exp_terms / S
    
    grad_centers = np.zeros_like(centers)
    
    # Pair contributions
    if np.any(pair_crit):
        # Map back to original pair indices
        pair_indices = np.where(pair_crit)[0]
        if len(pair_indices) > 0:
            # Get the corresponding weights from the filtered vals
            # We need to recompute which pairs are in the filtered set
            all_pair_weights = np.zeros(n_pairs)
            all_pair_weights[pair_indices] = weights[:len(pair_indices)]
            
            mask = all_pair_weights > 1e-12
            if np.any(mask):
                i_m = i_idx[mask]
                j_m = j_idx[mask]
                w_m = all_pair_weights[mask]
                d_m = dists[i_m, j_m]
                d_safe = np.where(d_m > 1e-10, d_m, 1e-10)
                vecs = (centers[i_m] - centers[j_m]) / d_safe[:, np.newaxis]
                contrib = w_m[:, np.newaxis] * 0.5 * vecs
                np.add.at(grad_centers, i_m, -contrib)
                np.add.at(grad_centers, j_m, contrib)
    
    # Boundary contributions
    if np.any(bound_crit):
        bound_indices = np.where(bound_crit)[0]
        if len(bound_indices) > 0:
            all_bound_weights = np.zeros(n)
            all_bound_weights[bound_indices] = weights[len(pair_indices):]
            
            mask_b = all_bound_weights > 1e-12
            if np.any(mask_b):
                idx_b = np.where(mask_b)[0]
                w_b = all_bound_weights[idx_b]
                norm_b = norms[idx_b]
                norm_safe = np.where(norm_b > 1e-10, norm_b, 1e-10)
                vecs_b = centers[idx_b] / norm_safe[:, np.newaxis]
                contrib_b = w_b[:, np.newaxis] * vecs_b
                np.add.at(grad_centers, idx_b, contrib_b)
            
    return obj_val, grad_centers.flatten()

def _identify_critical_constraints(centers, threshold=1.05):
    """
    Identify constraints that are within threshold * min_value of the minimum.
    Returns a boolean mask over all constraints (pairs + boundary).
    """
    n = centers.shape[0]
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    bound_dists = 1.0 - norms
    
    all_vals = np.concatenate([pair_dists, bound_dists])
    min_val = np.min(all_vals)
    
    # Constraints within threshold * min_val are "critical"
    critical = all_vals <= threshold * min_val
    
    return critical

def _get_ring_initialization(n):
    """
    Generate a ring-based initialization.
    """
    centers = []
    centers.append([0.0, 0.0])
    
    rings = [
        (6, 0.22),
        (12, 0.44),
        (12, 0.66),
        (9, 0.82),
        (10, 0.92)
    ]
    
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

def _force_directed_relax(centers, time_budget=3.0):
    """
    Pre-processing step: use force-directed relaxation to resolve overlaps.
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
        
        force_mag = 1.0 / (dists * dists + 1e-6)
        np.fill_diagonal(force_mag, 0.0)
        force_vec = diff * force_mag[:, :, np.newaxis] / np.maximum(dists, 1e-10)[:, :, np.newaxis]
        forces = np.sum(force_vec, axis=1)
        
        # Boundary constraint
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
    Optimize starting from a given configuration using L-BFGS with adaptive critical constraints.
    """
    start_time = time.time()
    current_centers = centers.copy()
    best_centers = centers.copy()
    best_score = _compute_score(centers)
    
    # Force-directed relaxation first (shorter budget)
    current_centers, _ = _force_directed_relax(current_centers, time_budget=min(2.0, time_budget * 0.1))
    score = _compute_score(current_centers)
    if score > best_score:
        best_score = score
        best_centers = current_centers.copy()
        
    # Phase 1: Broad soft-min with moderate beta
    betas_broad = [20.0, 100.0, 500.0]
    for beta in betas_broad:
        if time.time() - start_time > time_budget - 8.0:
            break
        remaining = time_budget - (time.time() - start_time)
        if remaining < 2.0:
            break
            
        def fun_and_grad(x):
            val, grad = _soft_min_obj_and_grad(x, N, beta)
            return val, grad
            
        try:
            result = minimize(
                fun_and_grad,
                current_centers.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1500, 'ftol': 1e-13, 'gtol': 1e-9, 'maxfun': 8000}
            )
            current_centers = result.x.reshape(N, 2)
        except Exception:
            break
            
        score = _compute_score(current_centers)
        if score > best_score:
            best_score = score
            best_centers = current_centers.copy()
    
    # Phase 2: Focused optimization on critical constraints with high beta
    if time.time() - start_time < time_budget - 5.0:
        critical_mask = _identify_critical_constraints(best_centers, threshold=1.02)
        
        betas_focused = [2000.0, 10000.0, 50000.0]
        for beta in betas_focused:
            if time.time() - start_time > time_budget - 4.0:
                break
            remaining = time_budget - (time.time() - start_time)
            if remaining < 2.0:
                break
                
            def fun_and_grad_focused(x):
                val, grad = _soft_min_obj_and_grad(x, N, beta, critical_mask=critical_mask)
                return val, grad
                
            try:
                result = minimize(
                    fun_and_grad_focused,
                    best_centers.flatten(),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 10000}
                )
                current_centers = result.x.reshape(N, 2)
            except Exception:
                break
                
            score = _compute_score(current_centers)
            if score > best_score:
                best_score = score
                best_centers = current_centers.copy()
                # Update critical mask
                critical_mask = _identify_critical_constraints(best_centers, threshold=1.02)
    
    # Phase 3: Final high-precision polish with very high beta on all constraints
    if time.time() - start_time < time_budget - 2.0:
        try:
            def fun_and_grad_polish(x):
                val, grad = _soft_min_obj_and_grad(x, N, 200000.0)
                return val, grad
                
            result = minimize(
                fun_and_grad_polish,
                best_centers.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 15000}
            )
            polished_centers = result.x.reshape(N, 2)
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
    initializations.append(_get_ring_initialization(N))
    initializations.append(_get_hex_lattice_initialization(N))
    
    for seed in [1, 2, 3, 4, 5]:
        initializations.append(_get_random_initialization(N, seed=seed))
    
    best_centers = None
    best_score = 0.0
    
    # First pass: quick optimization of all initializations
    n_inits = len(initializations)
    for i, init in enumerate(initializations):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 12.0:
            break
            
        remaining = DEADLINE - elapsed - 8.0
        time_per_opt = max(1.0, min(remaining / (n_inits - i), 12.0))
        
        optimized, score = _optimize_from_start(init, time_budget=time_per_opt)
        
        if score > best_score:
            best_score = score
            best_centers = optimized.copy()
    
    # Second pass: refine the best solution with more time
    if best_centers is not None and time.time() - start_time < DEADLINE - 8.0:
        remaining = DEADLINE - time.time() - 4.0
        if remaining > 4.0:
            # Do a final focused optimization with very high precision
            try:
                critical_mask = _identify_critical_constraints(best_centers, threshold=1.01)
                
                def fun_and_grad_final(x):
                    val, grad = _soft_min_obj_and_grad(x, N, 500000.0, critical_mask=critical_mask)
                    return val, grad
                    
                result = minimize(
                    fun_and_grad_final,
                    best_centers.flatten(),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 25000}
                )
                final_centers = result.x.reshape(N, 2)
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
    
    if best_centers is None:
        best_centers = initializations[0]
    
    # Ensure validity
    radii = np.sqrt(np.sum(best_centers * best_centers, axis=1))
    mask = radii > 1.0
    if np.any(mask):
        best_centers[mask] = best_centers[mask] / radii[mask][:, None] * 0.999999
        
    return best_centers
# EVOLVE-BLOCK-END

"""Improved program for packing 300 circles in a unit circle using smooth soft-min optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    """
    points = []
    # Estimate spacing. r ~ 1/sqrt(N). Diameter ~ 2r.
    # Hex lattice spacing s = 2r.
    # Area of hexagon per point ~ s^2 * sqrt(3)/2.
    # N * s^2 * sqrt(3)/2 ~ pi.
    # s ~ sqrt(2*pi / (N*sqrt(3))).
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    # Generate a large grid of hex lattice points
    # Hex lattice coordinates:
    # x = s * (i + 0.5 * (j % 2))
    # y = s * j * sqrt(3) / 2
    max_k = int(1.0 / (s * 0.5)) + 2
    
    candidates = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * j * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    if len(candidates) < n:
        # If we didn't generate enough, shrink spacing slightly and retry
        # This is a fallback, the estimate should be close
        s *= 0.9
        max_k = int(1.0 / (s * 0.5)) + 3
        candidates = []
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = s * (i + 0.5 * (j % 2))
                y = s * j * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    candidates.append([x, y])
        candidates = np.array(candidates)
        
    if len(candidates) >= n:
        # Sort by distance to origin to pick the most central ones?
        # Or just pick the first n? Picking central ones might be better for stability.
        radii = np.linalg.norm(candidates, axis=1)
        indices = np.argsort(radii)[:n]
        points = candidates[indices]
    else:
        # Fallback: random points
        points = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        points[mask] = points[mask] / norms[mask, np.newaxis] * 0.9

    # Add small random perturbation to break symmetry
    points += np.random.normal(0, 1e-4, points.shape)
    
    return points

def _ring_init(n):
    """
    Initialize points in concentric rings with optimized angular phases.
    """
    points = []
    # Estimate ring structure
    # Number of rings ~ sqrt(n)
    num_rings = int(np.sqrt(n * 2 / np.sqrt(3)))
    
    # Distribute points among rings
    # Ring k has roughly proportional to circumference
    ring_counts = []
    remaining = n
    for k in range(num_rings):
        if k == 0:
            count = 1
        else:
            # Approximate count based on circumference
            count = int(2 * np.pi * k / (2 * np.sqrt(3) * np.sqrt(n)))
            count = max(1, count)
        ring_counts.append(count)
    
    # Adjust to match total
    diff = n - sum(ring_counts)
    if diff > 0:
        # Add extra to outer rings
        for i in range(len(ring_counts)-1, -1, -1):
            add = min(diff, max(1, ring_counts[i]//3))
            ring_counts[i] += add
            diff -= add
            if diff <= 0:
                break
    elif diff < 0:
        # Remove from outer rings
        for i in range(len(ring_counts)-1, -1, -1):
            remove = min(-diff, ring_counts[i]-1)
            ring_counts[i] -= remove
            diff += remove
            if diff >= 0:
                break
    
    # Ensure we have exactly n points
    while sum(ring_counts) > n:
        for i in range(len(ring_counts)-1, -1, -1):
            if ring_counts[i] > 1:
                ring_counts[i] -= 1
                break
    while sum(ring_counts) < n:
        ring_counts[-1] += 1
    
    # Place points
    idx = 0
    for k, count in enumerate(ring_counts):
        if k == 0:
            if count >= 1:
                points.append([0.0, 0.0])
                idx += 1
        else:
            radius = k * np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
            # Phase offset to stagger rings
            phase = (k % 2) * np.pi / count
            for m in range(count):
                if idx >= n:
                    break
                angle = 2 * np.pi * m / count + phase
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                if x*x + y*y <= 1.0:
                    points.append([x, y])
                    idx += 1
    
    points = np.array(points)
    
    if len(points) < n:
        # Fill remaining with random points
        while len(points) < n:
            x, y = np.random.uniform(-0.9, 0.9, 2)
            if x*x + y*y <= 1.0:
                points = np.vstack([points, [[x, y]]])
    elif len(points) > n:
        points = points[:n]
    
    # Add small random perturbation
    points += np.random.normal(0, 1e-4, points.shape)
    
    return points

def _force_directed_relax(points, n_steps=100, dt=0.01):
    """
    Simple force-directed relaxation to resolve overlaps and push points inward.
    """
    n = len(points)
    points = points.copy()
    
    # Precompute upper triangle indices
    iu = np.triu_indices(n, k=1)
    
    for step in range(n_steps):
        forces = np.zeros_like(points)
        
        # Pairwise repulsion
        vecs = points[iu[0]] - points[iu[1]]
        dists = np.linalg.norm(vecs, axis=1)
        # Only apply force when overlapping or very close
        mask = dists < 0.1
        if np.any(mask):
            safe_dists = np.where(dists[mask] > 1e-10, dists[mask], 1e-10)
            # Repulsive force proportional to overlap
            repulse = (0.1 - dists[mask]) / safe_dists
            force_vecs = repulse[:, np.newaxis] * vecs[mask]
            np.add.at(forces, iu[0][mask], -force_vecs)
            np.add.at(forces, iu[1][mask], force_vecs)
        
        # Boundary attraction (push inward if too close to boundary)
        norms = np.linalg.norm(points, axis=1)
        boundary_mask = norms > 0.95
        if np.any(boundary_mask):
            # Pull inward
            pull_strength = (norms[boundary_mask] - 0.95) * 2.0
            forces[boundary_mask] -= pull_strength[:, np.newaxis] * points[boundary_mask] / norms[boundary_mask, np.newaxis]
        
        # Apply forces with damping
        points += dt * forces
        
        # Keep points inside unit circle
        norms = np.linalg.norm(points, axis=1)
        outside = norms > 1.0
        if np.any(outside):
            points[outside] /= norms[outside, np.newaxis]
    
    return points

def _epigraph_obj_and_grad(x, iu):
    """
    Epigraph formulation: minimize t subject to all constraints <= t.
    We minimize a smooth approximation: t + sum of soft-violations.
    Variables: x = [p_1, p_2, ..., p_N, t] where t is scalar.
    """
    n = N
    p = x[:2*n].reshape(n, 2)
    t = x[2*n]
    
    i_idx, j_idx = iu
    vecs = p[i_idx] - p[j_idx]
    pair_dists = np.sqrt(np.sum(vecs**2, axis=1)) / 2.0
    
    norms = np.linalg.norm(p, axis=1)
    bound_dists = 1.0 - norms
    
    # All constraints: we want c_i <= t for all i
    # Smooth penalty: sum of softplus(t - c_i) with large beta
    beta = 5000.0
    
    violations_pair = t - pair_dists
    violations_bound = t - bound_dists
    
    # softplus(x) = log(1 + exp(x)), but for stability:
    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    def _softplus(v):
        return np.maximum(v, 0) + np.log1p(np.exp(-np.abs(v)))
    
    sp_pairs = _softplus(violations_pair)
    sp_bounds = _softplus(violations_bound)
    
    val = t + np.sum(sp_pairs) + np.sum(sp_bounds)
    
    # Gradient
    grad = np.zeros_like(x)
    
    # d/dt: 1 + sum of sigmoid(violations)
    sig_pairs = 1.0 / (1.0 + np.exp(-violations_pair))
    sig_bounds = 1.0 / (1.0 + np.exp(-violations_bound))
    grad_t = 1.0 + np.sum(sig_pairs) + np.sum(sig_bounds)
    grad[2*n] = grad_t
    
    # d/dp for pairs: -sig_ij * d(pair_dist)/dp
    safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
    coeff = sig_pairs / (2.0 * safe_pair_dists)
    
    np.add.at(grad[:2*n].reshape(-1, 2), i_idx, -coeff[:, np.newaxis] * vecs)
    np.add.at(grad[:2*n].reshape(-1, 2), j_idx, coeff[:, np.newaxis] * vecs)
    
    # d/dp for bounds: -sig_k * d(bound_dist)/dp = -sig_k * (-p_k/norm_k) = sig_k * p_k/norm_k
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    bound_grad_vecs = sig_bounds[:, np.newaxis] * p / safe_norms[:, np.newaxis]
    grad[:2*n] += bound_grad_vecs
    
    return val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 147.0  # 3s safety margin
    
    # Try multiple initializations
    inits = [_hex_lattice_init(N), _ring_init(N)]
    
    best_points = None
    best_score = -np.inf
    
    for init_points in inits:
        if time.time() > deadline - 20:
            break
        
        points = init_points
        score = _compute_score(points)
        
        if score > best_score:
            best_points = points.copy()
            best_score = score
    
    if best_points is None:
        best_points = _hex_lattice_init(N)
        best_score = _compute_score(best_points)
    
    # Apply force-directed relaxation to improve initial configuration
    relaxed_points = _force_directed_relax(best_points, n_steps=50, dt=0.005)
    relaxed_score = _compute_score(relaxed_points)
    if relaxed_score > best_score:
        best_points = relaxed_points
        best_score = relaxed_score
    
    # Precompute upper triangle indices
    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])
    
    # Optimization using L-BFGS-B with smooth soft-min objective
    def objective_and_grad(x, beta):
        p = x.reshape(-1, 2)
        
        # Calculate pair distances efficiently
        i_idx, j_idx = iu
        vecs = p[i_idx] - p[j_idx]
        pair_dists = np.sqrt(np.sum(vecs**2, axis=1)) / 2.0
        
        # Boundary distances
        norms = np.linalg.norm(p, axis=1)
        bound_dists = 1.0 - norms
        
        # To avoid overflow in exp, we use the log-sum-exp trick.
        # Let z = -beta * c.
        # soft_min(c) = -1/beta * log(sum(exp(-beta*c)))
        
        z_pairs = -beta * pair_dists
        z_bounds = -beta * bound_dists
        
        # Max values for stability
        max_z_pair = np.max(z_pairs)
        max_z_bound = np.max(z_bounds)
        max_z = max(max_z_pair, max_z_bound)
        
        # exp terms
        exp_pairs = np.exp(z_pairs - max_z)
        exp_bounds = np.exp(z_bounds - max_z)
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
        
        # log_sum_exp = max_z + log(sum_exp)
        log_sum_exp = max_z + np.log(sum_exp)
        
        # Objective value: -soft_min = 1/beta * log_sum_exp
        val = log_sum_exp / beta
        
        # Gradient
        # Weights
        w_pairs = exp_pairs / sum_exp
        w_bounds = exp_bounds / sum_exp
        
        grad = np.zeros_like(p)
        
        # Pair contributions
        # d(pair_dist)/dp_i = (p_i - p_j) / (2 * dist)
        # d(-beta * pair_dist)/dp_i = -beta * (p_i - p_j) / (2 * dist)
        # soft_min gradient: sum(w_k * d(-beta*c_k)/dp) / beta = sum(w_k * d(-c_k)/dp)
        # So: grad_pair_i = sum over j of w_ij * (p_j - p_i) / (2 * pair_dist_ij)
        
        # Avoid division by zero
        safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
        
        # Coefficient for each pair: w / (2 * pair_dist)
        coeff = w_pairs / (2.0 * safe_pair_dists)
        
        # Accumulate gradients
        # For point i: sum of coeff * (p_j - p_i) for all j > i
        # For point j: sum of coeff * (p_i - p_j) for all i < j
        np.add.at(grad, i_idx, -coeff[:, np.newaxis] * vecs)
        np.add.at(grad, j_idx, coeff[:, np.newaxis] * vecs)
        
        # Boundary contributions
        # d(-bound_dist)/dp = d(-1 + norm)/dp = p / norm
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        bound_grad_vecs = p / safe_norms[:, np.newaxis]
        
        grad += w_bounds[:, np.newaxis] * bound_grad_vecs
        
        return val, grad

    # Optimization loop with increasing beta - improved schedule
    # Spend more time at high beta for better convergence
    betas = [5.0, 20.0, 100.0, 500.0, 2000.0, 5000.0, 10000.0, 50000.0, 100000.0, 200000.0]
    
    current_points = best_points.copy()
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
            
        try:
            # Use a tighter tolerance for higher betas
            if beta >= 100000:
                maxiter = 400
                gtol = 1e-14
                ftol = 1e-16
                maxls = 80
            elif beta >= 50000:
                maxiter = 400
                gtol = 1e-14
                ftol = 1e-16
                maxls = 70
            elif beta >= 10000:
                maxiter = 500
                gtol = 1e-13
                ftol = 1e-15
                maxls = 60
            elif beta >= 1000:
                maxiter = 600
                gtol = 1e-12
                ftol = 1e-14
                maxls = 50
            elif beta >= 100:
                maxiter = 700
                gtol = 1e-11
                ftol = 1e-13
                maxls = 40
            else:
                maxiter = 500
                gtol = 1e-10
                ftol = 1e-12
                maxls = 30
            
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                current_points.flatten(),
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': maxiter,
                    'gtol': gtol,
                    'ftol': ftol,
                    'maxls': maxls,
                    'disp': False
                }
            )
            
            if time.time() > deadline - 8:
                break
                
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > best_score:
                best_points = candidate
                best_score = candidate_score
                current_points = candidate
            else:
                current_points = best_points
                
        except Exception:
            pass

    # Epigraph polish: directly optimize the nonsmooth min with auxiliary t
    if time.time() < deadline - 10:
        try:
            # Initialize t slightly above current best score
            t_init = best_score + 1e-6
            x_init = np.concatenate([best_points.flatten(), [t_init]])
            
            res = minimize(
                lambda x: _epigraph_obj_and_grad(x, iu)[0],
                x_init,
                jac=lambda x: _epigraph_obj_and_grad(x, iu)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'gtol': 1e-14,
                    'ftol': 1e-16,
                    'maxls': 100,
                    'disp': False
                }
            )
            
            if time.time() < deadline - 5:
                candidate = res.x[:2*N].reshape(-1, 2)
                candidate_score = _compute_score(candidate)
                
                if candidate_score > best_score:
                    best_points = candidate
                    best_score = candidate_score
        except Exception:
            pass

    # Multi-start perturbation: try small perturbations of best solution
    if time.time() < deadline - 12:
        for attempt in range(4):
            if time.time() > deadline - 6:
                break
            try:
                perturbed = best_points + np.random.normal(0, 1e-5, (N, 2))
                # Ensure inside
                norms = np.linalg.norm(perturbed, axis=1)
                outside = norms > 1.0
                if np.any(outside):
                    perturbed[outside] /= norms[outside, np.newaxis]
                
                # Quick optimization with medium beta
                beta = 10000.0
                res = minimize(
                    lambda x: objective_and_grad(x, beta)[0],
                    perturbed.flatten(),
                    jac=lambda x: objective_and_grad(x, beta)[1],
                    method='L-BFGS-B',
                    options={
                        'maxiter': 300,
                        'gtol': 1e-13,
                        'ftol': 1e-15,
                        'maxls': 50,
                        'disp': False
                    }
                )
                candidate = res.x.reshape(-1, 2)
                candidate_score = _compute_score(candidate)
                
                if candidate_score > best_score:
                    best_points = candidate
                    best_score = candidate_score
            except Exception:
                pass

    # Final polish with very high beta if time permits
    if time.time() < deadline - 5:
        try:
            final_beta = 500000.0
            res = minimize(
                lambda x: objective_and_grad(x, final_beta)[0],
                best_points.flatten(),
                jac=lambda x: objective_and_grad(x, final_beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 400,
                    'gtol': 1e-15,
                    'ftol': 1e-17,
                    'maxls': 100,
                    'disp': False
                }
            )
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            if candidate_score > best_score:
                best_points = candidate
                best_score = candidate_score
        except Exception:
            pass

    # Ensure points are inside
    norms = np.linalg.norm(best_points, axis=1)
    outside = norms > 1.0
    if np.any(outside):
        best_points[outside] /= norms[outside, np.newaxis]
        best_score = _compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END

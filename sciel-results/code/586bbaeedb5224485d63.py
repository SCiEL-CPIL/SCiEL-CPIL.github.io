"""Packing 50 equal circles in a unit circle - ring-based init + soft-min L-BFGS annealing."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START = time.time()
_DEADLINE = _START + 135.0  # 15s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    # Pairwise distances
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r_pair = min_pair / 2.0
    return min(r_cont, r_pair)


def _soft_min(values, beta):
    """Compute the soft minimum of an array of values."""
    v = np.asarray(values, dtype=np.float64)
    m = np.min(v)
    # Clip to avoid overflow in exp
    exp_terms = np.exp(-beta * (v - m))
    return m - (1.0 / beta) * np.log(np.sum(exp_terms))


def _objective_and_gradient(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints)
    params: flat array of length 2*n
    Returns (objective_value, gradient)
    """
    centers = params.reshape(n, 2)
    
    # Pairwise distances
    i_idx, j_idx = np.triu_indices(n, k=1)
    diff = centers[i_idx, :] - centers[j_idx, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    boundary_dists = 1.0 - norms
    
    # Combine all constraints
    # We want to maximize min(dists/2, 1-norms)
    # So we minimize -min(...)
    vals = np.concatenate([dists / 2.0, boundary_dists])
    
    obj = -_soft_min(vals, beta)
    
    # Compute gradient of -soft_min
    # d/dx (-soft_min) = - (sum exp(-beta(v_i - m)) / sum exp(-beta(v_i - m))) * (-beta * dv_i/dx)
    #                  = beta * sum(weights_i * dv_i/dx)
    # where weights_i = exp(-beta(v_i - m)) / sum(exp(...))
    
    m = np.min(vals)
    exp_vals = np.exp(-beta * (vals - m))
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    grad = np.zeros_like(centers)
    
    # Pairwise part: v_i = dists_k / 2
    # d(dist_k)/d(centers_i) = (c_i - c_j) / dist_k
    # d(v_k)/d(centers_i) = 0.5 * (c_i - c_j) / dist_k
    pair_weights = weights[:len(i_idx)]
    safe_dists = np.where(dists > 1e-8, dists, 1.0)
    unit_vecs = diff / safe_dists[:, np.newaxis]
    # Gradient contribution for i_idx: + 0.5 * weight * unit_vec
    # Gradient contribution for j_idx: - 0.5 * weight * unit_vec
    weighted_vecs = pair_weights[:, np.newaxis] * 0.5 * unit_vecs
    
    np.add.at(grad, i_idx, weighted_vecs)
    np.add.at(grad, j_idx, -weighted_vecs)
    
    # Boundary part: v_i = 1 - norm_i
    # d(1 - norm_i)/d(centers_i) = - centers_i / norm_i
    bound_weights = weights[len(i_idx):]
    safe_norms = np.where(norms > 1e-8, norms, 1.0)
    # Gradient contribution: + weight * (- centers_i / norm_i)
    # Note: obj = -soft_min, so grad_obj = - grad_soft_min
    # grad_soft_min = sum w_i * dv_i/dx
    # grad_obj = - sum w_i * dv_i/dx
    # Wait, let's re-verify signs.
    # f(x) = -soft_min(v(x))
    # f'(x) = - [ sum w_i * v_i'(x) ]
    # For pairwise: v_k' = 0.5 * u_k
    # f'_pair = - sum w_k * 0.5 * u_k
    # My code above calculated grad = + sum w * 0.5 * u. This is grad_soft_min.
    # So I need to negate the whole gradient at the end or negate the contributions.
    
    # Let's stick to: grad_obj = - (beta * sum(w_i * dv_i/dx)) ? 
    # No, the beta cancels in the weights definition if we define w = exp/S.
    # soft_min = m - 1/beta * log(S)
    # d(soft_min)/dx = -1/beta * (1/S) * sum(exp(-beta(v-m)) * (-beta * dv/dx))
    #                = 1/S * sum(exp(-beta(v-m)) * dv/dx)
    #                = sum(w_i * dv_i/dx)
    # So grad_obj = - sum(w_i * dv_i/dx)
    
    # My current grad accumulation is sum(w_i * dv_i/dx).
    # So I just need to return -grad.
    
    # Boundary gradient: dv_i/dx_i = - c_i / norm_i
    # Contribution to grad_soft_min: w_i * (- c_i / norm_i)
    # Contribution to grad_obj: - w_i * (- c_i / norm_i) = + w_i * c_i / norm_i
    
    # Let's re-evaluate the pairwise part in the context of grad_obj.
    # dv_k/dx_i = 0.5 * u_k
    # grad_obj_pair_i = - w_k * 0.5 * u_k
    # grad_obj_pair_j = - w_k * (-0.5 * u_k) = + w_k * 0.5 * u_k
    
    # So:
    # grad_obj[i_idx] -= 0.5 * w * u
    # grad_obj[j_idx] += 0.5 * w * u
    # grad_obj[boundary] += w * c / norm
    
    # Let's restart the gradient calculation cleanly.
    grad = np.zeros_like(centers)
    
    # Pairwise
    # d obj / d c_i = - 0.5 * sum_{j>i} w_{ij} * (c_i - c_j)/d_{ij}
    # d obj / d c_j = + 0.5 * sum_{i<j} w_{ij} * (c_i - c_j)/d_{ij}
    
    np.add.at(grad, i_idx, -0.5 * pair_weights[:, np.newaxis] * unit_vecs)
    np.add.at(grad, j_idx, 0.5 * pair_weights[:, np.newaxis] * unit_vecs)
    
    # Boundary
    # d obj / d c_i = - w_i * d(1 - norm_i)/d c_i = - w_i * (- c_i / norm_i) = w_i * c_i / norm_i
    grad += bound_weights[:, np.newaxis] * centers / safe_norms[:, np.newaxis]
    
    return obj, grad


def _optimize_lbfgs(centers, beta, maxiter, gtol):
    """Optimize using L-BFGS-B."""
    try:
        res = minimize(
            _objective_and_gradient,
            centers.flatten(),
            args=(N, beta),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol, 'disp': False}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return centers


def _get_ring_init(n, seed=42):
    """
    Generate a ring-based initialization matching known optimal structure.
    Rings: 1, 6, 12, 18, 13 = 50
    """
    rng = np.random.default_rng(seed)
    
    # Standard ring counts for N=50
    rings = [1, 6, 12, 18, 13]
    while sum(rings) > n:
        rings[-1] -= 1
        if rings[-1] <= 0:
            rings.pop()
    while sum(rings) < n:
        rings.append(1)
    
    points = []
    # Radii tuned to be close to optimal spacing
    # r ~ 0.125, so ring radii should be roughly k * 2r
    # 0, 0.25, 0.5, 0.75, 0.95
    ring_radii = [0.0, 0.25, 0.50, 0.75, 0.92]
    
    for ring_idx, count in enumerate(rings):
        if count == 0:
            continue
        if ring_idx == 0:
            points.append([0.0, 0.0])
        else:
            radius = ring_radii[min(ring_idx, len(ring_radii) - 1)]
            # Randomize angle offset for diversity
            angle_offset = rng.uniform(0, 2 * np.pi)
            for k in range(count):
                angle = angle_offset + 2 * np.pi * k / count
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                # Add small jitter to break symmetry
                r_jitter = rng.normal(0, 0.002)
                a_jitter = rng.normal(0, 0.01)
                x += r_jitter * np.cos(angle)
                y += r_jitter * np.sin(angle)
                points.append([x, y])
    
    points = np.array(points[:n])
    
    # Ensure all points are strictly inside
    norms = np.linalg.norm(points, axis=1)
    mask = norms > 0.99
    if np.any(mask):
        points[mask] *= (0.99 / norms[mask])[:, np.newaxis]
        
    return points


def construct():
    """Construct 50 circle centers in unit circle maximizing common radius."""
    best_pts = None
    best_score = 0.0
    
    # Generate diverse initializations
    initializations = []
    for seed in [42, 123, 999, 314, 271]:
        initializations.append(_get_ring_init(N, seed=seed))
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() - _START > _DEADLINE - 30:
            break
        
        current_pts = init_pts.copy()
        current_score = _score(current_pts)
        
        # Annealing schedule: start with low beta, end with high beta
        # Low beta allows more freedom to move, high beta forces tight packing
        betas = [10.0, 30.0, 100.0, 300.0, 1000.0]
        
        for beta in betas:
            if time.time() - _START > _DEADLINE - 15:
                break
            
            # Increase maxiter for higher beta to ensure convergence
            maxiter = 100 if beta < 100 else 300
            gtol = 1e-5 if beta < 100 else 1e-7
            
            refined = _optimize_lbfgs(current_pts, beta, maxiter, gtol)
            ref_score = _score(refined)
            
            if ref_score > current_score:
                current_pts = refined
                current_score = ref_score
        
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()
            
    # Final polish on the best candidate found
    if best_pts is not None and time.time() - _START < _DEADLINE - 10:
        # Very high beta for final precision
        final_pts = _optimize_lbfgs(best_pts, 2000.0, maxiter=500, gtol=1e-8)
        final_score = _score(final_pts)
        if final_score > best_score:
            best_pts = final_pts
            best_score = final_score

    if best_pts is None:
        best_pts = _get_ring_init(N, seed=42)
        
    # Final validation: ensure inside unit circle
    r = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r > 1.0
    if np.any(mask):
        best_pts[mask] /= r[mask, None] * (1.0 + 1e-9)
        
    return best_pts

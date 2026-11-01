"""Combined approach: contact-graph weighted optimization + ring/hex initializations."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = _START_TIME + 112.0  # Safety margin before 120s limit


def _score(positions):
    """Compute the valid packing radius."""
    n = len(positions)
    if n != N:
        return 0.0
    if not np.all(np.isfinite(positions)):
        return 0.0
    radii = np.hypot(positions[:, 0], positions[:, 1])
    if np.any(radii > 1.0):
        return 0.0
    min_boundary = 1.0 - np.max(radii)
    if min_boundary < 0:
        return 0.0
    if n < 2:
        return min_boundary
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, min_boundary)


def _contact_weights(dists, norms, r_target, temperature=0.05):
    """
    Compute weights for constraints based on how close they are to the target radius.
    Contacts closer to r_target get higher weight (they are binding constraints).
    This focuses optimization effort on critical contacts.
    """
    # Pair distances / 2
    pair_vals = dists / 2.0
    # Boundary distances
    bound_vals = 1.0 - norms
    
    # Weight based on proximity to r_target
    # Use a soft-max-like weighting: constraints closer to r_target get more weight
    pair_diff = np.abs(pair_vals - r_target)
    bound_diff = np.abs(bound_vals - r_target)
    
    # Temperature controls how sharply we focus on binding constraints
    pair_w = np.exp(-pair_diff / temperature)
    bound_w = np.exp(-bound_diff / temperature)
    
    # Normalize so total weight is 1
    total = np.sum(pair_w) + np.sum(bound_w)
    if total > 0:
        pair_w /= total
        bound_w /= total
    
    return pair_w, bound_w


def _weighted_objective_and_grad(params, n, r_target, temperature):
    """
    Objective to minimize: negative weighted sum of constraint violations.
    Weights focus on binding constraints (those closest to r_target).
    """
    centers = params.reshape(n, 2)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    diff = centers[i_idx, :] - centers[j_idx, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    norms = np.linalg.norm(centers, axis=1)
    bound_dists = 1.0 - norms
    
    # Get contact weights
    pair_w, bound_w = _contact_weights(dists, norms, r_target, temperature)
    
    # Objective: minimize sum of weighted (r_target - constraint)^2 for violated constraints
    # For pairs: violation = max(0, r_target - dist/2)
    # For boundary: violation = max(0, r_target - (1-norm))
    
    pair_violations = np.maximum(0, r_target - dists / 2.0)
    bound_violations = np.maximum(0, r_target - bound_dists)
    
    obj = np.sum(pair_w * pair_violations ** 2) + np.sum(bound_w * bound_violations ** 2)
    
    # Gradient
    grad = np.zeros_like(centers)
    
    # Pair contributions
    for k in range(len(i_idx)):
        i, j = i_idx[k], j_idx[k]
        if pair_violations[k] > 0:
            d = dists[k]
            if d > 1e-10:
                vec = diff[k] / d
                factor = -2.0 * pair_w[k] * pair_violations[k] / 2.0
                grad[i] += factor * vec
                grad[j] -= factor * vec
    
    # Boundary contributions
    for i in range(n):
        if bound_violations[i] > 0:
            norm = norms[i]
            if norm > 1e-10:
                vec = centers[i] / norm
                factor = -2.0 * bound_w[i] * bound_violations[i]
                grad[i] += factor * vec
    
    return obj, grad.ravel()


def _soft_min_objective_and_grad(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints)
    Standard soft-min approach with stabilized log-sum-exp.
    """
    centers = params.reshape(n, 2)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    diff = centers[i_idx, :] - centers[j_idx, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    norms = np.linalg.norm(centers, axis=1)
    bound_dists = 1.0 - norms
    
    vals = np.concatenate([dists / 2.0, bound_dists])
    
    m = np.min(vals)
    shifted = vals - m
    exp_terms = np.exp(-beta * shifted)
    S = np.sum(exp_terms)
    
    obj = -(m - (1.0 / beta) * np.log(S))
    
    weights = exp_terms / S
    
    grad = np.zeros_like(centers)
    
    # Pair part
    pair_weights = weights[:len(i_idx)]
    safe_dists = np.where(dists > 1e-8, dists, 1.0)
    unit_vecs = diff / safe_dists[:, np.newaxis]
    weighted_vecs = pair_weights[:, np.newaxis] * 0.5 * unit_vecs
    
    np.add.at(grad, i_idx, weighted_vecs)
    np.add.at(grad, j_idx, -weighted_vecs)
    
    # Boundary part
    bound_weights = weights[len(i_idx):]
    safe_norms = np.where(norms > 1e-8, norms, 1.0)
    grad -= bound_weights[:, np.newaxis] * centers / safe_norms[:, np.newaxis]
    
    return obj, -grad.ravel()


def _optimize(centers, method='soft_min', beta=500.0, r_target=None, temperature=0.05, maxiter=300):
    """
    Optimize using either soft-min or contact-weighted approach.
    """
    try:
        if method == 'soft_min':
            res = minimize(
                _soft_min_objective_and_grad,
                centers.flatten(),
                args=(N, beta),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': maxiter, 'gtol': 1e-10, 'disp': False}
            )
        else:
            res = minimize(
                _weighted_objective_and_grad,
                centers.flatten(),
                args=(N, r_target, temperature),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': maxiter, 'gtol': 1e-10, 'disp': False}
            )
        
        result = res.x.reshape(N, 2)
        # Ensure points stay within unit disk
        radii = np.hypot(result[:, 0], result[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            result[mask] *= (0.999 / radii[mask])[:, None]
        
        return result
    except Exception:
        return centers


def _get_ring_init(n, seed=42):
    """Generate a ring-based initialization with approximate optimal structure."""
    rng = np.random.default_rng(seed)
    
    # Approximate ring structure for N=50
    rings = [1, 6, 12, 13, 18]
    while sum(rings) > n:
        rings[-1] -= 1
        if rings[-1] <= 0:
            rings.pop()
    while sum(rings) < n:
        rings.append(1)
    
    points = []
    ring_radii = [0.0, 0.22, 0.44, 0.65, 0.85, 0.95]
    
    for ring_idx, count in enumerate(rings):
        if count == 0:
            continue
        if ring_idx == 0:
            points.append([0.0, 0.0])
        else:
            radius = ring_radii[min(ring_idx, len(ring_radii)-1)]
            angle_offset = rng.uniform(0, 2 * np.pi)
            for k in range(count):
                angle = angle_offset + 2 * np.pi * k / count
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                # Small jitter
                jitter = rng.normal(0, 0.005)
                x += jitter * np.cos(angle)
                y += jitter * np.sin(angle)
                points.append([x, y])
    
    points = np.array(points[:n])
    
    # Ensure all points are within unit disk
    norms = np.linalg.norm(points, axis=1)
    mask = norms > 0.98
    if np.any(mask):
        points[mask] *= (0.98 / norms[mask])[:, np.newaxis]
    
    return points


def _get_hex_lattice(n, seed_offset=0):
    """Generate a hexagonal lattice initialization."""
    rng = np.random.default_rng(42 + seed_offset)
    s_est = np.sqrt(2 * np.pi / (n * np.sqrt(3)))
    
    points = []
    max_range = int(2.0 / s_est) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s_est + (j % 2) * s_est / 2.0
            y = j * s_est * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
    
    if len(points) < n:
        points = []
        for _ in range(n * 10):
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        while len(points) < n:
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points = np.vstack([points, [[x, y]]])
    
    points = points + rng.normal(0, 0.01, points.shape)
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]


def _get_random_init(n, seed):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.85)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)


def _polish_soft_min(centers, start_time):
    """Progressively polish with increasing beta using soft-min."""
    current = centers.copy()
    best_score = _score(current)
    
    betas = [50.0, 100.0, 200.0, 400.0, 800.0, 1500.0, 3000.0]
    
    for beta in betas:
        if time.time() - start_time > _DEADLINE - 8:
            break
        
        polished = _optimize(current, method='soft_min', beta=beta, maxiter=200)
        
        if np.all(np.isfinite(polished)):
            r = _score(polished)
            if r > best_score:
                best_score = r
                current = polished
    
    return current, best_score


def _polish_contact_weighted(centers, start_time):
    """Polish using contact-weighted optimization with adaptive r_target."""
    current = centers.copy()
    best_score = _score(current)
    
    # Start with current best score as r_target
    r_target = best_score
    temperatures = [0.1, 0.05, 0.02, 0.01]
    
    for temp in temperatures:
        if time.time() - start_time > _DEADLINE - 8:
            break
        
        for _ in range(3):  # Multiple iterations at each temperature
            if time.time() - start_time > _DEADLINE - 8:
                break
            
            polished = _optimize(current, method='contact', r_target=r_target, 
                                temperature=temp, maxiter=300)
            
            if np.all(np.isfinite(polished)):
                r = _score(polished)
                if r > best_score:
                    best_score = r
                    current = polished
                    # Update r_target to new best
                    r_target = best_score
    
    return current, best_score


def _high_beta_polish(centers, start_time):
    """Final high-precision polish with very high beta values."""
    current = centers.copy()
    best_score = _score(current)
    
    betas = [5000.0, 10000.0, 20000.0]
    
    for beta in betas:
        if time.time() - start_time > _DEADLINE - 4:
            break
        
        polished = _optimize(current, method='soft_min', beta=beta, maxiter=400)
        
        if np.all(np.isfinite(polished)):
            r = _score(polished)
            if r > best_score:
                best_score = r
                current = polished
    
    return current, best_score


def construct():
    """
    Construct positions for N circles in the unit disk.
    Returns an array of shape (N, 2).
    """
    start_time = time.time()
    
    # Generate diverse initializations
    inits = [
        _get_ring_init(N, seed=42),
        _get_ring_init(N, seed=123),
        _get_hex_lattice(N, seed_offset=0),
        _get_hex_lattice(N, seed_offset=1),
        _get_random_init(N, seed=7),
        _get_random_init(N, seed=99),
    ]
    
    best_points = None
    best_score = 0.0
    
    # Phase 1: Soft-min polishing from each initialization
    for init in inits:
        if time.time() - start_time > _DEADLINE - 20:
            break
        
        polished, score = _polish_soft_min(init, start_time)
        
        if score > best_score:
            best_score = score
            best_points = polished.copy()
    
    if best_points is None:
        best_points = inits[0]
        best_score = _score(best_points)
    
    # Phase 2: Contact-weighted polishing from the best solution
    if time.time() - start_time < _DEADLINE - 15:
        polished, score = _polish_contact_weighted(best_points, start_time)
        if score > best_score:
            best_score = score
            best_points = polished.copy()
    
    # Phase 3: Try contact-weighted from other initializations if time permits
    if time.time() - start_time < _DEADLINE - 10:
        for init in inits[:3]:
            if time.time() - start_time > _DEADLINE - 8:
                break
            
            # First do a quick soft-min polish
            quick_polished, _ = _polish_soft_min(init, start_time)
            
            # Then contact-weighted
            polished, score = _polish_contact_weighted(quick_polished, start_time)
            
            if score > best_score:
                best_score = score
                best_points = polished.copy()
    
    # Phase 4: Final high-precision polish
    if time.time() - start_time < _DEADLINE - 4:
        best_points, best_score = _high_beta_polish(best_points, start_time)
    
    # Final validation
    final_score = _score(best_points)
    if final_score > best_score:
        best_score = final_score
    
    # Ensure validity
    if not np.all(np.isfinite(best_points)):
        best_points = inits[0]
        best_score = _score(best_points)
    
    return best_points
# EVOLVE-BLOCK-END

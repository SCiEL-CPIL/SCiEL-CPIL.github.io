"""Improved packing: multi-start L-BFGS-B with smooth soft-min objective."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _score_raw(centers):
    """Compute the actual score (minimum radius) for a set of centers."""
    # Check validity
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Distance from origin
    norms = np.sqrt(np.sum(centers**2, axis=1))
    # Check all inside unit circle
    if np.any(norms >= 1.0):
        return 0.0
    
    # Pairwise distances
    # Use efficient computation
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    min_edge_dist = np.min(1.0 - norms)
    
    r = min(min_pair_dist / 2.0, min_edge_dist)
    return max(r, 0.0)


def _soft_min_objective(centers, beta=200.0):
    """
    Smooth approximation of the negative score using log-sum-exp.
    We want to maximize r = min(min_pair/2, min_edge).
    So we minimize -r. Using soft min: -soft_min(x) = -(-logsumexp(-x/beta)/beta)
    Actually, let's define it more carefully.
    
    We want to maximize min(all_constraints). 
    Let constraints be:
      - d_ij / 2 for all i<j  (pairwise)
      - (1 - |c_i|) for all i  (boundary)
    
    We want max(min(constraints)).
    
    Using log-sum-exp soft min:
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_k)))
    
    As beta -> inf, soft_min -> min.
    
    So objective = -soft_min(constraints)
    """
    centers = centers.reshape(-1, 2)
    
    # Boundary constraints: 1 - |c_i|
    norms = np.sqrt(np.sum(centers**2, axis=1))
    edge_constraints = 1.0 - norms
    
    # Pairwise constraints: |c_i - c_j| / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Take upper triangle
    iu = np.triu_indices(N, 1)
    pair_constraints = dists[iu] / 2.0
    
    # Combine all constraints
    all_constraints = np.concatenate([edge_constraints, pair_constraints])
    
    # Soft min
    neg_beta_x = -beta * all_constraints
    # For numerical stability
    max_val = np.max(neg_beta_x)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
    soft_min_val = -log_sum_exp / beta
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    return -soft_min_val


def _generate_hexagonal_initial(r_init):
    """Generate a hexagonal lattice packing with given circle radius."""
    points = []
    spacing = 2 * r_init
    # Generate hexagonal lattice points
    for j in range(-15, 16):
        for i in range(-15, 16):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.sqrt(x*x + y*y)
            if r <= 1.0 - r_init - 1e-9:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < N:
        return None
    
    # Sort by distance from origin and take N
    norms = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(norms, kind='stable')[:N]
    return points[idx]


def _generate_ring_initial():
    """Generate a ring-based initial configuration."""
    centers = []
    # Center
    centers.append([0.0, 0.0])
    
    # Rings
    ring_radii = [0.25, 0.5, 0.75, 0.95]
    counts = [6, 12, 18, 13]  # Adjust to get 50 total
    
    for r, count in zip(ring_radii, counts):
        for k in range(count):
            theta = 2 * np.pi * k / count
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            centers.append([x, y])
    
    centers = np.array(centers[:N])
    return centers


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_centers = None
    
    # Initial configurations to try
    initial_configs = []
    
    # Hexagonal with different radii
    for r_init in [0.12, 0.13, 0.14, 0.15]:
        cfg = _generate_hexagonal_initial(r_init)
        if cfg is not None:
            initial_configs.append(cfg)
    
    # Ring-based
    ring_cfg = _generate_ring_initial()
    if ring_cfg is not None:
        initial_configs.append(ring_cfg)
    
    # Random valid configurations
    np.random.seed(42)
    for _ in range(5):
        cfg = np.random.uniform(-0.9, 0.9, size=(N, 2))
        # Project inside circle
        norms = np.sqrt(np.sum(cfg**2, axis=1))
        mask = norms > 0.9
        cfg[mask] *= 0.9 / norms[mask, np.newaxis]
        initial_configs.append(cfg)
    
    # Optimize each initial configuration
    for idx, x0 in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        # Try with different beta values
        for beta in [100.0, 200.0, 500.0]:
            if time.time() > deadline:
                break
            
            try:
                result = minimize(
                    _soft_min_objective,
                    x0.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.status == 0:
                    centers = result.x.reshape(N, 2)
                    score = _score_raw(centers)
                    if score > best_score:
                        best_score = score
                        best_centers = centers
            except Exception:
                continue
    
    # If we still have time, do a final local refinement on the best solution
    if best_centers is not None and time.time() < deadline - 10.0:
        # Try a few more optimizations from the best solution with perturbations
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            perturbed = best_centers + np.random.normal(0, 0.01, size=(N, 2))
            # Project back inside circle
            norms = np.sqrt(np.sum(perturbed**2, axis=1))
            mask = norms > 0.98
            perturbed[mask] *= 0.98 / norms[mask, np.newaxis]
            
            try:
                result = minimize(
                    _soft_min_objective,
                    perturbed.flatten(),
                    args=(300.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if result.success or result.status == 0:
                    centers = result.x.reshape(N, 2)
                    score = _score_raw(centers)
                    if score > best_score:
                        best_score = score
                        best_centers = centers
            except Exception:
                continue
    
    # Final validation
    if best_centers is not None:
        final_score = _score_raw(best_centers)
        if final_score > 0.0 and np.all(np.isfinite(best_centers)):
            return best_centers
    
    # Fallback: return hexagonal if nothing worked
    fallback = _generate_hexagonal_initial(0.12)
    if fallback is not None:
        return fallback
    
    # Last resort: simple grid
    return np.random.uniform(-0.5, 0.5, size=(N, 2))

# EVOLVE-BLOCK-END

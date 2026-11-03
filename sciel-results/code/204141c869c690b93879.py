"""Circle packing in unit square using Delaunay-based radius computation and force optimization."""
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _compute_radii(centers):
    """
    Given N centers in [0,1]^2, compute the maximum radius for each circle
    such that all circles have disjoint interiors and stay inside [0,1]^2.
    
    r_i = 0.5 * min(
        min_j ||c_i - c_j|| for j != i,
        c_i[0], 1 - c_i[0], c_i[1], 1 - c_i[1]
    )
    """
    n = len(centers)
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Distance to boundaries
    boundary_dists = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    
    # Min neighbor distance
    min_neighbor = np.min(dists, axis=1)
    
    # Radius is half the minimum of neighbor dist and boundary dist
    radii = 0.5 * np.minimum(min_neighbor, boundary_dists)
    
    return radii


def _objective(centers_flat):
    """Objective: negative sum of radii (to maximize sum of radii)."""
    centers = centers_flat.reshape(-1, 2)
    radii = _compute_radii(centers)
    return -np.sum(radii)


def _objective_and_grad(centers_flat):
    """Objective and gradient using finite differences (simple but effective)."""
    centers = centers_flat.reshape(-1, 2)
    radii = _compute_radii(centers)
    obj = -np.sum(radii)
    
    # Numerical gradient
    eps = 1e-6
    grad = np.zeros_like(centers_flat)
    for i in range(len(centers_flat)):
        orig = centers_flat[i]
        centers_flat[i] = orig + eps
        f_plus = -np.sum(_compute_radii(centers_flat.reshape(-1, 2)))
        centers_flat[i] = orig - eps
        f_minus = -np.sum(_compute_radii(centers_flat.reshape(-1, 2)))
        centers_flat[i] = orig
        grad[i] = (f_plus - f_minus) / (2 * eps)
    
    return obj, grad


def construct():
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    # Initial layout: grid with slight random perturbation
    np.random.seed(42)
    side = int(np.ceil(np.sqrt(N)))
    centers = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            x = (i + 0.5) / side + np.random.uniform(-0.05, 0.05)
            y = (j + 0.5) / side + np.random.uniform(-0.05, 0.05)
            # Clamp to [0.05, 0.95]
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            centers.append([x, y])
            count += 1
    
    centers = np.array(centers)
    centers_flat = centers.flatten().copy()
    
    best_obj = _objective(centers_flat)
    best_centers = centers_flat.copy()
    
    # Multi-start with different perturbations
    n_starts = 5
    for start_idx in range(n_starts):
        if time.time() - start_time > time_limit:
            break
        
        # Perturb the initial layout
        perturbed = centers_flat.copy()
        noise_scale = 0.1 if start_idx > 0 else 0.0
        if noise_scale > 0:
            perturbed += np.random.uniform(-noise_scale, noise_scale, size=perturbed.shape)
            perturbed = np.clip(perturbed, 0.01, 0.99)
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                _objective,
                perturbed,
                method='L-BFGS-B',
                bounds=[(0.01, 0.99)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_centers = result.x.copy()
        except Exception:
            continue
    
    # Additional local refinement with smaller steps
    if time.time() - start_time < time_limit - 10:
        try:
            result = minimize(
                _objective,
                best_centers,
                method='L-BFGS-B',
                bounds=[(0.001, 0.999)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_centers = result.x.copy()
        except Exception:
            pass
    
    # Final refinement
    if time.time() - start_time < time_limit - 5:
        try:
            result = minimize(
                _objective,
                best_centers,
                method='L-BFGS-B',
                bounds=[(0.0001, 0.9999)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_centers = result.x.copy()
        except Exception:
            pass
    
    centers = best_centers.reshape(-1, 2)
    radii = _compute_radii(centers)
    
    # Verify validity and clamp
    result = np.column_stack([centers, radii])
    
    # Ensure circles are strictly inside and non-overlapping
    for i in range(N):
        x, y, r = result[i]
        # Clamp radius to ensure inside boundary
        r = min(r, x, 1 - x, y, 1 - y)
        r = max(r, 1e-6)
        result[i, 2] = r
    
    # Verify no overlaps (adjust if needed)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(result[i, :2] - result[j, :2])
            min_r = d / 2 - 1e-6
            if result[i, 2] + result[j, 2] > d:
                # Scale down radii
                scale = d / (result[i, 2] + result[j, 2] + 1e-8)
                result[i, 2] *= scale
                result[j, 2] *= scale
    
    return result

# EVOLVE-BLOCK-END

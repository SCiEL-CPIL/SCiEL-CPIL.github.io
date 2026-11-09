"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import random

DEADLINE_SECONDS = 115.0  # safety margin before 120s kill


def _score(centers):
    """Compute the packing score: min of (min pairwise dist / 2, min distance to boundary)."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check inside cube [-0.5, 0.5]^3
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    # Pairwise distances
    d = pdist(centers, metric='euclidean')
    min_pair = d.min() if d.size > 0 else 1.0
    # Distance to boundary
    dist_to_boundary = 0.5 - np.abs(centers).max(axis=1)
    min_boundary = dist_to_boundary.min()
    return min(min_pair / 2.0, min_boundary)


def _potential(centers, r_target):
    """Repulsive potential: sum of 1/d^2 for pairs closer than 2*r_target, plus boundary penalty."""
    # Pairwise
    d = pdist(centers, metric='euclidean')
    mask = d < 2.0 * r_target
    if np.any(mask):
        pair_energy = np.sum(1.0 / (d[mask] ** 2))
    else:
        pair_energy = 0.0
    # Boundary
    dist_to_boundary = 0.5 - np.abs(centers)
    boundary_violation = np.maximum(0.0, -dist_to_boundary)
    boundary_energy = 10.0 * np.sum(boundary_violation ** 2)
    return pair_energy + boundary_energy


def _objective(centers_flat, r_target):
    centers = centers_flat.reshape(N, 3)
    return _potential(centers, r_target)


def _make_grid(m, n_use):
    """Create a grid of m^3 points in [-0.5, 0.5]^3, take first n_use."""
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n_use].copy()


def _local_search(centers, r_target, max_iter=500, tol=1e-10):
    """Run a local optimization to improve the packing for a given target radius."""
    x0 = centers.flatten()
    
    def obj(x):
        return _objective(x, r_target)
    
    def grad(x):
        centers = x.reshape(N, 3)
        grad = np.zeros_like(x)
        # Pairwise gradient (vectorized)
        # Compute all pairwise differences and distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (N, N, 3)
        dist_mat = np.linalg.norm(diff, axis=2)  # (N, N)
        np.fill_diagonal(dist_mat, 1.0)
        mask = (dist_mat < 2.0 * r_target) & (dist_mat > 0)
        if np.any(mask):
            # force_ij = 2 / d_ij^3 * diff_ij
            # For each i, sum over j: grad_i += 2/d_ij^3 * (c_i - c_j)
            # grad_j -= 2/d_ij^3 * (c_i - c_j)
            force_mag = np.zeros_like(dist_mat)
            force_mag[mask] = 2.0 / (dist_mat[mask] ** 3)
            # grad[i] = sum_j force_mag[i,j] * diff[i,j]
            # diff[i,j] = c_i - c_j, so grad[i] = sum_j force_mag[i,j] * (c_i - c_j)
            # = c_i * sum_j force_mag[i,j] - sum_j force_mag[i,j] * c_j
            # This is equivalent to: grad[i] = sum_j force_mag[i,j] * diff[i,j]
            grad_flat = np.einsum('ij,ijk->ik', force_mag, diff)
            grad = grad_flat
        # Boundary gradient (vectorized)
        dist_to_boundary = 0.5 - np.abs(centers)
        boundary_violation = np.maximum(0.0, -dist_to_boundary)
        if np.any(boundary_violation > 0):
            sign = np.sign(centers)
            grad += 20.0 * boundary_violation * sign
        return grad
    
    result = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                       bounds=[(-0.5, 0.5)] * (N * 3),
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10})
    return result.x.reshape(N, 3)


def construct():
    start_time = time.time()
    
    # Start with a 5x5x5 grid (125 points, take 100)
    m = 5
    centers = _make_grid(m, N)
    best_score = _score(centers)
    best_centers = centers.copy()
    
    # Binary search on radius to find a good target, then local search
    # The known best is ~0.11138
    # Grid score: spacing = 1/5 = 0.2, so r = 0.1
    # Let's try to improve from there
    
    # Strategy: try several target radii, do local search, keep best
    target_radii = [0.105, 0.108, 0.110, 0.111, 0.1113, 0.11138, 0.1114, 0.1115, 0.112]
    
    # Also try different initial configurations
    initial_configs = [best_centers.copy()]
    
    # Add some perturbed versions
    rng = np.random.default_rng(42)
    for _ in range(5):
        perturbed = best_centers + rng.normal(0, 0.01, size=(N, 3))
        # Clip to cube
        perturbed = np.clip(perturbed, -0.49, 0.49)
        initial_configs.append(perturbed)
    
    # Try each initial config with each target radius
    for r_target in target_radii:
        if time.time() - start_time > DEADLINE_SECONDS:
            break
        for init in initial_configs:
            if time.time() - start_time > DEADLINE_SECONDS:
                break
            try:
                improved = _local_search(init.copy(), r_target, max_iter=300)
                s = _score(improved)
                if s > best_score:
                    best_score = s
                    best_centers = improved.copy()
            except Exception:
                pass
    
    # Final refinement: use the best found score as target and do one more local search
    if time.time() - start_time < DEADLINE_SECONDS - 10:
        try:
            r_final = best_score * 0.999  # slightly below to be safe
            improved = _local_search(best_centers.copy(), r_final, max_iter=500)
            s = _score(improved)
            if s > best_score:
                best_score = s
                best_centers = improved.copy()
        except Exception:
            pass
    
    # One more pass with a tighter target
    if time.time() - start_time < DEADLINE_SECONDS - 5:
        try:
            r_tight = best_score * 1.001
            improved = _local_search(best_centers.copy(), r_tight, max_iter=300)
            s = _score(improved)
            if s > best_score:
                best_score = s
                best_centers = improved.copy()
        except Exception:
            pass
    
    # High-precision polish: very tight tolerance, more iterations
    if time.time() - start_time < DEADLINE_SECONDS - 2:
        try:
            r_polish = best_score * 1.0001
            improved = _local_search(best_centers.copy(), r_polish, max_iter=1000, tol=1e-14)
            s = _score(improved)
            if s > best_score:
                best_score = s
                best_centers = improved.copy()
        except Exception:
            pass
    
    # Ensure validity
    best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_centers
# EVOLVE-BLOCK-END

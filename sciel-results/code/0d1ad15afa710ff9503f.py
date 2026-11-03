"""Optimized packing of 50 equal spheres in a unit cube."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 115.0  # safety margin before 120s limit


def compute_score(centers):
    """Compute the packing radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check all centers are within the cube [-0.5, 0.5]^3
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Compute minimum distance to boundary
    boundary_dist = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(boundary_dist)
    
    # Compute minimum pairwise distance
    # Vectorized computation
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to infinity
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists)
    
    r = min(min_pairwise / 2.0, min_boundary)
    return r


def neg_score(centers_flat):
    """Negative score for minimization."""
    centers = centers_flat.reshape(N, 3)
    # Check validity
    if np.any(centers < -0.5 + 1e-12) or np.any(centers > 0.5 - 1e-12):
        return 1e10
    return -compute_score(centers)


def neg_score_grad(centers_flat):
    """Negative score and gradient for minimization."""
    centers = centers_flat.reshape(N, 3)
    
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-30)
    np.fill_diagonal(dists, np.inf)
    
    # Minimum pairwise distance
    min_pairwise = np.min(dists)
    
    # Boundary distances
    boundary_dist = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dist)
    
    r = min(min_pairwise / 2.0, min_boundary)
    
    # Gradient computation
    grad = np.zeros_like(centers)
    
    # Gradient from pairwise distances
    if min_pairwise / 2.0 <= min_boundary:
        # The limiting factor is pairwise distance
        # Find the pair with minimum distance
        min_idx = np.argmin(dists)
        i, j = np.unravel_index(min_idx, (N, N))
        if i != j:
            d = centers[i] - centers[j]
            norm = np.linalg.norm(d)
            if norm > 1e-12:
                unit = d / norm
                grad[i] += -0.5 * unit
                grad[j] += 0.5 * unit
    
    # Gradient from boundary
    if min_boundary <= min_pairwise / 2.0:
        # The limiting factor is boundary distance
        # For each center, the gradient pushes it away from the closest boundary
        for k in range(3):
            # Find centers closest to boundary in each dimension
            for dim in range(3):
                dist_to_pos = 0.5 - centers[:, dim]
                dist_to_neg = centers[:, dim] + 0.5
                dist_to_boundary = np.minimum(dist_to_pos, dist_to_neg)
                min_idx_b = np.argmin(dist_to_boundary)
                if dist_to_boundary[min_idx_b] == min_boundary:
                    center_idx = min_idx_b
                    if dist_to_pos[min_idx_b] < dist_to_neg[min_idx_b]:
                        grad[center_idx, dim] += -1.0
                    else:
                        grad[center_idx, dim] += 1.0
    
    return -r, grad.reshape(-1)


def bcc_lattice_init(n_spheres):
    """Generate initial positions using a BCC-like lattice."""
    # Use a 4x4x4 grid with BCC structure
    # Each unit cell has 2 points: corner and body center
    m = 4
    points = []
    
    for i in range(m):
        for j in range(m):
            for k in range(m):
                # Corner
                x = (i + 0.5) / m - 0.5
                y = (j + 0.5) / m - 0.5
                z = (k + 0.5) / m - 0.5
                points.append([x, y, z])
                # Body center
                x_b = (i + 1.0) / m - 0.5
                y_b = (j + 1.0) / m - 0.5
                z_b = (k + 1.0) / m - 0.5
                points.append([x_b, y_b, z_b])
    
    points = np.array(points[:n_spheres])
    return points


def random_init(n_spheres, seed=0):
    """Generate random initial positions."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(-0.4, 0.4, size=(n_spheres, 3))
    return points


def spectral_init(n_spheres, seed=0):
    """Generate initial positions using eigenvectors of a random matrix."""
    rng = np.random.RandomState(seed)
    # Create a random positive definite matrix
    A = rng.randn(n_spheres, n_spheres)
    A = A @ A.T + n_spheres * np.eye(n_spheres)
    
    # Get eigenvectors
    eigvals, eigvecs = np.linalg.eigh(A)
    
    # Use top N eigenvectors (largest eigenvalues)
    idx = np.argsort(eigvals)[::-1][:n_spheres]
    eigvecs_top = eigvecs[:, idx]
    
    # Normalize and scale to fit in cube
    centers = eigvecs_top.T * 0.4
    # Center and scale
    centers = (centers - centers.mean(axis=0)) / (np.max(np.abs(centers)) + 1e-12) * 0.4
    return centers


def optimize_from_init(init_centers, start_time, deadline, max_iter=500):
    """Optimize from given initial centers."""
    x0 = init_centers.reshape(-1)
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        result = minimize(
            neg_score,
            x0,
            jac=neg_score_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
        )
        
        if time.time() - start_time > deadline:
            return init_centers
        
        optimized = result.x.reshape(N, 3)
        
        # Project back to valid region
        optimized = np.clip(optimized, -0.5 + 1e-10, 0.5 - 1e-10)
        
        return optimized
    
    except Exception:
        return init_centers


def construct():
    """Construct optimal packing of 50 spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_centers = None
    best_score = 0.0
    
    # Try multiple initialization strategies
    inits = []
    
    # BCC lattice initialization
    inits.append(bcc_lattice_init(N))
    
    # Random initializations with different seeds
    for seed in [1, 2, 3, 4, 5]:
        inits.append(random_init(N, seed=seed))
    
    # Spectral initialization
    inits.append(spectral_init(N, seed=42))
    
    for i, init_centers in enumerate(inits):
        if time.time() - start_time > deadline - 10:
            break
        
        # Optimize
        optimized = optimize_from_init(init_centers, start_time, deadline)
        
        # Compute score
        score = compute_score(optimized)
        
        if score > best_score:
            best_score = score
            best_centers = optimized.copy()
    
    # If no valid result, use a fallback
    if best_centers is None:
        best_centers = bcc_lattice_init(N)
        best_score = compute_score(best_centers)
    
    # Final local optimization with more iterations if time permits
    if time.time() - start_time < deadline - 15:
        x0 = best_centers.reshape(-1)
        bounds = [(-0.5, 0.5)] * (N * 3)
        try:
            result = minimize(
                neg_score,
                x0,
                jac=neg_score_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 100}
            )
            final_centers = result.x.reshape(N, 3)
            final_centers = np.clip(final_centers, -0.5 + 1e-10, 0.5 - 1e-10)
            final_score = compute_score(final_centers)
            
            if final_score > best_score:
                best_centers = final_centers
                best_score = final_score
        except Exception:
            pass
    
    # Ensure valid output
    best_centers = np.clip(best_centers, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_centers

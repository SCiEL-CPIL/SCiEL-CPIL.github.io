"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START

def _compute_score(centers):
    """
    Compute the score (largest common radius) for a set of centers in [-0.5, 0.5]^2.
    Score = min(
        min_{i<j} ||c_i - c_j|| / 2,
        min_i min_k (0.5 - |c_ik|)
    )
    Returns 0.0 if any center is outside or if there are invalid points.
    """
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check bounds strictly
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    
    # Clip to bounds to be safe for calculation, though they should be inside
    c = np.clip(centers, -0.5, 0.5)
    
    # Distance to walls
    wall_dists = np.minimum(0.5 - np.abs(c), 0.5)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # For N=200, 200x200 matrix is 40,000 elements. Very fast.
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)

def _energy_objective(x, p=10.0):
    """
    Smooth objective: sum of inverse powers of all constraint distances.
    Minimizing this drives the minimum distance up, since close pairs/walls
    contribute huge energy.
    """
    c = x.reshape(N, 2)
    # Wall constraints: 0.5 - |c_ik|
    walls = 0.5 - np.abs(c)
    # Clip to avoid zero/negative
    walls = np.maximum(walls, 1e-12)
    
    # Pair constraints: dist_ij / 2 for i < j
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Only upper triangle i < j
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0
    pair_dists = np.maximum(pair_dists, 1e-12)
    
    # Energy = sum(1/d^p) for all constraints
    energy = np.sum(1.0 / (walls ** p)) + np.sum(1.0 / (pair_dists ** p))
    return energy

def _energy_gradient(x, p=10.0):
    """
    Analytical gradient of the energy objective.
    """
    c = x.reshape(N, 2)
    grad = np.zeros_like(c)
    
    # Wall gradient: d/dx_k [ (0.5 - |x_k|)^(-p) ]
    # = -p * (0.5 - |x_k|)^(-p-1) * d/dx_k(0.5 - |x_k|)
    # d/dx_k(0.5 - |x_k|) = -sign(x_k)
    # So gradient = p * sign(x_k) * (0.5 - |x_k|)^(-p-1)
    walls = 0.5 - np.abs(c)
    walls = np.maximum(walls, 1e-12)
    signs = np.sign(c)
    # Where c is exactly 0, sign is 0, but we can use a small default
    signs = np.where(signs == 0, 1.0, signs)
    wall_grad = p * signs * (walls ** (-p - 1))
    grad += wall_grad
    
    # Pair gradient: d/dx_i [ (d_ij/2)^(-p) ] = -p * (d_ij/2)^(-p-1) * d/dx_i(d_ij/2)
    # d_ij = ||c_i - c_j||, so d/dx_i(d_ij) = (c_i - c_j) / d_ij
    # d/dx_i(d_ij/2) = (c_i - c_j) / (2 * d_ij)
    # So gradient contribution for pair (i,j) to c_i:
    #   -p * (d_ij/2)^(-p-1) * (c_i - c_j) / (2 * d_ij)
    # Simplify: -p * (d_ij/2)^(-p-1) * (c_i - c_j) / (2 * d_ij)
    # Note: (d_ij/2)^(-p-1) = 2^(p+1) * d_ij^(-p-1)
    # So: -p * 2^(p+1) * d_ij^(-p-1) * (c_i - c_j) / (2 * d_ij)
    #    = -p * 2^p * d_ij^(-p-2) * (c_i - c_j)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Avoid division by zero
    dists = np.maximum(dists, 1e-12)
    
    # For each pair (i,j) with i<j, the force on i is proportional to (c_i - c_j) / d_ij^(p+2)
    # and on j is proportional to (c_j - c_i) / d_ij^(p+2)
    # Coefficient: -p * 2^p
    coeff = -p * (2.0 ** p)
    
    # Build gradient contributions
    # grad[i] += coeff * (c_i - c_j) / dists[i,j]^(p+2) for all j != i
    # grad[j] += coeff * (c_j - c_i) / dists[i,j]^(p+2) for all i != j
    
    # Vectorized:
    # weight[i,j] = coeff / dists[i,j]^(p+2)
    weights = coeff / (dists ** (p + 2))
    np.fill_diagonal(weights, 0.0)
    
    # grad[i] = sum_j weights[i,j] * (c_i - c_j)
    # = sum_j weights[i,j] * c_i - sum_j weights[i,j] * c_j
    # = c_i * sum_j weights[i,j] - sum_j (weights[i,j] * c_j)
    
    sum_w = np.sum(weights, axis=1)  # shape (N,)
    # sum_j weights[i,j] * c_j
    weighted_c = weights @ c  # shape (N, 2)
    
    grad += c * sum_w[:, np.newaxis] - weighted_c
    
    return grad.flatten()

def _generate_hex_init():
    """
    Generate an initial hexagonal packing.
    """
    # Target N=200
    # Hex grid spacing s
    # Area per point in hex grid is s^2 * sqrt(3)/2
    # Total Area = 1.0
    # N * s^2 * sqrt(3)/2 = 1.0  => s = sqrt(2 / (N * sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    
    # Generate points in a grid larger than the square
    # Square is [-0.5, 0.5]
    # We need to cover this area.
    
    points = []
    # y range
    dy = s * np.sqrt(3.0) / 2.0
    # Start from -0.6 to 0.6 to ensure coverage
    y = -0.6
    row = 0
    while y <= 0.6:
        # x offset for alternating rows
        offset = 0.0 if row % 2 == 0 else s / 2.0
        x = -0.6 + offset
        while x <= 0.6:
            points.append((x, y))
            x += s
        y += dy
        row += 1
    
    points = np.array(points)
    
    # Filter points inside the square [-0.5, 0.5]
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    # If we have more than N points, remove the ones with the worst local packing
    if len(points) > N:
        # Calculate min distance to neighbor for each point
        # This is O(M^2) where M is number of points. M is approx 200-300.
        # 300^2 = 90,000. Fast.
        min_dists = np.full(len(points), np.inf)
        for i in range(len(points)):
            dists = np.linalg.norm(points[i] - points, axis=1)
            dists[i] = np.inf
            min_dists[i] = np.min(dists)
        
        # Also consider wall distance?
        # Let's just use min neighbor distance for now.
        # Sort indices by min_dist, remove the smallest ones
        idx = np.argsort(min_dists)
        # Keep the largest N
        keep_idx = idx[N:]
        points = points[keep_idx]
        
    # If we have fewer than N points (unlikely with this s), we might need to add
    # But for N=200, s is small enough that we usually have >200 points in the square.
    # If len(points) < N, we just pad with random points? Or just return what we have?
    # The task requires exactly N.
    if len(points) < N:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:N]
        
    # Jitter slightly to break symmetry
    points += np.random.normal(0, 0.001, points.shape)
    points = np.clip(points, -0.5, 0.5)
    
    return points

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # 1. Initialization
    centers = _generate_hex_init()
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    # 2. Optimize using inverse-power-sum energy with L-BFGS-B
    # Start with moderate p, then increase for finer resolution
    ps = [4, 8, 16, 32, 64]
    
    x0 = best_centers.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    for p in ps:
        if time.time() - start_time > deadline - 10:
            break
        
        def obj(x):
            return _energy_objective(x, p)
        
        def grad(x):
            return _energy_gradient(x, p)
        
        try:
            result = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-6, 'iprint': -1}
            )
            
            new_centers = result.x.reshape(N, 2)
            new_score = _compute_score(new_centers)
            
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
                x0 = new_centers.flatten()
                
        except Exception:
            pass
    
    # 3. Final polish with a high p for very sharp optimization
    if time.time() - start_time < deadline - 5:
        p_final = 128
        x0 = best_centers.flatten()
        
        def obj_final(x):
            return _energy_objective(x, p_final)
        
        def grad_final(x):
            return _energy_gradient(x, p_final)
        
        try:
            result = minimize(
                obj_final,
                x0,
                jac=grad_final,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-8, 'iprint': -1}
            )
            
            new_centers = result.x.reshape(N, 2)
            new_score = _compute_score(new_centers)
            
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
                
        except Exception:
            pass
    
    return best_centers

# EVOLVE-BLOCK-END

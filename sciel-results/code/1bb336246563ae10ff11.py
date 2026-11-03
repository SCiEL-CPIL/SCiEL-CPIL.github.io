"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_DEADLINE = 150.0  # seconds, must finish before evaluator kills


def _score(centers):
    """Compute the score: min pairwise distance/2 and min wall distance."""
    n = len(centers)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    # Wall distances
    wall = np.min(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(wall)
    return min(min_pair, min_wall)


def _bcc_initial(n_points):
    """Generate initial points on a BCC lattice scaled to fit the unit cube."""
    # BCC lattice: simple cubic + body centers
    # We need about 50 points. Let's use a 3x3x3 cubic grid (27 points) 
    # plus body centers of some cubes, or a different arrangement.
    
    # Better approach: generate a dense BCC lattice and pick the best n_points.
    # BCC unit cell has 2 atoms. For a 3x3x3 array of unit cells, we have 27*2=54 atoms.
    # Scale to fit in [-0.5, 0.5]^3.
    
    # Let's use a 3x3x3 grid of BCC unit cells.
    # Each unit cell has corners and one body center.
    # Total points: 3*3*3 corners + 3*3*3 body centers = 54 points.
    
    # Generate corner points (simple cubic grid)
    m = 3
    ticks = (np.arange(m) + 0.5) / m - 0.5  # centers of sub-cubes
    # Actually, for BCC, let's think of it differently.
    
    # BCC lattice points in a cube of side L:
    # Simple cubic lattice points at (i, j, k) for i,j,k in range(m)
    # Plus body center points at (i+0.5, j+0.5, k+0.5) for i,j,k in range(m-1)
    # But this gives m^3 + (m-1)^3 points.
    
    # For m=3: 27 + 8 = 35 points. Not enough.
    # For m=4: 64 + 27 = 91 points. Too many, but we can pick 50.
    
    # Let's use m=3 for the simple cubic part and add body centers.
    # Actually, let's just generate a BCC-like pattern.
    
    # Alternative: use a 3x3x3 grid of BCC unit cells.
    # Each cell contributes 2 points. 27 cells * 2 = 54 points.
    # Scale the whole thing to fit in [-0.5, 0.5]^3.
    
    # BCC unit cell: corners at (0,0,0) to (1,1,1), body center at (0.5,0.5,0.5).
    # For a 3x3x3 array of cells:
    # Corner points: (i, j, k) for i,j,k in 0,1,2 -> 27 points
    # Body center points: (i+0.5, j+0.5, k+0.5) for i,j,k in 0,1,2 -> 27 points
    # Total: 54 points.
    
    # These span [0, 2] in each dimension. Scale to [-0.5, 0.5].
    # Scale factor: 1.0 / 2.0 = 0.5. Offset: -0.5.
    
    i = np.arange(3)
    corners = np.array([(x, y, z) for z in i for y in i for x in i], dtype=float)
    body_centers = corners + 0.5
    
    all_points = np.vstack([corners, body_centers])
    
    # Scale to fit in [-0.5, 0.5]^3
    # Current range: [0, 2] in each dim. Target: [-0.5, 0.5], range 1.0.
    all_points = all_points * 0.5 - 0.5
    
    # We have 54 points, need 50. Pick the 50 that give the best initial score.
    # For simplicity, just take the first 50.
    if len(all_points) >= n_points:
        return all_points[:n_points].copy()
    else:
        # Pad if needed
        pad = np.zeros((n_points - len(all_points), 3))
        return np.vstack([all_points, pad])


def _soft_min_objective(centers, beta):
    """Smooth surrogate for the negative of the min distance.
    
    We want to maximize r = min(min_pair_dist/2, min_wall_dist).
    Equivalently, minimize -r.
    
    Using log-sum-exp: -min(d_i) = log-sum-exp(-d_i * beta) / beta for large beta.
    """
    n = len(centers)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_dists = dist / 2.0  # radius contribution from pairs
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    wall_dists = np.min(wall_dists, axis=1)  # min over 3 coords for each point
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists.ravel(), wall_dists.ravel()])
    all_dists = all_dists[all_dists > 1e-12]  # avoid log(0)
    
    # Soft min: -min(d) ≈ (1/beta) * log(sum(exp(-beta * d)))
    # For numerical stability:
    max_d = np.max(all_dists)
    exp_term = np.exp(-beta * (all_dists - max_d))
    soft_min_neg = (np.log(np.sum(exp_term)) - beta * max_d) / beta
    
    return soft_min_neg


def _soft_min_gradient(centers, beta):
    """Gradient of the soft min objective."""
    n = len(centers)
    dim = 3
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_dists = dist / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    wall_min_idx = np.argmin(np.abs(centers), axis=1)
    wall_dists_min = np.min(wall_dists, axis=1)
    
    # Combine
    all_dists = np.concatenate([pair_dists.ravel(), wall_dists_min.ravel()])
    all_dists = all_dists[all_dists > 1e-12]
    
    max_d = np.max(all_dists)
    exp_term = np.exp(-beta * (all_dists - max_d))
    weights = exp_term / np.sum(exp_term)
    
    grad = np.zeros((n, dim))
    
    # Pairwise gradient
    pair_weights = weights[:n*(n-1)//2] if n*(n-1)//2 < len(weights) else weights[:len(weights)-n]
    # This is getting complex. Let's use numerical gradient for simplicity and reliability.
    
    return None  # Use numerical gradient


def construct():
    start_time = time.time()
    
    # Initial configuration: BCC lattice
    centers = _bcc_initial(N)
    
    # Ensure valid
    centers = np.clip(centers, -0.499, 0.499)
    best_score = _score(centers)
    best_centers = centers.copy()
    
    # Multi-stage optimization with increasing beta
    betas = [5, 10, 20, 50, 100, 200]
    
    for beta in betas:
        if time.time() - start_time > _DEADLINE - 10:
            break
        
        def objective(x):
            c = x.reshape(-1, 3)
            return _soft_min_objective(c, beta)
        
        try:
            result = minimize(
                objective,
                centers.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8},
                bounds=[(-0.499, 0.499)] * (N * 3)
            )
            centers = result.x.reshape(-1, 3)
            centers = np.clip(centers, -0.499, 0.499)
            
            current_score = _score(centers)
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
                
        except Exception:
            break
    
    # Final polish: direct optimization of the raw min objective
    if time.time() - start_time < _DEADLINE - 5:
        def raw_objective(x):
            c = x.reshape(-1, 3)
            return -_score(c)
        
        try:
            result = minimize(
                raw_objective,
                best_centers.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10},
                bounds=[(-0.499, 0.499)] * (N * 3)
            )
            centers = result.x.reshape(-1, 3)
            centers = np.clip(centers, -0.499, 0.499)
            
            current_score = _score(centers)
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
                
        except Exception:
            pass
    
    # Ensure validity
    best_centers = np.clip(best_centers, -0.499, 0.499)
    
    return best_centers
# EVOLVE-BLOCK-END

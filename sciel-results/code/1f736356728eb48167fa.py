"""Packing 50 equal spheres in a unit cube via multi-start gradient optimization."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s limit


def _score(centers):
    """Compute the maximum radius for given centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall distances
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_wall)


def _init_cubic():
    """Initial cubic grid placement with slight jitter."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    centers += np.random.randn(N, 3) * 0.001
    return centers


def _init_fcc():
    """FCC lattice initialization."""
    # FCC unit cell has 4 points
    # Generate FCC points in a box and select 50
    n = 5
    points = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
                for bx, by, bz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    x = (i + bx) / n - 0.5
                    y = (j + by) / n - 0.5
                    z = (k + bz) / n - 0.5
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                        points.append([x, y, z])
    
    points = np.array(points)
    # Add jitter
    points += np.random.randn(len(points), 3) * 0.001
    
    # Select 50 points, preferring those more central
    # Sort by distance from center, take closest 50
    dists_to_center = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists_to_center)[:N]
    centers = points[idx].copy()
    return centers


def _init_bcc():
    """BCC lattice initialization."""
    n = 4
    points = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # BCC basis: (0,0,0), (0.5,0.5,0.5)
                for bx, by, bz in [(0,0,0), (0.5,0.5,0.5)]:
                    x = (i + bx) / n - 0.5
                    y = (j + by) / n - 0.5
                    z = (k + bz) / n - 0.5
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                        points.append([x, y, z])
    
    points = np.array(points)
    points += np.random.randn(len(points), 3) * 0.001
    
    dists_to_center = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists_to_center)[:N]
    centers = points[idx].copy()
    return centers


def _objective_and_grad(x, beta):
    """
    Objective and gradient for L-BFGS-B.
    Minimizes the negative soft-min of clearances.
    """
    centers = x.reshape(N, 3)
    
    # Wall distances for each point and each of 6 faces
    wall_dists = np.concatenate([
        0.5 - centers[:, 0],
        centers[:, 0] + 0.5,
        0.5 - centers[:, 1],
        centers[:, 1] + 0.5,
        0.5 - centers[:, 2],
        centers[:, 2] + 0.5
    ])
    
    # Pairwise distances
    iu = np.triu_indices(N, k=1)
    diff = centers[iu[0]] - centers[iu[1]]
    dists = np.linalg.norm(diff, axis=1)
    pair_dists = dists / 2.0
    
    # Combine all constraints
    g_all = np.concatenate([pair_dists, wall_dists])
    
    # Soft min
    neg_beta_g = -beta * g_all
    log_sum_exp = logsumexp(neg_beta_g)
    val = -log_sum_exp / beta
    
    # Gradient
    weights = np.exp(neg_beta_g - log_sum_exp)
    grad_x = np.zeros_like(x)
    
    # Pairwise part
    w_pairs = weights[:len(pair_dists)]
    
    # Vectorized gradient accumulation for pairs
    # We need to accumulate gradients for each pair
    # For pair (i,j) with weight w:
    # grad_x[i] += w * 0.5 * (c_i - c_j) / d_ij
    # grad_x[j] -= w * 0.5 * (c_i - c_j) / d_ij
    
    # Create a matrix of contributions
    # For each pair k with indices i_k, j_k:
    # contribution to i_k: w_k * 0.5 * diff_k / d_k
    # contribution to j_k: -w_k * 0.5 * diff_k / d_k
    
    # We can use np.add.at for accumulation
    # But it's easier to loop over pairs if N is small
    # N=50, pairs=1225, loop is fast
    
    for k in range(len(pair_dists)):
        i, j = iu[0][k], iu[1][k]
        w = w_pairs[k]
        if w < 1e-12:
            continue
        
        d_ij = dists[k]
        if d_ij > 1e-12:
            diff_ij = diff[k]
            grad_vec = 0.5 * diff_ij / d_ij
            
            grad_x[i*3:i*3+3] += w * grad_vec
            grad_x[j*3:j*3+3] -= w * grad_vec
    
    # Wall part
    w_start = len(pair_dists)
    w_walls = weights[w_start:]
    
    # x-coords
    grad_x[0::3] -= w_walls[0:N]
    grad_x[0::3] += w_walls[N:2*N]
    
    # y-coords
    grad_x[1::3] -= w_walls[2*N:3*N]
    grad_x[1::3] += w_walls[3*N:4*N]
    
    # z-coords
    grad_x[2::3] -= w_walls[4*N:5*N]
    grad_x[2::3] += w_walls[5*N:6*N]
    
    return -val, -grad_x


def _local_refinement(centers, start_time, time_budget):
    """Local optimization using scipy minimize on negative soft-min score."""
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Anneal beta from low to high
    beta_schedule = [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    
    current_centers = centers.copy()
    
    for beta in beta_schedule:
        if (time.time() - start_time) > time_budget:
            break
        
        x0 = current_centers.ravel()
        
        try:
            result = minimize(_objective_and_grad, x0, method='L-BFGS-B', 
                             args=(beta,),
                             options={'maxiter': 1000, 'maxls': 100, 'ftol': 1e-14, 'gtol': 1e-12})
            
            new_centers = result.x.reshape(N, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            new_score = _score(new_centers)
            
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
                current_centers = new_centers.copy()
            else:
                current_centers = best_centers.copy()
                
        except Exception:
            pass
            
        if (time.time() - start_time) > time_budget:
            break
            
    return best_centers, best_score


def construct():
    start_time = time.time()
    
    # Try multiple initializations
    initializations = [
        _init_cubic,
        _init_fcc,
        _init_bcc,
    ]
    
    best_centers = None
    best_score = 0.0
    
    # Allocate time per initialization
    n_inits = len(initializations)
    time_per_init = (DEADLINE - 10.0) / n_inits  # 10s safety buffer
    
    for init_func in initializations:
        if (time.time() - start_time) > DEADLINE - 5.0:
            break
        
        centers = init_func()
        score = _score(centers)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
        
        # Local refinement
        remaining = DEADLINE - 5.0 - (time.time() - start_time)
        if remaining > 2.0:
            refined_centers, refined_score = _local_refinement(
                centers, start_time, time.time() - start_time + remaining
            )
            if refined_score > best_score:
                best_score = refined_score
                best_centers = refined_centers.copy()
        
        # Check time
        if (time.time() - start_time) > DEADLINE - 5.0:
            break
    
    # Final safety check
    if best_centers is None:
        best_centers = _init_cubic()
    
    # Ensure all centers are within bounds
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

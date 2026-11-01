import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # Safety margin

def _score(pts):
    """Compute the packing radius for given centers."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    # Boundary constraint
    dist_origin = np.linalg.norm(pts, axis=1)
    if np.any(dist_origin >= 1.0):
        return 0.0
    boundary_r = 1.0 - dist_origin
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return float(min(min_pair, np.min(boundary_r)))

def _soft_min_obj_and_grad(pts, beta):
    """
    Compute the soft-min objective and gradient.
    Objective: Minimize -soft_min(distances)
    soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    We want to maximize min(dists), so we minimize -min(dists).
    Let g = -min(dists).
    Using soft min approximation:
    g_soft = 1/beta * log(sum(exp(-beta * d)))
    where d are the relevant distances (pairwise/2 and to boundary).
    
    Actually, standard formulation:
    We want to maximize r such that d_ij >= 2r and 1-|c_i| >= r.
    Equivalent to minimizing -r.
    Using soft min of the constraints:
    L = 1/beta * log( sum_{i<j} exp(-beta * (d_ij/2)) + sum_i exp(-beta * (1-|c_i|)) )
    
    Gradient w.r.t x_k:
    dL/dx_k = - sum_{j != k} w_j * (x_k - x_j) / d_kj - w_b * x_k / |x_k|
    where w_j are weights from the softmax.
    """
    pts = np.asarray(pts, dtype=np.float64)
    
    # 1. Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Pairwise constraints: d_ij / 2
    pair_dists = dists / 2.0
    
    # Boundary constraints: 1 - |c_i|
    dists_to_origin = np.linalg.norm(pts, axis=1)
    bound_dists = 1.0 - dists_to_origin
    
    # Combine all constraints into one array for log-sum-exp stability
    # We need to be careful with memory. N=50, pairs=1225, bounds=50. Total ~1275.
    # This is small enough to handle directly.
    
    # Flatten pairwise distances (upper triangle only to avoid double counting? 
    # The objective is symmetric, but gradients need care. 
    # Let's use the full matrix but mask diagonal.
    # Actually, for the log-sum-exp, we sum over all valid constraints.
    
    # Let's construct the vector of all constraint values 'v'
    # v contains d_ij/2 for i<j and 1-|c_i| for all i.
    
    # For efficiency and stability, we compute the log-sum-exp directly.
    # Let A be the matrix of pairwise distances / 2.
    # Let B be the vector of boundary distances.
    
    # To avoid overflow in exp(-beta * v), we use the max trick.
    # Let m = min(v). Then sum(exp(-beta * v)) = exp(-beta * m) * sum(exp(-beta * (v - m)))
    # log(sum(exp(-beta * v))) = -beta * m + log(sum(exp(-beta * (v - m))))
    
    # We need gradients.
    # Let S = sum_{k} exp(-beta * v_k)
    # L = (1/beta) * log(S)
    # dL/dv_k = (1/beta) * (1/S) * dS/dv_k = (1/beta) * (1/S) * (-beta * exp(-beta * v_k)) = - exp(-beta * v_k) / S
    # So dL/dv_k = - w_k, where w_k = exp(-beta * v_k) / S.
    
    # Now we need dL/dx_i.
    # v_k depends on x.
    
    # Let's compute weights for pairwise and boundary separately.
    
    # Pairwise part:
    # v_ij = ||x_i - x_j|| / 2
    # exp(-beta * v_ij)
    
    # Boundary part:
    # v_i = 1 - ||x_i||
    # exp(-beta * v_i)
    
    # Total Sum S = sum_{i<j} exp(-beta * ||x_i - x_j||/2) + sum_i exp(-beta * (1 - ||x_i||))
    
    # For numerical stability, let's find the global min of all v's.
    min_pair = np.min(pair_dists)
    min_bound = np.min(bound_dists)
    v_min = min(min_pair, min_bound)
    
    # Compute shifted values
    pair_shifted = pair_dists - v_min
    bound_shifted = bound_dists - v_min
    
    # Compute exponentials
    # To save memory/time, we can compute the sum of exps.
    # N=50, pairs ~ 1225.
    
    exp_pairs = np.exp(-beta * pair_shifted)
    exp_bounds = np.exp(-beta * bound_shifted)
    
    # Mask diagonal for pairs
    np.fill_diagonal(exp_pairs, 0.0)
    
    S_pairs = np.sum(exp_pairs)
    S_bounds = np.sum(exp_bounds)
    S_total = S_pairs + S_bounds
    
    # Objective value
    obj = (1.0 / beta) * (np.log(S_total) - beta * v_min)
    # Wait, log(sum(exp(-beta * v))) = -beta * v_min + log(sum(exp(-beta * (v - v_min))))
    # So L = 1/beta * (-beta * v_min + log(S_total)) = -v_min + (1/beta) * log(S_total)
    obj = -v_min + (1.0 / beta) * np.log(S_total)
    
    # Gradients
    # dL/dv_k = - w_k
    # w_k = exp(-beta * v_k) / S_total
    
    # We need dL/dx_i = sum_k (dL/dv_k) * (dv_k/dx_i)
    
    # For boundary term k=i:
    # v_i = 1 - ||x_i||
    # dv_i/dx_i = - x_i / ||x_i||
    # Contribution to grad from boundary: - w_i * (- x_i / ||x_i||) = w_i * x_i / ||x_i||
    
    # For pairwise term k=(i,j) where i<j:
    # v_ij = ||x_i - x_j|| / 2
    # dv_ij/dx_i = (x_i - x_j) / (2 * ||x_i - x_j||)
    # dv_ij/dx_j = (x_j - x_i) / (2 * ||x_i - x_j||)
    # Note: dL/dv_ij = - w_ij
    # Contribution to grad_i from pair (i,j): - w_ij * (x_i - x_j) / (2 * d_ij)
    # Contribution to grad_j from pair (i,j): - w_ij * (x_j - x_i) / (2 * d_ij)
    
    # Let's compute the gradient vector g.
    g = np.zeros_like(pts)
    
    # Boundary contributions
    norms = np.linalg.norm(pts, axis=1)
    # Avoid division by zero
    norms_safe = np.where(norms < 1e-10, 1.0, norms)
    w_bounds = exp_bounds / S_total
    # g_i += w_i * x_i / norms_i
    g += w_bounds[:, np.newaxis] * pts / norms_safe[:, np.newaxis]
    
    # Pairwise contributions
    # We have exp_pairs matrix.
    # w_pairs[i,j] = exp_pairs[i,j] / S_total
    w_pairs = exp_pairs / S_total
    
    # We need to sum over j for each i.
    # g_i += - sum_{j != i} w_pairs[i,j] * (x_i - x_j) / (2 * d_ij)
    # Note that w_pairs is symmetric? No, exp(-beta * d/2) is symmetric.
    # But we only summed i<j in the objective?
    # In my S_pairs calculation, I used the full matrix with diagonal 0.
    # So S_pairs = sum_{i != j} exp(...). This counts each pair twice (i,j) and (j,i).
    # The objective L = 1/beta log(sum_{i!=j} ... + sum_i ...).
    # This is effectively 2 * (sum_{i<j} ...).
    # The factor of 2 cancels out in the gradient direction?
    # Let's check.
    # If we use sum_{i!=j}, then w_{ij} = w_{ji}.
    # Term for i in pair (i,j): - w_{ij} * (x_i - x_j) / (2 d_ij)
    # Term for j in pair (j,i): - w_{ji} * (x_j - x_i) / (2 d_ij)
    # Sum for i: sum_{j!=i} - w_{ij} (x_i - x_j) / (2 d_ij)
    
    # Let's compute this efficiently.
    # (x_i - x_j) / d_ij is the unit vector u_ij.
    # u = diff / dists[:, :, np.newaxis]
    # Handle zeros in dists (diagonal).
    dists_safe = np.where(dists < 1e-10, 1.0, dists)
    unit_vecs = diff / dists_safe[:, :, np.newaxis]
    np.fill_diagonal(unit_vecs, 0.0) # Not strictly needed if w is 0, but safe
    
    # grad_pair_i = - sum_j w_pairs[i,j] * unit_vecs[i,j] / 2
    # w_pairs is symmetric.
    # grad_pair = -0.5 * (w_pairs @ unit_vecs) ?
    # Let's check dimensions.
    # w_pairs: (N, N)
    # unit_vecs: (N, N, 2)
    # We want for each i: sum_j w_pairs[i,j] * unit_vecs[i,j]
    # This is a tensor contraction.
    
    # grad_pair[i] = -0.5 * sum_j w_pairs[i,j] * unit_vecs[i,j]
    
    # We can use einsum or broadcasting.
    # w_pairs[:, :, np.newaxis] * unit_vecs -> (N, N, 2)
    # Sum over axis 1 (j) -> (N, 2)
    
    weighted_units = w_pairs[:, :, np.newaxis] * unit_vecs
    grad_pair_sum = np.sum(weighted_units, axis=1)
    
    g -= 0.5 * grad_pair_sum
    
    return obj, g

def _hex_lattice_init():
    """Generate initial positions from hexagonal lattice."""
    pts = []
    spacing = 0.2
    for j in range(-8, 9):
        for i in range(-8, 9):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y < 0.85:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) < N:
        # Fallback to random if lattice doesn't provide enough
        return _random_init(42)
    # Select N points closest to origin
    r = np.linalg.norm(pts, axis=1)
    idx = np.argsort(r)[:N]
    return pts[idx]

def _random_init(seed):
    """Random initialization inside unit disk."""
    rng = np.random.RandomState(seed)
    pts = []
    for _ in range(N):
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            if x * x + y * y < 0.95:
                pts.append([x, y])
                break
    return np.array(pts)

def _optimize_from_init(pts_init, betas):
    """Run L-BFGS-B optimization with a sequence of beta values."""
    pts = pts_init.copy()
    
    for beta in betas:
        if time.time() > _DEADLINE - 5.0:
            break
            
        # Define objective and gradient for scipy
        def obj_grad(x):
            p = x.reshape(N, 2)
            val, grad = _soft_min_obj_and_grad(p, beta)
            # Flatten gradient
            return val, grad.flatten()
            
        try:
            result = minimize(
                obj_grad, 
                pts.reshape(-1), 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            pts = result.x.reshape(N, 2)
        except Exception:
            pass
            
    return pts

def construct():
    """Main function: find optimal packing of 50 equal circles in unit disk."""
    global _DEADLINE
    _DEADLINE = time.time() + 115.0
    
    best_pts = None
    best_r = 0.0
    
    # Initialization
    pts_hex = _hex_lattice_init()
    pts_rand = _random_init(42)
    
    # Beta annealing schedule
    # Start smooth, end sharp
    betas = [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0]
    
    # Try hex lattice
    if time.time() < _DEADLINE - 20:
        pts_opt = _optimize_from_init(pts_hex, betas)
        r = _score(pts_opt)
        if r > best_r:
            best_r = r
            best_pts = pts_opt
    
    # Try random init
    if time.time() < _DEADLINE - 10:
        pts_opt2 = _optimize_from_init(pts_rand, betas)
        r2 = _score(pts_opt2)
        if r2 > best_r:
            best_r = r2
            best_pts = pts_opt2
            
    # Final polish with very high beta
    if best_pts is not None and time.time() < _DEADLINE - 2:
        # Use the best found so far
        final_betas = [100000.0, 500000.0]
        pts_final = _optimize_from_init(best_pts, final_betas)
        r_final = _score(pts_final)
        if r_final > best_r:
            best_r = r_final
            best_pts = pts_final
            
    if best_pts is None:
        best_pts = pts_hex
        
    return best_pts

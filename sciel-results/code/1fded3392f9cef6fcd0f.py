"""Sphere packing in a unit cube using SGLD + local polish."""
import numpy as np
import time

N = 100
DEADLINE = 150.0  # seconds
MARGIN = 5.0      # safety margin

def _clamp(x, lo, hi):
    return np.clip(x, lo, hi)

def _pairwise_dists(C):
    """Compute all pairwise distances efficiently."""
    # C has shape (N, 3)
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=2))
    return d

def _score(C):
    """Compute the packing score: min of pairwise distances/2 and wall distances."""
    C = np.asarray(C)
    if C.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(C)):
        return 0.0
    # Check inside cube
    if np.any(C < -0.5) or np.any(C > 0.5):
        return 0.0
    # Wall distances
    wall = 0.5 - np.abs(C)
    min_wall = np.min(wall)
    # Pairwise distances
    d = _pairwise_dists(C)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    return min(min_pair, min_wall)

def _energy_and_grad(C, beta=100.0):
    """
    Compute a smooth surrogate energy and its gradient.
    We use a soft-min approximation: E = -log(sum(exp(-beta * d_ij))) / beta for pairs,
    and similar for walls. Actually, let's use a simpler approach:
    
    We want to maximize r = min(min_pair, min_wall).
    Equivalently, minimize E = -r.
    
    For a smooth surrogate, we can use:
    E_soft = -(1/beta) * log( sum_{i<j} exp(-beta * d_ij) + sum_{i,k} exp(-beta * w_ik) )
    
    where d_ij are pairwise distances and w_ik are wall distances.
    
    Gradient: dE/dC_i = (1/beta) * [ sum_j exp(-beta*d_ij) * (C_i - C_j)/d_ij + sum_k exp(-beta*w_ik) * sign(C_ik) ] / sum(...)
    """
    C = np.asarray(C, dtype=np.float64)
    # Pairwise
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-15)
    np.fill_diagonal(d, np.inf)
    
    # Wall distances
    wall = 0.5 - np.abs(C)
    
    # Combine all constraints into one array for log-sum-exp
    # We'll treat each pair and each wall as a separate constraint
    # But this is O(N^2) which is fine for N=100
    
    # For efficiency, let's compute the soft-min separately for pairs and walls
    # and then combine.
    
    # Pair soft-min
    # Mask out diagonal
    mask = ~np.eye(N, dtype=bool)
    d_valid = d[mask]
    exp_d = np.exp(-beta * d_valid)
    log_sum_exp_pairs = np.log(np.sum(exp_d))
    soft_min_pairs = -log_sum_exp_pairs / beta
    
    # Wall soft-min
    wall_flat = wall.flatten()
    exp_w = np.exp(-beta * wall_flat)
    log_sum_exp_walls = np.log(np.sum(exp_w))
    soft_min_walls = -log_sum_exp_walls / beta
    
    # Combined soft-min (approximate)
    # E = -log(exp(beta*soft_min_pairs) + exp(beta*soft_min_walls)) / beta
    # = -log(exp(-log_sum_exp_pairs) + exp(-log_sum_exp_walls)) / beta
    # This is getting complex. Let's just use a weighted combination or just the min.
    
    # Actually, for SGLD, we can just use the gradient of the soft-min.
    # Let's compute gradient for pairs and walls separately.
    
    # Gradient for pairs
    # d(soft_min_pairs)/dC_i = (1/beta) * sum_j exp(-beta*d_ij) * (C_i - C_j)/d_ij / sum_j exp(-beta*d_ij)
    # We need to be careful with the diagonal.
    
    grad = np.zeros_like(C)
    
    # Pair contribution
    # For each pair (i,j), contribution to grad[i] is exp(-beta*d_ij) * (C_i - C_j)/d_ij
    # and to grad[j] is exp(-beta*d_ij) * (C_j - C_i)/d_ij
    # Normalize by sum of exps
    
    # Create a matrix of exp(-beta*d) with diagonal 0
    exp_d_mat = np.exp(-beta * d)
    np.fill_diagonal(exp_d_mat, 0)
    
    sum_exp = np.sum(exp_d_mat, axis=1)  # shape (N,)
    
    # Avoid division by zero
    sum_exp = np.where(sum_exp > 0, sum_exp, 1.0)
    
    # grad_pairs[i] = (1/beta) * sum_j exp_d_mat[i,j] * (C[i] - C[j]) / d[i,j] / sum_exp[i]
    # (C[i] - C[j]) / d[i,j] is the unit vector from j to i
    
    # Compute unit vectors
    unit = diff / (d[:, :, np.newaxis] + 1e-15)
    
    # Weighted sum
    weighted = exp_d_mat[:, :, np.newaxis] * unit  # shape (N, N, 3)
    grad_pairs = np.sum(weighted, axis=1) / (beta * sum_exp[:, np.newaxis])
    
    # Wall contribution
    # d(soft_min_walls)/dC_ik = (1/beta) * exp(-beta*w_ik) * sign(C_ik) / sum exp(-beta*w)
    # w_ik = 0.5 - |C_ik|
    # dw_ik/dC_ik = -sign(C_ik)
    # So d(soft_min_walls)/dC_ik = (1/beta) * exp(-beta*w_ik) * (-sign(C_ik)) / sum exp(-beta*w)
    # Wait, soft_min = -log(sum exp(-beta*w))/beta
    # d(soft_min)/dC_ik = -(1/beta) * (1/sum exp) * (-beta * exp(-beta*w)) * dw/dC
    # = (1/sum exp) * exp(-beta*w) * dw/dC
    # dw/dC_ik = -sign(C_ik)
    # So d(soft_min)/dC_ik = -exp(-beta*w_ik) * sign(C_ik) / sum exp(-beta*w)
    
    exp_w_mat = np.exp(-beta * wall)  # shape (N, 3)
    sum_exp_w = np.sum(exp_w_mat)
    sum_exp_w = max(sum_exp_w, 1e-15)
    
    sign_C = np.sign(C)
    grad_walls = -exp_w_mat * sign_C / sum_exp_w
    
    grad = grad_pairs + grad_walls
    
    energy = -(soft_min_pairs + soft_min_walls) / 2  # Just a representative value
    # Actually, let's just return the soft-min as the energy to minimize
    energy = min(soft_min_pairs, soft_min_walls)
    
    return energy, grad

def construct():
    start_time = time.time()
    
    # Initial guess: random points inside the cube
    np.random.seed(42)
    C = np.random.uniform(-0.45, 0.45, size=(N, 3))
    
    best_C = C.copy()
    best_score = _score(C)
    
    # SGLD parameters
    beta = 50.0  # inverse temperature for soft-min
    dt = 0.005   # step size
    noise_scale = 0.01  # noise amplitude
    n_iter = 5000
    
    elapsed = 0.0
    iter_count = 0
    
    for i in range(n_iter):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - MARGIN:
            break
        
        # Decay noise over time
        progress = i / n_iter
        current_noise = noise_scale * (1.0 - 0.8 * progress)
        
        # Compute energy and gradient
        E, grad = _energy_and_grad(C, beta)
        
        # SGLD update: C <- C - dt * grad + noise
        noise = np.random.normal(0, current_noise, size=C.shape)
        C_new = C - dt * grad + noise
        
        # Clamp to cube
        C_new = _clamp(C_new, -0.5, 0.5)
        
        C = C_new
        iter_count = i + 1
        
        # Periodically check score
        if i % 50 == 0:
            s = _score(C)
            if s > best_score:
                best_score = s
                best_C = C.copy()
    
    # Final polish: local optimization with smaller steps
    elapsed = time.time() - start_time
    remaining = DEADLINE - MARGIN - elapsed
    if remaining > 10:
        # Do a local polish with gradient descent (no noise)
        C = best_C.copy()
        beta_polish = 100.0
        dt_polish = 0.001
        n_polish = min(2000, int(remaining * 50))
        
        for i in range(n_polish):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - MARGIN:
                break
            
            E, grad = _energy_and_grad(C, beta_polish)
            C_new = C - dt_polish * grad
            C_new = _clamp(C_new, -0.5, 0.5)
            C = C_new
            
            if i % 20 == 0:
                s = _score(C)
                if s > best_score:
                    best_score = s
                    best_C = C.copy()
    
    return best_C

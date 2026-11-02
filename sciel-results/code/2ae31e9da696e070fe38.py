"""Packing 100 equal spheres in a unit cube: soft-min landscape + L-BFGS-B + polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def _pairwise_dists(C):
    """Compute all pairwise distances (N, N) from centre array (N, 3)."""
    # Using broadcasting: (N,1,3) - (1,N,3) -> (N,N,3)
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _radius(C):
    """Compute the packing radius for centres C (N,3) in cube [-0.5, 0.5]^3."""
    if C.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(C)):
        return 0.0
    # Check inside cube
    if np.any(C < -0.5 - 1e-12) or np.any(C > 0.5 + 1e-12):
        return 0.0
    # Pairwise distances
    D = _pairwise_dists(C)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    # Boundary distances
    bound = np.minimum.reduce([
        0.5 - C[:, 0],
        C[:, 0] + 0.5,
        0.5 - C[:, 1],
        C[:, 1] + 0.5,
        0.5 - C[:, 2],
        C[:, 2] + 0.5,
    ])
    min_bound = np.min(bound)
    return 0.5 * min(min_pair, min_bound)


def _soft_min_objective(flat, beta):
    """
    Negative soft-min of pairwise distances and boundary distances.
    We minimize -softmin, which is equivalent to maximizing softmin.
    softmin(x) = -(1/beta) * log(sum(exp(-beta * x_i)))
    """
    C = flat.reshape(N, 3)
    
    # Pairwise distances
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Exclude diagonal
    idx = np.triu_indices(N, k=1)
    pair_dists = dists[idx]
    
    # Boundary distances
    bounds = np.concatenate([
        0.5 - C[:, 0],
        C[:, 0] + 0.5,
        0.5 - C[:, 1],
        C[:, 1] + 0.5,
        0.5 - C[:, 2],
        C[:, 2] + 0.5,
    ])
    
    all_dists = np.concatenate([pair_dists, bounds])
    
    # Soft min: -(1/beta) * log(sum(exp(-beta * d)))
    # For numerical stability, subtract max
    neg_beta_d = -beta * all_dists
    max_val = np.max(neg_beta_d)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_d - max_val)))
    soft_min = -log_sum_exp / beta
    
    # We want to maximize soft_min, so minimize -soft_min
    return -soft_min


def _soft_min_grad(flat, beta):
    """Gradient of -soft_min_objective."""
    C = flat.reshape(N, 3)
    
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    idx = np.triu_indices(N, k=1)
    pair_dists = dists[idx]
    
    bounds = np.concatenate([
        0.5 - C[:, 0],
        C[:, 0] + 0.5,
        0.5 - C[:, 1],
        C[:, 1] + 0.5,
        0.5 - C[:, 2],
        C[:, 2] + 0.5,
    ])
    
    all_dists = np.concatenate([pair_dists, bounds])
    
    neg_beta_d = -beta * all_dists
    max_val = np.max(neg_beta_d)
    exps = np.exp(neg_beta_d - max_val)
    sum_exps = np.sum(exps)
    weights = exps / sum_exps  # These are the soft-min weights
    
    # d(soft_min)/d(all_dists_i) = weights_i
    # d(-soft_min)/d(all_dists_i) = -weights_i
    
    # Gradient w.r.t. C
    grad_C = np.zeros_like(C)
    
    # Pairwise contribution
    # For each pair (i,j), d/dC_i of dist_ij = (C_i - C_j) / dist_ij
    # d(-soft_min)/dC_i = -sum_j weights_j * (C_i - C_j)/dist_ij
    for k, (i, j) in enumerate(zip(idx[0], idx[1])):
        d = pair_dists[k]
        if d < 1e-12:
            continue
        w = weights[k]
        diff_ij = C[i] - C[j]
        grad_C[i] -= w * diff_ij / d
        grad_C[j] += w * diff_ij / d
    
    # Boundary contribution
    # For each boundary distance, gradient w.r.t. C
    # 0.5 - C[i,0]: d/dC[i,0] = -1
    # C[i,0] + 0.5: d/dC[i,0] = +1
    # etc.
    for i in range(N):
        # 0.5 - C[i,0], weight = weights[N*(N-1)/2 + 6*i + 0]
        base = N * (N - 1) // 2
        w0 = weights[base + 6 * i + 0]  # 0.5 - C[i,0]
        w1 = weights[base + 6 * i + 1]  # C[i,0] + 0.5
        w2 = weights[base + 6 * i + 2]  # 0.5 - C[i,1]
        w3 = weights[base + 6 * i + 3]  # C[i,1] + 0.5
        w4 = weights[base + 6 * i + 4]  # 0.5 - C[i,2]
        w5 = weights[base + 6 * i + 5]  # C[i,2] + 0.5
        
        grad_C[i, 0] -= (-w0 + w1)  # d(-soft_min)/dC[i,0] = -(-w0 + w1) ... wait
        # Let me re-derive: soft_min = -(1/beta)*log(sum exp(-beta*d_k))
        # d(soft_min)/d(d_k) = w_k where w_k = exp(-beta*d_k)/sum
        # d(-soft_min)/d(d_k) = -w_k
        # d(d_k)/dC for boundary:
        # d_k = 0.5 - C[i,0] => d(d_k)/dC[i,0] = -1
        # d_k = C[i,0] + 0.5 => d(d_k)/dC[i,0] = +1
        # So d(-soft_min)/dC[i,0] = -w0 * (-1) + -w1 * (1) = w0 - w1
        
        grad_C[i, 0] = w0 - w1
        grad_C[i, 1] = w2 - w3
        grad_C[i, 2] = w4 - w5
    
    return grad_C.ravel()


def _project_to_cube(C):
    """Project centres into the cube [-0.5, 0.5]^3."""
    C = np.clip(C, -0.5, 0.5)
    return C


def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # Safety margin before 150s limit
    
    best_C = None
    best_r = 0.0
    
    # Multiple random restarts
    np.random.seed(42)
    
    # Start with a reasonable initial configuration
    # Use a jittered grid as initial guess
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    # Add small random perturbation
    C_init = grid + np.random.randn(N, 3) * 0.01
    C_init = _project_to_cube(C_init)
    
    # Annealing schedule: start with low beta, increase
    betas = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    
    C_current = C_init.copy()
    
    for beta in betas:
        if time.time() > deadline - 20.0:
            break
        
        # Optimize using L-BFGS-B
        result = minimize(
            _soft_min_objective,
            C_current.ravel(),
            jac=_soft_min_grad,
            args=(beta,),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        C_new = _project_to_cube(result.x.reshape(N, 3))
        r_new = _radius(C_new)
        
        if r_new > best_r:
            best_r = r_new
            best_C = C_new.copy()
        
        # Accept if improved or with some probability (simulated annealing-like)
        if r_new >= best_r or np.random.random() < 0.1:
            C_current = C_new
    
    # High-precision polish on the best configuration
    if best_C is not None and time.time() < deadline - 10.0:
        C_polish = best_C.copy()
        for beta_polish in [20000, 50000, 100000, 200000]:
            if time.time() > deadline - 5.0:
                break
            try:
                result = minimize(
                    _soft_min_objective,
                    C_polish.ravel(),
                    jac=_soft_min_grad,
                    args=(beta_polish,),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                C_polish = _project_to_cube(result.x.reshape(N, 3))
                r_polish = _radius(C_polish)
                if r_polish > best_r:
                    best_r = r_polish
                    best_C = C_polish.copy()
            except Exception:
                break
    
    # Final validation
    if best_C is None:
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_C = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    # Ensure strict validity
    best_C = np.clip(best_C, -0.5 + 1e-15, 0.5 - 1e-15)
    
    return best_C
# EVOLVE-BLOCK-END

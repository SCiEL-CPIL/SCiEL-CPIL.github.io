# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from collections import defaultdict

def _entropy_from_pooled(pooled):
    """Compute Shannon entropy in bits from a dict of value -> weight."""
    total = sum(pooled.values())
    if total <= 1e-15:
        return 0.0
    probs = [w / total for w in pooled.values() if w > 1e-15]
    if not probs:
        return 0.0
    return -sum(p * np.log2(p) for p in probs)

def _compute_entropies(P):
    """Compute D and M entropies for a 10x10 distribution P[y,x]."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0, 0.0
    P = P / s
    
    n = 10
    y_idx, x_idx = np.indices((n, n))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    w = P.flatten()
    
    # D = H of (x - y)
    d_vals = x_flat - y_flat
    d_pooled = defaultdict(float)
    for v, wt in zip(d_vals, w):
        d_pooled[v] += wt
    D = _entropy_from_pooled(d_pooled)
    
    # M = max(H_0, H_1, H_2)
    # H_0: distribution of x
    x_pooled = defaultdict(float)
    for v, wt in zip(x_flat, w):
        x_pooled[v] += wt
    H0 = _entropy_from_pooled(x_pooled)
    
    # H_1: distribution of x + y
    xy_vals = x_flat + y_flat
    xy_pooled = defaultdict(float)
    for v, wt in zip(xy_vals, w):
        xy_pooled[v] += wt
    H1 = _entropy_from_pooled(xy_pooled)
    
    # H_2: distribution of x + 2*y
    x2y_vals = x_flat + 2 * y_flat
    x2y_pooled = defaultdict(float)
    for v, wt in zip(x2y_vals, w):
        x2y_pooled[v] += wt
    H2 = _entropy_from_pooled(x2y_pooled)
    
    M = max(H0, H1, H2)
    return D, M

def _ratio(P):
    """Compute D/M ratio."""
    D, M = _compute_entropies(P)
    if M <= 1e-9:
        return 0.0
    return D / M

def _soft_max_entropy(ents, temperature):
    """Soft max of entropies with temperature smoothing."""
    if temperature <= 0:
        return max(ents)
    # Use log-sum-exp style soft max
    scaled = np.array(ents) / temperature
    max_s = np.max(scaled)
    if max_s < -1000:
        return 0.0
    exp_vals = np.exp(scaled - max_s)
    sum_exp = np.sum(exp_vals)
    if sum_exp <= 0:
        return 0.0
    return temperature * (max_s + np.log(sum_exp))

def _objective(params, temperature):
    """Objective function for optimization: minimize -ratio with soft max."""
    n = 10
    # params are logits for softmax parameterization
    logits = params.reshape(n, n)
    # Softmax to get probabilities
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    P = exp_logits / np.sum(exp_logits)
    
    D, M_raw = _compute_entropies(P)
    
    if M_raw <= 1e-9:
        return 1e6  # Large penalty
    
    # Use soft max for the max term during optimization
    # But for the final ratio, we need actual max
    # For optimization, we use soft max to make it differentiable
    # Recompute the individual entropies for soft max
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 1e6
    P = P / s
    
    y_idx, x_idx = np.indices((n, n))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    w = P.flatten()
    
    # D = H of (x - y)
    d_vals = x_flat - y_flat
    d_pooled = defaultdict(float)
    for v, wt in zip(d_vals, w):
        d_pooled[v] += wt
    D = _entropy_from_pooled(d_pooled)
    
    # Individual entropies
    x_pooled = defaultdict(float)
    for v, wt in zip(x_flat, w):
        x_pooled[v] += wt
    H0 = _entropy_from_pooled(x_pooled)
    
    xy_vals = x_flat + y_flat
    xy_pooled = defaultdict(float)
    for v, wt in zip(xy_vals, w):
        xy_pooled[v] += wt
    H1 = _entropy_from_pooled(xy_pooled)
    
    x2y_vals = x_flat + 2 * y_flat
    x2y_pooled = defaultdict(float)
    for v, wt in zip(x2y_vals, w):
        x2y_pooled[v] += wt
    H2 = _entropy_from_pooled(x2y_pooled)
    
    # Soft max of H0, H1, H2
    M_soft = _soft_max_entropy([H0, H1, H2], temperature)
    
    if M_soft <= 1e-9:
        return 1e6
    
    return -D / M_soft

def _optimize_from_start(start_logits, temperature, deadline):
    """Run L-BFGS-B optimization from a starting point."""
    n = 10
    result = minimize(
        _objective,
        start_logits,
        args=(temperature,),
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    if time.time() > deadline:
        return None
    
    # Convert back to P
    logits = result.x.reshape(n, n)
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    P = exp_logits / np.sum(exp_logits)
    
    return P

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10 second safety margin
    
    n = 10
    best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_ratio = _ratio(best_P)
    
    # Try multiple temperatures for soft max annealing
    temperatures = [5.0, 2.0, 1.0, 0.5, 0.2, 0.1]
    
    # Multiple random restarts
    np.random.seed(42)
    for temp in temperatures:
        if time.time() > deadline:
            break
        
        # Try uniform start
        start_logits = np.zeros((n, n))
        P_opt = _optimize_from_start(start_logits, temp, deadline)
        if P_opt is not None:
            ratio = _ratio(P_opt)
            if ratio > best_ratio:
                best_ratio = ratio
                best_P = P_opt.copy()
        
        # Try a few random starts
        for _ in range(3):
            if time.time() > deadline:
                break
            start_logits = np.random.randn(n, n) * 0.5
            P_opt = _optimize_from_start(start_logits, temp, deadline)
            if P_opt is not None:
                ratio = _ratio(P_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_P = P_opt.copy()
    
    # Final validation and normalization
    best_P = np.clip(best_P, 0, None)
    total = best_P.sum()
    if total <= 1e-9:
        best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        best_P = best_P / total
    
    return best_P

# EVOLVE-BLOCK-END

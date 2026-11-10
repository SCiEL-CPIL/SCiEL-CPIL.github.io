# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_area(positions, n=64, M=20000):
    """Compute union area via quadrature at M heights."""
    y = (np.arange(M) + 0.5) / M  # (M,)
    w = (1.0 - y) / n             # (M,)
    a = (np.arange(n) + 1) / n    # (n,)
    # L[i, m] = x_i + a_i * y_m  -> (n, M)
    L = positions[:, None] + a[:, None] * y[None, :]
    # Sort along axis 0
    L_sorted = np.sort(L, axis=0)  # (n, M)
    # Consecutive gaps
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]  # (n-1, M)
    # min(gap, w): use np.minimum
    mins = np.minimum(gaps, w[None, :])  # (n-1, M)
    union_len = w + mins.sum(axis=0)     # (M,)
    area = union_len.mean()
    return area

def _smooth_area(positions, n=64, M=20000, beta=100.0):
    """Smooth surrogate for union area using log-sum-exp for soft-min."""
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n
    a = (np.arange(n) + 1) / n
    L = positions[:, None] + a[:, None] * y[None, :]
    L_sorted = np.sort(L, axis=0)
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]  # (n-1, M)
    # soft-min(gap, w) = -softmax(-gap, -w) = -logsumexp(-gap, -w) + log(2)
    # More precisely: soft_min(a,b) = -softmax(-a,-b) where softmax uses log-sum-exp
    # soft_min(g, w) ≈ -( -max(g,w) + log(exp(-(g-max)) + exp(-(w-max))) )
    # = max(g,w) - log(exp(-(g-max)) + exp(-(w-max)))
    # But we want min, so: soft_min = max - log(exp(-(max-g)) + exp(-(max-w)))
    # Let's use: soft_min(g, w) = max(g,w) - log(exp(g - max) + exp(w - max))... no
    # Actually: soft_min(a,b) = -soft_max(-a,-b) = -(max(-a,-b) + log(exp(-a-max(-a,-b)) + exp(-b-max(-a,-b))))
    # = max(a,b) - log(exp(-a + max(a,b)) + exp(-b + max(a,b)))
    # = max(a,b) - log(exp(-(a-max)) + exp(-(b-max)))
    # where max = max(a,b)
    mx = np.maximum(gaps, w[None, :])  # (n-1, M)
    # soft_min = mx - log(exp(-(gaps - mx)) + exp(-(w - mx)))
    # = mx - log(exp(mx - gaps) + exp(mx - w))... wait that's wrong
    # Let me redo: soft_min(a,b) = -soft_max(-a,-b)
    # soft_max(-a,-b) = max(-a,-b) + log(exp(-a - max(-a,-b)) + exp(-b - max(-a,-b)))
    # max(-a,-b) = -min(a,b)
    # soft_max(-a,-b) = -min(a,b) + log(exp(-a + min(a,b)) + exp(-b + min(a,b)))
    # soft_min(a,b) = min(a,b) - log(exp(-a + min(a,b)) + exp(-b + min(a,b)))
    # = min(a,b) - log(exp(-(a - min(a,b))) + exp(-(b - min(a,b))))
    # = min(a,b) - log(exp(-max(0, a-b)) + exp(-max(0, b-a)))
    # For numerical stability with large beta:
    # soft_min_beta(g, w) = - (1/beta) * log(exp(-beta*g) + exp(-beta*w))
    # = min(g,w) - (1/beta)*log(exp(-beta*(g-min)) + exp(-beta*(w-min)))
    # = min(g,w) - (1/beta)*log(exp(-beta*max(0,g-w)) + exp(-beta*max(0,w-g)))
    # When beta is large, this approaches min(g,w).
    # For computation:
    beta_f = beta
    # Use log-sum-exp trick
    # log(exp(-beta*g) + exp(-beta*w)) = -beta*min(g,w) + log(exp(-beta*(g-min)) + exp(-beta*(w-min)))
    # where min = min(g,w)
    mn = np.minimum(gaps, w[None, :])  # (n-1, M)
    # log(exp(-beta*(g-mn)) + exp(-beta*(w-mn)))
    # g-mn >= 0, w-mn >= 0
    # = log(exp(-beta*(g-mn)) + exp(-beta*(w-mn)))
    # For stability, factor out the larger term (which is exp(0) = 1 when one is 0)
    dg = gaps - mn  # >= 0
    dw = w[None, :] - mn  # >= 0
    # log(exp(-beta*dg) + exp(-beta*dw))
    # Since dg + dw = |g - w|, one of them is 0
    # Use: log(exp(-beta*dg) + exp(-beta*dw)) 
    # = -beta*max(dg, dw) + log(exp(-beta*(dg - max)) + exp(-beta*(dw - max)))
    # max(dg,dw) = max(dg,dw)
    mx_d = np.maximum(dg, dw)
    log_sum = -beta_f * mx_d + np.log(np.exp(-beta_f * (dg - mx_d)) + np.exp(-beta_f * (dw - mx_d)))
    soft_min_vals = mn - log_sum / beta_f
    union_len = w + soft_min_vals.sum(axis=0)
    area = union_len.mean()
    return area

def _keich_baseline(n=64):
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)

def kakeya_needle_2d_positions64() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0  # 10s safety margin
    
    n = 64
    # Start from Keich baseline
    x0 = _keich_baseline(n)
    
    # Compute baseline area
    base_area = _compute_area(x0, n)
    best_pos = x0.copy()
    best_area = base_area
    
    # Try smooth optimization with different betas
    for beta in [200.0, 500.0, 1000.0, 2000.0]:
        if time.time() > deadline - 60:
            break
        
        def obj(x):
            return _smooth_area(x, n, M=20000, beta=beta)
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success or res.fun < best_area:
                # Verify with exact area
                exact_a = _compute_area(res.x, n)
                if exact_a < best_area:
                    best_area = exact_a
                    best_pos = res.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 30:
            break
    
    # Local refinement with coordinate descent on exact objective
    if time.time() < deadline - 10:
        x = best_pos.copy()
        for it in range(200):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(n):
                # Try small perturbations
                for delta in [1e-4, -1e-4, 1e-5, -1e-5, 1e-6, -1e-6]:
                    x_trial = x.copy()
                    x_trial[i] += delta
                    a_trial = _compute_area(x_trial, n)
                    if a_trial < best_area:
                        best_area = a_trial
                        best_pos = x_trial.copy()
                        x = x_trial
                        improved = True
                        break
            if not improved:
                break
    
    return best_pos

# EVOLVE-BLOCK-END

import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50
DEADLINE = 145.0  # 150s limit, 5s margin

def compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * max|(h*h)|)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-9):
        return -1.0
    h = np.clip(h, 0.0, None)
    l1 = h.sum()
    if l1 < 1e-12:
        return -1.0
    l2_sq = float(np.dot(h, h))
    conv = np.convolve(h, h, mode='full')
    max_conv = float(np.max(conv))
    if max_conv < 1e-12:
        return -1.0
    return l2_sq / (l1 * max_conv)

def _neg_log_ratio(params):
    """Objective: minimize -log(ratio) = log(max_conv) - log(l2_sq) with l1=1 normalization."""
    h = np.exp(params)
    l1 = h.sum()
    if l1 < 1e-30:
        return 1e6
    h = h / l1
    l2_sq = float(np.dot(h, h))
    if l2_sq < 1e-30:
        return 1e6
    conv = np.convolve(h, h, mode='full')
    max_conv = float(np.max(conv))
    if max_conv < 1e-30:
        return 1e6
    return np.log(max_conv) - np.log(l2_sq)

def _soft_neg_log_ratio(params, beta=100.0):
    """Smooth surrogate using log-sum-exp for the max."""
    h = np.exp(params)
    l1 = h.sum()
    if l1 < 1e-30:
        return 1e6
    h = h / l1
    l2_sq = float(np.dot(h, h))
    if l2_sq < 1e-30:
        return 1e6
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    if m < 1e-30:
        return 1e6
    # log-sum-exp approximation of log(max)
    log_max = m + np.log(np.sum(np.exp((conv - m) * beta)) / beta)
    return log_max - np.log(l2_sq)

def _greedy_sidon(k_target):
    """Greedy construction of a near-Sidon set of size k_target."""
    used = set()
    chosen = []
    for i in range(N):
        if len(chosen) >= k_target:
            break
        # Check if adding i creates any sum collision
        ok = True
        for j in range(len(chosen)):
            s = chosen[j] + i
            if s in used:
                ok = False
                break
        if ok:
            chosen.append(i)
            for j in range(len(chosen)):
                used.add(chosen[j] + i)
    return chosen

def _weighted_sidon_initialization(k_target, weights=None):
    """Create initial vector with support on a Sidon set and given weights."""
    sidon = _greedy_sidon(k_target)
    if not sidon:
        return None
    h = np.zeros(N)
    if weights is None:
        weights = np.ones(len(sidon)) / np.sqrt(len(sidon))
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
    h[sidon] = weights
    return h

def construct():
    t0 = time.time()
    best_h = np.ones(N) / N
    best_val = compute_ratio(best_h)
    
    # --- Phase 0: Generate diverse initial candidates ---
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N) / N)
    
    # Various bump shapes
    x = np.linspace(-1, 1, N)
    for sigma in [0.2, 0.3, 0.5, 0.7, 1.0]:
        h_g = np.exp(-x**2 / (2 * sigma**2))
        h_g /= h_g.sum()
        candidates.append(h_g)
    
    # Cosine bump
    h_cos = 0.5 * (1 + np.cos(np.pi * x))
    h_cos /= h_cos.sum()
    candidates.append(h_cos)
    
    # Triangular
    h_tri = np.maximum(0, 1 - np.abs(x))
    h_tri /= h_tri.sum()
    candidates.append(h_tri)
    
    # Quadratic
    h_quad = np.maximum(0, 1 - x**2)
    h_quad /= h_quad.sum()
    candidates.append(h_quad)
    
    # Asymmetric Gaussians
    for mu in [-0.4, -0.2, 0.2, 0.4]:
        h_a = np.exp(-(x - mu)**2 / (2 * 0.4**2))
        h_a /= h_a.sum()
        candidates.append(h_a)
    
    # Sidon-based sparse initializations
    np.random.seed(42)
    for k in [4, 6, 8, 10, 12, 15, 20]:
        h_s = _weighted_sidon_initialization(k)
        if h_s is not None:
            candidates.append(h_s)
    
    # Random sparse
    for k in [3, 5, 7, 10, 12, 15, 20, 25, 30]:
        idx = np.random.choice(N, k, replace=False)
        h_sp = np.zeros(N)
        h_sp[idx] = 1.0 / np.sqrt(k)
        candidates.append(h_sp)
    
    # Evaluate all candidates
    for h in candidates:
        v = compute_ratio(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
    
    # --- Phase 1: Differential Evolution (global search) ---
    if time.time() < DEADLINE - 70:
        lb = np.full(N, -8.0)
        ub = np.full(N, 3.0)
        
        try:
            de_res = differential_evolution(
                _neg_log_ratio, bounds=list(zip(lb, ub)),
                maxiter=150, popsize=25, tol=1e-5,
                seed=42, polish=False,
                update_interval=10, init='sobol'
            )
            if de_res.fun < 1e5:
                h_de = np.exp(de_res.x)
                h_de = h_de / h_de.sum()
                v = compute_ratio(h_de)
                if v > best_val:
                    best_val = v
                    best_h = h_de.copy()
        except Exception:
            pass
    
    # --- Phase 2: Multi-start local optimization with smooth surrogate ---
    n_restarts = 25
    np.random.seed(123)
    for r in range(n_restarts):
        if time.time() > DEADLINE - 40:
            break
        
        if r == 0:
            x0 = np.log(best_h + 1e-10)
        elif r == 1:
            # Sidon with 8 elements
            h_s = _weighted_sidon_initialization(8)
            if h_s is not None:
                x0 = np.log(h_s + 1e-10)
            else:
                continue
        elif r == 2:
            # Sidon with 12 elements
            h_s = _weighted_sidon_initialization(12)
            if h_s is not None:
                x0 = np.log(h_s + 1e-10)
            else:
                continue
        elif r == 3:
            # Sidon with 15 elements
            h_s = _weighted_sidon_initialization(15)
            if h_s is not None:
                x0 = np.log(h_s + 1e-10)
            else:
                continue
        else:
            # Random sparse starts with varying sparsity
            k = np.random.randint(3, 25)
            idx = np.random.choice(N, k, replace=False)
            h_start = np.zeros(N)
            h_start[idx] = 1.0 / np.sqrt(k)
            # Add small random perturbation to break symmetry
            h_start += np.random.uniform(0, 0.01, N)
            h_start = np.clip(h_start, 0, None)
            h_start /= h_start.sum()
            x0 = np.log(h_start + 1e-10)
        
        try:
            # Use smooth surrogate for L-BFGS-B
            res = minimize(_soft_neg_log_ratio, x0, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < 1e5:
                h_opt = np.exp(res.x)
                h_opt = h_opt / h_opt.sum()
                v = compute_ratio(h_opt)
                if v > best_val:
                    best_val = v
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # --- Phase 3: High-precision polish with exact objective ---
    if time.time() < DEADLINE - 15:
        x0 = np.log(best_h + 1e-10)
        try:
            res = minimize(_neg_log_ratio, x0, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < 1e5:
                h_opt = np.exp(res.x)
                h_opt = h_opt / h_opt.sum()
                v = compute_ratio(h_opt)
                if v > best_val:
                    best_val = v
                    best_h = h_opt.copy()
        except Exception:
            pass
        
        # Second polish from a slightly perturbed start
        if time.time() < DEADLINE - 10:
            x0_pert = np.log(best_h + 1e-10) + np.random.normal(0, 0.01, N)
            try:
                res = minimize(_neg_log_ratio, x0_pert, method='L-BFGS-B',
                               options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
                if res.fun < 1e5:
                    h_opt = np.exp(res.x)
                    h_opt = h_opt / h_opt.sum()
                    v = compute_ratio(h_opt)
                    if v > best_val:
                        best_val = v
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    # --- Phase 4: Coordinate descent with line search ---
    if time.time() < DEADLINE - 5:
        h_cd = best_h.copy()
        golden = (1 + np.sqrt(5)) / 2
        inv_golden = 1 - 1 / golden
        
        for iteration in range(200):
            if time.time() > DEADLINE - 3:
                break
            improved = False
            for i in range(N):
                old = h_cd[i]
                if old < 1e-15:
                    # Try activating this coordinate
                    for trial in [1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1]:
                        h_try = h_cd.copy()
                        h_try[i] = trial
                        h_try /= h_try.sum()
                        v = compute_ratio(h_try)
                        if v > best_val + 1e-12:
                            best_val = v
                            best_h = h_try.copy()
                            h_cd = h_try.copy()
                            improved = True
                            break
                else:
                    # Golden-section search over [0, 3*old]
                    lo, hi = 0.0, 3.0 * old
                    for _ in range(25):
                        c = hi - inv_golden * (hi - lo)
                        d = lo + inv_golden * (hi - lo)
                        
                        h_c = h_cd.copy()
                        h_c[i] = c
                        h_c /= h_c.sum()
                        fc = compute_ratio(h_c)
                        
                        h_d = h_cd.copy()
                        h_d[i] = d
                        h_d /= h_d.sum()
                        fd = compute_ratio(h_d)
                        
                        if fc > fd:
                            hi = d
                        else:
                            lo = c
                    
                    best_trial = (lo + hi) / 2
                    h_try = h_cd.copy()
                    h_try[i] = best_trial
                    h_try /= h_try.sum()
                    v = compute_ratio(h_try)
                    if v > best_val + 1e-12:
                        best_val = v
                        best_h = h_try.copy()
                        h_cd = h_try.copy()
                        improved = True
            
            if not improved:
                break
    
    # Final safety
    best_h = np.clip(best_h, 0.0, None)
    if best_h.sum() < 1e-12:
        best_h = np.ones(N) / N
    
    return list(best_h)

import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.signal import fftconvolve

N = 50
DEADLINE_MARGIN = 5.0  # Return 5s before the 150s kill

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    The ratio is homogeneous of degree 0, so scaling h does not change the value.
    We assume h is non-negative.
    """
    h = np.asarray(h, dtype=np.float64)
    # Enforce non-negativity for safety
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    
    # Normalize to sum=1 to improve numerical stability and simplify ratio
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    
    # Full convolution
    # Using np.convolve for exactness. N=50 is small.
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    
    if conv_inf < 1e-15:
        return 0.0
        
    return h2_sq / conv_inf

def soft_max(values, alpha=100.0):
    """
    Soft approximation of max using log-sum-exp.
    As alpha -> infinity, this approaches max.
    """
    shifted = values - np.max(values)
    return np.max(values) + np.log(np.sum(np.exp(alpha * shifted))) / alpha

def compute_ratio_smooth(h, alpha=50.0):
    """
    Smooth version of compute_ratio using soft-max for the convolution peak.
    This makes the objective differentiable, enabling gradient-based optimization.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_soft_max = soft_max(conv, alpha)
    
    if conv_soft_max < 1e-15:
        return 0.0
        
    return h2_sq / conv_soft_max

def neg_ratio_normalized_smooth(x, alpha=50.0):
    """
    Objective for optimization on normalized vector x (sum=1, x>=0).
    Uses smooth max to make the objective differentiable.
    """
    h = np.maximum(x, 0)
    s = np.sum(h)
    if s < 1e-15:
        return 1e10
    h_norm = h / s
    return -compute_ratio_smooth(h_norm, alpha)

def neg_ratio_logspace_smooth(log_x, alpha=50.0):
    """
    Objective in log-space to handle non-negativity via exp.
    log_x are log of h values. We optimize over log_x with no constraints
    except bounds, then convert to h = exp(log_x) and normalize.
    """
    # Convert from log-space to h
    # Clip to avoid overflow
    log_x_clipped = np.clip(log_x, -30, 10)
    h_raw = np.exp(log_x_clipped)
    s = np.sum(h_raw)
    if s < 1e-15:
        return 1e10
    h_norm = h_raw / s
    return -compute_ratio_smooth(h_norm, alpha)

def construct():
    t0 = time.time()
    deadline = t0 + 150.0 - DEADLINE_MARGIN
    
    def elapsed():
        return time.time() - t0
        
    def time_left():
        return deadline - time.time()

    # Initial best: single spike
    best_h = np.zeros(N, dtype=np.float64)
    best_h[N // 2] = 1.0
    best_score = compute_ratio(best_h)
    
    candidates = []
    
    # Helper to add candidate
    def add_candidate(h):
        nonlocal best_h, best_score
        h = np.maximum(h, 0.0)
        if np.sum(h) < 1e-15:
            return
        sc = compute_ratio(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
        candidates.append((sc, h.copy()))
        
    # --- Phase 1: Generate Structured Candidates (reduced) ---
    
    # 1. Single spike (already set)
    add_candidate(best_h)
    
    # 2. Two spikes (sampled)
    rng = np.random.default_rng(42)
    for _ in range(50):
        i, j = rng.choice(N, size=2, replace=False)
        h = np.zeros(N)
        h[i] = 0.5
        h[j] = 0.5
        add_candidate(h)
        if elapsed() > deadline - 20:
            break
            
    # 3. Three spikes (sampled)
    for _ in range(50):
        idx = rng.choice(N, size=3, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0/3
        add_candidate(h)
        if elapsed() > deadline - 20:
            break

    # 4. Sidon-like sets
    def find_sidon_set_greedy(n, start_idx=0, step=1):
        A = []
        used_sums = set()
        for x in range(start_idx, N, step):
            valid = True
            new_sums = []
            for a in A:
                s = a + x
                if s in used_sums:
                    valid = False
                    break
                new_sums.append(s)
            if 2 * x in used_sums:
                valid = False
            if valid:
                A.append(x)
                used_sums.update(new_sums)
                used_sums.add(2 * x)
        return A

    for step in [1, 2, 3]:
        for start in range(0, 3, step):
            A = find_sidon_set_greedy(N, start_idx=start, step=step)
            if len(A) >= 2:
                h = np.zeros(N)
                for idx in A:
                    h[idx] = 1.0
                add_candidate(h)
                if elapsed() > deadline - 20:
                    break
        if elapsed() > deadline - 20:
            break

    # 5. Random Sparse Sets
    for _ in range(60):
        k = rng.integers(2, 25)
        idx = rng.choice(N, size=k, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0
        add_candidate(h)
        if elapsed() > deadline - 20:
            break

    # 6. Uniform
    add_candidate(np.ones(N))
    
    # 7. Gaussian
    x = np.arange(N)
    for sigma in [3, 8, 15]:
        h = np.exp(-0.5 * ((x - N/2) / sigma)**2)
        add_candidate(h)

    # Sort candidates by score, keep top few for refinement
    candidates.sort(key=lambda x: -x[0])
    
    # --- Phase 2: Global Optimization with Differential Evolution ---
    # DE can escape local minima better than SLSQP on non-smooth objectives
    
    if time_left() > 30:
        # Use smooth objective for DE
        alpha_de = 30.0
        
        def de_objective(x):
            return neg_ratio_normalized_smooth(x, alpha_de)
        
        try:
            de_res = differential_evolution(
                de_objective,
                bounds=[(0, 1)] * N,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=True,
                update_interval=1
            )
            
            if elapsed() > deadline - 10:
                # DE didn't finish in time, use best from candidates
                pass
            else:
                h_de = de_res.x
                h_de = np.maximum(h_de, 0.0)
                if np.sum(h_de) > 1e-15:
                    h_de = h_de / np.sum(h_de)
                sc_de = compute_ratio(h_de)
                if sc_de > best_score:
                    best_score = sc_de
                    best_h = h_de.copy()
                    add_candidate(h_de)
                    
        except Exception:
            pass

    # --- Phase 3: Local Refinement with Smooth Objective ---
    
    # Use the best candidate from Phase 1 and DE results
    # Refine using L-BFGS-B on the smooth objective
    
    top_candidates = candidates[:10]
    
    for sc, h_init in top_candidates:
        if time_left() < 5:
            break
            
        h_norm_init = h_init / np.sum(h_init)
        
        # Use SLSQP on smooth objective
        try:
            res = minimize(
                neg_ratio_normalized_smooth,
                h_norm_init,
                method='SLSQP',
                bounds=[(0, 1)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}],
                options={'maxiter': 300, 'ftol': 1e-14}
            )
            
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0.0)
            if np.sum(h_opt) > 1e-15:
                h_opt = h_opt / np.sum(h_opt)
            
            sc_opt = compute_ratio(h_opt)
            if sc_opt > best_score:
                best_score = sc_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # --- Phase 4: Log-space refinement with L-BFGS-B ---
    # This handles non-negativity naturally via exp
    
    if time_left() > 10:
        h_norm_best = best_h / np.sum(best_h)
        
        # Convert to log-space
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        log_init = np.log(h_norm_best + eps)
        
        alpha_ls = 50.0
        
        try:
            res = minimize(
                neg_ratio_logspace_smooth,
                log_init,
                method='L-BFGS-B',
                bounds=[(-30, 10)] * N,
                options={'maxiter': 500, 'ftol': 1e-15}
            )
            
            log_opt = res.x
            h_opt = np.exp(np.clip(log_opt, -30, 10))
            h_opt = np.maximum(h_opt, 0.0)
            if np.sum(h_opt) > 1e-15:
                h_opt = h_opt / np.sum(h_opt)
            
            sc_opt = compute_ratio(h_opt)
            if sc_opt > best_score:
                best_score = sc_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # --- Phase 5: Final Polish with tight tolerances ---
    if time_left() > 5:
        h_norm_best = best_h / np.sum(best_h)
        
        try:
            # Final SLSQP with very tight tolerance
            res = minimize(
                neg_ratio_normalized_smooth,
                h_norm_best,
                method='SLSQP',
                bounds=[(0, 1)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}],
                options={'maxiter': 500, 'ftol': 1e-16}
            )
            
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0.0)
            if np.sum(h_opt) > 1e-15:
                h_opt = h_opt / np.sum(h_opt)
            
            sc_opt = compute_ratio(h_opt)
            if sc_opt > best_score:
                best_score = sc_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # Ensure non-negative
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

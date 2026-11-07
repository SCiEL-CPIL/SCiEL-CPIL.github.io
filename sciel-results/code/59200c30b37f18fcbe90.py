import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 100
DEADLINE = 148.0

def _correlation_value(h):
    """Compute the evaluator's value: max(corr) / N * 2."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _project_to_feasible(h):
    """Project h onto feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    for _ in range(200):
        diff = np.sum(h) - N / 2.0
        if abs(diff) < 1e-15:
            break
        if diff > 0:
            mask = h > 1e-14
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
        else:
            mask = h < 1.0 - 1e-14
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_max_objective(h, T):
    """Smooth max approximation of the correlation value using log-sum-exp."""
    corr = np.correlate(h, 1 - h, mode="full")
    x = corr / T
    x_max = np.max(x)
    shifted = x - x_max
    sum_exp = np.sum(np.exp(shifted))
    lse = x_max + np.log(sum_exp)
    return T * lse / N * 2

def _subgradient_step(h, step_size, max_iters=200, deadline=None):
    """Subgradient descent on the convex objective max_k corr[k]."""
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        h = _project_to_feasible(h)
        corr = np.correlate(h, 1 - h, mode="full")
        max_idx = np.argmax(corr)
        
        # Subgradient of max_k corr[k] w.r.t. h
        # corr[k] = sum_i h[i] * (1-h[i+k]) for appropriate indices
        # d(corr[k])/dh[j] = (1-h[j+k']) - h[j] * (something)
        # Actually: corr[k] = sum_i h[i]*(1-h[i+k]) where i+k in [0,N-1]
        # Let's compute numerically for the active lag
        
        # For lag k (in full mode, k ranges from -(N-1) to N-1)
        # corr[k] = sum_{i: 0<=i<N, 0<=i+k<N} h[i]*(1-h[i+k])
        # d(corr[k])/dh[j] = sum_{i: i=j, i+k in range} (1-h[i+k]) + sum_{i: i+k=j, i in range} h[i]*(-1)
        #                   = (1-h[j+k]) if j+k in [0,N) else 0 + (-h[j-k]) if j-k in [0,N) else 0
        
        lag = max_idx - (N - 1)  # convert full mode index to lag
        grad = np.zeros(N)
        for j in range(N):
            # Term 1: j is the "i" in h[i]*(1-h[i+lag])
            if 0 <= j + lag < N:
                grad[j] += (1 - h[j + lag])
            # Term 2: j is the "i+lag" in h[i]*(1-h[i+lag])
            if 0 <= j - lag < N:
                grad[j] += (-h[j - lag])
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad = grad / grad_norm
        
        # Adaptive step size
        alpha = step_size / (1 + it * 0.05)
        h_new = h - alpha * grad
        h_new = _project_to_feasible(h_new)
        
        val_new = _correlation_value(h_new)
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
        h = h_new
    
    return best_h, best_val

def _pairwise_coordinate_descent(h, deadline, max_rounds=30):
    """Pairwise coordinate descent: optimize pairs of coordinates."""
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        # Try random pairs
        for _ in range(200):
            if time.time() > deadline:
                break
            i, j = np.random.choice(N, size=2, replace=False)
            if i == j:
                continue
            
            # For fixed other coordinates, the objective as a function of (h[i], h[j])
            # is convex. We can solve a small 2D optimization.
            best_pair_val = best_val
            best_hi, best_hj = best_h[i], best_h[j]
            
            # Grid search over a small range
            for di in np.linspace(-0.15, 0.15, 7):
                for dj in np.linspace(-0.15, 0.15, 7):
                    trial_i = np.clip(best_hi + di, 0.0, 1.0)
                    trial_j = np.clip(best_hj + dj, 0.0, 1.0)
                    h_trial = best_h.copy()
                    h_trial[i] = trial_i
                    h_trial[j] = trial_j
                    h_trial = _project_to_feasible(h_trial)
                    val_trial = _correlation_value(h_trial)
                    if val_trial < best_pair_val - 1e-13:
                        best_pair_val = val_trial
                        best_hi, best_hj = trial_i, trial_j
                        best_h = h_trial.copy()
                        best_val = val_trial
                        improved = True
        
        if not improved:
            break
    
    return best_h, best_val

def _slsqp_polish(h, deadline, T=20.0):
    """SLSQP polish with proper constraints."""
    def obj(x):
        x_proj = _project_to_feasible(x)
        return _smooth_max_objective(x_proj, T)
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
         'jac': lambda x: np.ones(N)}
    ]
    
    try:
        res = minimize(
            obj,
            h,
            method='SLSQP',
            bounds=[(0.0, 1.0)] * N,
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-14, 'disp': False}
        )
        h_opt = _project_to_feasible(res.x)
        val_opt = _correlation_value(h_opt)
        return h_opt, val_opt
    except Exception:
        return h, _correlation_value(h)

def _lbfgs_polish(h, deadline, T=10.0):
    """L-BFGS-B polish with smooth max."""
    def obj(x):
        x_proj = _project_to_feasible(x)
        return _smooth_max_objective(x_proj, T)
    
    try:
        res = minimize(
            obj,
            h,
            method='L-BFGS-B',
            bounds=[(0.0, 1.0)] * N,
            options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-12, 'disp': False}
        )
        h_opt = _project_to_feasible(res.x)
        val_opt = _correlation_value(h_opt)
        return h_opt, val_opt
    except Exception:
        return h, _correlation_value(h)

def _differential_evolution_search(deadline, rng):
    """Global search using differential evolution."""
    bounds = [(0.0, 1.0)] * N
    
    def de_obj(x):
        x_proj = _project_to_feasible(x)
        return _correlation_value(x_proj)
    
    try:
        res = differential_evolution(
            de_obj,
            bounds,
            seed=rng.integers(0, 2**31),
            maxiter=50,
            popsize=15,
            tol=1e-8,
            polish=False,
            update_interval=10,
            workers=1
        )
        h_opt = _project_to_feasible(res.x)
        val_opt = _correlation_value(h_opt)
        return h_opt, val_opt
    except Exception:
        return None, None

def _generate_structural_candidates(rng):
    """Generate diverse structural candidates."""
    candidates = []
    
    # Alternating blocks
    for block_size in [2, 3, 4, 5, 7, 10, 14, 20, 25, 50]:
        h_block = np.zeros(N)
        for i in range(N):
            if (i // block_size) % 2 == 0:
                h_block[i] = 1.0
        candidates.append(_project_to_feasible(h_block))
    
    # Cosine patterns
    for period in [2, 3, 4, 5, 6, 7, 8, 10, 14, 20, 25]:
        h_pat = np.zeros(N)
        for i in range(N):
            phase = (i % period) / period
            h_pat[i] = 0.5 + 0.48 * np.cos(2 * np.pi * phase)
        candidates.append(_project_to_feasible(h_pat))
    
    # Sine waves
    for f in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 25, 30]:
        h_sin = 0.5 + 0.48 * np.sin(2 * np.pi * f * np.arange(N) / N)
        candidates.append(_project_to_feasible(h_sin))
    
    # Cosine waves
    for f in [1, 2, 3, 4, 5, 7, 10, 14, 20]:
        h_cos = 0.5 + 0.48 * np.cos(2 * np.pi * f * np.arange(N) / N)
        candidates.append(_project_to_feasible(h_cos))
    
    # Random candidates
    for _ in range(30):
        h_rand = rng.uniform(0, 1, N)
        candidates.append(_project_to_feasible(h_rand))
    
    # Random smooth candidates
    for _ in range(15):
        h_smooth = 0.5 + 0.4 * rng.normal(0, 0.3, N)
        kernel = np.array([1, 2, 3, 2, 1], dtype=float)
        kernel /= kernel.sum()
        h_smooth = np.convolve(h_smooth, kernel, mode='same')
        candidates.append(_project_to_feasible(h_smooth))
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    rng = np.random.default_rng(100)
    
    # Phase 1: Generate and evaluate structural candidates
    candidates = _generate_structural_candidates(rng)
    candidate_vals = np.array([_correlation_value(h) for h in candidates])
    sorted_indices = np.argsort(candidate_vals)
    
    # Phase 2: Polish top candidates with multiple methods
    top_k = min(12, len(candidates))
    for idx in sorted_indices[:top_k]:
        if time.time() > deadline - 25:
            break
        
        h_candidate = candidates[idx].copy()
        
        # Subgradient step
        h_sg, val_sg = _subgradient_step(h_candidate, step_size=0.1, max_iters=100, deadline=deadline - 2)
        if val_sg < best_val:
            best_val = val_sg
            best_h = h_sg.copy()
        
        # L-BFGS-B polish
        h_lbfgs, val_lbfgs = _lbfgs_polish(best_h, deadline - 2, T=10.0)
        if val_lbfgs < best_val:
            best_val = val_lbfgs
            best_h = h_lbfgs.copy()
    
    # Phase 3: Differential evolution for global search
    if time.time() < deadline - 30:
        h_de, val_de = _differential_evolution_search(deadline - 5, rng)
        if h_de is not None and val_de < best_val:
            best_val = val_de
            best_h = h_de.copy()
    
    # Phase 4: SLSQP polish
    if time.time() < deadline - 10:
        h_sl, val_sl = _slsqp_polish(best_h, deadline - 3, T=20.0)
        if val_sl < best_val:
            best_val = val_sl
            best_h = h_sl.copy()
    
    # Phase 5: Pairwise coordinate descent
    if time.time() < deadline - 5:
        h_pc, val_pc = _pairwise_coordinate_descent(best_h, deadline - 2, max_rounds=20)
        if val_pc < best_val:
            best_val = val_pc
            best_h = h_pc.copy()
    
    # Phase 6: Final SLSQP polish
    if time.time() < deadline - 3:
        h_final, val_final = _slsqp_polish(best_h, deadline - 1, T=50.0)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final.copy()
    
    # Final projection and validation
    best_h = _project_to_feasible(best_h)
    diff = np.sum(best_h) - N / 2.0
    if abs(diff) > 1e-9:
        mask = (best_h > 1e-8) & (best_h < 1.0 - 1e-8)
        if np.any(mask):
            best_h[mask] -= diff / np.sum(mask)
        else:
            best_h -= diff / N
        best_h = np.clip(best_h, 0, 1)
    
    if not np.all(np.isfinite(best_h)):
        best_h = np.full(N, 0.5)
    
    return best_h

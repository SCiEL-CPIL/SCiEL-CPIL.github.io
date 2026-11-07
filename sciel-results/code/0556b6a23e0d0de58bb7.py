import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0

def _correlation_value(h):
    """Compute the evaluator's value: max(corr) / N * 2."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _project_to_feasible(h):
    """Project h onto feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    for _ in range(200):
        diff = np.sum(h) - N / 2.0
        if abs(diff) < 1e-14:
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

def _epigraph_objective(params, T):
    """
    Epigraph formulation: params = [h (N), t]
    Minimize t + penalty for corr > t
    """
    h = params[:N]
    t = params[N]
    h = np.clip(h, 0.0, 1.0)
    corr = np.correlate(h, 1 - h, mode="full")
    # Penalty for violations where corr > t
    violations = np.maximum(corr - t, 0.0)
    penalty = np.sum(np.exp(violations / T)) * T
    return t + penalty

def _epigraph_grad(params, T):
    """Gradient for epigraph objective."""
    h = params[:N]
    t = params[N]
    h = np.clip(h, 0.0, 1.0)
    corr = np.correlate(h, 1 - h, mode="full")
    violations = np.maximum(corr - t, 0.0)
    exp_v = np.exp(violations / T)
    
    # d/dt of t + sum(exp(violations/T)*T)
    dt = 1.0 + T * np.sum(exp_v * (-1.0/T) * (violations > 0))
    
    # d/dh: chain rule through corr
    # corr[k] = sum_i h[i] * (1-h)[i + (N-1-k)] for appropriate indices
    # This is complex; use numerical gradient as fallback
    return None

def _local_polish(h, rng, start_time, deadline):
    """Local polishing using L-BFGS-B with smooth max objective."""
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    
    # Try different temperatures for the smooth max
    temperatures = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    for T in temperatures:
        if time.time() > deadline - 2:
            break
        
        def obj(x):
            x_proj = _project_to_feasible(x)
            return _smooth_max_objective(x_proj, T)
        
        try:
            res = minimize(
                obj,
                best_h,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 150, 'ftol': 1e-16, 'gtol': 1e-12, 'disp': False}
            )
            h_opt = _project_to_feasible(res.x)
            val_opt = _correlation_value(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Also try SLSQP with tighter tolerance
    if time.time() < deadline - 5:
        def obj_sl(x):
            x_proj = _project_to_feasible(x)
            return _smooth_max_objective(x_proj, 20.0)
        
        try:
            res = minimize(
                obj_sl,
                best_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 200, 'ftol': 1e-14, 'disp': False}
            )
            h_opt = _project_to_feasible(res.x)
            val_opt = _correlation_value(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    return best_h, best_val

def _coordinate_descent(h, start_time, deadline):
    """Coordinate descent with finer steps."""
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    
    improved = True
    max_rounds = 50
    for _ in range(max_rounds):
        if time.time() > deadline - 2:
            break
        if not improved:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline - 1:
                break
            # Try a range of values with finer granularity
            current = best_h[i]
            for step in np.linspace(-0.2, 0.2, 21):
                trial_val = np.clip(current + step, 0.0, 1.0)
                h_trial = best_h.copy()
                h_trial[i] = trial_val
                h_trial = _project_to_feasible(h_trial)
                val_trial = _correlation_value(h_trial)
                if val_trial < best_val - 1e-13:
                    best_val = val_trial
                    best_h = h_trial.copy()
                    improved = True
                    current = trial_val
    
    return best_h, best_val

def _random_restart_polish(h, rng, start_time, deadline):
    """Apply random perturbations and polish to escape local minima."""
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    
    n_restarts = 25
    for r in range(n_restarts):
        if time.time() > deadline - 8:
            break
        
        # Create a perturbed version with varying strength
        strength = 0.1 + 0.1 * (r / max(1, n_restarts))
        h_pert = h.copy()
        n_perturb = rng.integers(3, 12)
        indices = rng.choice(N, size=n_perturb, replace=False)
        for idx in indices:
            h_pert[idx] += rng.uniform(-strength, strength)
        h_pert = _project_to_feasible(h_pert)
        
        # Polish
        h_polished, val_polished = _local_polish(h_pert, rng, start_time, deadline)
        
        if val_polished < best_val:
            best_val = val_polished
            best_h = h_polished.copy()
    
    return best_h, best_val

def _generate_initial_candidates(rng):
    """Generate diverse initial candidates."""
    initial_candidates = []
    
    # Block patterns
    for block_size in [2, 3, 4, 5, 7, 10, 14, 20, 25, 50]:
        h_block = np.zeros(N)
        for i in range(N):
            if (i // block_size) % 2 == 0:
                h_block[i] = 1.0
        initial_candidates.append(_project_to_feasible(h_block))
    
    # Cosine patterns
    for period in [2, 3, 4, 5, 6, 7, 8, 10, 14, 20, 25]:
        h_pat = np.zeros(N)
        for i in range(N):
            phase = (i % period) / period
            h_pat[i] = 0.5 + 0.48 * np.cos(2 * np.pi * phase)
        initial_candidates.append(_project_to_feasible(h_pat))
    
    # Sine waves
    for f in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 25, 30]:
        h_sin = 0.5 + 0.48 * np.sin(2 * np.pi * f * np.arange(N) / N)
        initial_candidates.append(_project_to_feasible(h_sin))
    
    # Cosine waves
    for f in [1, 2, 3, 4, 5, 7, 10, 14, 20]:
        h_cos = 0.5 + 0.48 * np.cos(2 * np.pi * f * np.arange(N) / N)
        initial_candidates.append(_project_to_feasible(h_cos))
    
    # Bimodal distributions
    for center1, center2, width in [(20, 80, 10), (25, 75, 15), (30, 70, 8), (15, 85, 12), (20, 80, 20)]:
        h_bimodal = np.zeros(N)
        for i in range(N):
            d1 = abs(i - center1)
            d2 = abs(i - center2)
            h_bimodal[i] = 0.5 + 0.45 * (np.exp(-d1**2 / (2*width**2)) + np.exp(-d2**2 / (2*width**2)))
        initial_candidates.append(_project_to_feasible(h_bimodal))
    
    # Random candidates
    for _ in range(40):
        h_rand = rng.uniform(0, 1, N)
        initial_candidates.append(_project_to_feasible(h_rand))
    
    # Random smooth candidates
    for _ in range(20):
        h_smooth = 0.5 + 0.4 * rng.normal(0, 0.3, N)
        # Apply simple smoothing
        kernel = np.array([1, 2, 3, 2, 1], dtype=float)
        kernel /= kernel.sum()
        h_smooth = np.convolve(h_smooth, kernel, mode='same')
        initial_candidates.append(_project_to_feasible(h_smooth))
    
    return initial_candidates

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    rng = np.random.default_rng(100)
    
    # Phase 1: Generate diverse initial candidates
    initial_candidates = _generate_initial_candidates(rng)
    
    # Evaluate and select top candidates
    candidate_vals = np.array([_correlation_value(h) for h in initial_candidates])
    sorted_indices = np.argsort(candidate_vals)
    
    # Keep top 15 diverse candidates
    top_indices = sorted_indices[:15]
    
    # Phase 2: Polish each top candidate
    for idx in top_indices:
        if time.time() > deadline - 15:
            break
        
        h_candidate = initial_candidates[idx].copy()
        h_polished, val_polished = _local_polish(h_candidate, rng, start_time, deadline)
        
        if val_polished < best_val:
            best_val = val_polished
            best_h = h_polished.copy()
    
    # Phase 3: Random restart polishing from best solution
    if time.time() < deadline - 20:
        h_restart, val_restart = _random_restart_polish(best_h, rng, start_time, deadline)
        if val_restart < best_val:
            best_val = val_restart
            best_h = h_restart.copy()
    
    # Phase 4: Coordinate descent for fine-tuning
    if time.time() < deadline - 5:
        h_cd, val_cd = _coordinate_descent(best_h, start_time, deadline)
        if val_cd < best_val:
            best_val = val_cd
            best_h = h_cd.copy()
    
    # Phase 5: Final high-precision polish
    if time.time() < deadline - 3:
        h_final, val_final = _local_polish(best_h, rng, start_time, deadline)
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

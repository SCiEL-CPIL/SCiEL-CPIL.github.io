import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_linear_overlaps(h):
    """
    Compute linear cross-correlation of h with (1-h) for all non-zero shifts using FFT.
    Returns an array of overlap values C_k for k = 1..N-1 (both positive and negative).
    """
    g = 1.0 - h
    # Use FFT for O(N log N) correlation instead of O(N^2)
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    G = np.fft.rfft(g, n=n_fft)
    c_full = np.fft.irfft(H * G, n=n_fft)
    # c_full has length n_fft, but the correlation is in c_full[0:N-1] and c_full[n_fft-N+1:n_fft]
    # Actually for linear correlation of two length-N sequences:
    # np.correlate(h, g, 'full') gives length 2N-1
    # With FFT of size n_fft >= 2N-1, we get circular correlation
    # The linear correlation is embedded: c[0:N-1] and c[n_fft-(N-1):n_fft]
    c_neg = c_full[:N-1]
    c_pos = c_full[n_fft-(N-1):]
    return np.concatenate([c_neg, c_pos])

def _max_overlap(h):
    """Return the maximum linear overlap value."""
    overlaps = _compute_linear_overlaps(h)
    if len(overlaps) == 0:
        return 0.0
    return np.max(overlaps)

def _project_to_feasible(h):
    """
    Project h onto the feasible set: 0 <= h_i <= 1 and sum(h) = N/2.
    Uses bisection on a Lagrange multiplier for exact projection.
    """
    h = np.clip(h, 0, 1).copy()
    target = N / 2.0
    current = np.sum(h)
    
    if abs(current - target) < 1e-12:
        return h
    
    lo, hi = -max(h) - 1, 1 - min(h) + 1
    
    def sum_at_lambda(lam):
        return np.sum(np.clip(h - lam, 0, 1))
    
    for _ in range(60):
        mid = (lo + hi) / 2
        s = sum_at_lambda(mid)
        if abs(s - target) < 1e-12:
            break
        if s > target:
            lo = mid
        else:
            hi = mid
    
    lam = (lo + hi) / 2
    h = np.clip(h - lam, 0, 1)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-14:
        mask = (h > 1e-10) & (h < 1 - 1e-10)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
    
    return h

def _objective_lse(x, beta, lam_sum):
    """
    Objective function using LogSumExp approximation of the max overlap.
    """
    h = np.clip(x, 0, 1)
    overlaps = _compute_linear_overlaps(h)
    
    if len(overlaps) == 0:
        max_ov = 0.0
    else:
        max_ov = np.max(overlaps)
    
    if beta > 0:
        shifted = overlaps - max_ov
        shifted = np.clip(shifted, -50, 50)
        lse = max_ov + np.log(np.sum(np.exp(beta * shifted))) / beta
    else:
        lse = max_ov
    
    sum_penalty = lam_sum * (np.sum(h) - N / 2.0) ** 2
    return lse + sum_penalty

def _objective_epigraph(x, beta, lam_sum):
    """
    Epigraph formulation: minimize t + smooth_constraint + penalty
    where smooth_constraint approximates max(0, overlaps - t).
    """
    h = np.clip(x[:N], 0, 1)
    t = x[N]
    overlaps = _compute_linear_overlaps(h)
    
    if len(overlaps) == 0:
        return t + lam_sum * (np.sum(h) - N / 2.0) ** 2
    
    # Soft constraint: penalize overlaps exceeding t
    shifted = overlaps - t
    
    # Use smooth max approximation via log-sum-exp for the constraint
    # constraint ≈ (1/beta) * log(sum(exp(beta * max(shifted, 0))))
    # But better: use softplus for each term and sum
    # For numerical stability, use the identity:
    # softplus(x) = log(1 + exp(x)) = x + log(1 + exp(-x)) for x > 0
    #               = log(1 + exp(x)) for x <= 0
    
    # Clip to avoid overflow
    beta_shifted = beta * shifted
    beta_shifted = np.clip(beta_shifted, -50, 50)
    
    # softplus: log(1 + exp(x)) computed stably
    # For x > 0: x + log(1 + exp(-x))
    # For x <= 0: log(1 + exp(x))
    pos_mask = beta_shifted > 0
    softplus = np.empty_like(beta_shifted)
    softplus[pos_mask] = beta_shifted[pos_mask] + np.log1p(np.exp(-beta_shifted[pos_mask]))
    softplus[~pos_mask] = np.log1p(np.exp(beta_shifted[~pos_mask]))
    
    constraint_term = np.sum(softplus) / (beta * len(overlaps))
    
    sum_penalty = lam_sum * (np.sum(h) - N / 2.0) ** 2
    return t + constraint_term + sum_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    bounds_h = [(0.0, 1.0) for _ in range(N)]
    bounds_full = bounds_h + [(0.0, 2.0)]  # t is the max overlap, bounded
    
    rng = np.random.default_rng(123)
    
    # Initial best: uniform
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Generate diverse initial candidates
    candidates = []
    candidates.append(np.full(N, 0.5))
    
    # Alternating patterns
    h_alt = np.zeros(N)
    h_alt[::2] = 0.6
    h_alt[1::2] = 0.4
    candidates.append(h_alt)
    
    # Sinusoidal
    t_arr = np.arange(N)
    for freq in range(1, 30):
        for amp in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
            for phase in [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]:
                trial = 0.5 + amp * np.sin(2*np.pi*freq*t_arr/N + phase)
                candidates.append(trial.copy())
    
    # Multi-harmonic
    for num_harmonics in [2, 3, 4]:
        for freq1 in range(1, 15):
            for freq2 in range(1, 15):
                if freq1 == freq2:
                    continue
                amp1 = 0.2
                amp2 = 0.1
                trial = 0.5 + amp1 * np.sin(2*np.pi*freq1*t_arr/N) + amp2 * np.sin(2*np.pi*freq2*t_arr/N)
                candidates.append(trial.copy())
    
    # Random
    for i in range(80):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        candidates.append(h_rand)
    
    # Evaluate all candidates
    for h_cand in candidates:
        h_proj = _project_to_feasible(h_cand)
        score = _max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj
    
    # Optimization phases with increasing beta
    beta_schedule = [5, 10, 20, 50, 100, 200, 400]
    
    for beta in beta_schedule:
        if time.time() > deadline - 20.0:
            break
        
        # Try epigraph formulation from best solution
        x0_ep = np.concatenate([best_h, [best_score]])
        
        def obj_ep(x):
            return _objective_epigraph(x, beta, 100.0)
        
        try:
            result = minimize(
                obj_ep, x0_ep, method='L-BFGS-B', bounds=bounds_full,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_new = _project_to_feasible(result.x[:N])
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
        
        # Try LSE formulation from best solution
        x0_lse = best_h.copy()
        
        def obj_lse(x):
            return _objective_lse(x, beta, 100.0)
        
        try:
            result = minimize(
                obj_lse, x0_lse, method='L-BFGS-B', bounds=bounds_h,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
        
        # Random restarts
        num_restarts = 8 if beta <= 50 else 4
        for _ in range(num_restarts):
            if time.time() > deadline - 20.0:
                break
            
            # Diverse random initialization
            r = rng.random()
            if r < 0.3:
                # Sinusoidal random
                freq = rng.integers(1, 50)
                amp = rng.uniform(0.05, 0.4)
                phase = rng.uniform(0, 2*np.pi)
                x0 = 0.5 + amp * np.sin(2*np.pi*freq*np.arange(N)/N + phase)
            elif r < 0.5:
                # Multi-harmonic random
                num_h = rng.integers(2, 5)
                x0 = np.full(N, 0.5)
                for _ in range(num_h):
                    freq = rng.integers(1, 50)
                    amp = rng.uniform(0.05, 0.3)
                    phase = rng.uniform(0, 2*np.pi)
                    x0 += amp * np.sin(2*np.pi*freq*np.arange(N)/N + phase)
            elif r < 0.75:
                # Random
                x0 = rng.uniform(0.0, 1.0, size=N)
            else:
                # Perturb best
                x0 = best_h + rng.normal(0, 0.1, size=N)
            
            x0 = _project_to_feasible(x0)
            
            # Use epigraph for random restarts
            x0_ep = np.concatenate([x0, [best_score]])
            
            def obj_rand_ep(x):
                return _objective_epigraph(x, beta, 100.0)
            
            try:
                result = minimize(
                    obj_rand_ep, x0_ep, method='L-BFGS-B', bounds=bounds_full,
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                h_new = _project_to_feasible(result.x[:N])
                score_new = _max_overlap(h_new)
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new
            except Exception:
                pass
    
    # Final polish with high beta using epigraph
    if time.time() < deadline - 10.0:
        x0_polish = np.concatenate([best_h, [best_score]])
        
        def obj_polish(x):
            return _objective_epigraph(x, 1000.0, 5000.0)
        
        try:
            result = minimize(
                obj_polish, x0_polish, method='L-BFGS-B', bounds=bounds_full,
                options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            h_polished = _project_to_feasible(result.x[:N])
            score_polished = _max_overlap(h_polished)
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished
        except Exception:
            pass
        
        # Second polish pass with even tighter tolerance if time permits
        if time.time() < deadline - 5.0:
            x0_polish2 = np.concatenate([best_h, [best_score]])
            
            def obj_polish2(x):
                return _objective_epigraph(x, 1500.0, 10000.0)
            
            try:
                result2 = minimize(
                    obj_polish2, x0_polish2, method='L-BFGS-B', bounds=bounds_full,
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14}
                )
                h_polished2 = _project_to_feasible(result2.x[:N])
                score_polished2 = _max_overlap(h_polished2)
                if score_polished2 < best_score:
                    best_score = score_polished2
                    best_h = h_polished2
            except Exception:
                pass
    
    # Additional local search: adaptive coordinate descent on individual elements
    if time.time() < deadline - 5.0:
        improved = True
        max_cd_iterations = 50
        cd_iter = 0
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        while improved and cd_iter < max_cd_iterations and time.time() < deadline - 5.0:
            improved = False
            cd_iter += 1
            for i in range(N):
                old_val = best_h[i]
                # Try adaptive steps: start large, go smaller if no improvement
                for delta in step_sizes + [-s for s in step_sizes]:
                    trial = best_h.copy()
                    trial[i] = np.clip(old_val + delta, 0, 1)
                    trial = _project_to_feasible(trial)
                    score = _max_overlap(trial)
                    if score < best_score - 1e-14:
                        best_h = trial
                        best_score = score
                        improved = True
                        break
                # Also try a few random small perturbations
                for _ in range(3):
                    delta = rng.normal(0, 0.01)
                    trial = best_h.copy()
                    trial[i] = np.clip(old_val + delta, 0, 1)
                    trial = _project_to_feasible(trial)
                    score = _max_overlap(trial)
                    if score < best_score - 1e-14:
                        best_h = trial
                        best_score = score
                        improved = True
                        break
    
    # Ensure final validity
    best_h = _project_to_feasible(best_h)
    
    # Final verification: ensure sum is exactly N/2
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-15:
        # Distribute the difference among interior points
        mask = (best_h > 1e-12) & (best_h < 1 - 1e-12)
        if np.any(mask):
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0, 1)
    
    return best_h

import numpy as np
from scipy.optimize import minimize
import time

N = 100
_NFFT = 256  # next power of 2 >= 2N-1

def _compute_linear_overlaps(h):
    """Compute linear cross-correlation of h with (1-h) for all non-zero shifts using FFT."""
    g = 1.0 - h
    # Use FFT-based correlation for speed
    H = np.fft.rfft(h, _NFFT)
    G = np.fft.rfft(g, _NFFT)
    c = np.fft.irfft(H * G.conj(), _NFFT)
    # c has length _NFFT; the linear correlation is in c[0:N-1] and c[_NFFT-N+1:]
    # Center (zero shift) is at index N-1 in the full correlation
    # Full correlation: c_full[k] for k=0..2N-2, mapped to FFT output
    # c_full[0..N-2] -> c[0..N-2]
    # c_full[N-1] -> c[N-1] (zero shift)
    # c_full[N..2N-2] -> c[_NFFT-N+1 .. _NFFT-1]
    c_neg = c[:N-1]
    c_pos = c[_NFFT-N+1:]
    return np.concatenate([c_neg, c_pos])

def _max_overlap(h):
    """Return the maximum linear overlap value."""
    overlaps = _compute_linear_overlaps(h)
    if len(overlaps) == 0:
        return 0.0
    return np.max(overlaps)

def _project_to_feasible(h):
    """Project h onto feasible set: 0 <= h_i <= 1 and sum(h) = N/2."""
    h = np.clip(h, 0, 1).copy()
    target = N / 2.0
    current = np.sum(h)
    
    if abs(current - target) < 1e-14:
        return h
    
    # Bisection on Lagrange multiplier for faster convergence
    lo, hi = -1.0, 1.0
    
    def sum_at_lambda(lam):
        return np.sum(np.clip(h - lam, 0, 1))
    
    for _ in range(60):
        mid = (lo + hi) / 2
        s = sum_at_lambda(mid)
        if abs(s - target) < 1e-15:
            break
        if s > target:
            lo = mid
        else:
            hi = mid
    
    lam = (lo + hi) / 2
    h = np.clip(h - lam, 0, 1)
    
    # Final adjustment
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 1e-10) & (h < 1 - 1e-10)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
    
    return h

def _objective(x, beta, lam_sum):
    """Objective: LSE-smoothed max overlap + sum constraint penalty."""
    h = np.clip(x, 0, 1)
    overlaps = _compute_linear_overlaps(h)
    
    if len(overlaps) == 0:
        max_ov = 0.0
    else:
        max_ov = np.max(overlaps)
    
    # Smooth max via LSE
    if beta > 0 and len(overlaps) > 0:
        shifted = overlaps - max_ov
        shifted = np.clip(shifted, -50, 50)
        lse = max_ov + np.log(np.sum(np.exp(beta * shifted))) / beta
    else:
        lse = max_ov
    
    sum_penalty = lam_sum * (np.sum(h) - N / 2.0) ** 2
    return lse + sum_penalty

def _objective_epigraph(x, beta, lam_sum):
    """Epigraph formulation: minimize t + penalties, where t >= max overlap."""
    # x = [h (N vars), t (1 var)]
    h = np.clip(x[:N], 0, 1)
    t = x[N]
    overlaps = _compute_linear_overlaps(h)
    
    if len(overlaps) == 0:
        max_ov = 0.0
    else:
        max_ov = np.max(overlaps)
    
    # Smooth max via LSE for the epigraph constraint
    if beta > 0 and len(overlaps) > 0:
        shifted = overlaps - t
        shifted = np.clip(shifted, -50, 50)
        lse = t + np.log(np.sum(np.exp(beta * shifted))) / beta
    else:
        lse = t
    
    sum_penalty = lam_sum * (np.sum(h) - N / 2.0) ** 2
    return lse + sum_penalty

def _spectral_init(rng):
    """Generate candidates using spectral flatness design."""
    candidates = []
    t = np.arange(N)
    
    # Uniform baseline
    candidates.append(np.full(N, 0.5))
    
    # Alternating patterns
    for base in [0.3, 0.4, 0.45, 0.5]:
        h_alt = np.zeros(N)
        h_alt[::2] = base + 0.1
        h_alt[1::2] = base - 0.1
        candidates.append(h_alt.copy())
    
    # Multi-harmonic with spectral flatness
    for n_harm in range(1, 8):
        for total_amp in [0.2, 0.3, 0.4, 0.45, 0.49]:
            # Distribute amplitude across harmonics with some randomness
            amps = rng.dirichlet(np.ones(n_harm)) * total_amp
            phases = rng.uniform(0, 2*np.pi, size=n_harm)
            h = np.full(N, 0.5)
            for k in range(n_harm):
                freq = (k + 1)
                h += amps[k] * np.sin(2*np.pi*freq*t/N + phases[k])
            candidates.append(h.copy())
    
    # Pure sinusoids
    for freq in range(1, 20):
        for amp in [0.1, 0.2, 0.3, 0.4, 0.45]:
            for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                h = 0.5 + amp * np.sin(2*np.pi*freq*t/N + phase)
                candidates.append(h.copy())
    
    # Random
    for _ in range(50):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        candidates.append(h_rand)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    bounds = [(0.0, 1.0) for _ in range(N)]
    rng = np.random.default_rng(42)
    
    # Generate and evaluate initial candidates
    candidates = _spectral_init(rng)
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    for h_cand in candidates:
        h_proj = _project_to_feasible(h_cand)
        score = _max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj
    
    # Multi-phase optimization with increasing beta
    beta_schedule = [3, 8, 20, 50, 100, 200, 400, 800]
    
    for beta in beta_schedule:
        if time.time() > deadline - 20.0:
            break
        
        # Optimize from best solution
        x0 = best_h.copy()
        
        def obj_func(x):
            return _objective(x, beta, 500.0)
        
        try:
            result = minimize(
                obj_func, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
        
        # Random restarts
        num_restarts = 5 if beta <= 50 else 3
        for _ in range(num_restarts):
            if time.time() > deadline - 20.0:
                break
            
            x0 = rng.uniform(0.1, 0.9, size=N)
            x0 = _project_to_feasible(x0)
            
            def obj_rand(x):
                return _objective(x, beta, 500.0)
            
            try:
                result = minimize(
                    obj_rand, x0, method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                h_new = _project_to_feasible(result.x)
                score_new = _max_overlap(h_new)
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new
            except Exception:
                pass
    
    # Epigraph-based optimization for better max constraint handling
    if time.time() < deadline - 25.0:
        for beta_ep in [50, 100, 200]:
            if time.time() > deadline - 25.0:
                break
            x0 = np.concatenate([best_h, [best_score]])
            bounds_ep = bounds + [(-1.0, 1.0)]
            
            def obj_ep(x):
                return _objective_epigraph(x, beta_ep, 500.0)
            
            try:
                result = minimize(
                    obj_ep, x0, method='L-BFGS-B', bounds=bounds_ep,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                h_new = _project_to_feasible(result.x[:N])
                score_new = _max_overlap(h_new)
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new
            except Exception:
                pass
    
    # Coordinate descent polish - more efficient with fewer delta steps
    if time.time() < deadline - 8.0:
        improved = True
        max_cd = 50
        cd_iter = 0
        deltas = [0.02, -0.02, 0.01, -0.01, 0.005, -0.005]
        while improved and cd_iter < max_cd and time.time() < deadline - 8.0:
            improved = False
            cd_iter += 1
            for i in range(N):
                old_val = best_h[i]
                for delta in deltas:
                    trial = best_h.copy()
                    trial[i] = np.clip(old_val + delta, 0, 1)
                    trial = _project_to_feasible(trial)
                    score = _max_overlap(trial)
                    if score < best_score - 1e-14:
                        best_h = trial
                        best_score = score
                        improved = True
                        break
    
    # Final high-precision polish with epigraph
    if time.time() < deadline - 5.0:
        x0 = np.concatenate([best_h, [best_score]])
        bounds_ep = bounds + [(-1.0, 1.0)]
        
        def obj_final_ep(x):
            return _objective_epigraph(x, 1000.0, 5000.0)
        
        try:
            result = minimize(
                obj_final_ep, x0, method='L-BFGS-B', bounds=bounds_ep,
                options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            h_polished = _project_to_feasible(result.x[:N])
            score_polished = _max_overlap(h_polished)
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished
        except Exception:
            pass
    
    # Final standard polish
    if time.time() < deadline - 3.0:
        def obj_final(x):
            return _objective(x, 2000.0, 5000.0)
        
        try:
            result = minimize(
                obj_final, best_h, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            h_polished = _project_to_feasible(result.x)
            score_polished = _max_overlap(h_polished)
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished
        except Exception:
            pass
    
    # Ensure validity
    best_h = _project_to_feasible(best_h)
    return best_h

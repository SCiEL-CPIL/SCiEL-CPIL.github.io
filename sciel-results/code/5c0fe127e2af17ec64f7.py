import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 100

def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr)

def _smooth_max_corr(h, lam=50.0):
    """Compute smooth approximation of max correlation using log-sum-exp."""
    corr = np.correlate(h, 1 - h, mode="full")
    m = np.max(corr)
    # Soft max: m + (1/lam) * log(sum(exp((corr - m) * lam)))
    return m + np.log(np.sum(np.exp((corr - m) * lam))) / lam

def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    for _ in range(50):
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            mask = h > 1e-9
            if not np.any(mask): break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            mask = h < 1 - 1e-9
            if not np.any(mask): break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - N / 2.0
    return h

def _fourier_basis(i, n_harm):
    """Generate Fourier basis matrix for N points and n_harm harmonics."""
    cols = []
    for k in range(1, n_harm + 1):
        cols.append(np.cos(2 * np.pi * k * i / N))
        cols.append(np.sin(2 * np.pi * k * i / N))
    return np.column_stack(cols)

def _fast_corr(h):
    """
    Fast correlation using FFT.
    corr = np.correlate(h, 1-h, mode='full')
    We can compute this via FFT: correlate(a,b) = ifft(fft(a) * conj(fft(b)))
    But for full mode, we need to zero-pad.
    """
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    G = np.fft.rfft(1 - h, size)
    corr = np.fft.irfft(H * np.conj(G), size)
    # The result is circular correlation, but with zero-padding it gives linear correlation
    # The 'full' mode result has length 2*n - 1
    # The FFT-based result has length size = 2*n
    # The linear correlation is embedded in the circular one
    # For correlate(h, 1-h, 'full'), the result is:
    # [h0*(1-h0), h0*(1-h1)+h1*(1-h0), ..., h0*(1-h_{n-1})+...+h_{n-1}*(1-h0), ..., h_{n-1}*(1-h_{n-1})]
    # The FFT approach gives circular correlation, but with zero-padding of size 2n,
    # the first n elements and last n-1 elements correspond to the linear correlation.
    # Actually, let's just use np.correlate for correctness and rely on other optimizations.
    return np.correlate(h, 1 - h, mode="full")

def _targeted_local_search(h, max_iters=5000, time_limit=None):
    """
    Targeted local search that focuses on the peak lag.
    Uses incremental correlation updates for speed.
    """
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    
    start_time = time.time()
    if time_limit is None:
        time_limit = start_time + 10.0
    
    rng = np.random.default_rng(42)
    
    for it in range(max_iters):
        if time.time() > time_limit:
            break
        
        # Compute correlations
        corr = np.correlate(best_h, 1 - best_h, mode="full")
        max_idx = np.argmax(corr)
        max_val = corr[max_idx]
        
        # The lag corresponding to max_idx in 'full' mode
        lag = max_idx - (N - 1)
        
        improved = False
        
        # Strategy: Target the peak lag directly
        # S_lag = sum_i h_i * h_{i-lag} (with appropriate bounds)
        # We want to reduce S_lag by adjusting h_i and h_j
        
        # Find indices i where h_i and h_{i-lag} are both large
        # These contribute most to S_lag
        
        if lag >= 0:
            valid_i = np.arange(lag, N)
        else:
            valid_i = np.arange(0, N + lag)
        
        # Compute contribution of each i to S_lag
        if lag >= 0:
            contrib = best_h[valid_i] * best_h[valid_i - lag]
        else:
            contrib = best_h[valid_i] * best_h[valid_i + (-lag)]
        
        # Pick top few contributors
        n_top = min(5, len(valid_i))
        top_indices = valid_i[np.argsort(contrib)[-n_top:]]
        
        # For each top index, try to reduce it
        for i in top_indices:
            # Determine direction: if h_i is high, try to decrease it
            # If h_i is low, try to increase it (but this might increase other terms)
            
            # Try moving h_i towards 0 or 1
            if best_h[i] > 0.5:
                d_target = -best_h[i]  # move to 0
            else:
                d_target = 1.0 - best_h[i]  # move to 1
            
            # Try several step sizes
            for frac in [0.1, 0.3, 0.5, 0.8]:
                d = d_target * frac
                if abs(d) < 1e-6:
                    continue
                
                # Find j to compensate
                # Prefer j that doesn't affect S_lag much
                # j should be far from i and i±lag
                candidates = [j for j in range(N) if j != i and abs(j - i) > max(abs(lag), 5)]
                if not candidates:
                    candidates = [j for j in range(N) if j != i]
                
                j = candidates[rng.integers(len(candidates))]
                
                # Check bounds
                new_i = best_h[i] + d
                new_j = best_h[j] - d
                if new_i < -1e-9 or new_i > 1 + 1e-9 or new_j < -1e-9 or new_j > 1 + 1e-9:
                    # Scale d
                    if d > 0:
                        d_max = min(1.0 - best_h[i], best_h[j])
                    else:
                        d_max = min(best_h[i], 1.0 - best_h[j])
                    d = d * d_max / abs(d) if d_max > 0 else 0
                    if abs(d) < 1e-6:
                        continue
                    new_i = best_h[i] + d
                    new_j = best_h[j] - d
                    if new_i < -1e-9 or new_i > 1 + 1e-9 or new_j < -1e-9 or new_j > 1 + 1e-9:
                        continue
                
                new_h = best_h.copy()
                new_h[i] = new_i
                new_h[j] = new_j
                
                val = _correlation_value(new_h)
                if val < best_val - 1e-12:
                    best_val = val
                    best_h = new_h
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            # Try random moves as fallback
            for _ in range(10):
                i = rng.integers(N)
                j = rng.integers(N)
                if i == j:
                    continue
                
                d = rng.uniform(-0.1, 0.1)
                new_i = best_h[i] + d
                new_j = best_h[j] - d
                if new_i < -1e-9 or new_i > 1 + 1e-9 or new_j < -1e-9 or new_j > 1 + 1e-9:
                    continue
                
                new_h = best_h.copy()
                new_h[i] = new_i
                new_h[j] = new_j
                
                val = _correlation_value(new_h)
                if val < best_val - 1e-12:
                    best_val = val
                    best_h = new_h
                    improved = True
                    break
        
        if not improved and it > 200:
            break
    
    return best_h

def _construct_from_basis(basis, params):
    """Construct h from Fourier basis and parameters."""
    h = np.full(N, 0.5)
    h += params @ basis
    return _project_to_feasible(h)

def _objective_from_coeffs(params, basis):
    """Objective function for Fourier coefficient optimization."""
    h = _construct_from_basis(basis, params)
    return _correlation_value(h)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Phase 1: Differential Evolution on Fourier Coefficients
    n_harm = 8
    n_params = 2 * n_harm
    i = np.arange(N)
    basis = _fourier_basis(i, n_harm)
    
    bounds = [(-0.5, 0.5)] * n_params
    
    de_deadline = start_time + 40.0
    
    def timed_fourier_obj(params):
        if time.time() > de_deadline:
            raise TimeoutError()
        return _objective_from_coeffs(params, basis)
        
    try:
        result = differential_evolution(
            timed_fourier_obj,
            bounds=bounds,
            maxiter=150,
            popsize=8,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=False,
            init='sobol'
        )
        if result.x is not None:
            h_de = _construct_from_basis(basis, result.x)
            val_de = _correlation_value(h_de)
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
    except TimeoutError:
        pass
    except Exception:
        pass

    # Phase 2: Targeted Local Search
    ls_deadline = deadline - 20.0
    if time.time() < ls_deadline:
        best_h = _targeted_local_search(best_h, time_limit=ls_deadline)
        best_val = _correlation_value(best_h)
        
    # Phase 3: SLSQP with ramping lambda for smooth max approximation
    if time.time() < deadline - 15.0:
        try:
            lambdas = [10.0, 30.0, 100.0]
            for lam in lambdas:
                if time.time() >= deadline - 15.0:
                    break
                
                def smooth_obj(x, lam=lam):
                    return _smooth_max_corr(x, lam)
                
                def constraint(x):
                    return np.sum(x) - N / 2.0
                
                res = minimize(
                    smooth_obj,
                    best_h,
                    method='SLSQP',
                    bounds=[(0.0, 1.0)] * N,
                    constraints=[{'type': 'eq', 'fun': constraint}],
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                if res.x is not None:
                    h_opt = _project_to_feasible(res.x)
                    val_opt = _correlation_value(h_opt)
                    if val_opt < best_val:
                        best_val = val_opt
                        best_h = h_opt.copy()
        except Exception:
            pass

    # Phase 4: L-BFGS-B with multiple restarts and ramping lambda
    if time.time() < deadline - 10.0:
        try:
            lambdas = [50.0, 100.0, 200.0]
            for lam in lambdas:
                if time.time() >= deadline - 10.0:
                    break
                
                def smooth_obj_lbfgs(x, lam=lam):
                    return _smooth_max_corr(x, lam)
                
                # Use L-BFGS-B with bounds (no explicit constraint needed if we project)
                # But we need to handle the sum constraint. Use penalty method.
                def penalized_obj(x, lam=lam):
                    h = _project_to_feasible(x)
                    return _smooth_max_corr(h, lam)
                
                res = minimize(
                    penalized_obj,
                    best_h,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.x is not None:
                    h_opt = _project_to_feasible(res.x)
                    val_opt = _correlation_value(h_opt)
                    if val_opt < best_val:
                        best_val = val_opt
                        best_h = h_opt.copy()
        except Exception:
            pass

    # Phase 5: High-precision SLSQP with very high lambda
    if time.time() < deadline - 5.0:
        try:
            lam_final = 300.0
            def smooth_obj_final(x):
                return _smooth_max_corr(x, lam_final)
            
            def constraint_final(x):
                return np.sum(x) - N / 2.0
            
            res = minimize(
                smooth_obj_final,
                best_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': constraint_final}],
                options={'maxiter': 400, 'ftol': 1e-15}
            )
            if res.x is not None:
                h_opt = _project_to_feasible(res.x)
                val_opt = _correlation_value(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 6: Final targeted local search polish
    if time.time() < deadline - 2.0:
        best_h = _targeted_local_search(best_h, max_iters=500, time_limit=deadline - 1.0)
        best_val = _correlation_value(best_h)
        
    best_h = _project_to_feasible(best_h)
    
    return best_h

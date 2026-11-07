import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 100

def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    # The evaluator uses max(corr) / N * 2
    # We just need to minimize max(corr)
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
    # Returns matrix of shape (N, 2*n_harm)
    # Columns: cos(2*pi*k*i/N), sin(2*pi*k*i/N) for k=1..n_harm
    cols = []
    for k in range(1, n_harm + 1):
        cols.append(np.cos(2 * np.pi * k * i / N))
        cols.append(np.sin(2 * np.pi * k * i / N))
    return np.column_stack(cols)

def _local_search(h, max_iters=2000, time_limit=None):
    """
    Local search using pair flips and binarization.
    """
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    
    start_time = time.time()
    if time_limit is None:
        time_limit = start_time + 10.0
    
    for it in range(max_iters):
        if time.time() > time_limit:
            break
            
        # Compute correlations
        corr = np.correlate(best_h, 1 - best_h, mode="full")
        max_idx = np.argmax(corr)
        max_val = corr[max_idx]
        
        # The lag corresponding to max_idx in 'full' mode
        # full mode lags are -(N-1) ... 0 ... N-1
        # index max_idx corresponds to lag = max_idx - (N-1)
        lag = max_idx - (N - 1)
        
        improved = False
        
        # Strategy 1: Adjust pairs separated by 'lag'
        # If we decrease h_i and increase h_j (or vice versa) where j = i + lag,
        # we directly affect S_lag.
        # S_lag = sum_i h_i h_{i-lag}
        # Term involves h_i h_{i-lag} and h_{i+lag} h_i
        # So changing h_i affects S_lag via h_{i-lag} and h_{i+lag}.
        
        # Try to find a pair (i, j) such that flipping them reduces max correlation.
        # We focus on the lag 'lag' and its symmetric counterparts.
        
        # Candidate indices for i
        if lag >= 0:
            range_i = range(lag, N) # i-lag >= 0 and i < N
        else:
            range_i = range(0, N + lag) # i+lag < N and i >= 0
            
        # To keep sum constant, we change h_i by d and h_j by -d.
        # Let's try small d steps.
        step_sizes = [0.05, 0.1, 0.2, 0.5]
        
        # We can only pick j such that it helps. 
        # Ideally, j is close to i to minimize disruption to other lags? 
        # Or j = i + lag to directly target the peak?
        # If j = i + lag, then the change in S_lag is:
        # Delta S_lag = (h_i+d)(h_{i-lag}) + (h_{i+lag})(h_i+d) - h_i h_{i-lag} - h_{i+lag} h_i
        #            = d(h_{i-lag} + h_{i+lag})
        # So if we pick d with sign opposite to (h_{i-lag} + h_{i+lag}), we reduce S_lag.
        # But we also change other S_k.
        
        # Let's just try random pairs and small steps, accepting if max(corr) decreases.
        # This is slower but more robust.
        
        # Optimization: Only try pairs involving the 'hot' indices.
        # Indices i that contribute significantly to S_lag are those where h_i and h_{i-lag} are large.
        
        # Simpler: Try to binarize.
        # Pick a random i. Move h_i towards 0 or 1.
        # Pick a random j != i. Move h_j in opposite direction to keep sum.
        # Accept if max(corr) decreases.
        
        # To be faster, we sample a few candidates.
        for _ in range(20): # 20 candidates per iteration
            i = np.random.randint(N)
            # Determine direction: towards 0 or 1
            if best_h[i] < 0.5:
                target = 0.0
            else:
                target = 1.0
            
            d = target - best_h[i]
            if abs(d) < 1e-6: continue
            
            # Find j
            # Prefer j such that h_j moves in a way that doesn't violate bounds too much
            # Just pick random j
            j = np.random.randint(N)
            if j == i: continue
            
            # Adjust d so that j stays in [0,1]
            new_j = best_h[j] - d
            if new_j < 0 or new_j > 1:
                # Scale d
                if d > 0:
                    max_d = best_h[i] - 0.0
                    min_d = best_h[j] - 1.0 # new_j = h_j - d >= 0 => d <= h_j
                    # Actually:
                    # h_i + d >= 0 => d >= -h_i
                    # h_i + d <= 1 => d <= 1 - h_i
                    # h_j - d >= 0 => d <= h_j
                    # h_j - d <= 1 => d >= h_j - 1
                    d_min = max(-best_h[i], best_h[j] - 1.0)
                    d_max = min(1.0 - best_h[i], best_h[j])
                    if d > d_max: d = d_max
                    if d < d_min: d = d_min
                    if abs(d) < 1e-6: continue
                else:
                    d_min = max(-best_h[i], best_h[j] - 1.0)
                    d_max = min(1.0 - best_h[i], best_h[j])
                    if d < d_min: d = d_min
                    if d > d_max: d = d_max
                    if abs(d) < 1e-6: continue
            
            new_h = best_h.copy()
            new_h[i] += d
            new_h[j] -= d
            
            val = _correlation_value(new_h)
            if val < best_val - 1e-12:
                best_val = val
                best_h = new_h
                improved = True
                break
        
        if not improved:
            # Try a more targeted move on the peak lag
            # If we didn't improve with random flips, try forcing reduction on peak lag
            # We identified 'lag'.
            # Let's try to adjust h_i where h_i * h_{i-lag} is high.
            # This is complex. Let's just break if stuck for a while? 
            # No, try again with smaller steps?
            # For now, if no improvement in 100 iters, break to save time.
            if it > 100:
                 break
            
    return best_h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Phase 1: Differential Evolution on Fourier Coefficients
    # h_i = 0.5 + sum_k (a_k cos(...) + b_k sin(...))
    n_harm = 10
    n_params = 2 * n_harm
    i = np.arange(N)
    basis = _fourier_basis(i, n_harm)
    
    def fourier_objective(params):
        # params are a_1..a_n, b_1..b_n
        # Reconstruct h
        h = np.full(N, 0.5)
        # a_1 is params[0], b_1 is params[1], etc? 
        # Let's map: params[0] -> a_1, params[1] -> b_1
        # basis col 0 is cos(1), col 1 is sin(1)
        # So h += params[0]*basis[:,0] + params[1]*basis[:,1] ...
        h += params @ basis
        h = _project_to_feasible(h)
        return _correlation_value(h)
    
    bounds = [(-0.5, 0.5)] * n_params
    
    de_deadline = start_time + 60.0
    
    def timed_fourier_obj(params):
        if time.time() > de_deadline:
            raise TimeoutError()
        return fourier_objective(params)
        
    try:
        result = differential_evolution(
            timed_fourier_obj,
            bounds=bounds,
            maxiter=200,
            popsize=10,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=False, # We will do local search later
            init='sobol'
        )
        if result.x is not None:
            h_de = 0.5 + result.x @ basis
            h_de = _project_to_feasible(h_de)
            val_de = _correlation_value(h_de)
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
    except TimeoutError:
        pass
    except Exception:
        pass

    # Phase 2: Local Search Polish
    # Run local search on the best candidate
    ls_deadline = deadline - 5.0
    if time.time() < ls_deadline:
        best_h = _local_search(best_h, time_limit=ls_deadline)
        best_val = _correlation_value(best_h)
        
    # Phase 3: SLSQP with ramping lambda for smooth max approximation
    if time.time() < deadline - 10.0:
        try:
            # Use ramping lambda to gradually sharpen the objective
            lambdas = [10.0, 30.0, 100.0]
            for lam in lambdas:
                if time.time() >= deadline - 10.0:
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

    # Final Polish: High-precision SLSQP with very high lambda
    if time.time() < deadline - 5.0:
        try:
            lam_final = 200.0
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
                options={'maxiter': 300, 'ftol': 1e-15}
            )
            if res.x is not None:
                h_opt = _project_to_feasible(res.x)
                val_opt = _correlation_value(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final local search polish
    if time.time() < deadline - 2.0:
        best_h = _local_search(best_h, max_iters=200, time_limit=deadline - 1.0)
        best_val = _correlation_value(best_h)
        
    best_h = _project_to_feasible(best_h)
    
    return best_h

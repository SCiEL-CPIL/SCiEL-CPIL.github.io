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

def _local_search_targeted(h, max_iters=5000, time_limit=None):
    """
    Targeted local search:
    1. Compute R_k = sum h_i (1-h_{i+k}) for all k.
    2. Find k_max = argmax R_k.
    3. Try to decrease R_k_max by increasing P_k_max = sum h_i h_{i+k_max}.
    4. Move delta from j (where neighbors at k_max are small) to i (where neighbors at k_max are large).
    """
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    
    start_time = time.time()
    if time_limit is None:
        time_limit = start_time + 30.0
        
    for it in range(max_iters):
        if time.time() > time_limit:
            break
            
        R = np.correlate(best_h, 1 - best_h, mode="full")
        k_max = np.argmax(R)
        
        # Map k_max to lag l.
        lag = k_max - (N - 1)
        abs_lag = abs(lag)
        
        # Calculate neighbor sums for the specific lag
        # ns[i] = h[i+lag] + h[i-lag] (valid terms only)
        ns = np.zeros(N)
        if abs_lag == 0:
            ns = best_h.copy()
        else:
            ns[:N-abs_lag] += best_h[abs_lag:]
            ns[abs_lag:] += best_h[:N-abs_lag]
        
        # We want to increase P_lag.
        # Delta P_lag ~ d * (ns[i] - ns[j])
        # We want d > 0 if ns[i] > ns[j].
        # So move mass from j (low ns) to i (high ns).
        
        # Sort indices by ns
        idx_sorted = np.argsort(ns)[::-1]
        
        found_improvement = False
        
        # Candidates for i (destination, high ns)
        top_i_indices = []
        for idx in idx_sorted:
            if best_h[idx] < 1.0 - 1e-6:
                top_i_indices.append(idx)
            if len(top_i_indices) >= 5:
                break
                
        # Candidates for j (source, low ns)
        bottom_j_indices = []
        for idx in idx_sorted[::-1]:
            if best_h[idx] > 1e-6:
                bottom_j_indices.append(idx)
            if len(bottom_j_indices) >= 5:
                break
                
        if not top_i_indices or not bottom_j_indices:
            # Try random jitter
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j: continue
            d = 0.01 * np.random.choice([-1, 1])
            new_h = best_h.copy()
            new_h[i] += d
            new_h[j] -= d
            if np.all((new_h >= 0) & (new_h <= 1)):
                val = _correlation_value(new_h)
                if val < best_val:
                    best_val = val
                    best_h = new_h
                    found_improvement = True
            if not found_improvement and it > 100:
                break
            continue
            
        for i in top_i_indices:
            for j in bottom_j_indices:
                if i == j: continue
                max_d = min(1.0 - best_h[i], best_h[j])
                if max_d <= 1e-9: continue
                
                for frac in [1.0, 0.5, 0.1]:
                    d = max_d * frac
                    if d < 1e-9: continue
                    new_h = best_h.copy()
                    new_h[i] += d
                    new_h[j] -= d
                    
                    val = _correlation_value(new_h)
                    if val < best_val - 1e-12:
                        best_val = val
                        best_h = new_h
                        found_improvement = True
                        break
                if found_improvement:
                    break
            if found_improvement:
                break
                
        if not found_improvement:
            # Random kick
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j: continue
            d = 0.05 * np.random.choice([-1, 1])
            new_h = best_h.copy()
            new_h[i] += d
            new_h[j] -= d
            if np.all((new_h >= 0) & (new_h <= 1)):
                val = _correlation_value(new_h)
                if val < best_val:
                    best_val = val
                    best_h = new_h
                    found_improvement = True
            if not found_improvement and it > 200:
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

    # Phase 2: Targeted Local Search
    ls_deadline = deadline - 2.0
    if time.time() < ls_deadline:
        best_h = _local_search_targeted(best_h, time_limit=ls_deadline)
        best_val = _correlation_value(best_h)
        
    best_h = _project_to_feasible(best_h)
    
    return best_h

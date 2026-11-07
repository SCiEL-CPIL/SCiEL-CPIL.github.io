"""Model-and-optimize approach for Erdos minimum overlap.

Key insight: the max correlation value is driven by the peaks of the
correlation function. We can parameterize h as a smooth function of a
few basis coefficients, then directly optimize those parameters.
"""
N = 100
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution


def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2


def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    for _ in range(20):
        if abs(diff) < 1e-14:
            break
        # Only adjust elements that can absorb the correction
        if diff > 0:
            # Need to decrease sum: reduce elements that are > 0
            mask = h > 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            # Need to increase sum: increase elements that are < 1
            mask = h < 1 - 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)  # diff is negative, so this adds
            h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - N / 2.0
    return h


def _construct_from_basis(coeffs):
    """
    Build h from a basis expansion:
    h[i] = 0.5 + sum_k coeffs[k] * basis_k(i)
    
    Basis functions are chosen to capture the key features:
    - Constant (handled by offset)
    - Low-frequency sinusoids
    - Linear ramp
    - Quadratic
    """
    n = len(coeffs)
    i = np.arange(N, dtype=float)
    h = np.full(N, 0.5)
    
    # Up to 12 basis functions
    if n > 0:
        h += coeffs[0] * (i / N - 0.5)  # Linear ramp
    if n > 1:
        h += coeffs[1] * ((i / N - 0.5) ** 2 - 1/12)  # Quadratic
    if n > 2:
        h += coeffs[2] * np.sin(2 * np.pi * i / N)  # freq 1
    if n > 3:
        h += coeffs[3] * np.cos(2 * np.pi * i / N)
    if n > 4:
        h += coeffs[4] * np.sin(4 * np.pi * i / N)  # freq 2
    if n > 5:
        h += coeffs[5] * np.cos(4 * np.pi * i / N)
    if n > 6:
        h += coeffs[6] * np.sin(6 * np.pi * i / N)  # freq 3
    if n > 7:
        h += coeffs[7] * np.cos(6 * np.pi * i / N)
    if n > 8:
        h += coeffs[8] * np.sin(10 * np.pi * i / N)  # freq 5
    if n > 9:
        h += coeffs[9] * np.cos(10 * np.pi * i / N)
    if n > 10:
        h += coeffs[10] * np.sin(14 * np.pi * i / N)  # freq 7
    if n > 11:
        h += coeffs[11] * np.cos(14 * np.pi * i / N)
    
    return h


def _objective_from_coeffs(coeffs):
    """Objective: compute correlation value from basis coefficients."""
    h = _construct_from_basis(coeffs)
    h = _project_to_feasible(h)
    return _correlation_value(h)


def _local_polish(h, max_iters=50, step_sizes=None):
    """
    Local search using derivative-aware pair swaps.
    The correlation at lag k is sum_i h[i]*(1-h[i+k]).
    We find the lag(s) achieving the max, then try swaps that reduce it.
    """
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    n = len(best_h)
    corr_full = np.correlate(best_h, 1 - best_h, mode="full")
    max_lags = np.where(corr_full == np.max(corr_full))[0]
    # Convert full-mode indices to lag offsets: index i corresponds to lag (i - (n-1))
    max_lag_offsets = max_lags - (n - 1)
    
    if step_sizes is None:
        step_sizes = [0.01, 0.02, 0.05, 0.1]
    
    for it in range(max_iters):
        improved = False
        # Recompute which lags are at max
        corr_full = np.correlate(best_h, 1 - best_h, mode="full")
        cmax = np.max(corr_full)
        max_lags = np.where(np.abs(corr_full - cmax) < 1e-12)[0]
        
        for i in range(n):
            for d in step_sizes:
                for sign in [+1, -1]:
                    new_val_i = best_h[i] + sign * d
                    if new_val_i < -1e-12 or new_val_i > 1 + 1e-12:
                        continue
                    new_val_i = np.clip(new_val_i, 0, 1)
                    actual_d = new_val_i - best_h[i]
                    if abs(actual_d) < 1e-14:
                        continue
                    needed = -actual_d
                    # Try each j as partner
                    best_j = -1
                    best_new_val = best_val
                    for j in range(n):
                        if j == i:
                            continue
                        new_val_j = best_h[j] + needed
                        if new_val_j < -1e-12 or new_val_j > 1 + 1e-12:
                            continue
                        new_val_j = np.clip(new_val_j, 0, 1)
                        # Quick check: does this help the max lags?
                        # For lag L (offset), corr_L = sum_k h[k]*(1-h[k+L])
                        # Changing h[i] by d and h[j] by -d affects lags where i or j appears
                        # Just compute the new value
                        new_h = best_h.copy()
                        new_h[i] = new_val_i
                        new_h[j] = new_val_j
                        val = _correlation_value(new_h)
                        if val < best_new_val - 1e-14:
                            best_new_val = val
                            best_j = j
                            best_new_h = new_h
                    if best_j >= 0 and best_new_val < best_val - 1e-14:
                        best_val = best_new_val
                        best_h = best_new_h
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_h


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Phase 1: Differential evolution over basis coefficients
    # Use a moderate number of basis functions
    n_basis = 12
    bounds = [(-0.5, 0.5)] * n_basis
    
    def de_objective(coeffs):
        return _objective_from_coeffs(coeffs)
    
    # Run differential evolution with time budget
    de_deadline = start_time + 50.0
    
    try:
        # Custom wrapper that checks time
        def timed_de_objective(coeffs):
            if time.time() > de_deadline:
                raise TimeoutError()
            return de_objective(coeffs)
        
        result = differential_evolution(
            timed_de_objective,
            bounds=bounds,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            mutation=(0.3, 1.0),
            recombination=0.7,
            seed=42,
            polish=True,
            init='sobol'
        )
        
        if result.x is not None:
            h_de = _project_to_feasible(_construct_from_basis(result.x))
            val_de = _correlation_value(h_de)
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
    except TimeoutError:
        pass
    except Exception:
        pass
    
    # Phase 2: Multi-start SLSQP from good starting points
    rng = np.random.default_rng(123)
    
    # Generate diverse starting points
    starts = [best_h.copy()]
    
    # Random feasible starts
    for _ in range(8):
        h_rand = rng.uniform(0, 1, N)
        h_rand = _project_to_feasible(h_rand)
        starts.append(h_rand)
    
    # Structured: alternating 0s and 1s
    h_alt = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
    starts.append(_project_to_feasible(h_alt))
    
    # Structured: blocks
    for block_size in [4, 5, 10]:
        h_block = np.zeros(N)
        for i in range(N):
            h_block[i] = 1.0 if (i // block_size) % 2 == 0 else 0.0
        starts.append(_project_to_feasible(h_block))
    
    # Sine waves at various frequencies
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        amp = min(0.4, 0.5 / freq)
        h_sin = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
        starts.append(_project_to_feasible(h_sin))
    
    # Sawtooth / triangular
    for period in [10, 20, 25]:
        h_tri = np.zeros(N)
        for i in range(N):
            t = (i % period) / period
            if t < 0.5:
                h_tri[i] = 2 * t
            else:
                h_tri[i] = 2 - 2 * t
        starts.append(_project_to_feasible(h_tri))
    
    # Optimize from each start with epigraph formulation
    slsqp_deadline = start_time + 115.0
    
    def _epigraph_objective_and_grad(x, lam=50.0):
        """
        Epigraph formulation: minimize t + smooth_penalty
        where t >= max_k corr_k.
        We use a smooth surrogate: t + (1/lam)*log(sum(exp(lam*corr_k)))
        but constrain t >= each corr_k approximately.
        
        Actually, simpler: minimize the smooth max directly.
        """
        h = x
        corr = np.correlate(h, 1 - h, mode="full")
        m = np.max(corr)
        if m > 1e15:
            return m
        # Smooth max via log-sum-exp
        scaled = (corr - m) * lam
        # Clip for numerical stability
        scaled = np.clip(scaled, -50, 50)
        log_sum = np.log(np.sum(np.exp(scaled)) / len(corr)) / lam
        return m + log_sum
    
    def _epigraph_grad(x, lam=50.0):
        """Gradient of the smooth max objective."""
        h = x
        corr = np.correlate(h, 1 - h, mode="full")
        m = np.max(corr)
        if m > 1e15:
            return np.zeros(N)
        
        # Gradient of corr_k w.r.t. h: 
        # corr_k = sum_i h[i] * (1 - h[i+k]) for lag k
        # d corr_k / d h[j] = (1 - h[j+k]) - h[j-k] (with boundary handling)
        
        # For the smooth max: grad = sum_k w_k * grad_corr_k
        # where w_k = exp(lam*(corr_k - m)) / sum_l exp(lam*(corr_l - m))
        
        scaled = (corr - m) * lam
        scaled = np.clip(scaled, -50, 50)
        weights = np.exp(scaled)
        weights /= np.sum(weights)
        
        # Compute gradient of each corr_k w.r.t. h
        # corr at full-mode index p corresponds to lag L = p - (N-1)
        # corr_p = sum_{i} h[i] * (1 - h[i + L]) where valid i range
        # d corr_p / d h[j] = sum over i where i=j: (1-h[j+L]) + sum over i where i+L=j: -h[j-L]
        # = (1 - h[j+L]) - h[j-L] (if both indices valid, with 0 otherwise)
        
        grad_h = np.zeros(N)
        for p, w in enumerate(weights):
            if w < 1e-15:
                continue
            L = p - (N - 1)  # lag offset
            for j in range(N):
                g = 0.0
                # Term from h[j] as first factor: (1 - h[j+L])
                idx1 = j + L
                if 0 <= idx1 < N:
                    g += (1.0 - h[idx1])
                # Term from h[j] as second factor (1-h[j]): -h[j-L]
                idx2 = j - L
                if 0 <= idx2 < N:
                    g -= h[idx2]
                grad_h[j] += w * g
        
        # Add gradient of the log-sum-exp correction (it's already in the weights)
        # The smooth max grad is: sum_k w_k * grad_corr_k + (1/lam)*grad_log_sum
        # But since we defined obj = m + log_sum/lam, and m is constant w.r.t. h at the max point...
        # Actually the derivative of the soft-max IS sum_k w_k * grad_corr_k
        # The extra log term's gradient is (1/lam) * sum_k (exp(scaled_k)/sum) * lam * grad_corr_k
        # = sum_k w_k * grad_corr_k which is the same!
        # Wait no: d/dh [m + (1/lam) log(sum exp(lam(corr_k - m)))] 
        # = sum_k w_k * grad_corr_k (the m terms cancel in the log-sum-exp derivative)
        
        return grad_h
    
    for idx, h_start in enumerate(starts):
        if time.time() > slsqp_deadline - 12.0:
            break
        
        h_start = _project_to_feasible(h_start)
        current_h = h_start.copy()
        
        # Stage 1: Smooth optimization with moderate lambda
        for lam in [30.0, 100.0, 300.0]:
            if time.time() > slsqp_deadline - 8.0:
                break
            
            def smooth_obj(x, lam=lam):
                return _epigraph_objective_and_grad(x, lam)
            
            def smooth_grad(x, lam=lam):
                return _epigraph_grad(x, lam)
            
            try:
                res = minimize(
                    smooth_obj,
                    smooth_grad,
                    current_h,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-12}
                )
                h_opt = _project_to_feasible(res.x)
                val = _correlation_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                current_h = h_opt
            except Exception:
                pass
        
        # Stage 2: High-precision SLSQP with tighter tolerances
        if time.time() > slsqp_deadline - 5.0:
            break
        
        try:
            def exact_obj(x):
                h = x
                corr = np.correlate(h, 1 - h, mode="full")
                return np.max(corr)
            
            res_exact = minimize(
                exact_obj,
                current_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 500, 'ftol': 1e-16}
            )
            h_exact = _project_to_feasible(res_exact.x)
            val_exact = _correlation_value(h_exact)
            if val_exact < best_val:
                best_val = val_exact
                best_h = h_exact.copy()
            current_h = h_exact
        except Exception:
            pass
        
        # Stage 3: Final polish with even tighter SLSQP
        if time.time() > slsqp_deadline - 2.0:
            break
        
        try:
            res_final = minimize(
                exact_obj,
                current_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 300, 'ftol': 1e-18}
            )
            h_final = _project_to_feasible(res_final.x)
            val_final = _correlation_value(h_final)
            if val_final < best_val:
                best_val = val_final
                best_h = h_final.copy()
        except Exception:
            pass
    
    # Phase 3: Local search polish
    if time.time() < deadline - 8.0:
        try:
            best_h = _local_polish(best_h, max_iters=40)
            best_val = _correlation_value(best_h)
        except Exception:
            pass
    
    # Phase 4: One more round of SLSQP from the polished point
    if time.time() < deadline - 3.0:
        try:
            def exact_obj2(x):
                h = x
                corr = np.correlate(h, 1 - h, mode="full")
                return np.max(corr)
            
            res_last = minimize(
                exact_obj2,
                best_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 200, 'ftol': 1e-16}
            )
            h_last = _project_to_feasible(res_last.x)
            val_last = _correlation_value(h_last)
            if val_last < best_val:
                best_val = val_last
                best_h = h_last.copy()
        except Exception:
            pass
    
    # Final projection
    best_h = _project_to_feasible(best_h)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h

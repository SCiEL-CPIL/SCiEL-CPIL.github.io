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

def _smooth_max_correlation(h, lam):
    """Smooth surrogate for max(corr) using log-sum-exp."""
    corr = np.correlate(h, 1 - h, mode="full")
    m = np.max(corr)
    if m > 1e15:
        return m
    # log-sum-exp for numerical stability
    shifted = corr - m
    log_sum = m + np.log(np.sum(np.exp(shifted * lam)) / len(corr)) / lam
    return log_sum


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


def _local_polish(h, max_iters=50, step_sizes=None, deadline=None):
    """
    Local search: try small adjustments to individual elements.
    Uses incremental correlation updates for speed.
    """
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    n = len(best_h)
    
    if step_sizes is None:
        step_sizes = [0.002, 0.005, 0.01, 0.02, 0.05]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Try adjusting pairs of elements to maintain sum
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(i + 1, n):
                if deadline is not None and time.time() > deadline:
                    break
                for d in step_sizes:
                    # Try: h[i] += d, h[j] -= d
                    if best_h[i] + d > 1 or best_h[j] - d < 0:
                        continue
                    new_h = best_h.copy()
                    new_h[i] += d
                    new_h[j] -= d
                    val = _correlation_value(new_h)
                    if val < best_val - 1e-14:
                        best_val = val
                        best_h = new_h
                        improved = True
                        break
                    # Try: h[i] -= d, h[j] += d
                    if best_h[i] - d < 0 or best_h[j] + d > 1:
                        continue
                    new_h = best_h.copy()
                    new_h[i] -= d
                    new_h[j] += d
                    val = _correlation_value(new_h)
                    if val < best_val - 1e-14:
                        best_val = val
                        best_h = new_h
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_h


def _high_precision_polish(h, deadline=None):
    """
    High-precision polish using L-BFGS-B with a smooth surrogate.
    Uses ramping lambda to gradually sharpen the max objective.
    """
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    
    # Try different lambda values for the smooth max
    lambdas = [100.0, 500.0, 1000.0, 2000.0, 5000.0]
    
    for lam in lambdas:
        if deadline is not None and time.time() > deadline:
            break
        
        def smooth_obj(x, lam=lam):
            return _smooth_max_correlation(x, lam)
        
        def smooth_grad(x, lam=lam):
            # Compute gradient of smooth max correlation
            corr = np.correlate(x, 1 - x, mode="full")
            m = np.max(corr)
            shifted = corr - m
            weights = np.exp(shifted * lam)
            weights /= np.sum(weights)
            
            # Gradient of corr with respect to x
            # corr[k] = sum_i x[i] * (1-x)[i+k-N+1] for appropriate k
            # This is complex, so we use numerical gradient as fallback
            # For now, let L-BFGS-B use finite differences
            return None
        
        try:
            res = minimize(
                smooth_obj,
                best_h,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            h_opt = _project_to_feasible(res.x)
            val = _correlation_value(h_opt)
            if val < best_val - 1e-15:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final exact local search with very tight tolerance
    if deadline is not None and time.time() < deadline:
        try:
            def exact_obj(x):
                return np.max(np.correlate(x, 1 - x, mode="full"))
            
            res_exact = minimize(
                exact_obj,
                best_h,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            h_exact = _project_to_feasible(res_exact.x)
            val_exact = _correlation_value(h_exact)
            if val_exact < best_val - 1e-15:
                best_val = val_exact
                best_h = h_exact.copy()
        except Exception:
            pass
    
    return best_h


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Phase 1: Differential evolution over basis coefficients
    n_basis = 12
    bounds = [(-0.5, 0.5)] * n_basis
    
    def de_objective(coeffs):
        return _objective_from_coeffs(coeffs)
    
    # Run differential evolution with time budget
    de_deadline = start_time + 45.0
    
    try:
        def timed_de_objective(coeffs):
            if time.time() > de_deadline:
                raise TimeoutError()
            return de_objective(coeffs)
        
        result = differential_evolution(
            timed_de_objective,
            bounds=bounds,
            maxiter=80,
            popsize=12,
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
    
    # Phase 2: Multi-start SLSQP with smooth surrogate
    rng = np.random.default_rng(123)
    
    # Generate diverse starting points
    starts = [best_h.copy()]
    
    # Random feasible starts
    for _ in range(6):
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
    
    # Triangular waves
    for period in [10, 20, 25]:
        h_tri = np.zeros(N)
        for i in range(N):
            t = (i % period) / period
            if t < 0.5:
                h_tri[i] = 2 * t
            else:
                h_tri[i] = 2 - 2 * t
        starts.append(_project_to_feasible(h_tri))
    
    # Optimize from each start with SLSQP using smooth surrogate
    slsqp_deadline = start_time + 115.0
    
    for idx, h_start in enumerate(starts):
        if time.time() > slsqp_deadline - 8.0:
            break
        
        h_start = _project_to_feasible(h_start)
        current_h = h_start
        
        # Smooth objective with increasing lambda for better convergence
        for lam in [50.0, 200.0, 1000.0]:
            if time.time() > slsqp_deadline - 4.0:
                break
            
            def smooth_obj(x, lam=lam):
                return _smooth_max_correlation(x, lam)
            
            try:
                res = minimize(
                    smooth_obj,
                    current_h,
                    method='SLSQP',
                    bounds=[(0.0, 1.0)] * N,
                    constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                    options={'maxiter': 150, 'ftol': 1e-12}
                )
                h_opt = _project_to_feasible(res.x)
                val = _correlation_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                current_h = h_opt
            except Exception:
                pass
        
        # Final exact optimization with tight tolerance
        if time.time() > slsqp_deadline - 2.0:
            break
        
        try:
            def exact_obj(x):
                return np.max(np.correlate(x, 1 - x, mode="full"))
            
            res_exact = minimize(
                exact_obj,
                current_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            h_exact = _project_to_feasible(res_exact.x)
            val_exact = _correlation_value(h_exact)
            if val_exact < best_val:
                best_val = val_exact
                best_h = h_exact.copy()
        except Exception:
            pass
    
    # Phase 3: High-precision polish with smooth surrogate
    if time.time() < deadline - 5.0:
        try:
            polish_deadline = deadline - 1.0
            best_h = _high_precision_polish(best_h, deadline=polish_deadline)
            best_val = _correlation_value(best_h)
        except Exception:
            pass
    
    # Phase 4: Final local search with very tight steps
    if time.time() < deadline - 1.0:
        try:
            final_deadline = deadline - 0.5
            best_h = _local_polish(best_h, max_iters=50, step_sizes=[0.0005, 0.001, 0.002], deadline=final_deadline)
            best_val = _correlation_value(best_h)
        except Exception:
            pass
    
    # Final projection and validation
    best_h = _project_to_feasible(best_h)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val
    
    # Ensure sum constraint is met exactly
    diff = np.sum(best_h) - N / 2.0
    if abs(diff) > 1e-12:
        # Distribute correction evenly among elements that can absorb it
        if diff > 0:
            mask = best_h > 1e-10
        else:
            mask = best_h < 1 - 1e-10
        if np.any(mask):
            best_h[mask] -= diff / np.sum(mask)
            best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h

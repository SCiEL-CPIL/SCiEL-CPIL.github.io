"""
Combined Erdos minimum overlap solver.
Strategy:
1. Global search: Differential Evolution over a low-dimensional basis to find a good basin.
2. Local Refinement: SLSQP with increasing smoothness (log-sum-exp) to navigate the max-correlation landscape.
3. High-Precision Polish: Tighter SLSQP tolerances to squeeze out remaining value.
4. Local Search: Fast, pairwise coordinate descent to fix discrete-like issues and fine-tune.
"""

import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
DEADLINE_OFFSET = 145.0  # Safety margin for 150s limit


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
        if diff > 0:
            mask = h > 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            mask = h < 1 - 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - N / 2.0
    return h


def _construct_from_basis(coeffs):
    """
    Build h from a basis expansion.
    """
    n = len(coeffs)
    i = np.arange(N, dtype=float)
    h = np.full(N, 0.5)
    
    if n > 0:
        h += coeffs[0] * (i / N - 0.5)
    if n > 1:
        h += coeffs[1] * ((i / N - 0.5) ** 2 - 1/12)
    if n > 2:
        h += coeffs[2] * np.sin(2 * np.pi * i / N)
    if n > 3:
        h += coeffs[3] * np.cos(2 * np.pi * i / N)
    if n > 4:
        h += coeffs[4] * np.sin(4 * np.pi * i / N)
    if n > 5:
        h += coeffs[5] * np.cos(4 * np.pi * i / N)
    if n > 6:
        h += coeffs[6] * np.sin(6 * np.pi * i / N)
    if n > 7:
        h += coeffs[7] * np.cos(6 * np.pi * i / N)
    if n > 8:
        h += coeffs[8] * np.sin(10 * np.pi * i / N)
    if n > 9:
        h += coeffs[9] * np.cos(10 * np.pi * i / N)
    if n > 10:
        h += coeffs[10] * np.sin(14 * np.pi * i / N)
    if n > 11:
        h += coeffs[11] * np.cos(14 * np.pi * i / N)
        
    return h


def _objective_from_coeffs(coeffs):
    """Objective: compute correlation value from basis coefficients."""
    h = _construct_from_basis(coeffs)
    h = _project_to_feasible(h)
    return _correlation_value(h)


def _fast_local_search(h, time_limit=10.0):
    """
    Fast local search using coordinate descent.
    For each index i, find the best value in [0,1] that minimizes the objective,
    adjusting a partner index j to maintain the sum constraint.
    """
    start_time = time.time()
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    n = len(best_h)
    
    # Pre-compute some constants for speed? No, np.correlate is fast enough for N=100.
    
    # We will iterate over i, and for each i, try a few candidate steps.
    # To maintain sum, we adjust a "partner" j.
    # A good heuristic for partner j is one that is far away in index (to decorrelate) 
    # or simply the one that allows the largest step.
    # Here we use a simple approach: pick j = (i + n//2) % n as a default partner, 
    # but also try a few others if time permits.
    
    step_sizes = [0.01, 0.05, 0.1, 0.2, 0.4]
    
    for _ in range(100): # Max iterations
        if time.time() > start_time + time_limit:
            break
        improved = False
        for i in range(n):
            if time.time() > start_time + time_limit:
                break
            current_val = best_h[i]
            # Try moving i by delta, and adjusting j by -delta
            # We pick j such that it can absorb the change.
            # Let's try a few partner indices.
            partners = [(i + 1) % n, (i + n//2) % n, (i + n//3) % n]
            for j in partners:
                if j == i: continue
                for delta in step_sizes:
                    for sign in [1, -1]:
                        new_i_val = current_val + sign * delta
                        new_j_val = best_h[j] - sign * delta
                        if 0 <= new_i_val <= 1 and 0 <= new_j_val <= 1:
                            new_h = best_h.copy()
                            new_h[i] = new_i_val
                            new_h[j] = new_j_val
                            val = _correlation_value(new_h)
                            if val < best_val - 1e-12:
                                best_val = val
                                best_h = new_h
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
            
    return best_h


def _high_precision_polish(h, time_budget=20.0):
    """
    High-precision polish using SLSQP with tighter tolerances.
    Uses log-sum-exp smoothing with increasing lambda to approximate max.
    """
    start_time = time.time()
    current_h = h.copy()
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    
    # Increasing lambda makes the surrogate closer to the true max
    for lam in [100.0, 200.0, 500.0, 1000.0, 2000.0]:
        if time.time() - start_time > time_budget - 5.0:
            break
        
        def smooth_obj(x, lam=lam):
            h_local = x
            corr = np.correlate(h_local, 1 - h_local, mode="full")
            m = np.max(corr)
            if m > 1e15:
                return m
            shifted = corr - m
            log_sum_exp = m + np.log(np.sum(np.exp(shifted)) / len(corr))
            return log_sum_exp
        
        try:
            res = minimize(
                smooth_obj,
                current_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 500, 'ftol': 1e-16}
            )
            h_opt = _project_to_feasible(res.x)
            val = _correlation_value(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
            current_h = h_opt
        except Exception:
            pass
    
    # Final exact optimization
    if time.time() - start_time < time_budget - 2.0:
        try:
            def exact_obj(x):
                h_local = x
                corr = np.correlate(h_local, 1 - h_local, mode="full")
                return np.max(corr)
            
            res_exact = minimize(
                exact_obj,
                current_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 1000, 'ftol': 1e-18}
            )
            h_exact = _project_to_feasible(res_exact.x)
            val_exact = _correlation_value(h_exact)
            if val_exact < best_val:
                best_val = val_exact
                best_h = h_exact.copy()
        except Exception:
            pass
    
    return best_h


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_OFFSET
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Phase 1: Differential Evolution
    n_basis = 12
    bounds = [(-0.5, 0.5)] * n_basis
    
    def de_objective(coeffs):
        return _objective_from_coeffs(coeffs)
    
    de_deadline = start_time + 40.0
    try:
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
    
    # Phase 2: Multi-start SLSQP
    rng = np.random.default_rng(123)
    starts = [best_h.copy()]
    
    # Random starts
    for _ in range(5):
        h_rand = rng.uniform(0, 1, N)
        starts.append(_project_to_feasible(h_rand))
    
    # Structured starts
    h_alt = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
    starts.append(_project_to_feasible(h_alt))
    
    for block_size in [4, 5, 10]:
        h_block = np.zeros(N)
        for i in range(N):
            h_block[i] = 1.0 if (i // block_size) % 2 == 0 else 0.0
        starts.append(_project_to_feasible(h_block))
        
    for freq in [1, 2, 3, 5, 7]:
        amp = min(0.4, 0.5 / freq)
        h_sin = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
        starts.append(_project_to_feasible(h_sin))

    slsqp_deadline = start_time + 110.0
    
    for idx, h_start in enumerate(starts):
        if time.time() > slsqp_deadline - 10.0:
            break
        
        h_start = _project_to_feasible(h_start)
        current_h = h_start
        
        # Coarse to fine smoothing
        for lam in [20.0, 100.0, 500.0]:
            if time.time() > slsqp_deadline - 5.0:
                break
            
            def smooth_obj(x, lam=lam):
                h = x
                corr = np.correlate(h, 1 - h, mode="full")
                m = np.max(corr)
                if m > 1e15:
                    return m
                return m + np.log(np.sum(np.exp((corr - m) * lam)) / len(corr)) / lam
            
            try:
                res = minimize(
                    smooth_obj,
                    current_h,
                    method='SLSQP',
                    bounds=[(0.0, 1.0)] * N,
                    constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                    options={'maxiter': 200, 'ftol': 1e-14}
                )
                h_opt = _project_to_feasible(res.x)
                val = _correlation_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                current_h = h_opt
            except Exception:
                pass
        
        # Final exact optimization for this start
        if time.time() > slsqp_deadline - 3.0:
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
                options={'maxiter': 300, 'ftol': 1e-15}
            )
            h_exact = _project_to_feasible(res_exact.x)
            val_exact = _correlation_value(h_exact)
            if val_exact < best_val:
                best_val = val_exact
                best_h = h_exact.copy()
        except Exception:
            pass

    # Phase 3: High Precision Polish on the global best
    if time.time() < deadline - 25.0:
        try:
            best_h = _high_precision_polish(best_h, time_budget=20.0)
            best_val = _correlation_value(best_h)
        except Exception:
            pass
            
    # Phase 4: Fast Local Search
    if time.time() < deadline - 5.0:
        try:
            best_h = _fast_local_search(best_h, time_limit=10.0)
            best_val = _correlation_value(best_h)
        except Exception:
            pass
            
    # Final Projection
    best_h = _project_to_feasible(best_h)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val
        
    return best_h

"""Model-and-optimize approach for Erdos minimum overlap.

Combines the control-point parameterization with sinusoidal basis expansion.
Key improvements: vectorized local polish, epigraph-based SLSQP for smoother
gradient, and tighter final polish.  A smooth log-sum-exp surrogate is used
in early stages, followed by epigraph SLSQP and coordinate-descent polish.
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
    for _ in range(30):
        if abs(diff) < 1e-15:
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
    # Final fine adjustment on a single element to hit sum exactly
    diff = np.sum(h) - N / 2.0
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
            h[idx] = max(0.0, h[idx] - diff)
        else:
            idx = np.argmax(h)
            h[idx] = min(1.0, h[idx] - diff)
    return h


def _interp_control_to_full(control):
    """
    Interpolate 10 control points to 100 samples via linear interpolation.
    Control points at indices 0, 9, 19, ..., 79, 99.
    """
    ctrl_positions = np.array([0, 9, 19, 29, 39, 49, 59, 69, 79, 99], dtype=float)
    x = np.arange(N, dtype=float)
    h = np.interp(x, ctrl_positions, control)
    return h


def _basis_terms(x, coeffs):
    """
    Add sinusoidal basis terms to a base signal x.
    coeffs: array of 8 coefficients for frequencies 1, 2, 3, 5, 7 (sin+cos each).
    """
    n = len(coeffs)
    i = np.arange(N, dtype=float)
    t = i / N
    result = np.zeros(N)
    if n > 0:
        result += coeffs[0] * np.sin(2 * np.pi * 1 * t)
    if n > 1:
        result += coeffs[1] * np.cos(2 * np.pi * 1 * t)
    if n > 2:
        result += coeffs[2] * np.sin(2 * np.pi * 2 * t)
    if n > 3:
        result += coeffs[3] * np.cos(2 * np.pi * 2 * t)
    if n > 4:
        result += coeffs[4] * np.sin(2 * np.pi * 3 * t)
    if n > 5:
        result += coeffs[5] * np.cos(2 * np.pi * 3 * t)
    if n > 6:
        result += coeffs[6] * np.sin(2 * np.pi * 5 * t)
    if n > 7:
        result += coeffs[7] * np.cos(2 * np.pi * 5 * t)
    return result


def _construct_hybrid(params):
    """
    Build h from a hybrid parameterization:
    - First 10 params: control points for piecewise-linear interpolation
    - Next 8 params: sinusoidal basis coefficients
    """
    n_ctrl = 10
    n_basis = 8
    control = params[:n_ctrl]
    basis_coeffs = params[n_ctrl:n_ctrl + n_basis]
    h_base = _interp_control_to_full(control)
    h_basis = _basis_terms(h_base, basis_coeffs)
    h = h_base + h_basis
    return h


def _smooth_max_correlation(h, lam):
    """
    Compute a smooth approximation of max(corr) using log-sum-exp.
    """
    corr = np.correlate(h, 1 - h, mode="full")
    m = np.max(corr)
    scaled = (corr - m) * lam
    scaled = np.clip(scaled, -50, 50)
    exp_scaled = np.exp(scaled)
    log_sum = np.log(np.sum(exp_scaled) / len(corr))
    smooth_val = m + log_sum / lam
    return smooth_val


def _smooth_grad(h, lam):
    """
    Vectorized gradient of the smooth max correlation w.r.t. h.
    corr[p] = sum_j h[j] * (1-h[j+L]) where L = p-(N-1)
    d corr[p] / d h[j] = (1-h[j+L]) - h[j-L]  (when indices valid)
    """
    corr = np.correlate(h, 1 - h, mode="full")
    m = np.max(corr)
    scaled = (corr - m) * lam
    scaled = np.clip(scaled, -50, 50)
    weights = np.exp(scaled)
    weights /= np.sum(weights)
    
    grad_h = np.zeros(N)
    # For each lag L = p - (N-1), p in [0, 2N-2], L in [-(N-1), N-1]
    for p in range(2 * N - 1):
        w = weights[p]
        if w < 1e-16:
            continue
        L = p - (N - 1)
        # d corr[p] / d h[j] = (1 - h[j+L]) if 0<=j+L<N else 0
        #                    - h[j-L] if 0<=j-L<N else 0
        # j+L in [0,N) => j in [-L, N-L) => j in [max(0,-L), min(N, N-L))
        # j-L in [0,N) => j in [L, L+N) => j in [max(0,L), min(N, L+N))
        
        # Term 1: (1 - h[j+L]) for j where 0 <= j+L < N
        j_start1 = max(0, -L)
        j_end1 = min(N, N - L)
        if j_start1 < j_end1:
            idx1 = np.arange(j_start1, j_end1) + L
            grad_h[j_start1:j_end1] += w * (1.0 - h[idx1])
        
        # Term 2: -h[j-L] for j where 0 <= j-L < N
        j_start2 = max(0, L)
        j_end2 = min(N, L + N)
        if j_start2 < j_end2:
            idx2 = np.arange(j_start2, j_end2) - L
            grad_h[j_start2:j_end2] -= w * h[idx2]
    
    return grad_h


def _local_polish(h, max_iters=20, step_sizes=None):
    """
    Vectorized coordinate-descent polish with pair moves.
    For each coordinate i, try moving h[i] by +/-d and compensating on the
    best other coordinate j.  Uses vectorized evaluation.
    """
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    n = len(best_h)
    
    if step_sizes is None:
        step_sizes = [0.01, 0.02, 0.05, 0.1, 0.005]
    
    for it in range(max_iters):
        improved = False
        
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
                    
                    # Find best j to compensate
                    candidate_js = np.arange(n)
                    candidate_js = candidate_js[candidate_js != i]
                    new_vals_j = best_h[candidate_js] + needed
                    valid = (new_vals_j >= -1e-12) & (new_vals_j <= 1 + 1e-12)
                    if not np.any(valid):
                        continue
                    
                    # Try the best few candidates (those closest to bounds)
                    # Just try all valid ones but vectorize the correlation
                    best_new_val = best_val
                    best_j = -1
                    best_new_h = None
                    
                    for j in candidate_js[valid]:
                        new_val_j = np.clip(best_h[j] + needed, 0, 1)
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
    
    # Phase 1: Differential evolution over hybrid parameters (10 control + 8 basis)
    n_ctrl = 10
    n_basis = 8
    n_params = n_ctrl + n_basis
    bounds = [(0.0, 1.0)] * n_ctrl + [(-0.3, 0.3)] * n_basis
    
    def de_objective(params):
        h = _construct_hybrid(params)
        h = _project_to_feasible(h)
        val, _, _ = _smooth_max_correlation(h, lam=100.0) if False else (_smooth_max_correlation(h, 100.0), None, None)
        return _smooth_max_correlation(h, 100.0)
    
    de_deadline = start_time + 45.0
    
    try:
        def timed_de_objective(params):
            if time.time() > de_deadline:
                raise TimeoutError()
            h = _construct_hybrid(params)
            h = _project_to_feasible(h)
            return _smooth_max_correlation(h, 100.0)
        
        result = differential_evolution(
            timed_de_objective,
            bounds=bounds,
            maxiter=80,
            popsize=15,
            tol=1e-8,
            mutation=(0.3, 1.0),
            recombination=0.7,
            seed=42,
            polish=True,
            init='sobol'
        )
        
        if result.x is not None:
            h_de = _project_to_feasible(_construct_hybrid(result.x))
            val_de = _correlation_value(h_de)
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
    except TimeoutError:
        pass
    except Exception:
        pass
    
    # Phase 2: Multi-start smooth optimization from diverse starting points
    rng = np.random.default_rng(123)
    
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
    for block_size in [4, 5, 10, 20]:
        h_block = np.zeros(N)
        for i in range(N):
            h_block[i] = 1.0 if (i // block_size) % 2 == 0 else 0.0
        starts.append(_project_to_feasible(h_block))
    
    # Sine waves
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        amp = min(0.4, 0.5 / freq)
        h_sin = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
        starts.append(_project_to_feasible(h_sin))
    
    # Triangular waves
    for period in [10, 20, 25, 50]:
        h_tri = np.zeros(N)
        for i in range(N):
            t = (i % period) / period
            if t < 0.5:
                h_tri[i] = 2 * t
            else:
                h_tri[i] = 2 - 2 * t
        starts.append(_project_to_feasible(h_tri))
    
    # Bimodal: two bumps
    for center in [20, 30, 40]:
        h_bi = 0.5 * np.ones(N)
        for i in range(N):
            d1 = abs(i - center)
            d2 = abs(i - (N - 1 - center))
            h_bi[i] = 0.5 + 0.4 * np.exp(-d1**2 / 100) + 0.4 * np.exp(-d2**2 / 100)
        starts.append(_project_to_feasible(h_bi))
    
    slsqp_deadline = start_time + 115.0
    
    for idx, h_start in enumerate(starts):
        if time.time() > slsqp_deadline - 10.0:
            break
        
        h_start = _project_to_feasible(h_start)
        current_h = h_start.copy()
        
        # Stage 1: Smooth optimization with increasing lambda
        for lam in [30.0, 100.0, 300.0]:
            if time.time() > slsqp_deadline - 8.0:
                break
            
            def smooth_obj(x, lam=lam):
                return _smooth_max_correlation(x, lam)
            
            def smooth_grad(x, lam=lam):
                return _smooth_grad(x, lam)
            
            try:
                res = minimize(
                    smooth_obj,
                    smooth_grad,
                    current_h,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 150, 'gtol': 1e-10, 'ftol': 1e-12}
                )
                h_opt = _project_to_feasible(res.x)
                val = _correlation_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                current_h = h_opt
            except Exception:
                pass
        
        # Stage 2: Epigraph-based SLSQP for smoother optimization
        if time.time() > slsqp_deadline - 5.0:
            break
        
        try:
            # Epigraph: minimize t subject to corr[p] <= t for all p,
            # sum(h) = N/2, 0 <= h <= 1
            # Variables: [h(100), t(1)] = 101 vars
            def epigraph_obj(x):
                return x[N]  # t
            
            def epigraph_obj_grad(x):
                g = np.zeros(N + 1)
                g[N] = 1.0
                return g
            
            # Constraints: t - corr[p] >= 0 for each p
            # That's 199 inequality constraints
            def make_ineq(p):
                def con(x):
                    h = x[:N]
                    t = x[N]
                    corr = np.correlate(h, 1 - h, mode="full")
                    return t - corr[p]
                return con
            
            ineq_constraints = [{'type': 'ineq', 'fun': make_ineq(p)} for p in range(2*N-1)]
            eq_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x[:N]) - N/2.0}
            all_constraints = ineq_constraints + [eq_constraint]
            
            x0 = np.concatenate([current_h, [_correlation_value(current_h) + 0.01]])
            bounds_ep = [(0.0, 1.0)] * N + [(-1.0, 1.0)]
            
            res_ep = minimize(
                epigraph_obj,
                epigraph_obj_grad,
                x0,
                method='SLSQP',
                bounds=bounds_ep,
                constraints=all_constraints,
                options={'maxiter': 300, 'ftol': 1e-14}
            )
            if res_ep.x is not None:
                h_ep = _project_to_feasible(res_ep.x[:N])
                val_ep = _correlation_value(h_ep)
                if val_ep < best_val:
                    best_val = val_ep
                    best_h = h_ep.copy()
                current_h = h_ep
        except Exception:
            pass
        
        # Stage 3: Exact SLSQP polish
        if time.time() > slsqp_deadline - 2.0:
            break
        
        try:
            def exact_obj2(x):
                corr = np.correlate(x, 1 - x, mode="full")
                return np.max(corr)
            
            res_final = minimize(
                exact_obj2,
                current_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 200, 'ftol': 1e-16}
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
            best_h = _local_polish(best_h, max_iters=30)
            best_val = _correlation_value(best_h)
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish with epigraph + coordinate descent
    if time.time() < deadline - 3.0:
        try:
            # Epigraph polish
            def epigraph_obj4(x):
                return x[N]
            
            def epigraph_obj_grad4(x):
                g = np.zeros(N + 1)
                g[N] = 1.0
                return g
            
            def make_ineq4(p):
                def con(x):
                    h = x[:N]
                    t = x[N]
                    corr = np.correlate(h, 1 - h, mode="full")
                    return t - corr[p]
                return con
            
            ineq_constraints4 = [{'type': 'ineq', 'fun': make_ineq4(p)} for p in range(2*N-1)]
            eq_constraint4 = {'type': 'eq', 'fun': lambda x: np.sum(x[:N]) - N/2.0}
            all_constraints4 = ineq_constraints4 + [eq_constraint4]
            
            x0_4 = np.concatenate([best_h, [_correlation_value(best_h) + 0.005]])
            bounds_ep4 = [(0.0, 1.0)] * N + [(-1.0, 1.0)]
            
            res_ep4 = minimize(
                epigraph_obj4,
                epigraph_obj_grad4,
                x0_4,
                method='SLSQP',
                bounds=bounds_ep4,
                constraints=all_constraints4,
                options={'maxiter': 400, 'ftol': 1e-16}
            )
            if res_ep4.x is not None:
                h_ep4 = _project_to_feasible(res_ep4.x[:N])
                val_ep4 = _correlation_value(h_ep4)
                if val_ep4 < best_val:
                    best_val = val_ep4
                    best_h = h_ep4.copy()
        except Exception:
            pass
        
        # Coordinate descent final polish
        if time.time() < deadline - 1.0:
            try:
                best_h = _local_polish(best_h, max_iters=10, step_sizes=[0.005, 0.01, 0.02])
                best_val = _correlation_value(best_h)
            except Exception:
                pass
    
    # Final projection and validation
    best_h = _project_to_feasible(best_h)
    final_val = _correlation_value(best_h)
    
    return best_h

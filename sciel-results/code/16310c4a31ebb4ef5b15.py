import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_laguerre
import time

def _compute_c4_bound(roots):
    """Compute the C4 upper bound for given double-root positions."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        return np.inf
    
    # Use Gauss-Laguerre quadrature with enough points
    n_quad = 300
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Compute P(x) = prod (x - r_i)^2
    # Use log-space for numerical stability
    diff = x_quad[:, np.newaxis] - roots[np.newaxis, :]
    abs_diff = np.abs(diff)
    
    # Handle zeros carefully
    log_P = 2.0 * np.sum(np.log(abs_diff + 1e-300), axis=1)
    
    P = np.exp(log_P)
    P2 = P * P
    
    int_P = np.sum(w_quad * P)
    int_P2 = np.sum(w_quad * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return np.inf
    
    c4 = int_P2 / (int_P * int_P)
    return c4

def _log_roots_from_params(params):
    """Convert parameters to log-roots for smooth optimization."""
    log_roots = params
    roots = np.exp(log_roots)
    if np.any(roots <= 0) or np.any(roots > 1e8):
        return None
    return roots

def _objective_log(params):
    """Objective in log-space."""
    roots = _log_roots_from_params(params)
    if roots is None:
        return 1e10
    return _compute_c4_bound(roots)

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    # Initialize with Laguerre roots as a good starting point
    x_lag, _ = roots_laguerre(12)
    x_lag = np.sort(x_lag)
    log_x_lag = np.log(x_lag)
    
    best_roots = x_lag.copy()
    best_score = _compute_c4_bound(best_roots)
    
    if time.time() > deadline - 100:
        return [float(xi) for xi in best_roots]
    
    # Phase 1: Local optimization in log-space from Laguerre roots
    try:
        result1 = minimize(
            _objective_log,
            log_x_lag,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
        )
        if result1.fun < best_score:
            cand_roots = np.exp(result1.x)
            if np.all(cand_roots > 0) and np.all(cand_roots < 1e8):
                cand_score = _compute_c4_bound(cand_roots)
                if cand_score < best_score:
                    best_score = cand_score
                    best_roots = cand_roots.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 80:
        return [float(xi) for xi in best_roots]
    
    # Phase 2: Differential evolution in log-space for global search
    # Use bounds based on reasonable ranges
    log_lower = np.log(0.01)
    log_upper = np.log(100.0)
    bounds_de = [(log_lower, log_upper)] * 12
    
    # Generate diverse initial population
    rng = np.random.default_rng(42)
    
    # Create diverse starting points
    def _diverse_init(popsize, dim):
        pop = np.zeros((popsize, dim))
        # Some near Laguerre roots
        for i in range(min(popsize // 3, 5)):
            pop[i] = log_x_lag + 0.1 * rng.standard_normal(dim)
        # Some geometric progressions
        for i in range(popsize // 3, 2 * popsize // 3):
            ratio = np.exp(rng.uniform(-0.5, 0.5))
            base = np.exp(rng.uniform(-2, 2))
            idx = np.arange(dim)
            pop[i] = np.log(base) + idx * np.log(ratio)
        # Some random
        for i in range(2 * popsize // 3, popsize):
            pop[i] = rng.uniform(log_lower, log_upper, dim)
        return pop
    
    try:
        result_de = differential_evolution(
            _objective_log,
            bounds_de,
            maxiter=100,
            popsize=20,
            tol=1e-12,
            seed=42,
            polish=True,
            init=_diverse_init(20, 12),
            time_limit=max(10, deadline - time.time() - 30)
        )
        if result_de.fun < best_score:
            cand_roots = np.exp(result_de.x)
            if np.all(cand_roots > 0) and np.all(cand_roots < 1e8):
                cand_score = _compute_c4_bound(cand_roots)
                if cand_score < best_score:
                    best_score = cand_score
                    best_roots = cand_roots.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 20:
        return [float(xi) for xi in best_roots]
    
    # Phase 3: Final polish from best result
    try:
        log_best = np.log(best_roots)
        result_final = minimize(
            _objective_log,
            log_best,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16}
        )
        if result_final.fun < best_score:
            cand_roots = np.exp(result_final.x)
            if np.all(cand_roots > 0) and np.all(cand_roots < 1e8):
                cand_score = _compute_c4_bound(cand_roots)
                if cand_score < best_score:
                    best_score = cand_score
                    best_roots = cand_roots.copy()
    except Exception:
        pass
    
    return [float(xi) for xi in best_roots]

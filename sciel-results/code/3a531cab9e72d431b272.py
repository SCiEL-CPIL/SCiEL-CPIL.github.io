# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _shannon_entropy(probs):
    """Compute Shannon entropy in bits, ignoring zero probabilities."""
    probs = probs[probs > 1e-15]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

def _project_entropy(P, slope):
    """Compute entropy of projection x + slope*y."""
    n = P.shape[0]
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')  # X[y,x]=x, Y[y,x]=y
    vals = X + slope * Y
    unique_vals, inverse = np.unique(vals, return_inverse=True)
    pooled = np.zeros(len(unique_vals))
    np.add.at(pooled, inverse, P)
    return _shannon_entropy(pooled)

def _soft_max(vals, temperature=0.01):
    """Soft max using log-sum-exp for smoothness."""
    m = np.max(vals)
    return m + temperature * np.log(np.sum(np.exp((vals - m) / temperature)))

def _objective(params, temperature=0.01):
    """Objective to minimize: -D + M_smooth (we want to maximize D/M, use D - M as proxy)."""
    P = np.exp(params)
    P = P / P.sum()
    D = _project_entropy(P, -1)
    H0 = _project_entropy(P, 0)
    H1 = _project_entropy(P, 1)
    H2 = _project_entropy(P, 2)
    M_smooth = _soft_max(np.array([H0, H1, H2]), temperature)
    # We want to maximize D - M, so minimize -(D - M)
    return -(D - M_smooth)

def _make_initial_patterns():
    """Create structured initial patterns for optimization."""
    n = 10
    patterns = []
    
    # Uniform
    patterns.append(np.ones((n, n)) / (n * n))
    
    # Diagonal (x-y constant): high concentration in x-y bin
    for d in range(-4, 5):
        P = np.zeros((n, n))
        for y in range(n):
            x = y + d
            if 0 <= x < n:
                P[y, x] = 1.0
        if P.sum() > 0:
            P /= P.sum()
            patterns.append(P.copy())
    
    # Anti-diagonal (x+y constant)
    for s in range(0, 19):
        P = np.zeros((n, n))
        for y in range(n):
            x = s - y
            if 0 <= x < n:
                P[y, x] = 1.0
        if P.sum() > 0:
            P /= P.sum()
            patterns.append(P.copy())
    
    # Horizontal lines (y constant) - high x entropy
    for y in range(n):
        P = np.zeros((n, n))
        P[y, :] = 1.0 / n
        patterns.append(P.copy())
    
    # Vertical lines (x constant)
    for x in range(n):
        P = np.zeros((n, n))
        P[:, x] = 1.0 / n
        patterns.append(P.copy())
    
    # Two-line combinations
    for y1 in range(0, n, 3):
        for y2 in range(y1+1, n, 3):
            P = np.zeros((n, n))
            P[y1, :] = 0.5 / n
            P[y2, :] = 0.5 / n
            patterns.append(P.copy())
    
    return patterns

def _evaluate(P):
    """Evaluate the actual score D/M for a distribution P."""
    P = np.clip(P, 0, None)
    total = P.sum()
    if total <= 1e-9:
        return 0.0
    P = P / total
    D = _project_entropy(P, -1)
    H0 = _project_entropy(P, 0)
    H1 = _project_entropy(P, 1)
    H2 = _project_entropy(P, 2)
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 0.0
    return D / M

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    deadline = start_time + 600  # 600 second budget with safety margin
    
    best_P = np.ones((n, n)) / (n * n)
    best_score = _evaluate(best_P)
    
    # Get initial patterns
    patterns = _make_initial_patterns()
    
    # Evaluate all patterns
    for P in patterns:
        score = _evaluate(P)
        if score > best_score:
            best_score = score
            best_P = P.copy()
    
    # Check time
    if time.time() > deadline - 30:
        return best_P
    
    # Convert to log-space parameters for optimization
    # Use a soft lower bound to keep probabilities positive
    eps = 1e-6
    
    def _log_params_from_P(P):
        P_safe = np.clip(P, eps, None)
        P_safe = P_safe / P_safe.sum()
        return np.log(P_safe)
    
    def _P_from_log_params(log_params):
        P = np.exp(log_params)
        P = P / P.sum()
        return P
    
    # Try optimization from multiple starting points
    start_points = [best_P]
    # Add some random perturbations of best
    for _ in range(3):
        P_pert = best_P * (1 + 0.1 * np.random.randn(n, n))
        P_pert = np.clip(P_pert, 0, None)
        if P_pert.sum() > 0:
            P_pert /= P_pert.sum()
            start_points.append(P_pert)
    
    for init_P in start_points:
        if time.time() > deadline - 60:
            break
        
        log_params = _log_params_from_P(init_P)
        
        # SLSQP with soft-max surrogate
        try:
            result = minimize(
                _objective,
                log_params,
                method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
            )
            
            P_opt = _P_from_log_params(result.x)
            score = _evaluate(P_opt)
            if score > best_score:
                best_score = score
                best_P = P_opt.copy()
        except Exception:
            pass
    
    # Nelder-Mead polish for non-smooth max
    if time.time() > deadline - 30:
        pass
    else:
        try:
            def _nm_objective(params):
                P = np.exp(params)
                P = P / P.sum()
                D = _project_entropy(P, -1)
                H0 = _project_entropy(P, 0)
                H1 = _project_entropy(P, 1)
                H2 = _project_entropy(P, 2)
                M = max(H0, H1, H2)
                if M <= 1e-9:
                    return 0.0
                return -(D / M)
            
            log_params = _log_params_from_P(best_P)
            result = minimize(
                _nm_objective,
                log_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False}
            )
            
            P_opt = _P_from_log_params(result.x)
            score = _evaluate(P_opt)
            if score > best_score:
                best_score = score
                best_P = P_opt.copy()
        except Exception:
            pass
    
    # Final high-precision SLSQP polish
    if time.time() > deadline - 10:
        pass
    else:
        try:
            log_params = _log_params_from_P(best_P)
            result = minimize(
                _objective,
                log_params,
                method='SLSQP',
                options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
            )
            
            P_opt = _P_from_log_params(result.x)
            score = _evaluate(P_opt)
            if score > best_score:
                best_score = score
                best_P = P_opt.copy()
        except Exception:
            pass
    
    # Final cleanup: clip and normalize
    best_P = np.clip(best_P, 0, None)
    total = best_P.sum()
    if total > 1e-9:
        best_P = best_P / total
    else:
        best_P = np.ones((n, n)) / (n * n)
    
    return best_P

# EVOLVE-BLOCK-END

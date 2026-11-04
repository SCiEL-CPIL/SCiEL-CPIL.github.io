import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import roots_genlaguerre
import time

def _compute_c4_bound(roots, n_quad=200):
    """Compute the C4 upper bound: integral(P^2 * e^{-x}) / (integral(P * e^{-x}))^2."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
        return np.inf
    
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # P(x) = prod_{i=0}^{11} (x - r_i)^2
    # Use log-sum for numerical stability
    diff = x_quad[:, np.newaxis] - roots[np.newaxis, :]
    log_abs_diff = np.log(np.abs(diff) + 1e-300)
    log_P = 2.0 * np.sum(log_abs_diff, axis=1)
    
    # Clip to avoid overflow
    log_P = np.clip(log_P, -700, 700)
    P = np.exp(log_P)
    P2 = P * P
    
    int_P = np.sum(w_quad * P)
    int_P2 = np.sum(w_quad * P2)
    
    if int_P <= 0 or not np.isfinite(int_P) or not np.isfinite(int_P2):
        return np.inf
    
    c4 = int_P2 / (int_P * int_P)
    return c4

def construct():
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    def roots_from_beta(beta):
        """Get 12 double-root locations as nodes of generalized Laguerre polynomial L_11^(beta)."""
        # roots_genlaguerre(n, alpha) gives n roots of L_n^(alpha)(x)
        # We need 12 roots, so n=12, alpha=beta
        try:
            roots = roots_genlaguerre(12, beta)
            roots = np.sort(roots)
            if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
                return None
            return roots
        except Exception:
            return None
    
    def objective_beta(beta_val):
        """Objective: C4 bound as function of beta."""
        beta = float(beta_val[0])
        roots = roots_from_beta(beta)
        if roots is None:
            return 1e10
        c4 = _compute_c4_bound(roots)
        if not np.isfinite(c4):
            return 1e10
        return c4
    
    # Phase 1: Coarse scan over beta to find promising region
    best_c4 = np.inf
    best_beta = None
    
    # Beta must be > -1 for generalized Laguerre roots to be positive
    # Scan a reasonable range
    beta_scan = np.linspace(-0.5, 10.0, 100)
    
    for b in beta_scan:
        if time.time() > deadline - 120:
            break
        roots = roots_from_beta(b)
        if roots is None:
            continue
        c4 = _compute_c4_bound(roots)
        if c4 < best_c4 and np.isfinite(c4):
            best_c4 = c4
            best_beta = b
    
    # Phase 2: Local optimization around best beta
    if best_beta is not None:
        # Try multiple local optimizations from nearby starting points
        n_local = 20
        for i in range(n_local):
            if time.time() > deadline - 60:
                break
            
            if i == 0:
                beta0 = best_beta
            else:
                beta0 = best_beta + np.random.uniform(-2.0, 2.0)
                if beta0 < -0.9:
                    beta0 = -0.5
            
            try:
                result = minimize(objective_beta, np.array([beta0]), 
                                 method='L-BFGS-B',
                                 bounds=[(-0.9, 20.0)],
                                 options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
                
                if result.fun < best_c4 and np.isfinite(result.fun):
                    best_c4 = result.fun
                    best_beta = float(result.x[0])
            except Exception:
                pass
        
        # Phase 3: High-precision polish with tighter tolerances
        if time.time() < deadline - 30:
            try:
                result = minimize(objective_beta, np.array([best_beta]),
                                 method='L-BFGS-B',
                                 bounds=[(-0.9, 20.0)],
                                 options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-15})
                if result.fun < best_c4 and np.isfinite(result.fun):
                    best_c4 = result.fun
                    best_beta = float(result.x[0])
            except Exception:
                pass
    
    # Get final roots
    if best_beta is not None:
        roots = roots_from_beta(best_beta)
        if roots is not None and len(roots) == 12:
            return [float(r) for r in roots]
    
    # Fallback: use standard Laguerre nodes (beta=0)
    try:
        roots = roots_genlaguerre(12, 0.0)
        roots = np.sort(roots)
        if np.all(roots > 0):
            return [float(r) for r in roots]
    except Exception:
        pass
    
    # Final fallback
    return [4.0 * (i + 1) for i in range(12)]

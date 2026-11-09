# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximize the arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    D = H_{-1} (entropy of x - y), M = max(H_0, H_1, H_2).
    Uses a flexible parametric ansatz with polynomial log-weights and multi-stage optimization.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin for 630s limit

    # Precompute grid indices
    X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing='xy')
    Xc = X - (n - 1) / 2.0
    Yc = Y - (n - 1) / 2.0

    # Precompute value matrices
    vals_minus1 = X - Y       # x - y, range -9..9 (19 values)
    vals_0 = X                # x, range 0..9 (10 values)
    vals_1 = X + Y            # x + y, range 0..18 (19 values)
    vals_2 = X + 2 * Y        # x + 2y, range 0..27 (28 values)

    # Precompute bin indices for fast pooling
    # For each slope, map each cell to its bin index
    _, inv_minus1 = np.unique(vals_minus1.ravel(), return_inverse=True)
    _, inv_0 = np.unique(vals_0.ravel(), return_inverse=True)
    _, inv_1 = np.unique(vals_1.ravel(), return_inverse=True)
    _, inv_2 = np.unique(vals_2.ravel(), return_inverse=True)
    
    n_bins_minus1 = len(np.unique(vals_minus1))
    n_bins_0 = len(np.unique(vals_0))
    n_bins_1 = len(np.unique(vals_1))
    n_bins_2 = len(np.unique(vals_2))

    def fast_entropy(p_flat, inv, n_bins):
        """Compute Shannon entropy using precomputed bin indices."""
        counts = np.bincount(inv, weights=p_flat, minlength=n_bins)
        mask = counts > 1e-15
        if not np.any(mask):
            return 0.0
        c = counts[mask]
        return -np.sum(c * np.log2(c))

    def compute_ratio_from_logP(logP):
        """Compute D/M from log-probabilities (unnormalized)."""
        logP = logP - np.max(logP)
        P = np.exp(logP)
        total = P.sum()
        if total <= 1e-15:
            return 0.0
        P = P / total
        p_flat = P.ravel()
        
        D = fast_entropy(p_flat, inv_minus1, n_bins_minus1)
        H0 = fast_entropy(p_flat, inv_0, n_bins_0)
        H1 = fast_entropy(p_flat, inv_1, n_bins_1)
        H2 = fast_entropy(p_flat, inv_2, n_bins_2)
        
        M = max(H0, H1, H2)
        if M < 1e-9:
            return 0.0
        return D / M

    def logP_from_params(params):
        """
        Convert parameters to log-probabilities.
        params = [a0, a1, a2, b0, b1, b2, c0, c1, c2]
        logP = a0 + a1*x + a2*x^2 + b0 + b1*y + b2*y^2 + c0 + c1*x*y + c2*x^2*y + c3*x*y^2
        Simplified: quadratic in x and y with cross term
        """
        a0, a1, a2, b0, b1, b2, c0, c1, c2 = params
        return (a0 + a1*Xc + a2*Xc**2 + 
                b0 + b1*Yc + b2*Yc**2 + 
                c0 + c1*Xc*Yc + c2*Xc**2*Yc + c3*Xc*Yc**2) if len(params) > 9 else \
               (a0 + a1*Xc + a2*Xc**2 + 
                b0 + b1*Yc + b2*Yc**2 + 
                c0 + c1*Xc*Yc + c2*Xc**2*Yc)

    def objective(params):
        logP = logP_from_params(params)
        ratio = compute_ratio_from_logP(logP)
        if ratio <= 0:
            return 1e6
        return -ratio

    best_score = -1.0
    best_P = None

    # Strategy 1: 3-parameter Gaussian (fast baseline)
    def logP_gaussian(params):
        alpha, beta, gamma = params
        return -alpha * Xc**2 - beta * Yc**2 - gamma * Xc * Yc

    def objective_gaussian(params):
        logP = logP_gaussian(params)
        ratio = compute_ratio_from_logP(logP)
        if ratio <= 0:
            return 1e6
        return -ratio

    bounds_gauss = [(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)]
    
    # Run DE for Gaussian
    try:
        res = differential_evolution(
            objective_gaussian, 
            bounds_gauss, 
            maxiter=80, 
            popsize=12, 
            tol=1e-4, 
            seed=42,
            polish=False
        )
        if res.fun < 1e5:
            score = -res.fun
            if score > best_score:
                best_score = score
                logP = logP_gaussian(res.x)
                logP = logP - np.max(logP)
                best_P = np.exp(logP)
                best_P /= best_P.sum()
    except Exception:
        pass

    # Strategy 2: 6-parameter quadratic
    def logP_quadratic(params):
        a0, a1, a2, b0, b1, b2 = params
        return a0 + a1*Xc + a2*Xc**2 + b0 + b1*Yc + b2*Yc**2

    def objective_quadratic(params):
        logP = logP_quadratic(params)
        ratio = compute_ratio_from_logP(logP)
        if ratio <= 0:
            return 1e6
        return -ratio

    bounds_quad = [(-2.0, 2.0)] * 6
    
    if time.time() < deadline - 60:
        try:
            res = differential_evolution(
                objective_quadratic, 
                bounds_quad, 
                maxiter=60, 
                popsize=12, 
                tol=1e-4, 
                seed=123,
                polish=False
            )
            if res.fun < 1e5:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    logP = logP_quadratic(res.x)
                    logP = logP - np.max(logP)
                    best_P = np.exp(logP)
                    best_P /= best_P.sum()
        except Exception:
            pass

    # Strategy 3: 9-parameter with cross terms
    def logP_full(params):
        a0, a1, a2, b0, b1, b2, c0, c1, c2 = params
        return (a0 + a1*Xc + a2*Xc**2 + 
                b0 + b1*Yc + b2*Yc**2 + 
                c0 + c1*Xc*Yc + c2*Xc**2*Yc)

    def objective_full(params):
        logP = logP_full(params)
        ratio = compute_ratio_from_logP(logP)
        if ratio <= 0:
            return 1e6
        return -ratio

    bounds_full = [(-2.0, 2.0)] * 9

    if time.time() < deadline - 30:
        try:
            res = differential_evolution(
                objective_full, 
                bounds_full, 
                maxiter=50, 
                popsize=12, 
                tol=1e-4, 
                seed=456,
                polish=False
            )
            if res.fun < 1e5:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    logP = logP_full(res.x)
                    logP = logP - np.max(logP)
                    best_P = np.exp(logP)
                    best_P /= best_P.sum()
        except Exception:
            pass

    # Strategy 4: Direct optimization on diagonal structure
    # P(y,x) depends only on d = x - y (the D-slope value)
    # This maximizes D while keeping M controlled
    # Try: P(y,x) = w[x-y] for weights w over -9..9
    def compute_ratio_from_diag_weights(w):
        """w is array of length 19 for d = -9..9"""
        # Build P from diagonal weights
        P = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                d = x - y  # range -9..9
                P[y, x] = w[d + 9]
        total = P.sum()
        if total <= 1e-15:
            return 0.0
        P = P / total
        p_flat = P.ravel()
        
        D = fast_entropy(p_flat, inv_minus1, n_bins_minus1)
        H0 = fast_entropy(p_flat, inv_0, n_bins_0)
        H1 = fast_entropy(p_flat, inv_1, n_bins_1)
        H2 = fast_entropy(p_flat, inv_2, n_bins_2)
        
        M = max(H0, H1, H2)
        if M < 1e-9:
            return 0.0
        return D / M

    def objective_diag(params):
        """params are log-weights for 19 diagonal values"""
        w = np.exp(params - np.max(params))
        ratio = compute_ratio_from_diag_weights(w)
        if ratio <= 0:
            return 1e6
        return -ratio

    bounds_diag = [(-5.0, 5.0)] * 19

    if time.time() < deadline - 20:
        try:
            res = differential_evolution(
                objective_diag, 
                bounds_diag, 
                maxiter=40, 
                popsize=10, 
                tol=1e-4, 
                seed=789,
                polish=False
            )
            if res.fun < 1e5:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    w = np.exp(res.x - np.max(res.x))
                    P = np.zeros((n, n))
                    for y in range(n):
                        for x in range(n):
                            d = x - y
                            P[y, x] = w[d + 9]
                    total = P.sum()
                    if total > 1e-15:
                        best_P = P / total
        except Exception:
            pass

    # Fallback
    if best_P is None:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P = np.clip(best_P, 0, None)
        total = best_P.sum()
        if total <= 1e-9:
            best_P = np.ones((n, n)) / (n * n)
        else:
            best_P /= total

    return best_P

# EVOLVE-BLOCK-END

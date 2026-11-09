# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 600.0  # safety margin before 630s kill

def _shannon_entropy(p):
    """Shannon entropy in bits, ignoring zeros."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-15]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _compute_D_M(P):
    """Compute D and M for distribution P[y,x] of shape (10,10)."""
    n = 10
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')  # X[y,x]=x, Y[y,x]=y
    
    # D: entropy of x - y (values in -9..9, 19 bins)
    diff = X - Y  # shape (n,n)
    diff_flat = diff.ravel()
    P_flat = P.ravel()
    # bin indices: diff + 9 -> 0..18
    diff_probs = np.zeros(2 * n - 1)
    np.add.at(diff_probs, diff_flat + (n - 1), P_flat)
    D = _shannon_entropy(diff_probs)
    
    # H_0: entropy of x (10 bins)
    x_flat = X.ravel()
    x_probs = np.zeros(n)
    np.add.at(x_probs, x_flat, P_flat)
    H0 = _shannon_entropy(x_probs)
    
    # H_1: entropy of x + y (values 0..18, 19 bins)
    s1 = X + Y
    s1_flat = s1.ravel()
    s1_probs = np.zeros(2 * n - 1)
    np.add.at(s1_probs, s1_flat, P_flat)
    H1 = _shannon_entropy(s1_probs)
    
    # H_2: entropy of x + 2y (values 0..27, 28 bins)
    s2 = X + 2 * Y
    s2_flat = s2.ravel()
    s2_probs = np.zeros(2 * n + n - 1)  # 0..27 -> 28 bins
    np.add.at(s2_probs, s2_flat, P_flat)
    H2 = _shannon_entropy(s2_probs)
    
    M = max(H0, H1, H2)
    return D, M


def _value(P):
    """Compute D/M ratio, return 0 if invalid."""
    D, M = _compute_D_M(P)
    if M <= 1e-9:
        return 0.0
    return D / M


def _build_P_from_params(params):
    """Build P from parameter vector. params encodes weights for a set of basis patterns."""
    n = 10
    # We'll use a flexible parameterization: 
    # params[0:100] = raw weights for each cell (will be normalized)
    # This is the most flexible approach
    w = np.exp(params)  # ensure positivity
    P = w.reshape(n, n)
    total = P.sum()
    if total < 1e-15:
        return np.ones((n, n)) / (n * n)
    P = P / total
    return P


def _neg_value(params):
    """Negative value for minimization."""
    P = _build_P_from_params(params)
    return -_value(P)


def _try_pattern(pattern_type, **kwargs):
    """Try a specific parametric pattern and return best P found."""
    n = 10
    best_val = 0.0
    best_P = np.ones((n, n)) / (n * n)
    
    if pattern_type == 'diagonal_concentrate':
        # Put mass on anti-diagonals (x-y = const) but weight them unevenly
        # to maximize entropy of x-y while keeping x, x+y, x+2y low
        # Actually, if we put mass on a single anti-diagonal x-y=c, then D=0
        # We need to spread across multiple anti-diagonals
        
        # Try: mass proportional to exp(t * (x-y)^2) normalized
        for t in np.linspace(-2, 2, 50):
            X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing='xy')
            d = X - Y
            w = np.exp(t * d * d)
            P = w / w.sum()
            v = _value(P)
            if v > best_val:
                best_val = v
                best_P = P.copy()
    
    elif pattern_type == 'corner_concentrate':
        # Concentrate near corners to make x, x+y, x+2y have low entropy
        # while x-y still has some spread
        for alpha in np.linspace(0.5, 5.0, 30):
            X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing='xy')
            # Weight by distance from corners
            d00 = np.sqrt(X**2 + Y**2)
            d10 = np.sqrt((X-9)**2 + Y**2)
            d01 = np.sqrt(X**2 + (Y-9)**2)
            d11 = np.sqrt((X-9)**2 + (Y-9)**2)
            d_min = np.minimum(np.minimum(d00, d10), np.minimum(d01, d11))
            w = np.exp(-alpha * d_min)
            P = w / w.sum()
            v = _value(P)
            if v > best_val:
                best_val = v
                best_P = P.copy()
    
    elif pattern_type == 'line_concentrate':
        # Concentrate on lines where one projection is constant
        # e.g., x = c (vertical line): H0=0, but D, H1, H2 all ~log2(10)
        # Try mixing a few lines
        for c1 in range(n):
            for c2 in range(n):
                if c1 == c2:
                    continue
                for w1 in np.linspace(0.1, 0.9, 10):
                    X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing='xy')
                    w = np.zeros((n, n))
                    w[:, c1] = w1
                    w[:, c2] = 1 - w1
                    P = w / w.sum()
                    v = _value(P)
                    if v > best_val:
                        best_val = v
                        best_P = P.copy()
    
    elif pattern_type == 'sparse_cells':
        # Try putting mass on a small number of strategically chosen cells
        # Use random sampling of cell subsets
        np.random.seed(42)
        for trial in range(2000):
            # Pick 3-7 random cells
            k = np.random.randint(3, 8)
            cells = np.random.choice(n * n, size=k, replace=False)
            # Random weights
            w = np.random.dirichlet(np.ones(k))
            P = np.zeros((n, n))
            for idx, wt in zip(cells, w):
                P[idx // n, idx % n] = wt
            v = _value(P)
            if v > best_val:
                best_val = v
                best_P = P.copy()
    
    return best_P, best_val


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Returns P[y, x] for x, y in 0..9, normalized to sum 1.
    """
    n = 10
    best_P = np.ones((n, n)) / (n * n)
    best_val = _value(best_P)
    
    # Try various pattern families
    for ptype in ['diagonal_concentrate', 'corner_concentrate', 'line_concentrate', 'sparse_cells']:
        if time.time() > _DEADLINE - 60:
            break
        P, v = _try_pattern(ptype)
        if v > best_val:
            best_val = v
            best_P = P.copy()
    
    # Local optimization using scipy on the best pattern found
    if time.time() < _DEADLINE - 30:
        try:
            from scipy.optimize import minimize
            
            # Parameterize using log-softmax for stability
            # Start from best_P, convert to log-weights
            P_init = best_P.copy()
            P_init[P_init < 1e-15] = 1e-15
            P_init = P_init / P_init.sum()
            log_w = np.log(P_init.ravel())
            
            def _neg_log_value(log_w_params):
                w = np.exp(log_w_params - np.max(log_w_params))  # stability
                P = w.reshape(n, n)
                total = P.sum()
                if total < 1e-15:
                    return 0.0
                P = P / total
                return -_value(P)
            
            # Multiple restarts
            for restart in range(5):
                if time.time() > _DEADLINE - 30:
                    break
                if restart == 0:
                    x0 = log_w.copy()
                else:
                    x0 = log_w + np.random.randn(n * n) * 0.1
                
                res = minimize(
                    _neg_log_value, x0,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if res.fun < -best_val:
                    w = np.exp(res.x - np.max(res.x))
                    P = w.reshape(n, n)
                    total = P.sum()
                    if total > 1e-15:
                        P = P / total
                        v = _value(P)
                        if v > best_val:
                            best_val = v
                            best_P = P.copy()
        except Exception:
            pass
    
    # Final validation and cleanup
    best_P = np.clip(best_P, 0, None)
    total = best_P.sum()
    if total < 1e-9:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P = best_P / total
    
    return best_P

# EVOLVE-BLOCK-END

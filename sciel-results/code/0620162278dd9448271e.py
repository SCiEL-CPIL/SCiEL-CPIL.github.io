# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly for discrete measure."""
    n = len(support)
    w = weights / weights.sum()
    s = support
    
    # P = sum_{k,l} w_k * w_l * (sum_{i,j: s_i + s_j < 2*s_l - s_k} w_i * w_j)
    # For each (k, l), compute the inner sum
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            # Sum w_i * w_j over pairs where s_i + s_j < threshold
            inner = 0.0
            for i in range(n):
                for j in range(n):
                    if s[i] + s[j] < threshold:
                        inner += w[i] * w[j]
            P += w[k] * w[l] * inner
    return P


def _compute_P_fast(support, weights):
    """Vectorized computation of P."""
    n = len(support)
    w = weights / weights.sum()
    s = support
    
    # Pairwise sums
    s_sum = s[:, None] + s[None, :]  # n x n
    w_outer = w[:, None] * w[None, :]  # n x n
    
    # For each (k, l), threshold = 2*s[l] - s[k]
    # P = sum_{k,l} w[k]*w[l] * sum_{i,j: s_sum[i,j] < 2*s[l]-s[k]} w_outer[i,j]
    
    # Build thresholds matrix: T[k,l] = 2*s[l] - s[k]
    T = 2.0 * s[None, :] - s[:, None]  # n x n, T[k,l] = 2*s[l] - s[k]
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            mask = s_sum < T[k, l]
            inner = np.sum(w_outer[mask])
            P += w[k] * w[l] * inner
    return P


def _objective(weights, support):
    """Negative P for minimization."""
    w = np.abs(weights)
    if np.sum(w) < 1e-15:
        return 1.0
    return -_compute_P_fast(support, w)


def _optimize_weights(support, multi_start=True):
    """Optimize weights for a given support using SLSQP."""
    n = len(support)
    best_P = -1.0
    best_w = None
    
    if multi_start:
        # Multiple starting points
        starts = [
            np.ones(n) / n,
            np.array([0.5, 0.25, 0.25] + [0.0]*(n-3)) if n >= 3 else np.ones(n)/n,
            np.array([0.1, 0.8, 0.1] + [0.0]*(n-3)) if n >= 3 else np.ones(n)/n,
            np.array([0.8, 0.1, 0.1] + [0.0]*(n-3)) if n >= 3 else np.ones(n)/n,
        ]
        # Add some random starts
        rng = np.random.RandomState(42)
        for _ in range(10):
            r = rng.dirichlet(np.ones(n))
            starts.append(r)
    else:
        starts = [np.ones(n) / n]
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    
    for w0 in starts:
        w0 = np.abs(w0)
        w0 = w0 / np.sum(w0)
        try:
            res = minimize(
                _objective, w0, args=(support,),
                method='SLSQP', bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            w_opt = np.abs(res.x)
            w_opt = w_opt / np.sum(w_opt)
            P_val = _compute_P_fast(support, w_opt)
            if P_val > best_P:
                best_P = P_val
                best_w = w_opt
        except Exception:
            continue
    
    if best_w is None:
        best_w = np.ones(n) / n
        best_P = _compute_P_fast(support, best_w)
    
    return best_w, best_P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin
    
    # Try different support configurations
    candidates = [
        # 3-point supports
        np.array([0.0, 1.0, 10.0]),
        np.array([0.0, 1.0, 5.0]),
        np.array([0.0, 1.0, 20.0]),
        np.array([0.0, 0.5, 10.0]),
        np.array([0.0, 2.0, 10.0]),
        np.array([0.0, 1.0, 3.0]),
        np.array([0.0, 1.0, 2.0]),
        # 4-point supports
        np.array([0.0, 1.0, 5.0, 10.0]),
        np.array([0.0, 0.5, 1.0, 10.0]),
        np.array([0.0, 1.0, 2.0, 10.0]),
        # 5-point supports
        np.array([0.0, 0.5, 1.0, 5.0, 10.0]),
        np.array([0.0, 1.0, 2.0, 5.0, 10.0]),
    ]
    
    best_P = -1.0
    best_support = None
    best_weights = None
    
    for support in candidates:
        if time.time() > deadline - 30.0:
            break
        
        w, P = _optimize_weights(support, multi_start=True)
        
        # Validate
        if np.any(support < -1e-9) or np.any(w < -1e-9) or np.sum(w) <= 1e-9:
            continue
        if np.any(np.isnan(support)) or np.any(np.isinf(support)):
            continue
        if np.any(np.isnan(w)) or np.any(np.isinf(w)):
            continue
        
        if P > best_P:
            best_P = P
            best_support = support
            best_weights = w
    
    # Fallback to baseline if nothing worked
    if best_support is None:
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    
    # Final validation
    best_weights = np.abs(best_weights)
    if np.sum(best_weights) < 1e-9:
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)


# EVOLVE-BLOCK-END

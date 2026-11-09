# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

_DEADLINE = time.time() + 570  # 570s deadline, 60s safety margin


def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] exactly in float64.
    P = sum_{k,l} w_k * w_l * C(2*s_l - s_k)
    where C(t) = sum_{i,j} w_i * w_j * I(s_i + s_j < t)
    """
    n = len(support)
    if n == 0:
        return 0.0
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    s = support.astype(np.float64)
    
    # Pair sums and probabilities
    s_col = s.reshape(-1, 1)
    s_row = s.reshape(1, -1)
    pair_sums = s_col + s_row  # (n, n)
    pair_probs = np.outer(w, w)  # (n, n)
    
    # Sort pair_sums for efficient CDF lookup
    ps_flat = pair_sums.ravel()
    pp_flat = pair_probs.ravel()
    order = np.argsort(ps_flat)
    ps_sorted = ps_flat[order]
    pp_sorted = pp_flat[order]
    pp_cumsum = np.concatenate([[0.0], np.cumsum(pp_sorted)])
    
    # For each (k, l): threshold = 2*s_l - s_k
    # C(threshold) = pp_cumsum[searchsorted(ps_sorted, threshold, side='left')]
    
    # Vectorize over (k, l)
    kk, ll = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    kk = kk.ravel()
    ll = ll.ravel()
    
    thresholds = 2.0 * s[ll] - s[kk]  # shape (n^2,)
    idx = np.searchsorted(ps_sorted, thresholds, side='left')
    C_vals = pp_cumsum[idx]  # shape (n^2,)
    
    # P = sum_{k,l} w_k * w_l * C_vals
    P = np.sum(w[kk] * w[ll] * C_vals)
    
    return float(P)


def _decode(params, n):
    """
    Decode raw parameters into valid support and weights.
    params[0:n] -> support (non-negative, sorted)
    params[n:2n] -> weights (positive, normalized)
    """
    s_raw = params[:n]
    w_raw = params[n:2*n]
    
    # Support: softplus to ensure non-negative, then sort
    s = np.log1p(np.exp(s_raw))
    s = np.sort(s)
    
    # Weights: softplus to ensure positive, then normalize
    w = np.log1p(np.exp(w_raw))
    w = w / w.sum()
    
    return s, w


def _negP(params, n):
    """Objective function for optimization."""
    s, w = _decode(params, n)
    return -_compute_P_fast(s, w)


def _differential_evolution_search(n, max_iter=200, timeout=120):
    """Run differential evolution for given n."""
    start_time = time.time()
    
    # Bounds for raw parameters
    bounds = [(-5.0, 5.0)] * (2 * n)
    
    def _obj(params):
        if time.time() - start_time > timeout:
            raise TimeoutError()
        return _negP(params, n)
    
    try:
        result = differential_evolution(
            _obj, bounds,
            maxiter=max_iter,
            tol=1e-8,
            popsize=15,
            mutation=(0.5, 1.0),
            recombination=0.8,
            seed=42,
            polish=True,
            init='sobol'
        )
        if result.success or result.fun < 0:
            s, w = _decode(result.x, n)
            P = _compute_P_fast(s, w)
            return s, w, P
    except Exception:
        pass
    
    return None, None, -1.0


def _nelder_mead_refine(s0, w0, max_iter=500, timeout=30):
    """Refine using Nelder-Mead on decoded parameters."""
    n = len(s0)
    
    # Encode back to raw parameters (inverse of softplus)
    s_raw = np.log(np.expm1(s0)) if np.all(s0 > 0) else np.zeros(n)
    w_raw = np.log(np.expm1(w0 * w0.sum())) if np.all(w0 > 0) else np.zeros(n)
    
    x0 = np.concatenate([s_raw, w_raw])
    
    start_time = time.time()
    
    def _obj(params):
        if time.time() - start_time > timeout:
            raise TimeoutError()
        return _negP(params, n)
    
    try:
        result = minimize(
            _obj, x0, method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        s, w = _decode(result.x, n)
        P = _compute_P_fast(s, w)
        return s, w, P
    except Exception:
        pass
    
    return s0, w0, _compute_P_fast(s0, w0)


def _try_two_point():
    """Two-point distribution {0, L} with optimal p."""
    p_opt = (15.0 - np.sqrt(33.0)) / 16.0
    s = np.array([0.0, 1.0])
    w = np.array([p_opt, 1.0 - p_opt])
    P = _compute_P_fast(s, w)
    return s, w, P


def _try_three_point():
    """Three-point distribution with grid search + refinement."""
    best_s = None
    best_w = None
    best_P = -1.0
    
    # Coarse grid
    for a in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
        for b in [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]:
            if b <= a:
                continue
            for p0 in np.linspace(0.05, 0.85, 18):
                for p1 in np.linspace(0.02, 0.85 - p0, 14):
                    if p1 <= 0:
                        continue
                    p2 = 1.0 - p0 - p1
                    if p2 <= 0:
                        continue
                    s = np.array([0.0, a, b])
                    w = np.array([p0, p1, p2])
                    P = _compute_P_fast(s, w)
                    if P > best_P:
                        best_P = P
                        best_s = s.copy()
                        best_w = w.copy()
    
    if best_s is not None:
        # Refine
        s_ref, w_ref, P_ref = _nelder_mead_refine(best_s, best_w, max_iter=300, timeout=20)
        if P_ref > best_P:
            best_P = P_ref
            best_s = s_ref
            best_w = w_ref
    
    return best_s, best_w, best_P


def _try_four_point():
    """Four-point distribution with structured initialization."""
    best_s = None
    best_w = None
    best_P = -1.0
    
    # Try various geometric-like structures
    for ratio in [1.5, 2.0, 2.5, 3.0, 4.0]:
        for scale in [0.5, 1.0, 2.0]:
            s = np.array([0.0, scale, scale * ratio, scale * ratio * ratio])
            # Try different weight patterns
            for r in [0.6, 0.7, 0.75, 0.8]:
                w = np.array([r**i for i in range(4)])
                w = w / w.sum()
                P = _compute_P_fast(s, w)
                if P > best_P:
                    best_P = P
                    best_s = s.copy()
                    best_w = w.copy()
    
    if best_s is not None:
        s_ref, w_ref, P_ref = _nelder_mead_refine(best_s, best_w, max_iter=400, timeout=25)
        if P_ref > best_P:
            best_P = P_ref
            best_s = s_ref
            best_w = w_ref
    
    return best_s, best_w, best_P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    global _DEADLINE
    
    best_support = None
    best_weights = None
    best_P = -1.0
    
    # Try two-point
    if time.time() < _DEADLINE:
        s2, w2, P2 = _try_two_point()
        if P2 > best_P:
            best_P = P2
            best_support = s2.copy()
            best_weights = w2.copy()
    
    # Try three-point
    if time.time() < _DEADLINE:
        s3, w3, P3 = _try_three_point()
        if s3 is not None and P3 > best_P:
            best_P = P3
            best_support = s3.copy()
            best_weights = w3.copy()
    
    # Try four-point
    if time.time() < _DEADLINE:
        s4, w4, P4 = _try_four_point()
        if s4 is not None and P4 > best_P:
            best_P = P4
            best_support = s4.copy()
            best_weights = w4.copy()
    
    # Differential evolution for n=3
    if time.time() < _DEADLINE - 60:
        s_de, w_de, P_de = _differential_evolution_search(3, max_iter=100, timeout=60)
        if s_de is not None and P_de > best_P:
            best_P = P_de
            best_support = s_de.copy()
            best_weights = w_de.copy()
    
    # Differential evolution for n=4
    if time.time() < _DEADLINE - 40:
        s_de, w_de, P_de = _differential_evolution_search(4, max_iter=80, timeout=50)
        if s_de is not None and P_de > best_P:
            best_P = P_de
            best_support = s_de.copy()
            best_weights = w_de.copy()
    
    # Differential evolution for n=5
    if time.time() < _DEADLINE - 30:
        s_de, w_de, P_de = _differential_evolution_search(5, max_iter=60, timeout=40)
        if s_de is not None and P_de > best_P:
            best_P = P_de
            best_support = s_de.copy()
            best_weights = w_de.copy()
    
    # Final Nelder-Mead refinement on best found
    if time.time() < _DEADLINE - 10 and best_support is not None:
        s_ref, w_ref, P_ref = _nelder_mead_refine(
            best_support, best_weights, max_iter=500, timeout=30
        )
        if P_ref > best_P:
            best_P = P_ref
            best_support = s_ref
            best_weights = w_ref
    
    # If we still haven't found anything valid, use baseline
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([(15.0 - np.sqrt(33.0)) / 16.0, 1.0 - (15.0 - np.sqrt(33.0)) / 16.0])
        best_P = _compute_P_fast(best_support, best_weights)
    
    # Final verification
    P_final = _compute_P_fast(best_support, best_weights)
    
    # Ensure validity
    assert len(best_support) == len(best_weights)
    assert len(best_support) >= 1
    assert not np.any(np.isnan(best_support))
    assert not np.any(np.isinf(best_support))
    assert not np.any(np.isnan(best_weights))
    assert not np.any(np.isinf(best_weights))
    assert np.all(best_support >= -1e-9)
    assert np.all(best_weights >= -1e-9)
    assert np.sum(best_weights) > 1e-9
    
    # Clip any tiny negative values from numerical error
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 1e-15)
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)


# EVOLVE-BLOCK-END

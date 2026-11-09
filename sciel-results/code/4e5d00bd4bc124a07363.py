# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] for given measure."""
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    p = w / total
    n = len(s)
    
    # Pairwise sums and probabilities
    pair_sums = s[:, None] + s[None, :]  # shape (n, n)
    pair_probs = p[:, None] * p[None, :]  # shape (n, n)
    
    # For each (k, l), compute sum of pair_probs where pair_sums < 2*s_l - s_k
    # P = sum_{k,l} p_k * p_l * (sum_{i,j: s_i+s_j < 2*s_l-s_k} p_i*p_j)
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            mask = pair_sums < threshold
            P += p[k] * p[l] * pair_probs[mask].sum()
    return P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 600.0 - 30.0  # 30s safety margin
    
    # Two-point baseline: {0, a} with optimized p
    # P = (1-p) * (1 - (1-p)^3)
    # Optimal: 1-p = 4^(-1/3)
    inv_cbrt4 = 4.0 ** (-1.0/3.0)
    p0 = 1.0 - inv_cbrt4  # prob of 0
    p1 = inv_cbrt4         # prob of a
    
    # Try various two-point configurations
    best_P = 0.0
    best_support = np.array([0.0, 1.0])
    best_weights = np.array([p0, p1])
    
    # Search over two-point measures {0, a}
    for a_val in [0.5, 1.0, 2.0, 5.0]:
        for p in np.linspace(0.01, 0.99, 50):
            s = np.array([0.0, a_val])
            w = np.array([p, 1.0-p])
            P = _compute_P(s, w)
            if P > best_P:
                best_P = P
                best_support = s.copy()
                best_weights = w.copy()
    
    # Try three-point measures {0, a, b}
    for b_val in [3.0, 5.0, 10.0]:
        for a_val in [0.5, 1.0, 2.0]:
            for p0 in np.linspace(0.05, 0.6, 15):
                for pa in np.linspace(0.01, 0.5, 10):
                    pb = 1.0 - p0 - pa
                    if pb <= 0.001:
                        continue
                    s = np.array([0.0, a_val, b_val])
                    w = np.array([p0, pa, pb])
                    P = _compute_P(s, w)
                    if P > best_P:
                        best_P = P
                        best_support = s.copy()
                        best_weights = w.copy()
                    if time.time() > deadline - 60.0:
                        break
                if time.time() > deadline - 60.0:
                    break
        if time.time() > deadline - 60.0:
            break
    
    # Try four-point measures {0, a, b, c}
    if time.time() < deadline - 120.0:
        for c_val in [5.0, 10.0]:
            for b_val in [2.0, 3.0]:
                for a_val in [0.5, 1.0]:
                    for p0 in np.linspace(0.1, 0.5, 8):
                        for pa in np.linspace(0.05, 0.4, 6):
                            for pb in np.linspace(0.05, 0.3, 5):
                                pc = 1.0 - p0 - pa - pb
                                if pc <= 0.001:
                                    continue
                                s = np.array([0.0, a_val, b_val, c_val])
                                w = np.array([p0, pa, pb, pc])
                                P = _compute_P(s, w)
                                if P > best_P:
                                    best_P = P
                                    best_support = s.copy()
                                    best_weights = w.copy()
                                if time.time() > deadline - 60.0:
                                    break
                            if time.time() > deadline - 60.0:
                                break
                        if time.time() > deadline - 60.0:
                            break
                    if time.time() > deadline - 60.0:
                        break
                if time.time() > deadline - 60.0:
                    break
            if time.time() > deadline - 60.0:
                break
    
    return best_support, best_weights


# EVOLVE-BLOCK-END

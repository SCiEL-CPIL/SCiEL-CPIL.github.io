# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1..X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s kill

    def compute_P(support, weights):
        """Compute P[X1+X2+X3 < 2*X4] for a discrete measure."""
        n = len(support)
        total_P = 0.0
        for k in range(n):
            w4 = weights[k]
            if w4 <= 0:
                continue
            threshold = 2.0 * support[k]
            p_sum = 0.0
            for i in range(n):
                for j in range(n):
                    for l in range(n):
                        s = support[i] + support[j] + support[l]
                        if s < threshold:
                            p_sum += weights[i] * weights[j] * weights[l]
            total_P += w4 * p_sum
        return total_P

    def neg_P_3point(params):
        """Negative P for 3-point {0, 1, M} with masses (p0, p1, pM)."""
        p0, p1, logM = params
        pM = 1.0 - p0 - p1
        if p0 < 0 or p1 < 0 or pM < 0:
            return 1e10
        M = np.exp(logM)
        if M < 1.001:
            return 1e10
        support = np.array([0.0, 1.0, M])
        weights = np.array([p0, p1, pM])
        return -compute_P(support, weights)

    # Start with a good initial guess
    # From analysis: p0 ~ 0.4, p1 ~ 0.3, pM ~ 0.3, M ~ 2-3
    best_result = None
    best_P = -1.0

    # Multiple restarts with different initial points
    initial_points = [
        (0.3, 0.3, np.log(2.0)),
        (0.4, 0.2, np.log(3.0)),
        (0.2, 0.4, np.log(2.5)),
        (0.5, 0.2, np.log(1.5)),
        (0.3, 0.4, np.log(4.0)),
        (0.2, 0.3, np.log(2.0)),
        (0.4, 0.3, np.log(2.5)),
        (0.35, 0.25, np.log(3.5)),
    ]

    for init in initial_points:
        if time.time() > deadline - 30.0:
            break
        try:
            result = minimize(
                neg_P_3point,
                init,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            P_val = -result.fun
            if P_val > best_P and P_val > 0:
                best_P = P_val
                best_result = result.x
        except Exception:
            continue

    # Also try a 2-point {0, M} distribution for comparison
    def neg_P_2point(params):
        p0, logM = params
        pM = 1.0 - p0
        if p0 < 0 or pM < 0:
            return 1e10
        M = np.exp(logM)
        if M < 0.001:
            return 1e10
        support = np.array([0.0, M])
        weights = np.array([p0, pM])
        return -compute_P(support, weights)

    for p0_init in [0.3, 0.4, 0.5, 0.6]:
        if time.time() > deadline - 20.0:
            break
        for logM_init in [np.log(1.5), np.log(2.0), np.log(3.0)]:
            try:
                result = minimize(
                    neg_P_2point,
                    (p0_init, logM_init),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                P_val = -result.fun
                if P_val > best_P and P_val > 0:
                    best_P = P_val
                    # Store as 3-point with p1=0
                    best_result = (result.x[0], 0.0, result.x[1])
            except Exception:
                continue

    # Also try 4-point {0, 1, M, 2M} for broader exploration
    def neg_P_4point(params):
        p0, p1, pM, logM = params
        p2M = 1.0 - p0 - p1 - pM
        if p0 < 0 or p1 < 0 or pM < 0 or p2M < 0:
            return 1e10
        M = np.exp(logM)
        if M < 1.001:
            return 1e10
        support = np.array([0.0, 1.0, M, 2.0 * M])
        weights = np.array([p0, p1, pM, p2M])
        return -compute_P(support, weights)

    for init4 in [
        (0.2, 0.2, 0.2, np.log(2.0)),
        (0.3, 0.1, 0.3, np.log(2.5)),
        (0.2, 0.3, 0.2, np.log(3.0)),
    ]:
        if time.time() > deadline - 20.0:
            break
        try:
            result = minimize(
                neg_P_4point,
                init4,
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            P_val = -result.fun
            if P_val > best_P and P_val > 0:
                best_P = P_val
                # Need to handle 4-point separately
                p0, p1, pM, logM = result.x
                p2M = 1.0 - p0 - p1 - pM
                M = np.exp(logM)
                support = np.array([0.0, 1.0, M, 2.0 * M])
                weights = np.array([p0, p1, pM, p2M])
                # Store as final answer directly
                support = support.astype(np.float64)
                weights = weights.astype(np.float64)
                # Validate
                if np.all(support >= -1e-9) and np.all(weights >= -1e-9) and np.sum(weights) > 1e-9:
                    return support, weights
        except Exception:
            continue

    # Use best 3-point result
    if best_result is not None:
        p0, p1, logM = best_result
        pM = 1.0 - p0 - p1
        M = np.exp(logM)
        support = np.array([0.0, 1.0, M])
        weights = np.array([p0, p1, pM])
    else:
        # Fallback: 2-point
        support = np.array([0.0, 1.0])
        weights = np.array([0.4, 0.6])

    # Clean up: remove points with negligible weight
    support = support.astype(np.float64)
    weights = weights.astype(np.float64)

    # Remove any points with weight < 1e-10 to keep it clean
    mask = weights > 1e-10
    support = support[mask]
    weights = weights[mask]

    # Normalize
    weights = weights / weights.sum()

    # Final validation
    assert len(support) == len(weights)
    assert len(support) >= 1
    assert np.all(np.isfinite(support))
    assert np.all(np.isfinite(weights))
    assert np.all(support >= -1e-9)
    assert np.all(weights >= -1e-9)
    assert np.sum(weights) > 1e-9

    return support, weights

# EVOLVE-BLOCK-END

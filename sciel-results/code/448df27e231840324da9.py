# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_VALS = np.arange(_N, dtype=np.float64)
_YS = np.repeat(_VALS, _N)
_XS = np.tile(_VALS, _N)

# Precompute bin indices for each projection
# x - y ranges from -9 to 9 -> 19 bins
_IDX_D = (_XS - _YS).astype(int) + 9
# x ranges 0..9 -> 10 bins
_IDX_H0 = _XS.astype(int)
# x + y ranges 0..18 -> 19 bins
_IDX_H1 = (_XS + _YS).astype(int)
# x + 2y ranges 0..27 -> 28 bins
_IDX_H2 = (_XS + 2 * _YS).astype(int)

_N_BINS_D = 19
_N_BINS_H0 = 10
_N_BINS_H1 = 19
_N_BINS_H2 = 28


def _entropy_from_counts(counts):
    """Compute Shannon entropy in bits from a count vector, ignoring zeros."""
    s = counts.sum()
    if s <= 1e-30:
        return 0.0
    p = counts / s
    p = p[p > 0.0]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def _compute_value(P):
    """Compute D/M ratio for a given 10x10 distribution P."""
    p = P.ravel()
    total = p.sum()
    if total <= 1e-30:
        return 0.0
    p = p / total

    # D: distribution of x - y
    d_counts = np.zeros(_N_BINS_D)
    np.add.at(d_counts, _IDX_D, p)
    hd = _entropy_from_counts(d_counts)

    # H_0: distribution of x
    h0_counts = np.zeros(_N_BINS_H0)
    np.add.at(h0_counts, _IDX_H0, p)
    h0 = _entropy_from_counts(h0_counts)

    # H_1: distribution of x + y
    h1_counts = np.zeros(_N_BINS_H1)
    np.add.at(h1_counts, _IDX_H1, p)
    h1 = _entropy_from_counts(h1_counts)

    # H_2: distribution of x + 2y
    h2_counts = np.zeros(_N_BINS_H2)
    np.add.at(h2_counts, _IDX_H2, p)
    h2 = _entropy_from_counts(h2_counts)

    m = max(h0, h1, h2)
    if m <= 1e-9:
        return 0.0
    return hd / m


def _negative_value(theta):
    """Objective to minimize: -D/M. theta is a 100-dim vector of log-probabilities."""
    # Softmax to get probabilities
    theta = np.asarray(theta, dtype=np.float64)
    theta_shifted = theta - theta.max()
    exp_theta = np.exp(theta_shifted)
    p = exp_theta / exp_theta.sum()
    P = p.reshape(_N, _N)
    val = _compute_value(P)
    if val <= 1e-9:
        return 1e6  # large penalty for invalid
    return -val


def _uniform_init():
    return np.zeros(_N * _N, dtype=np.float64)


def _random_init():
    return np.random.randn(_N * _N) * 0.1


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s

    best_P = np.ones((_N, _N), dtype=np.float64) / (_N * _N)
    best_score = _compute_value(best_P)

    # Try multiple random restarts with L-BFGS-B
    np.random.seed(42)

    n_restarts = 20
    for i in range(n_restarts):
        if time.time() > deadline - 30.0:
            break

        if i == 0:
            theta0 = _uniform_init()
        else:
            theta0 = _random_init()

        try:
            result = minimize(
                _negative_value,
                theta0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            theta_opt = result.x
            # Convert back to probability distribution
            theta_shifted = theta_opt - theta_opt.max()
            exp_theta = np.exp(theta_shifted)
            p = exp_theta / exp_theta.sum()
            P = p.reshape(_N, _N)

            # Clip and normalize
            P = np.clip(P, 0.0, None)
            s = P.sum()
            if s <= 1e-9:
                continue
            P = P / s

            score = _compute_value(P)
            if score > best_score:
                best_score = score
                best_P = P.copy()

        except Exception:
            continue

    # Additional local refinement around the best solution found
    if time.time() < deadline - 60.0:
        try:
            # Re-optimize from the best found solution
            p_flat = best_P.ravel()
            # Convert to log-space for optimization
            p_safe = np.maximum(p_flat, 1e-15)
            theta_best = np.log(p_safe / p_safe.sum())

            result = minimize(
                _negative_value,
                theta_best,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'gtol': 1e-10, 'ftol': 1e-12}
            )
            theta_opt = result.x
            theta_shifted = theta_opt - theta_opt.max()
            exp_theta = np.exp(theta_shifted)
            p = exp_theta / exp_theta.sum()
            P = p.reshape(_N, _N)
            P = np.clip(P, 0.0, None)
            s = P.sum()
            if s > 1e-9:
                P = P / s
                score = _compute_value(P)
                if score > best_score:
                    best_score = score
                    best_P = P.copy()
        except Exception:
            pass

    # Final cleanup: ensure validity
    best_P = np.clip(best_P, 0.0, None)
    s = best_P.sum()
    if s > 1e-9:
        best_P = best_P / s
    else:
        best_P = np.ones((_N, _N), dtype=np.float64) / (_N * _N)

    return best_P


# EVOLVE-BLOCK-END

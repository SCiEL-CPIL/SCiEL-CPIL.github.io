# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_VALS = np.arange(_N, dtype=np.float64)
_YS = np.repeat(_VALS, _N)
_XS = np.tile(_VALS, _N)

# Precompute bin indices for each projection
_IDX_D = (_XS - _YS).astype(int) + 9
_IDX_H0 = _XS.astype(int)
_IDX_H1 = (_XS + _YS).astype(int)
_IDX_H2 = (_XS + 2 * _YS).astype(int)

_N_BINS_D = 19
_N_BINS_H0 = 10
_N_BINS_H1 = 19
_N_BINS_H2 = 28

# Build sparse projection matrices (100 x n_bins)
def _build_proj_matrix(idx, n_bins):
    M = np.zeros((100, n_bins), dtype=np.float64)
    for i, b in enumerate(idx):
        M[i, b] = 1.0
    return M

_PROJ_D = _build_proj_matrix(_IDX_D, _N_BINS_D)
_PROJ_H0 = _build_proj_matrix(_IDX_H0, _N_BINS_H0)
_PROJ_H1 = _build_proj_matrix(_IDX_H1, _N_BINS_H1)
_PROJ_H2 = _build_proj_matrix(_IDX_H2, _N_BINS_H2)


def _entropy_and_grad(q):
    """
    Compute Shannon entropy in bits and its gradient w.r.t. q.
    q is a 1D array of probabilities (may have zeros).
    Returns (H, grad_q) where grad_q has same shape as q.
    """
    q = np.asarray(q, dtype=np.float64)
    mask = q > 1e-300
    H = 0.0
    grad_q = np.zeros_like(q)
    if np.any(mask):
        qm = q[mask]
        H = -float(np.sum(qm * np.log2(qm)))
        grad_q[mask] = -(np.log2(qm) + 1.0 / np.log(2.0))
    return H, grad_q


def _softmax_and_grad(z):
    """
    Numerically stable softmax and its Jacobian.
    z is a 1D array. Returns (p, jac) where jac is a 2D array (100x100) 
    but we'll compute gradients via chain rule without forming full Jacobian.
    """
    z = np.asarray(z, dtype=np.float64)
    z_shifted = z - z.max()
    exp_z = np.exp(z_shifted)
    p = exp_z / exp_z.sum()
    return p


def _compute_value(P):
    """Compute D/M ratio for a given 10x10 distribution P."""
    p = P.ravel()
    total = p.sum()
    if total <= 1e-30:
        return 0.0
    p = p / total

    d_counts = _PROJ_D @ p
    hd, _ = _entropy_and_grad(d_counts)

    h0_counts = _PROJ_H0 @ p
    h0, _ = _entropy_and_grad(h0_counts)

    h1_counts = _PROJ_H1 @ p
    h1, _ = _entropy_and_grad(h1_counts)

    h2_counts = _PROJ_H2 @ p
    h2, _ = _entropy_and_grad(h2_counts)

    m = max(h0, h1, h2)
    if m <= 1e-9:
        return 0.0
    return hd / m


def _negative_value_and_grad(theta, beta=10.0):
    """
    Objective to minimize: -D/M_smooth. theta is a 100-dim vector of log-probabilities.
    Uses smooth max via log-sum-exp with temperature beta.
    Returns (value, gradient).
    """
    theta = np.asarray(theta, dtype=np.float64)
    
    # Softmax
    z_shifted = theta - theta.max()
    exp_theta = np.exp(z_shifted)
    p = exp_theta / exp_theta.sum()
    
    # Compute projected distributions
    d_counts = _PROJ_D @ p
    h0_counts = _PROJ_H0 @ p
    h1_counts = _PROJ_H1 @ p
    h2_counts = _PROJ_H2 @ p
    
    # Entropies and gradients w.r.t. projected counts
    hd, grad_d = _entropy_and_grad(d_counts)
    h0, grad_h0 = _entropy_and_grad(h0_counts)
    h1, grad_h1 = _entropy_and_grad(h1_counts)
    h2, grad_h2 = _entropy_and_grad(h2_counts)
    
    # Smooth max: M = (1/beta) * log(sum_k exp(beta * H_k))
    hs = np.array([h0, h1, h2])
    beta_hs = beta * hs
    max_beta_hs = np.max(beta_hs)
    exp_beta_hs = np.exp(beta_hs - max_beta_hs)
    sum_exp = np.sum(exp_beta_hs)
    M = (1.0 / beta) * (max_beta_hs + np.log(sum_exp))
    
    if M <= 1e-9:
        return 1e6, np.zeros_like(theta)
    
    value = -hd / M
    
    # Gradient computation
    # dH_k/dp = PROJ_k^T @ grad_Hk
    grad_p_d = _PROJ_D.T @ grad_d
    grad_p_h0 = _PROJ_H0.T @ grad_h0
    grad_p_h1 = _PROJ_H1.T @ grad_h1
    grad_p_h2 = _PROJ_H2.T @ grad_h2
    
    # dM/dp = (1/M) * sum_k (dH_k/dp) * w_k where w_k = exp(beta*H_k)/sum_exp
    weights = exp_beta_hs / sum_exp
    grad_p_M = (1.0 / M) * (weights[0] * grad_p_h0 + weights[1] * grad_p_h1 + weights[2] * grad_p_h2)
    
    # d(value)/dp = -[grad_p_d * M - hd * grad_p_M] / M^2 = -grad_p_d/M + hd*grad_p_M/M^2
    grad_p_val = -grad_p_d / M + hd * grad_p_M / (M * M)
    
    # dp/dtheta = diag(p) - p p^T
    # grad_theta = (diag(p) - p p^T) @ grad_p_val = p * (grad_p_val - dot(p, grad_p_val))
    dot_p_gp = np.dot(p, grad_p_val)
    grad_theta = p * (grad_p_val - dot_p_gp)
    
    return value, grad_theta


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

    np.random.seed(42)

    n_restarts = 30
    for i in range(n_restarts):
        if time.time() > deadline - 30.0:
            break

        if i == 0:
            theta0 = _uniform_init()
        elif i == 1:
            # Structured init: concentrate on diagonal
            theta0 = np.zeros(100)
            for k in range(10):
                theta0[k * 10 + k] = 2.0
        elif i == 2:
            # Structured init: concentrate on anti-diagonal
            theta0 = np.zeros(100)
            for k in range(10):
                theta0[k * 10 + (9 - k)] = 2.0
        else:
            theta0 = _random_init()

        try:
            def obj_func(t):
                val, _ = _negative_value_and_grad(t)
                return val

            def obj_grad(t):
                _, g = _negative_value_and_grad(t)
                return g

            result = minimize(
                obj_func,
                obj_grad,
                theta0,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'gtol': 1e-10, 'ftol': 1e-12}
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
            p_flat = best_P.ravel()
            p_safe = np.maximum(p_flat, 1e-15)
            theta_best = np.log(p_safe / p_safe.sum())

            def obj_func(t):
                val, _ = _negative_value_and_grad(t)
                return val

            def obj_grad(t):
                _, g = _negative_value_and_grad(t)
                return g

            result = minimize(
                obj_func,
                obj_grad,
                theta_best,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-14}
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

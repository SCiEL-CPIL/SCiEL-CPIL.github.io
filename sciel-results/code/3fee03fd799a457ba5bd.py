# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _shannon_entropy(p):
    """Compute Shannon entropy in bits, ignoring zero probabilities."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-15]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _compute_D_M(P):
    """Compute D = H_{-1} and M = max(H_0, H_1, H_2) for distribution P[y, x]."""
    n = 10
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)  # X[y, x] = x, Y[y, x] = y

    # D: distribution of x - y (values in -9..9, 19 bins)
    diff = X - Y
    diff_probs = np.zeros(2 * n - 1)
    for i in range(2 * n - 1):
        mask = (diff == (i - (n - 1)))
        diff_probs[i] = np.sum(P * mask)
    D = _shannon_entropy(diff_probs)

    # H_0: distribution of x (10 bins)
    x_probs = np.zeros(n)
    for i in range(n):
        x_probs[i] = np.sum(P[:, i])
    H0 = _shannon_entropy(x_probs)

    # H_1: distribution of x + y (values 0..18, 19 bins)
    s1 = X + Y
    s1_probs = np.zeros(2 * n - 1)
    for i in range(2 * n - 1):
        mask = (s1 == i)
        s1_probs[i] = np.sum(P * mask)
    H1 = _shannon_entropy(s1_probs)

    # H_2: distribution of x + 2y (values 0..27, 28 bins)
    s2 = X + 2 * Y
    s2_probs = np.zeros(3 * n - 2)
    for i in range(3 * n - 2):
        mask = (s2 == i)
        s2_probs[i] = np.sum(P * mask)
    H2 = _shannon_entropy(s2_probs)

    M = max(H0, H1, H2)
    return D, M


def _ratio_from_P(P):
    """Compute D/M ratio for a normalized distribution P."""
    P = np.clip(P, 0, None)
    s = np.sum(P)
    if s <= 1e-9:
        return 0.0
    P = P / s
    D, M = _compute_D_M(P)
    if M <= 1e-9:
        return 0.0
    return D / M


def _logsumexp_max_surrogate(p, bins, temperature=0.1):
    """
    Smooth surrogate for max entropy.
    p: array of bin probabilities (sum to 1)
    bins: not used directly, just for clarity
    Returns a smooth approximation of max(H_i) using log-sum-exp over entropies.
    Actually, we want to minimize M, so we maximize -M.
    We use a soft-min over the entropies to get a smooth M.
    """
    # p is already the bin probabilities for one slope
    # This function is a placeholder; we'll compute entropies separately
    pass


def _objective(theta, n=10):
    """
    Objective to minimize: -D/M
    theta: flattened parameter vector
    We use a softmax parameterization: P[y,x] = exp(theta[y,x]) / sum(exp(theta))
    To avoid overflow, we subtract max.
    """
    theta = theta.reshape(n, n)
    theta_shifted = theta - np.max(theta)
    P_unnorm = np.exp(theta_shifted)
    P = P_unnorm / np.sum(P_unnorm)
    ratio = _ratio_from_P(P)
    return -ratio


def _objective_and_P(theta, n=10):
    """Return (objective_value, P) for debugging/inspection."""
    theta = theta.reshape(n, n)
    theta_shifted = theta - np.max(theta)
    P_unnorm = np.exp(theta_shifted)
    P = P_unnorm / np.sum(P_unnorm)
    ratio = _ratio_from_P(P)
    return -ratio, P


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Returns a 10x10 array P of finite weights, P[y, x] for integers x, y in 0..9.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit, with safety margin

    # Initial best: uniform
    P_uniform = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P_uniform.copy()
    best_score = _ratio_from_P(P_uniform)

    # Try several initializations and optimize from each
    # Strategy: anti-diagonal structures to maximize H(x-y)
    # Also try random perturbations of uniform

    initializations = []

    # 1. Uniform
    initializations.append(np.zeros((n, n)))  # theta=0 gives uniform

    # 2. Anti-diagonal emphasis: P[y, x] ~ exp(alpha * (x - y)) for various alpha
    for alpha in [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0]:
        theta = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                theta[y, x] = alpha * (x - y)
        initializations.append(theta.copy())

    # 3. Random initializations
    rng = np.random.RandomState(42)
    for _ in range(5):
        theta = rng.randn(n, n) * 0.5
        initializations.append(theta.copy())

    # 4. Structured: emphasize cells where x-y is spread out
    #    i.e., make P concentrate on cells with extreme x-y values
    for spread in [1.0, 2.0, 3.0]:
        theta = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                diff = x - y
                theta[y, x] = spread * abs(diff) / (n - 1)
        initializations.append(theta.copy())

    for init_theta in initializations:
        if time.time() > deadline - 30:
            break

        try:
            result = minimize(
                _objective,
                init_theta.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8},
            )
            if result.success or result.fun < best_score:
                P_candidate = None
                theta_opt = result.x.reshape(n, n)
                theta_shifted = theta_opt - np.max(theta_opt)
                P_candidate = np.exp(theta_shifted)
                P_candidate = P_candidate / np.sum(P_candidate)
                score_candidate = _ratio_from_P(P_candidate)
                if score_candidate > best_score:
                    best_score = score_candidate
                    best_P = P_candidate.copy()
        except Exception:
            continue

    # Final validation and normalization
    P = np.clip(best_P, 0, None)
    total = np.sum(P)
    if total <= 1e-9:
        # Fallback to uniform
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        P = P / total

    # Ensure no NaN or inf
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 0, None)
    total = np.sum(P)
    if total <= 1e-9:
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        P = P / total

    return P


# EVOLVE-BLOCK-END

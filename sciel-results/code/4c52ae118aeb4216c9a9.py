# EVOLVE-BLOCK-START
import numpy as np
import time


def _shannon_entropy_from_counts(counts: np.ndarray) -> float:
    """Compute Shannon entropy in bits from a count array, ignoring zeros."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 1e-15]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _compute_D_M(P: np.ndarray):
    """
    Compute D = H(x-y) and M = max(H_0, H_1, H_2) for distribution P[y, x].
    Returns (D, M).
    """
    n = 10
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)  # X[y,x] = x, Y[y,x] = y

    # D: distribution of x - y, values in -9..9
    diff = X - Y  # shape (10, 10), values -9..9
    diff_probs = np.zeros(2 * n - 1, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            idx = diff[i, j] + (n - 1)
            diff_probs[idx] += P[i, j]
    D = _shannon_entropy_from_counts(diff_probs)

    # H_0: distribution of x (marginal over y)
    x_probs = P.sum(axis=0)  # sum over y for each x
    H0 = _shannon_entropy_from_counts(x_probs)

    # H_1: distribution of x + y, values 0..18
    sum_xy = X + Y
    sum_probs = np.zeros(2 * n - 1, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            sum_probs[sum_xy[i, j]] += P[i, j]
    H1 = _shannon_entropy_from_counts(sum_probs)

    # H_2: distribution of x + 2y, values 0..27
    sum_2y = X + 2 * Y
    max_val_2y = 27
    two_probs = np.zeros(max_val_2y + 1, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            two_probs[sum_2y[i, j]] += P[i, j]
    H2 = _shannon_entropy_from_counts(two_probs)

    M = max(H0, H1, H2)
    return D, M


def _objective(P: np.ndarray) -> float:
    """Compute D/M ratio. Returns 0 if M is too small."""
    D, M = _compute_D_M(P)
    if M <= 1e-9:
        return 0.0
    return D / M


def _softmax(theta: np.ndarray) -> np.ndarray:
    """Convert parameter vector to a probability distribution via softmax."""
    # Subtract max for numerical stability
    theta_shifted = theta - theta.max()
    exp_theta = np.exp(theta_shifted)
    return exp_theta / exp_theta.sum()


def _compute_D_M_fast(P: np.ndarray):
    """
    Optimized computation of D and M using vectorized operations.
    """
    n = 10
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)

    # D: x - y
    diff = X - Y
    diff_probs = np.bincount((diff + (n - 1)).ravel().astype(int), weights=P.ravel(), minlength=2 * n - 1)
    D = _shannon_entropy_from_counts(diff_probs)

    # H_0: marginal of x
    x_probs = P.sum(axis=0)
    H0 = _shannon_entropy_from_counts(x_probs)

    # H_1: x + y
    sum_xy = X + Y
    sum_probs = np.bincount(sum_xy.ravel().astype(int), weights=P.ravel(), minlength=2 * n - 1)
    H1 = _shannon_entropy_from_counts(sum_probs)

    # H_2: x + 2y
    sum_2y = X + 2 * Y
    two_probs = np.bincount(sum_2y.ravel().astype(int), weights=P.ravel(), minlength=28)
    H2 = _shannon_entropy_from_counts(two_probs)

    M = max(H0, H1, H2)
    return D, M


def _objective_fast(theta: np.ndarray) -> float:
    """Fast objective using softmax parameterization."""
    P = _softmax(theta)
    D, M = _compute_D_M_fast(P)
    if M <= 1e-9:
        return 0.0
    return D / M


def _neg_objective(theta: np.ndarray) -> float:
    """Negative objective for minimization."""
    return -_objective_fast(theta)


def _neg_objective_with_grad(theta: np.ndarray):
    """Objective and gradient for L-BFGS-B."""
    P = _softmax(theta)
    D, M = _compute_D_M_fast(P)
    if M <= 1e-9:
        return 0.0, np.zeros_like(theta)

    val = D / M

    # Numerical gradient
    eps = 1e-7
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        theta_minus = theta.copy()
        theta_minus[i] -= eps
        grad[i] = (_neg_objective(theta_plus) - _neg_objective(theta_minus)) / (2 * eps)

    return -val, grad


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    The baseline is the uniform distribution over the grid.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit

    # Baseline: uniform distribution
    P_uniform = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P_uniform.copy()
    best_score = _objective(P_uniform)

    # Try various initializations and optimize
    from scipy.optimize import minimize

    # Different initializations based on structured patterns
    initializations = []

    # 1. Uniform
    initializations.append(np.zeros(n * n))

    # 2. Anti-diagonal concentration (high x-y spread)
    for offset in range(-n + 1, n):
        theta = np.zeros(n * n)
        for y in range(n):
            for x in range(n):
                if x - y == offset:
                    theta[y * n + x] = 1.0
        initializations.append(theta.copy())

    # 3. Corner concentration
    for cx in [0, n - 1]:
        for cy in [0, n - 1]:
            theta = np.zeros(n * n)
            for y in range(n):
                for x in range(n):
                    dist = abs(x - cx) + abs(y - cy)
                    theta[y * n + x] = -dist * 0.5
            initializations.append(theta.copy())

    # 4. Random initializations
    rng = np.random.RandomState(42)
    for _ in range(5):
        initializations.append(rng.randn(n * n) * 0.5)

    # 5. Diagonal concentration
    theta_diag = np.zeros(n * n)
    for y in range(n):
        for x in range(n):
            if x == y:
                theta_diag[y * n + x] = 1.0
    initializations.append(theta_diag.copy())

    # 6. Opposite diagonal
    theta_opp = np.zeros(n * n)
    for y in range(n):
        for x in range(n):
            if x + y == n - 1:
                theta_opp[y * n + x] = 1.0
    initializations.append(theta_opp.copy())

    # 7. Banded distributions
    for bandwidth in [1, 2, 3]:
        theta_band = np.zeros(n * n)
        for y in range(n):
            for x in range(n):
                if abs(x - y) <= bandwidth:
                    theta_band[y * n + x] = 1.0
        initializations.append(theta_band.copy())

    best_theta = None
    best_obj = -np.inf

    for init_idx, theta0 in enumerate(initializations):
        if time.time() > deadline - 10:
            break

        try:
            # First, evaluate the initial point
            P0 = _softmax(theta0)
            score0 = _objective(P0)

            if score0 > best_score:
                best_score = score0
                best_P = P0.copy()
                best_theta = theta0.copy()

            # Run L-BFGS-B optimization
            result = minimize(
                _neg_objective,
                theta0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50},
            )

            if result.success or result.fun < best_obj:
                theta_opt = result.x
                P_opt = _softmax(theta_opt)
                score_opt = _objective(P_opt)

                if score_opt > best_score:
                    best_score = score_opt
                    best_P = P_opt.copy()
                    best_theta = theta_opt.copy()

                if result.fun < best_obj:
                    best_obj = result.fun

        except Exception:
            continue

    # Try a few more random restarts if time permits
    if best_theta is not None:
        rng2 = np.random.RandomState(123)
        for _ in range(10):
            if time.time() > deadline - 10:
                break
            theta_perturbed = best_theta + rng2.randn(n * n) * 0.3
            try:
                result = minimize(
                    _neg_objective,
                    theta_perturbed,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50},
                )
                theta_opt = result.x
                P_opt = _softmax(theta_opt)
                score_opt = _objective(P_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_P = P_opt.copy()
                    best_theta = theta_opt.copy()
            except Exception:
                continue

    # Final cleanup: clip negatives and normalize
    best_P = np.clip(best_P, 0, None)
    total = best_P.sum()
    if total <= 1e-9:
        # Fallback to uniform
        best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        best_P = best_P / total

    return best_P


# EVOLVE-BLOCK-END

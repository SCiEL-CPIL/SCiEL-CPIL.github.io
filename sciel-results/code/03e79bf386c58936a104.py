# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_SLOPES = np.array([0.0, 1.0, 2.0], dtype=np.float64)
_EPS = 1e-12


def _shannon_entropy_bits(counts: np.ndarray) -> float:
    """Shannon entropy (bits) of a non-negative count vector, ignoring zeros."""
    total = counts.sum()
    if total <= _EPS:
        return 0.0
    p = counts / total
    p = p[p > _EPS]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _projection_counts(P: np.ndarray, slope: float) -> np.ndarray:
    """
    P is shape (n, n) with P[y, x]. Value v = x + slope * y.
    Returns a 1D array of counts indexed by (v - v_min) where v_min is the
    minimum possible value. We use a fixed offset so the index is stable.
    """
    n = P.shape[0]
    xs = np.arange(n, dtype=np.float64)
    ys = np.arange(n, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')  # XX[y, x] = x, YY[y, x] = y
    V = XX + slope * YY
    v_min = V.min()
    v_max = V.max()
    # Quantize to integer index; values are integers when slope is integer.
    idx = np.round(V - v_min).astype(np.int64)
    size = int(round(v_max - v_min)) + 1
    counts = np.zeros(size, dtype=np.float64)
    np.add.at(counts, idx.ravel(), P.ravel())
    return counts


def _soft_max(vals: np.ndarray, temperature: float = 1.0) -> float:
    """Smooth approximation of max via log-sum-exp scaled by temperature."""
    m = np.max(vals)
    if m <= _EPS:
        return 0.0
    z = np.exp((vals - m) / temperature)
    return float(m + temperature * np.log(np.sum(z)))


def _objective(theta: np.ndarray) -> float:
    """
    theta: flattened (n, n) parameter vector.
    Returns D / M (higher is better). We minimize the negative.
    """
    P = theta.reshape(_N, _N)
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= _EPS:
        return 0.0
    P = P / s

    D = _shannon_entropy_bits(_projection_counts(P, -1.0))
    H0 = _shannon_entropy_bits(_projection_counts(P, 0.0))
    H1 = _shannon_entropy_bits(_projection_counts(P, 1.0))
    H2 = _shannon_entropy_bits(_projection_counts(P, 2.0))
    M = _soft_max(np.array([H0, H1, H2]), temperature=1.0)
    if M <= 1e-9:
        return 0.0
    return D / M


def _gradient(theta: np.ndarray) -> np.ndarray:
    """Numerical gradient (central difference) of the objective."""
    P = theta.reshape(_N, _N)
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= _EPS:
        return np.zeros_like(theta)
    P = P / s

    D = _shannon_entropy_bits(_projection_counts(P, -1.0))
    H0 = _shannon_entropy_bits(_projection_counts(P, 0.0))
    H1 = _shannon_entropy_bits(_projection_counts(P, 1.0))
    H2 = _shannon_entropy_bits(_projection_counts(P, 2.0))
    M = _soft_max(np.array([H0, H1, H2]), temperature=1.0)
    if M <= 1e-9:
        return np.zeros_like(theta)
    f0 = D / M

    grad = np.zeros_like(theta)
    h = 1e-6
    for i in range(theta.size):
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[i] += h
        theta_m[i] -= h
        grad[i] = (_objective(theta_p) - _objective(theta_m)) / (2 * h)
    return grad


def _softmax_parameterize(theta: np.ndarray) -> np.ndarray:
    """Convert free parameters to a valid probability distribution via softmax."""
    P = np.exp(theta - theta.max())
    s = P.sum()
    if s <= _EPS:
        return np.ones_like(theta) / theta.size
    return P / s


def _objective_softmax(theta: np.ndarray) -> float:
    P = _softmax_parameterize(theta)
    D = _shannon_entropy_bits(_projection_counts(P, -1.0))
    H0 = _shannon_entropy_bits(_projection_counts(P, 0.0))
    H1 = _shannon_entropy_bits(_projection_counts(P, 1.0))
    H2 = _shannon_entropy_bits(_projection_counts(P, 2.0))
    M = _soft_max(np.array([H0, H1, H2]), temperature=1.0)
    if M <= 1e-9:
        return 0.0
    return D / M


def _negative_objective_softmax(theta: np.ndarray) -> float:
    return -_objective_softmax(theta)


def _negative_gradient_softmax(theta: np.ndarray) -> np.ndarray:
    """Numerical gradient of the negative objective (for minimization)."""
    h = 1e-5
    grad = np.zeros_like(theta)
    for i in range(theta.size):
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[i] += h
        theta_m[i] -= h
        grad[i] = (_negative_objective_softmax(theta_p) - _negative_objective_softmax(theta_m)) / (2 * h)
    return grad


def _uniform_theta() -> np.ndarray:
    """Theta that yields a uniform distribution under softmax."""
    return np.zeros((_N, _N), dtype=np.float64)


def _random_theta(scale: float = 1.0) -> np.ndarray:
    return np.random.randn(_N, _N) * scale


def _optimize_from_theta(theta0: np.ndarray, deadline: float, maxiter: int = 200) -> tuple:
    """
    Run L-BFGS-B from theta0. Returns (best_theta, best_value, elapsed).
    """
    start = time.time()
    best_theta = theta0.copy()
    best_val = _objective_softmax(theta0)

    try:
        def fun(x):
            return _negative_objective_softmax(x)

        def jac(x):
            return _negative_gradient_softmax(x)

        res = minimize(
            fun,
            theta0.copy().ravel(),
            method='L-BFGS-B',
            jac=jac,
            options={'maxiter': maxiter, 'maxfun': maxiter * 10, 'ftol': 1e-12, 'gtol': 1e-8},
        )
        if res.x is not None:
            cand = res.x.reshape(_N, _N)
            cand_val = _objective_softmax(cand)
            if cand_val > best_val:
                best_theta = cand
                best_val = cand_val
    except Exception:
        pass

    elapsed = time.time() - start
    return best_theta, best_val, elapsed


def arithmetic_kakeya_distribution() -> np.ndarray:
    n = _N
    deadline = time.time() + 600.0  # safety margin before 630s kill

    # Start with uniform
    best_theta = _uniform_theta()
    best_val = _objective_softmax(best_theta)

    # Try several random restarts with L-BFGS-B
    rng = np.random.default_rng(42)
    attempt = 0
    while time.time() < deadline - 30.0 and attempt < 200:
        attempt += 1
        # Vary the scale of random initialization
        scale = 0.5 + 1.0 * (attempt % 5)
        theta0 = _random_theta(scale=scale)
        # Also try a structured initialization: emphasize diagonal or off-diagonal
        if attempt % 4 == 0:
            # Diagonal emphasis
            theta0 = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                theta0[i, i] = 2.0
            theta0 += rng.standard_normal((n, n)) * 0.3
        elif attempt % 4 == 1:
            # Anti-diagonal emphasis (x - y = 0 region)
            theta0 = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                theta0[i, n - 1 - i] = 2.0
            theta0 += rng.standard_normal((n, n)) * 0.3
        elif attempt % 4 == 2:
            # Corner emphasis
            theta0 = np.zeros((n, n), dtype=np.float64)
            for corner in [(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)]:
                theta0[corner] = 2.0
            theta0 += rng.standard_normal((n, n)) * 0.3
        else:
            # Pure random
            theta0 = _random_theta(scale=scale)

        theta_opt, val, elapsed = _optimize_from_theta(theta0, deadline)
        if val > best_val:
            best_theta = theta_opt
            best_val = val

    # Convert best theta to probability distribution
    P = _softmax_parameterize(best_theta)
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= _EPS:
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        P = P / s

    # Ensure validity
    P = np.clip(P, 0.0, None)
    total = P.sum()
    if total <= 1e-9:
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        P = P / total

    return P


# EVOLVE-BLOCK-END

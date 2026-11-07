"""Optimized program for the second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill


def _compute_value(h):
    """Compute the objective value for given heights h."""
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0, None)
    if not np.all(np.isfinite(h)):
        return 0.0

    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m + 1] = g
    w = 1.0 / (m + 1)

    l2 = (w / 3.0) * np.sum(y[:m + 1] ** 2 + y[:m + 1] * y[1:m + 2] + y[1:m + 2] ** 2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))

    if norm_1 * norm_inf < 1e-30:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def _neg_value(h):
    """Negative of the objective for minimization."""
    return -_compute_value(h)


def _neg_value_smooth(h, beta):
    """
    Smooth objective using log-sum-exp for the max term.
    Maximizes l2 / (norm_1 * norm_inf_smooth)
    """
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m + 1] = g
    w = 1.0 / (m + 1)

    l2 = (w / 3.0) * np.sum(y[:m + 1] ** 2 + y[:m + 1] * y[1:m + 2] + y[1:m + 2] ** 2)
    norm_1 = np.sum(g) / (m + 1)

    max_g = np.max(g)
    if max_g < 1e-30:
        return 0.0

    # log-sum-exp for numerical stability
    shifted = beta * (g - max_g)
    log_sum_exp = max_g + (1.0 / beta) * np.log(np.sum(np.exp(shifted)))
    norm_inf_smooth = log_sum_exp

    if norm_1 * norm_inf_smooth < 1e-30:
        return 0.0

    val = l2 / (norm_1 * norm_inf_smooth)
    return -val


def _numerical_gradient(f, x, eps=1e-7):
    """Compute numerical gradient."""
    grad = np.zeros_like(x)
    f0 = f(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        grad[i] = (f(x_plus) - f0) / eps
    return grad


def _optimize_from_start(h0, beta, maxiter=300):
    """Run L-BFGS-B optimization from a starting point."""
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return h0, _compute_value(h0)

    def obj(x):
        return _neg_value_smooth(x, beta)

    def grad(x):
        return _numerical_gradient(obj, x)

    try:
        result = minimize(
            obj,
            h0,
            jac=grad,
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
        )
        candidate = np.maximum(result.x, 0.0)
        val = _compute_value(candidate)
        return candidate, val
    except Exception:
        return h0, _compute_value(h0)


def construct():
    global START_TIME, TIME_LIMIT

    best_h = np.ones(N, dtype=np.float64)
    best_val = _compute_value(best_h)

    # Multi-stage optimization with increasing beta for smoothness
    betas = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0]

    current_h = best_h.copy()

    for beta in betas:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break

        candidate, val = _optimize_from_start(current_h, beta, maxiter=400)
        if val > best_val:
            best_val = val
            best_h = candidate.copy()
        current_h = candidate

    # Final polish with exact objective (no smoothing)
    if time.time() - START_TIME < TIME_LIMIT - 10:
        try:
            result = minimize(
                _neg_value,
                best_h,
                jac=lambda x: _numerical_gradient(_neg_value, x),
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000}
            )
            candidate = np.maximum(result.x, 0.0)
            val = _compute_value(candidate)
            if val > best_val:
                best_val = val
                best_h = candidate.copy()
        except Exception:
            pass

    # Try random perturbations from best
    if time.time() - START_TIME < TIME_LIMIT - 5:
        rng = np.random.RandomState(42)
        for _ in range(100):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            perturbation = rng.randn(N) * 0.05
            candidate = np.maximum(best_h + perturbation, 0.0)
            val = _compute_value(candidate)
            if val > best_val:
                best_val = val
                best_h = candidate.copy()

    # Ensure nonnegative and finite
    best_h = np.maximum(best_h, 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)

    return list(best_h)


# Ensure the function is defined at module level for the evaluator
if __name__ == "__main__":
    result = construct()
    print(f"Value: {_compute_value(result)}")

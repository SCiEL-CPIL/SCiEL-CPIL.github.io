"""
Minimizes the third autocorrelation inequality constant C3 upper bound.

Strategy:
Represent h as a constant-amplitude square wave (all 1s) multiplied by a low-order
polynomial window. The constant amplitude maximizes the denominator (sum h)^2 for a
given support, while the polynomial window tapers the edges to reduce the large
sidelobes in the autocorrelation, thereby lowering the peak max(conv).

We optimize the coefficients of the polynomial window using a smooth logsumexp
surrogate for the max operator, which allows gradient-based optimization.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 400
DEADLINE = 120.0  # Safety margin before 150s kill

def _evaluate(h):
    """Compute the exact objective value."""
    h = np.asarray(h, dtype=np.float64)
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _build_h_from_poly(coeffs):
    """
    Build h from polynomial coefficients.
    coeffs[0]: amplitude (scale factor)
    coeffs[1:]: coefficients for the polynomial window of order up to 3.
    
    The window is defined on x in [-1, 1], centered at the middle of the array.
    w(x) = c1 + c2*x^2 + c3*x^4 + c4*x^6  (using even powers for symmetry)
    We ensure w(x) > 0 to avoid sign flips that might complicate the sum.
    """
    n = np.arange(N)
    # Map indices to [-1, 1]
    x = 2.0 * (n - (N - 1) / 2.0) / (N - 1)
    
    # Polynomial window: w(x) = 1 + c1*x^2 + c2*x^4 + c3*x^6
    # Using even powers to maintain symmetry and positivity near center.
    # We will optimize c1, c2, c3.
    # To ensure positivity, we can use a softplus or just constrain them.
    # Let's define w = 1 + c1*x^2 + c2*x^4 + c3*x^6
    # We want w(0) to be close to 1 and w(1) to be small but positive.
    
    c1 = coeffs[1]
    c2 = coeffs[2]
    c3 = coeffs[3]
    
    w = 1.0 + c1 * x**2 + c2 * x**4 + c3 * x**6
    
    # Ensure positivity: clip to a small positive value
    w = np.maximum(w, 0.01)
    
    # The "square wave" part is just 1s.
    # h = amplitude * w
    amplitude = coeffs[0]
    h = amplitude * w
    
    return h

def _surrogate_objective(coeffs):
    """
    Smooth surrogate objective using logsumexp to approximate max(conv).
    """
    h = _build_h_from_poly(coeffs)
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e10
    
    conv = np.convolve(h, h, mode='full')
    
    # Normalize conv by s^2 to make the objective scale-invariant
    # value = 2*N * max(conv) / s^2
    # Let g = conv / s^2
    # We want to minimize max(g).
    # Use logsumexp with a large beta for sharpness.
    beta = 50.0
    g = conv / (s * s)
    
    # logsumexp trick: max(g) approx = (1/beta) * log(sum(exp(beta * g)))
    # Shift for numerical stability
    g_max = np.max(g)
    log_sum_exp = g_max + np.log(np.sum(np.exp(beta * (g - g_max))))
    max_approx = log_sum_exp / beta
    
    obj = 2.0 * N * max_approx
    return obj

def _surrogate_grad(coeffs):
    """
    Numerical gradient for the surrogate objective.
    """
    grad = np.zeros_like(coeffs)
    eps = 1e-5
    for i in range(len(coeffs)):
        coeffs_plus = coeffs.copy()
        coeffs_minus = coeffs.copy()
        coeffs_plus[i] += eps
        coeffs_minus[i] -= eps
        f_plus = _surrogate_objective(coeffs_plus)
        f_minus = _surrogate_objective(coeffs_minus)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad

def construct():
    """
    Returns the optimal step heights.
    """
    t0 = time.time()
    deadline = t0 + DEADLINE - 5.0  # 5s safety margin

    np.random.seed(42)

    best_h = np.ones(N)
    best_val = _evaluate(best_h)  # 2.0

    # Initial guess for polynomial coefficients
    # We want a window that tapers off.
    # w(x) = 1 + c1*x^2 + c2*x^4 + c3*x^6
    # At x=1, we want w(1) to be small, e.g., 0.1.
    # 1 + c1 + c2 + c3 = 0.1
    # Let's try a simple quadratic first: c2=c3=0, c1 = -0.9
    # w(x) = 1 - 0.9*x^2. w(1)=0.1.
    
    x0 = np.array([1.0, -0.9, 0.0, 0.0])
    
    # Optimization
    try:
        # Use L-BFGS-B with numerical gradient
        res = minimize(
            _surrogate_objective,
            x0,
            method='L-BFGS-B',
            jac=_surrogate_grad,
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5, 'maxfun': 1000}
        )
        
        if res.fun < best_val:
            best_h = _build_h_from_poly(res.x)
            best_val = _evaluate(best_h)
    except Exception:
        pass

    # Local refinement on the full 400-dim h using coordinate descent
    # This allows escaping the polynomial subspace if beneficial
    if time.time() < deadline - 10.0:
        h = np.array(best_h, dtype=np.float64, copy=True)
        current_val = best_val
        start = time.time()
        step = 0.01 * np.mean(np.abs(h))
        
        improved = True
        while improved and (time.time() - start) < (deadline - time.time() - 5.0):
            improved = False
            # Random coordinate descent
            for _ in range(500):
                if (time.time() - start) > (deadline - time.time() - 5.0):
                    break
                i = np.random.randint(0, N)
                for delta in [-step, step]:
                    h_new = h.copy()
                    h_new[i] += delta
                    v = _evaluate(h_new)
                    if v < current_val - 1e-10:
                        h = h_new
                        current_val = v
                        improved = True
                        break
            step *= 0.9
            if step < 1e-8:
                break
        
        if current_val < best_val:
            best_val = current_val
            best_h = h

    return best_h

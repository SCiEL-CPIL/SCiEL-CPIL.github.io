"""Improved program for third autocorrelation inequality task."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _value_from_h(h):
    """Compute the evaluator's objective value for a given height vector."""
    h = np.asarray(h, dtype=np.float64)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    # Full autoconvolution
    c = np.convolve(h, h)
    # The maximum over signed entries (no abs inside max)
    peak = np.max(c)
    return abs(2.0 * N * peak / (s * s))


def _neg_score_from_params(params):
    """Negative score (we minimize -score, i.e. maximize score = seed/value)."""
    h = np.asarray(params, dtype=np.float64)
    v = _value_from_h(h)
    if not np.isfinite(v):
        return -1e10  # Very bad
    return -v  # We want to minimize v, so minimize -(-v) = v... wait.
    # Actually score = seed_value / v = 2.0 / v. Higher is better.
    # We want to maximize score = 2.0 / v, equivalent to minimizing v.
    # Let's just minimize v directly.


def _objective(params):
    """Directly minimize the objective value v."""
    h = np.asarray(params, dtype=np.float64)
    v = _value_from_h(h)
    if not np.isfinite(v):
        return 1e10
    return v


def _construct_hamming_chirp(gamma, chirp_rate):
    """
    Construct a height vector using a raised-cosine window times a chirp.
    h[k] = (1 + gamma * cos(2*pi*k/(N-1))) * cos(2*pi*chirp_rate*k*(k+1)/2)
    But we need real values. Let's use a product of a window and a cosine chirp.
    
    Actually, let's think about what minimizes the peak autoconvolution.
    The peak is at lag 0 (R(0) = sum h_i^2). 
    Wait, is the peak always at lag 0?
    
    For any real sequence, R(0) = sum h_i^2 >= 0.
    For k != 0, R(k) = sum h_i * h_{i+k}.
    By Cauchy-Schwarz, |R(k)| <= sqrt(sum h_i^2 * sum h_{i+k}^2) <= R(0) (since the sums are subsets).
    Actually, sum h_{i+k}^2 <= sum h_i^2 = R(0) for the valid indices.
    So |R(k)| <= R(0) for all k.
    Therefore max(c) = R(0) = sum h_i^2.
    
    So the objective is:
    v = 2*N * sum(h_i^2) / (sum(h_i))^2
    
    By Cauchy-Schwarz, (sum h_i)^2 <= N * sum(h_i^2).
    So sum(h_i^2) >= (sum h_i)^2 / N.
    Therefore v >= 2*N * (sum h_i)^2 / N / (sum h_i)^2 = 2.
    
    Equality holds when all h_i are equal.
    So the minimum possible value is exactly 2, achieved by any constant vector.
    The seed gives exactly 2.
    
    This means it's mathematically impossible to get a value less than 2.
    The best possible score is 2.0/2.0 = 1.0.
    
    Any perturbation will make v > 2, giving score < 1.0.
    
    So the optimal solution IS the seed. But the task says "lower is better" for the value,
    and seed gives 2.0. Since 2.0 is the theoretical minimum, no improvement is possible.
    
    However, I should still write a non-trivial program that attempts optimization,
    in case I'm missing something (e.g., floating point effects, or the problem allows
    complex-valued or something else). But the problem says "real, possibly negative step heights".
    
    Let me just verify: is there any scenario where max(c) < R(0)?
    No, because R(0) is always in the convolution array, and R(0) >= |R(k)| for all k.
    
    So the answer is definitively: the constant vector is optimal, value = 2.0, score = 1.0.
    
    I'll write a program that does a quick optimization attempt but will converge to the constant vector.
    """
    k = np.arange(N)
    window = 1.0 + gamma * np.cos(2.0 * np.pi * k / (N - 1))
    chirp = np.cos(2.0 * np.pi * chirp_rate * k * (k + 1) / (2.0 * N))
    h = window * chirp
    # Ensure non-zero sum
    if abs(np.sum(h)) < 1e-10:
        h += 1.0
    return h


def construct():
    """
    Construct 400 step heights minimizing the autocorrelation peak value.
    
    Mathematical analysis shows the minimum value is 2.0 (achieved by constant vectors),
    so this function returns a near-constant vector after a brief optimization attempt.
    """
    # The theoretical minimum is 2.0, achieved by any constant vector.
    # We'll do a quick check/optimization but expect to stay near the constant solution.
    
    # Start from ones
    h0 = np.ones(N)
    
    # Try a few random perturbations to confirm they don't improve
    best_h = h0.copy()
    best_v = _value_from_h(h0)  # Should be 2.0
    
    # Quick local optimization attempt
    try:
        result = minimize(_objective, h0, method='L-BFGS-B', 
                          options={'maxiter': 50, 'gtol': 1e-8})
        if result.fun < best_v - 1e-12:
            best_h = result.x
            best_v = result.fun
    except Exception:
        pass
    
    # Try a few structured perturbations
    for gamma in [0.1, 0.3, 0.5, -0.1, -0.3]:
        for cr in [0.0, 0.01, 0.05, 0.1]:
            h_test = _construct_hamming_chirp(gamma, cr)
            v_test = _value_from_h(h_test)
            if v_test < best_v - 1e-12:
                best_h = h_test
                best_v = v_test
    
    return best_h
# EVOLVE-BLOCK-END

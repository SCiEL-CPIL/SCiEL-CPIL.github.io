"""
Strategy: Square wave base modulated by a low-order polynomial window.

We represent h[k] as a constant-amplitude square wave (alternating +1 and -1)
multiplied by a low-order polynomial window w[k]. The square wave has a large
energy (sum of squares) which helps the denominator, but its autocorrelation
has large sidelobes. The polynomial window tapers the edges to suppress these
sidelobes. We optimize the polynomial coefficients to minimize the peak of the
autoconvolution.

Specifically:
- Base square wave: s[k] = 1 if k is even, -1 if k is odd (or similar).
- Window: w[k] = p(x_k) where x_k = (k - (N-1)/2) / ((N-1)/2) in [-1, 1].
- h[k] = s[k] * w[k].

We optimize the coefficients of p (a polynomial of degree D) to minimize V.
The sum of h must be non-zero. Since the square wave has zero mean (if N is even),
the mean of h is determined by the window. If the window is even (symmetric),
the sum of h will be zero because s[k] is antisymmetric around the center for
odd N, but for even N, s[k] = (-1)^k has sum 0.
Wait, if s[k] = (-1)^k, sum s[k] = 0 for even N.
Then sum(h) = sum((-1)^k * w[k]).
If w[k] is an even polynomial in x_k (symmetric), then w[k] = w[N-1-k].
The term (-1)^k * w[k] + (-1)^(N-1-k) * w[N-1-k].
Since N is even, N-1 is odd. (-1)^(N-1-k) = (-1)^(odd - k) = -(-1)^k.
So the pair is (-1)^k w[k] - (-1)^k w[k] = 0.
Thus, if w is symmetric, sum(h) = 0. This is invalid.

We need an asymmetric window or a different base.
Let's use a base that is not exactly alternating.
Or, we can add a DC offset to the window?
If h[k] = s[k] * (c + p(x_k)), then sum(h) = c * sum(s) + sum(s * p).
If sum(s)=0, then sum(h) = sum(s * p).
If p is odd, s * p is even (symmetric), so sum is non-zero.
So let's use an odd polynomial window?
Or simply: h[k] = s[k] * w[k] where w[k] is NOT symmetric.
Or, better: The strategy says "product of a constant-amplitude square wave ... and a low-order polynomial window".
If the window breaks the symmetry that causes the sum to vanish, we are good.
Let's allow the window to be a general polynomial of degree D.
We optimize the D+1 coefficients.

Actually, a simpler approach to ensure non-zero sum:
Let h[k] = s[k] * w[k] + A, where A is a small constant?
No, the strategy specifies the product.
Let's stick to h[k] = s[k] * w[k].
We need sum(h) != 0.
If we choose s[k] = 1 for k < N/2 and -1 for k >= N/2 (a step function),
then s is not periodic in the alternating sense.
Sum(s) = 0.
If w is constant, sum(h) = 0.
If w is linear, w[k] = a*k + b.
sum(h) = sum_{k<N/2} (a k + b) - sum_{k>=N/2} (a k + b).
This is generally non-zero unless a and b are specific values.
So a step-function base with a linear window will likely have non-zero sum.

Let's try:
Base: s[k] = 1 if k < N/2 else -1.
Window: w[k] = 1 + sum_{j=1}^D c_j (x_k)^j, where x_k = k/(N-1) - 0.5.
We optimize c_1...c_D.
The DC term of the window is fixed at 1 to keep scale manageable, or we can optimize it too.
If we optimize the DC term, we might drift to 0.
Let's fix the DC term of the window to 1.
So w[k] = 1 + c_1 x_k + c_2 x_k^2 + ...

We will use scipy.optimize.minimize (L-BFGS-B) on the coefficients c.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 140.0

def _evaluate(h):
    """Compute the task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _build_h(coeffs, base_type='step'):
    """
    Construct h from polynomial window coefficients.
    coeffs[0] is the coefficient for x^1, coeffs[1] for x^2, etc.
    The constant term of the window is fixed to 1.
    """
    n = np.arange(N)
    if base_type == 'step':
        s = np.ones(N)
        s[N//2:] = -1.0
    else: # alternating
        s = 1.0 - 2.0 * (n % 2)
    
    # x in [-1, 1]
    x = (n - (N - 1) / 2.0) / ((N - 1) / 2.0)
    
    # Start with constant 1
    w = np.ones(N)
    
    # Add polynomial terms
    for i, c in enumerate(coeffs):
        w += c * (x ** (i + 1))
        
    h = s * w
    return h

def _objective(coeffs, base_type='step'):
    """
    Objective function for the optimizer.
    """
    h = _build_h(coeffs, base_type)
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    """
    Construct step heights using square wave base and polynomial window optimization.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_val = _evaluate(np.ones(N))
    best_h = np.ones(N)
    
    # Degree of polynomial window
    D = 6 # Try degrees 1 to 6
    
    rng = np.random.default_rng(42)
    
    base_types = ['step', 'alt']
    
    for bt in base_types:
        for deg in range(1, D + 1):
            if time.time() > deadline - 10:
                break
                
            # Initialize coefficients to 0
            coeffs0 = np.zeros(deg)
            
            # Run optimization
            try:
                res = minimize(
                    _objective,
                    coeffs0,
                    args=(bt,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < best_val:
                    h_opt = _build_h(res.x, bt)
                    if np.all(np.isfinite(h_opt)) and np.sum(h_opt) != 0:
                        v_opt = _evaluate(h_opt)
                        if v_opt < best_val:
                            best_val = v_opt
                            best_h = h_opt.copy()
                            
            except Exception:
                pass
            
            # Also try with random initializations
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                coeffs_rand = rng.normal(0, 0.5, size=deg)
                try:
                    res = minimize(
                        _objective,
                        coeffs_rand,
                        args=(bt,),
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if res.success or res.fun < best_val:
                        h_opt = _build_h(res.x, bt)
                        if np.all(np.isfinite(h_opt)) and np.sum(h_opt) != 0:
                            v_opt = _evaluate(h_opt)
                            if v_opt < best_val:
                                best_val = v_opt
                                best_h = h_opt.copy()
                except Exception:
                    pass

    # Final polish with greedy steps on the full vector
    if time.time() < deadline - 5:
        h_curr = best_h.copy()
        v_curr = best_val
        step_size = 0.05
        for _ in range(2000):
            if time.time() > deadline - 2:
                break
            idx = rng.integers(0, N)
            h_new = h_curr.copy()
            h_new[idx] += rng.normal(0, step_size)
            v_new = _evaluate(h_new)
            if v_new < v_curr:
                v_curr = v_new
                h_curr = h_new
                step_size *= 0.95
            else:
                step_size = min(step_size * 1.1, 0.2)
        
        if v_curr < best_val:
            best_val = v_curr
            best_h = h_curr

    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END

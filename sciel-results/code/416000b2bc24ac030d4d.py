"""
Strategy: Spectral Domain Optimization
The autoconvolution of h is the inverse DFT of |H|^2, where H is the DFT of h.
To minimize the peak of the autoconvolution (for a fixed sum, which is fixed by H[0]),
we want |H|^2 to be as "flat" as possible across non-zero frequencies, avoiding large
peaks in the spectrum which correspond to oscillatory components in h that can create
peaks in the convolution.
However, a perfectly flat spectrum (white noise) leads to a convolution that is a spike
at lag 0 (delta function), which has a huge peak. 
Wait, let's re-evaluate.
conv(h, h)[k] = sum h[i] h[i+k].
This is the autocorrelation.
The peak is at lag 0: sum h[i]^2.
We want to minimize 2*N * max(conv) / (sum h)^2.
Since max(conv) >= conv[0] = sum h[i]^2.
By Cauchy-Schwarz, (sum h)^2 <= N * sum h[i]^2.
So sum h[i]^2 >= (sum h)^2 / N.
Thus max(conv) >= (sum h)^2 / N.
Then value >= 2*N * ((sum h)^2 / N) / (sum h)^2 = 2.
The seed gives 2.0.
Is it possible to go below 2?
The inequality is derived from the fact that max(conv) is at least the average value of the convolution?
No, the average value of the full convolution (length 2N-1) is not simply (sum h)^2 / (2N-1).
The sum of all entries in conv(h,h) is (sum h)^2.
So the average entry is (sum h)^2 / (2N-1).
Therefore, max(conv) >= (sum h)^2 / (2N-1).
Then value >= 2*N * ((sum h)^2 / (2N-1)) / (sum h)^2 = 2N / (2N-1).
For N=400, 2N/(2N-1) = 800/799 ≈ 1.00125.
So the theoretical lower bound is ~1.00125.
To achieve this, the convolution must be as flat as possible, i.e., all entries of conv(h,h) should be equal to (sum h)^2 / (2N-1).
This implies h must be a "perfect sequence" or similar, but for real numbers, we can approximate a flat convolution.
A flat convolution in time domain corresponds to a flat magnitude spectrum in frequency domain?
No. Convolution in time is multiplication in frequency.
Let H be DFT(h).
Conv(h,h) is not simply related to |H|^2 in a way that makes flatness easy.
Actually, the circular autocorrelation is the inverse DFT of |H|^2.
The linear autocorrelation is the circular autocorrelation zero-padded? No.
Linear autocorrelation r[k] = sum h[i] h[i+k].
This is the inverse DFT of |H|^2 only if we consider circular convolution.
For linear convolution, the relationship is more complex.
However, if we choose h such that its linear autoconvolution is flat, we achieve the bound.
This is related to "low autocorrelation" sequences.
But we can optimize directly in the time domain with a better initialization or a different optimizer.
The previous program used L-BFGS-B on a soft-max surrogate.
A different approach: Use Simulated Annealing or Differential Evolution to directly minimize the true objective function, which is non-smooth but well-defined.
Given the 120s limit, we can run a global optimizer like Differential Evolution for a short time, or use a multi-start local search with a better cost function.
Let's try a different structural change:
Represent h as a low-rank combination of basis functions that naturally have flat autocorrelation properties, or simply use a global optimizer (Differential Evolution) on the raw 400-dim space, which is feasible if we limit the population size and generations.
Alternatively, we can use the fact that we want to minimize the max of the convolution.
We can formulate this as a convex optimization problem if we relax the max?
Minimize t
s.t. conv(h,h)[k] <= t for all k
     sum(h) = 1 (normalization, since value is homogeneous of degree 0 in h)
     h is real
This is a Second Order Cone Program (SOCP) or Linear Program if we fix the max?
conv(h,h)[k] is quadratic in h. So it's a Quadratic Constraint.
Minimize t
s.t. h^T A_k h <= t
     sum(h) = 1
This is a Quadratic Program (QP) with quadratic constraints.
We can solve this using scipy.optimize.minimize with SLSQP or trust-constr, enforcing the constraints.
This is a structural change: from "minimize a surrogate" to "solve a constrained optimization problem directly".
"""

import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

N = 400
_DEADLINE = 140.0

def _evaluate(h):
    """Compute the task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _conv_matrix(k):
    """
    Return the matrix A_k such that h^T A_k h = conv(h,h)[k].
    conv(h,h)[k] = sum_{i} h[i] h[i+k] for k >= 0.
    For k < 0, conv(h,h)[k] = conv(h,h)[N-1-k] ? No, linear conv.
    Let's define the indices properly.
    h has length N.
    conv = np.convolve(h, h) has length 2N-1.
    conv[k] = sum_{i=max(0, k-N+1)}^{min(N-1, k)} h[i] h[k-i] ? 
    Standard definition: (f*g)[n] = sum_m f[m] g[n-m].
    Here f=h, g=h.
    conv[k] = sum_{m} h[m] h[k-m].
    Indices m such that 0<=m<N and 0<=k-m<N.
    So max(0, k-(N-1)) <= m <= min(N-1, k).
    
    We can construct the matrix A_k where (A_k)_{ij} = 1 if i+j = k? No, h[i]h[j] with i+j=k?
    Wait, conv[k] = sum_m h[m] h[k-m]. Let j = k-m. Then m+j=k.
    So conv[k] = sum_{m+j=k} h[m] h[j].
    This is a quadratic form h^T A_k h where A_k has 1s at (m, k-m) and (k-m, m)?
    Since h[m]h[j] is symmetric, A_k should be symmetric.
    (A_k)_{m, j} = 0.5 if m+j=k and m!=j?
    Actually, h^T A h = sum_{m,j} h[m] A_{m,j} h[j].
    If we set A_{m,j} = 1 if m+j=k, then h^T A h = sum_{m+j=k} h[m]h[j].
    This includes the diagonal terms once if m=j, and off-diagonal twice?
    No, sum_{m,j} includes both (m,j) and (j,m).
    If m != j, h[m]h[j] appears twice.
    But conv[k] = sum_{m+j=k} h[m]h[j].
    If m != j, the term h[m]h[j] is just one term in the sum over m.
    Wait, the sum is over m. j is determined by k-m.
    So for a specific pair (m, j) with m+j=k, it appears once in the sum over m (when m is chosen).
    So conv[k] = sum_{m} h[m] h[k-m].
    This is not h^T A h in the standard symmetric form directly unless we handle the factor of 2.
    Let's just use the explicit formula for the constraint gradient.
    """
    pass

def _objective_and_grad(x):
    """
    Objective: t
    Constraints:
    1. sum(x) = 1
    2. conv(x,x)[k] <= t for all k
    We minimize t.
    We use scipy.optimize.minimize with method='SLSQP'.
    We need to define the constraints.
    SLSQP handles nonlinear constraints.
    """
    h = x[:-1]
    t = x[-1]
    return t

def _constraint_sum(x):
    return np.sum(x[:-1]) - 1.0

def _constraint_conv(x):
    """
    Return an array of values: conv(h,h)[k] - t <= 0 for all k.
    """
    h = x[:-1]
    t = x[-1]
    conv = np.convolve(h, h)
    return conv - t

def _construct():
    t0 = time.time()
    
    # Initial guess:
    # We want h to be such that conv(h,h) is flat.
    # A uniform h gives a triangular conv.
    # A random h gives a spike.
    # Let's start with h = 1/N (sum=1).
    h_init = np.ones(N) / N
    t_init = 2.0 * N * np.max(np.convolve(h_init, h_init)) / (np.sum(h_init)**2)
    # For uniform, max conv is at 0: N*(1/N)^2 = 1/N.
    # value = 2*N * (1/N) / 1 = 2.
    
    x_init = np.concatenate([h_init, [t_init]])
    
    # Define constraints for SLSQP
    # Inequality: conv(h,h) - t <= 0
    # This is a vector of 2N-1 constraints.
    # SLSQP might be slow with 799 constraints and 401 variables.
    # Let's try it.
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[:-1]) - 1.0},
        {'type': 'ineq', 'fun': lambda x: x[-1] - np.convolve(x[:-1], x[:-1])}
        # Note: SLSQP expects g(x) >= 0 for ineq.
        # So t - conv >= 0 => t >= conv.
    ]
    
    try:
        res = minimize(
            _objective_and_grad,
            x_init,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        
        if res.success:
            h_opt = res.x[:-1]
            val_opt = _evaluate(h_opt)
            if val_opt < 2.0:
                return h_opt
        else:
            # If SLSQP fails, try a different initialization or method
            pass
            
    except Exception as e:
        pass

    # Fallback: Differential Evolution on a smaller set of parameters?
    # Or just return the seed if optimization fails.
    # Let's try a simpler local search from a better start.
    # Start with a "flat" spectrum approximation.
    # H = 1 for low freqs, 0 for high freqs?
    # This is a low-pass filter.
    # The time domain h will be a sinc-like function.
    # The autocorrelation of a sinc is a triangle? No, sinc autocorr is sinc.
    # We want flat autocorr.
    
    # Let's just return the seed if the constrained optimization didn't work well.
    return np.ones(N)

def construct():
    return _construct()

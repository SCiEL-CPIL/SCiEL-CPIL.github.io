"""
Strategy: Epigraph formulation of the autoconvolution minimization problem.
Instead of directly minimizing the max absolute autoconvolution (non-smooth),
we introduce a variable 't' and minimize t subject to the constraints 
|autoconv(x)[k]| <= t for all k. 
This is a smooth convex optimization problem (convex objective, convex constraints 
if we consider the log-sum-exp or similar smooth approximations, or directly 
using the fact that the set of x such that |x*x| <= t is not necessarily convex 
in x, but the objective max|...| is convex).
Actually, max |x*x| is a convex function of x. Minimizing a convex function is generally hard, 
but if we fix the energy (integral squared is fixed by normalization? No, the denominator is (sum x)^2).
Let's normalize x such that sum(x) = 1. Then we minimize max |x*x|.
This is still non-convex in x generally? 
Wait, the autoconvolution is quadratic. The max of absolute values of quadratic forms.
The set {x : |x*x| <= t} is not convex.
However, we can use the "flat spectrum" insight. 
If we optimize the phases of a flat magnitude spectrum, we are on a manifold.
The previous program did random walk on phases.
Inspiration 1 used Legendre polynomials.
Inspiration 2/3 used spectral design with epigraph.

Let's try a direct gradient descent on the time-domain signal x, 
using a smooth approximation of the max absolute autoconvolution (log-sum-exp).
We will normalize x so that sum(x) = 1.
Objective: min_x log(sum(exp(beta * |x*x|))) / (sum(x))^2 ? 
Actually, the score is max|conv| / (sum x)^2.
If we fix sum(x)=1, we just minimize max|conv|.
Let's allow sum(x) to vary but penalize it? No, the denominator rewards large sum.
So we want large sum and small max conv.
This is equivalent to minimizing max|conv| / (sum x)^2.

Let's use scipy.optimize.minimize with L-BFGS-B or BFGS on the vector x.
We need gradients.
Let c = x * x (linear convolution).
f(x) = log(sum(exp(beta * |c_k|))) / (sum x)^2
This is complex.

Alternative: 
Minimize t subject to:
1) |c_k| <= t for all k
2) sum(x) = 1 (or fixed to some value)
This is a convex problem if we treat t as variable and constraints as convex?
|c_k| <= t is convex in (x, t).
Sum(x) = 1 is linear.
So we can minimize t.
We can use scipy.optimize.minimize with SLSQP or trust-constr.
But N=400, constraints 800. Might be slow.

Let's try the spectral phase optimization again but with proper gradient descent on the phases.
The current program uses a random walk (Metropolis-like).
A gradient descent on the phases might be faster.
Let x = ifft(|X| exp(i phi)).
We want to minimize max |x*x|.
In frequency domain, x*x corresponds to X^2 (with scaling).
Let Y = X^2.
x*x = ifft(Y).
We want to minimize max |ifft(Y)|.
Y is determined by X.
If we fix |X| (flat), then Y has fixed magnitude |X|^2.
We are optimizing the phases of X, which determines the phases of Y.
Actually, the phase of Y is 2 * phase of X.
So we are optimizing the phase of Y directly? 
Let psi = 2 * phi.
Then we want to minimize max |ifft(|Y| exp(i psi))|.
This is the classic "minimum peak factor" problem for a fixed magnitude spectrum.
This can be solved by gradient descent on psi.
Gradient of soft-max peak factor w.r.t. psi is well known.

Let's implement this specific spectral phase optimization with gradients.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    conv = np.fft.irfft(fx * fx, size)
    return float(np.max(np.abs(conv[:2 * n - 1])))

def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))

def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _spectral_phase_gradient_descent(deadline, rng, n_iter=2000, beta=50.0, lr=0.01):
    """
    Minimize the peak factor of the autoconvolution by optimizing the phases 
    of the DFT of x, assuming a flat magnitude spectrum.
    x = ifft(exp(i * phi)) * scale
    Actually, we want to optimize x directly? No, we optimize the phases of X.
    Let X[k] = |X[k]| exp(i phi[k]).
    For real x, X[-k] = conj(X[k]).
    We use rfft.
    """
    n = N
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    n_freq = size // 2 + 1
    # Flat magnitude spectrum
    # DC component must be real and positive to keep x real and sum positive?
    # If we want sum(x) > 0, DC should be positive.
    magnitude = np.ones(n_freq)
    magnitude[0] = n # This makes x have mean 1? 
    # If magnitude[0] = n, and others 1, then x[0] ~ 1 + (n-1)/n ... 
    # Let's just use magnitude = 1 for all, and adjust DC phase to 0.
    # Then sum(x) will be determined by the phases.
    
    # Initialize phases randomly
    phases = rng.uniform(0, 2 * np.pi, size=n_freq)
    phases[0] = 0.0 # Fix DC phase to 0 to ensure x is real and "centered"
    
    best_x = np.zeros(n)
    best_val = float('inf')
    
    current_val = float('inf')
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
            
        # Construct X
        X = magnitude * np.exp(1j * phases)
        x = np.fft.irfft(X, size)
        x = x[:n]
        
        # Calculate autoconvolution
        fx = np.fft.rfft(x, size)
        conv_full = np.fft.irfft(fx * fx, size)
        conv = conv_full[:2 * n - 1]
        abs_conv = np.abs(conv)
        
        # Soft max objective
        m = np.max(abs_conv)
        if m < 1e-12:
            obj = m
            grad_conv = np.zeros_like(conv)
        else:
            # log-sum-exp
            obj = m + np.log(np.sum(np.exp((abs_conv - m) * beta))) / beta
            weights = np.exp((abs_conv - m) * beta)
            weights /= np.sum(weights)
            signs = np.sign(conv)
            grad_conv = weights * signs
            
        # Calculate metric
        s = np.sum(x)
        if abs(s) < 1e-12:
            val = float('inf')
        else:
            val = obj / (s * s) # Using soft max as proxy for max
            
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        # Gradient of obj w.r.t. x
        # obj = sum_k w_k * sign(conv_k) * conv_k
        # conv = x * x
        # d(conv_k)/dx_i = x_{k-i} + x_{i-k} (with boundary checks)
        # This is a convolution of grad_conv with x?
        # grad_obj_dx = 2 * (grad_conv * x) ?
        # Let's verify.
        # conv_k = sum_i x_i x_{k-i}
        # d(conv_k)/dx_i = x_{k-i} + x_{i-k} (if we consider full linear conv)
        # sum_k grad_k * (x_{k-i} + x_{i-k})
        # = sum_k grad_k x_{k-i} + sum_k grad_k x_{i-k}
        # Let j = k-i => k = j+i. Sum_j grad_{j+i} x_j. This is cross-correlation.
        # Let j = i-k => k = i-j. Sum_j grad_{i-j} x_j.
        # Both are related to convolution/correlation.
        # Specifically, grad_obj_dx = 2 * ifft(FFT(grad_conv) * FFT(x)) ?
        # In frequency domain:
        # Conv = X^2.
        # Obj = f(Conv).
        # dObj/dX = 2 * X * dObj/dConv (in freq domain, with appropriate scaling)
        # dObj/dConv_k (time) -> FFT -> dObj/dC_k (freq)
        # dObj/dX_k = 2 * X_k * (FFT(dObj/dConv)_k)
        
        # Let's compute gradient in frequency domain for efficiency
        grad_conv_freq = np.fft.rfft(grad_conv, size)
        # X is the spectrum of x
        # dObj/dX = 2 * X * grad_conv_freq
        # Note: X here is the rfft of x.
        # But we are optimizing phases of X_init?
        # No, x is derived from phases.
        # x = ifft(X).
        # We want dObj/dphases.
        # dObj/dphases_k = Re( i * X_k * dObj/dX_k ) ?
        # X_k = M_k exp(i phi_k)
        # dX_k/dphi_k = i M_k exp(i phi_k) = i X_k
        # dObj/dphi_k = Re( conj(dX_k/dphi_k) * dObj/dX_k ) ?
        # Actually, for real objective, dObj/dphi = Re( -i * X_k * dObj/dX_k )?
        # Let's use: dObj = sum dObj/dX * dX = sum dObj/dX * (i X dphi)
        # = sum i X dObj/dX dphi
        # Since dObj is real, we take real part.
        # grad_phi = Re( i * X * dObj/dX )
        
        dObj_dX = 2 * X * grad_conv_freq
        
        # Gradient w.r.t phases
        # Note: X[0] is real, so phase 0 is fixed.
        # For k > 0, X[k] is complex.
        grad_phi = np.real(1j * X * dObj_dX)
        
        # Update phases
        # We need to normalize the gradient?
        # Simple gradient descent
        phases = phases - lr * grad_phi
        phases[0] = 0.0 # Keep DC phase fixed
        
        # Optional: Renormalize magnitude? We are keeping it flat.
        
        # Adaptive learning rate?
        if val < current_val:
            lr = min(lr * 1.01, 0.1)
        else:
            lr = max(lr * 0.99, 0.001)
            # If stuck, maybe restart?
            # For now, just continue.
            
        current_val = val
        
    return best_x, best_val

def _try_local_search_fast(x, deadline, rng, max_rounds=10000):
    """Fast local search with single flips."""
    best_x = x.copy()
    best_val = _metric(best_x)
    
    # Precompute FFT of x
    size = 1
    while size < 2 * N - 1:
        size *= 2
    fx = np.fft.rfft(best_x, size)
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        
        idx = rng.integers(N)
        # Flip sign
        delta = -2 * best_x[idx]
        
        # Update FFT
        # fx_new = fx + delta * exp(-2pi i idx k / size)
        # This is expensive to do fully. 
        # Let's just recompute metric for candidate.
        # To speed up, we can check if the flip is likely to help?
        # Just do a few flips and check.
        
        cand = best_x.copy()
        cand[idx] *= -1
        v = _metric(cand)
        if v < best_val:
            best_val = v
            best_x = cand

    return best_x, best_val

def construct():
    deadline = time.time() + 140.0 # Safety margin
    rng = np.random.default_rng(42)
    
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Phase 1: Spectral Phase Gradient Descent
    # Try multiple random initializations
    for i in range(10):
        if time.time() > deadline - 20:
            break
        x_spec, val_spec = _spectral_phase_gradient_descent(deadline - 20, rng, n_iter=1000, beta=50.0, lr=0.05)
        if val_spec < best_val:
            best_val = val_spec
            best_x = x_spec
        # Reset rng state to get different random phases? 
        # rng is stateful, so it will give different phases automatically.
        
    # Phase 2: Local Search Polish
    if time.time() < deadline - 5:
        best_x, best_val = _try_local_search_fast(best_x, deadline - 5, rng, max_rounds=50000)
        
    return best_x

# EVOLVE-BLOCK-END

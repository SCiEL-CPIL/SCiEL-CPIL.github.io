"""
Strategy: Spectral Peak Minimization via Gradient Descent on Soft-Max Objective.

The current program uses discrete heuristics (M-sequences, blocks) and local search.
This new approach treats the sequence as a continuous vector x in R^N.
We minimize a smooth approximation of the maximum absolute autoconvolution.
By Parseval's theorem, the autoconvolution in time domain corresponds to the 
power spectrum in frequency domain. Minimizing the peak of the power spectrum 
(minimax spectral shaping) leads to sequences with low sidelobes.

We use a "soft-max" function to make the objective differentiable:
  f(x) = log(sum(exp(beta * |ac|^2))) / beta  (approximates max |ac|^2)
Actually, the metric is max|ac| / (sum x)^2.
Let's minimize max|ac| directly using a soft-max on |ac|.
Or better, minimize the peak of the power spectrum P(f) = |FFT(x)|^2.
max|ac| is proportional to max P(f) (roughly, depending on normalization).
Actually, ac = IFFT(|FFT(x)|^2). So max|ac| is the peak of the inverse FFT of the power spectrum.
This is the "peak-to-average power ratio" of the spectrum, but in time domain of the spectrum.

Let's define the objective as:
  J(x) = max_k |ac[k]| / (sum(x))^2
To make it smooth, we use:
  J_soft(x) = (1/beta) * log( sum_k exp(beta * |ac[k]|) ) / (sum(x))^2
We perform gradient descent on J_soft(x) with respect to x.
Then we project x to signed steps: x[i] = 1 if x[i] >= 0 else -1.
We alternate between gradient steps on the continuous vector and projection.

This is a "spectral shaping" or "minimax" approach, distinct from local search.
"""

import numpy as np
import time

N = 400
_DEADLINE = time.time() + 140.0

def _autoconv(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n
    fx = np.fft.rfft(x, size)
    ac = np.fft.irfft(fx * np.conj(fx), size)
    return ac[:n]

def _metric(x):
    """max |autoconv| / (sum x)^2"""
    ac = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_abs_ac(x, beta):
    """
    Compute a smooth approximation of max |ac| and its gradient.
    We minimize max |ac| / S^2.
    Let M = max |ac|.
    We use log-sum-exp trick for stability.
    Let y = |ac|.
    LSE = (1/beta) * log(sum(exp(beta * y)))
    Gradient of LSE w.r.t x:
    dLSE/dx = (1/beta) * (1/sum(exp(beta*y))) * sum( beta * exp(beta*y) * dy/dx )
    dy/dx = sign(ac) * d(ac)/dx
    d(ac)/dx is the convolution of x with x? No, ac = x * x.
    d(ac)/dx is a linear operator.
    Specifically, ac[k] = sum_i x[i] x[i+k].
    d(ac[k])/dx[j] = x[j+k] + x[j-k] (with appropriate bounds).
    This is equivalent to the convolution of x with a shifted version?
    Actually, the gradient of the quadratic form x^T T_k x is (T_k + T_k^T)x.
    For autoconvolution, the operator is related to the circulant matrix of x?
    Let's use the FFT domain.
    ac = IFFT(|FFT(x)|^2).
    Let F be the FFT matrix. ac = F^H |F x|^2.
    This is getting complex to derive analytically.
    
    Alternative: Use the fact that we want to flatten the spectrum.
    If we minimize max |ac|, we are minimizing the peak of the time-domain representation of the power spectrum.
    This is equivalent to minimizing the peak of the "energy" of the spectrum in time domain.
    
    Let's try a simpler gradient:
    We want to reduce the peak of ac.
    The gradient of max|ac| w.r.t x can be approximated by:
    Let k_max = argmax |ac|.
    grad = sign(ac[k_max]) * d(ac[k_max])/dx.
    d(ac[k_max])/dx = x shifted by k_max + x shifted by -k_max.
    Specifically, if ac[k] = sum_i x[i] x[i+k], then
    d(ac[k])/dx[j] = x[j+k] + x[j-k] (if indices valid).
    This is a vector.
    
    Let's implement this subgradient step.
    """
    ac = _autoconv(x)
    abs_ac = np.abs(ac)
    # Soft max weights
    # Shift for numerical stability
    max_val = np.max(abs_ac)
    if max_val < 1e-10:
        return 0.0, np.zeros_like(x)
    
    weights = np.exp(beta * (abs_ac - max_val))
    sum_weights = np.sum(weights)
    
    # Gradient of LSE w.r.t ac
    # d/dac [ (1/beta) log sum exp(beta |ac|) ] = (exp(beta |ac|) / sum) * sign(ac)
    grad_ac = (weights / sum_weights) * np.sign(ac)
    
    # Now map grad_ac to grad_x
    # ac[k] = sum_i x[i] x[i+k]
    # d(ac[k])/dx[j] = x[j+k] + x[j-k]
    # So grad_x[j] = sum_k grad_ac[k] * (x[j+k] + x[j-k])
    # This is a convolution!
    # grad_x = convolve(grad_ac, x) + convolve(grad_ac, x[::-1])?
    # Let's check.
    # sum_k grad_ac[k] x[j+k] is the cross-correlation of grad_ac and x.
    # sum_k grad_ac[k] x[j-k] is the convolution of grad_ac and x.
    
    # Using FFT for efficiency
    size = 2 * N
    g_ac_fft = np.fft.rfft(grad_ac, size)
    x_fft = np.fft.rfft(x, size)
    
    # Term 1: sum_k grad_ac[k] x[j+k]
    # This is correlation. Correlation is convolution with reversed kernel.
    # Or simply: if we define cross_corr(a, b)[j] = sum_k a[k] b[j+k]
    # In FFT: IFFT(FFT(a) * conj(FFT(b)))? No.
    # Let's stick to direct convolution definition in time domain for clarity, 
    # but N=400, so O(N^2) is 160k ops, which is fast.
    # However, we are doing this in a loop. FFT is safer.
    
    # Let's use the property:
    # d/dx || x * x ||_inf
    # Let's just compute the gradient via finite differences? No, too slow.
    
    # Let's use the analytic gradient via FFT.
    # ac = ifft(fft(x) * conj(fft(x)))
    # Let X = fft(x). ac = ifft(X * conj(X))
    # Let LSE = f(ac)
    # grad_x = ifft( X * (dLSE/dac) * 2 ? )
    # Chain rule:
    # ac = ifft(P) where P = X * conj(X)
    # dLSE/dP = ?
    # P is real and non-negative.
    # ac is real.
    # The mapping P -> ac is linear (IFFT).
    # So dLSE/dP = fft(grad_ac) (in real domain, careful with factors)
    # Actually, d(ac)/dP is the IFFT matrix.
    # So dLSE/dP = FFT(grad_ac) (with appropriate scaling)
    
    # Then dLSE/dX = dLSE/dP * dP/dX
    # P = X * conj(X)
    # dP/dX = 2 * conj(X) (for real X? No, X is complex)
    # For complex variables, dP/dX = conj(X) and dP/dconj(X) = X.
    # Since x is real, X has conjugate symmetry.
    # The gradient w.r.t x (real) is:
    # grad_x = ifft( 2 * conj(X) * dLSE/dP ) ?
    # Let's verify with a simple case.
    # If we minimize sum ac[k]^2, grad_x = 4 * x * x (circular)?
    # Here we have a non-linear function of ac.
    
    # Let's use the following known result for minimizing f(ifft(X conj(X))):
    # grad_x = ifft( X * g ) where g = fft(grad_ac) * 2?
    # Let's test this numerically in the code or use a robust finite difference check for the first few steps?
    # No, let's use the standard Wirtinger calculus.
    # L = f(ac), ac = ifft(X * conj(X))
    # dL/dX = conj( dL/dconj(X) )
    # dL/dconj(X) = ifft( dL/dac * conj(X) ) ? No.
    # d(ac)/dconj(X) = ifft( X ) = x (shifted?)
    # Actually, ac = x * x (linear).
    # The gradient of f(x*x) w.r.t x is 2 * (f'(x*x) * x) ? No, that's for scalar.
    # For vectors, grad_x = 2 * (grad_ac * x) ? No, grad_ac is a vector.
    # grad_x = 2 * (convolution of grad_ac and x)?
    # Let's check:
    # ac[k] = sum_i x[i] x[i+k]
    # d(ac[k])/dx[j] = x[j+k] + x[j-k]
    # grad_x[j] = sum_k grad_ac[k] (x[j+k] + x[j-k])
    # This is indeed 2 * correlation(grad_ac, x) if grad_ac is symmetric.
    # But grad_ac is not necessarily symmetric.
    # So grad_x = correlation(grad_ac, x) + convolution(grad_ac, x)
    
    # Let's implement this using FFT.
    # corr(a, b)[j] = sum_k a[k] b[j+k]
    # conv(a, b)[j] = sum_k a[k] b[j-k]
    
    # FFT of corr(a, b) = FFT(a) * conj(FFT(b)) ?
    # Let's just use np.convolve for simplicity since N=400.
    # But we need linear convolution, not circular.
    # The autoconvolution we computed is linear (size 2N, take first N).
    # So the gradient should also be based on linear convolution.
    
    # Let's compute grad_x directly using the formula.
    # To make it fast, we can use FFT for the correlation/convolution.
    
    # Create full arrays of size 2N
    grad_ac_full = np.zeros(2 * N)
    grad_ac_full[:N] = grad_ac
    
    x_full = np.zeros(2 * N)
    x_full[:N] = x
    
    # Term 1: sum_k grad_ac[k] x[j+k]
    # This is cross-correlation.
    # Using FFT: IFFT(FFT(grad_ac_full) * conj(FFT(x_full)))
    # This gives circular correlation. Since we zero-padded to 2N, it is linear.
    g1_fft = np.fft.rfft(grad_ac_full, 2 * N)
    x_fft = np.fft.rfft(x_full, 2 * N)
    g1 = np.fft.irfft(g1_fft * np.conj(x_fft), 2 * N)
    
    # Term 2: sum_k grad_ac[k] x[j-k]
    # This is convolution.
    # Using FFT: IFFT(FFT(grad_ac_full) * FFT(x_full))
    g2 = np.fft.irfft(g1_fft * x_fft, 2 * N)
    
    grad_x = g1[:N] + g2[:N]
    
    # Normalize the gradient?
    # The objective is J = LSE / S^2.
    # We need to divide by S^2 and account for the derivative of 1/S^2.
    # Let L = LSE. J = L * S^-2.
    # dJ/dx = (dL/dx * S^-2) + L * (-2) * S^-3 * dS/dx
    # dS/dx = 1 (vector of ones)
    
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 0.0, np.zeros_like(x)
        
    dL_dx = grad_x
    dJ_dx = dL_dx / (s * s) - 2 * LSE_val / (s ** 3) * np.ones(N)
    
    # Compute LSE_val for return
    LSE_val = max_val + (1.0 / beta) * np.log(sum_weights)
    
    return LSE_val, dJ_dx

def _gradient_descent_step(x, lr, beta):
    """
    Perform a gradient descent step on the continuous vector x.
    """
    obj, grad = _soft_max_abs_ac(x, beta)
    
    # Gradient descent
    x_new = x - lr * grad
    
    # Optional: Project to have zero mean or something?
    # The metric is invariant to scaling, but sensitive to sign flips.
    # We will project to signs later.
    
    return x_new, obj

def construct():
    start_time = time.time()
    
    # Initialize with random signs
    np.random.seed(42)
    x = np.random.choice([-1.0, 1.0], size=N)
    
    # Also try starting from ones
    x_ones = np.ones(N)
    
    best_x = x
    best_m = _metric(x)
    
    # Parameters for gradient descent
    beta = 10.0  # Temperature for soft-max
    lr = 0.1     # Learning rate
    
    # We will alternate between:
    # 1. Continuous gradient descent steps
    # 2. Projection to signs
    # 3. Local search on signs (optional)
    
    # Let's run a loop
    # We have ~140 seconds.
    # Each step: FFT (fast), gradient computation (fast), update (fast).
    # We can do many iterations.
    
    current_x = x.copy()
    
    # Adaptive beta: start low, increase?
    # Or fixed beta.
    
    for i in range(10000):
        if time.time() > _DEADLINE - 10:
            break
        
        # Gradient step
        current_x, obj = _gradient_descent_step(current_x, lr, beta)
        
        # Project to signs
        x_proj = np.sign(current_x)
        x_proj[x_proj == 0] = 1.0
        
        # Evaluate
        m_proj = _metric(x_proj)
        
        if m_proj < best_m:
            best_m = m_proj
            best_x = x_proj.copy()
            
        # Occasionally, re-seed from best_x to avoid getting stuck in a bad basin?
        # Or just continue.
        
        # Adjust learning rate?
        # If obj increases, reduce lr?
        # For now, fixed lr.
        
        # Every 100 steps, try a few random restarts?
        if i % 500 == 0:
            # Try a random restart
            x_rand = np.random.choice([-1.0, 1.0], size=N)
            m_rand = _metric(x_rand)
            if m_rand < best_m:
                best_m = m_rand
                best_x = x_rand.copy()
                current_x = x_rand.copy()

    # Final polish with local search
    if time.time() < _DEADLINE - 5:
        # Local search on best_x
        x_ls = best_x.copy()
        m_ls = best_m
        for j in range(5000):
            if time.time() > _DEADLINE - 2:
                break
            idx = np.random.randint(0, N)
            x_ls[idx] *= -1.0
            m_new = _metric(x_ls)
            if m_new < m_ls:
                m_ls = m_new
                best_x = x_ls.copy()
                best_m = m_ls
            else:
                x_ls[idx] *= -1.0

    return best_x

# Keep the old functions just in case, but they are not used in the main loop
# except for _autoconv and _metric which are reused.

def _try_constant():
    x = np.ones(N)
    return x, _metric(x)

def _try_m_sequence():
    return None, np.inf

def _try_alternating_blocks():
    return None, np.inf

def _try_gaussian_weighted():
    return None, np.inf

def _try_local_search(x0, max_iter=5000):
    return x0, _metric(x0)

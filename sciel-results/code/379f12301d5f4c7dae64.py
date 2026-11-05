import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 145.0

def _evaluate(h):
    """
    Computes the exact value used by the evaluator.
    value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    """
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _h_from_coeffs(coeffs, bins):
    """
    Reconstructs h from sparse frequency coefficients.
    coeffs: array of complex numbers corresponding to the bins.
    bins: array of integer frequency indices (0 to N-1).
    
    h is the IDFT of the spectrum X, where X is zero except at 'bins'.
    X[bins[i]] = coeffs[i]
    h = np.fft.ifft(X).real * N  (Standard IDFT definition: h[k] = (1/N) sum X[m] exp(2pi i mk/N))
    Wait, np.fft.ifft(X) gives h[k] = (1/N) sum ...
    So if we set X[m] = coeffs, then h = ifft(X).
    The sum of h is h[0] * N? No.
    sum(h) = N * X[0] if X[0] is DC.
    Actually, sum(h) = N * X[0] is only true if we define DFT as X[k] = sum h[n] exp(-2pi i nk/N).
    Then h[n] = (1/N) sum X[k] exp(2pi i nk/N).
    Sum_{n=0}^{N-1} h[n] = (1/N) sum_k X[k] sum_n exp(2pi i nk/N).
    sum_n exp(2pi i nk/N) is N if k=0, else 0.
    So sum(h) = X[0].
    
    Therefore, to normalize sum(h)=1, we just need X[0]=1.
    But we want to optimize the shape.
    Let's allow X[0] to be part of the optimization or fix it?
    The value is scale invariant.
    val = |2 N max(conv) / S^2|.
    If we scale h by c, val is invariant.
    So we can fix S=1 by setting the DC component X[0] = 1.
    However, the DC component affects the autocorrelation.
    conv(h,h) in time domain corresponds to |H(f)|^2 in frequency domain?
    No, autocorrelation R[n] = sum h[k] h[k+n].
    DFT of R is |H(f)|^2.
    So max(conv) is related to the peak of the inverse DFT of |H(f)|^2.
    
    Let's optimize the complex coefficients of the non-DC bins, and fix DC=1.
    Or optimize all, and normalize at the end?
    If we optimize all, we have scale ambiguity.
    Better: Fix DC component X[0] = 1.
    Then sum(h) = 1.
    Then val = 2 * N * max(conv).
    We want to minimize max(conv).
    
    Parameters:
    - bins: list of indices.
    - coeffs: complex values for these bins.
    
    We will select a fixed set of bins for the optimization to keep the parameter count low and stable.
    Which bins?
    Low frequencies usually correspond to smooth signals.
    Let's pick a few low frequency bins: [1, 2, 3, 4, 5] and their conjugates?
    Since h is real, X[k] = conj(X[N-k]).
    So we only need to specify bins 1 to N/2.
    Let's choose bins: [1, 2, 3, 4, 5].
    And fix X[0] = 1.
    Parameters: real and imag parts of X[1], X[2], X[3], X[4], X[5].
    Total 10 real parameters.
    
    This is a very low-dimensional optimization problem.
    """
    X = np.zeros(N, dtype=complex)
    X[0] = 1.0  # Fix DC to ensure sum(h) = 1
    
    for i, b in enumerate(bins):
        X[b] = coeffs[i]
        # Ensure real h by setting conjugate symmetry
        if b != 0 and b != N//2:
            X[N-b] = np.conj(coeffs[i])
        elif b == N//2:
            # Nyquist frequency, must be real
            X[b] = coeffs[i].real
            
    h = np.fft.ifft(X).real
    return h

def _objective_and_grad(params, bins):
    """
    Objective function and gradient for L-BFGS-B.
    params: real array of length 2 * len(bins).
    First half: real parts of coeffs.
    Second half: imag parts of coeffs.
    """
    num_bins = len(bins)
    coeffs = params[:num_bins] + 1j * params[num_bins:]
    
    h = _h_from_coeffs(coeffs, bins)
    
    # Evaluate value
    conv = np.convolve(h, h)
    max_val = np.max(conv)
    val = 2.0 * N * max_val  # Since sum(h)=1, S^2=1
    
    # Gradient
    # We need d(val)/d(params).
    # val = 2 N max_k conv_k(h)
    # Let k* be the index of the max.
    # d(val)/d(params) = 2 N * d(conv_{k*}(h))/d(params)
    # conv_k(h) = sum_j h[j] h[j+k] (with appropriate boundaries)
    # h depends on params via FFT.
    # h = ifft(X(params)).
    # This chain rule is complex.
    # Alternative: Use numerical gradient?
    # With only 10 parameters, numerical gradient is very fast and robust.
    # L-BFGS-B with numerical gradient on 10 params is extremely fast.
    
    return val

def _numerical_grad(params, bins, eps=1e-7):
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _objective_and_grad(params_plus, bins)[0]
        
        params_minus = params.copy()
        params_minus[i] -= eps
        f_minus = _objective_and_grad(params_minus, bins)[0]
        
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # Select frequency bins to optimize
    # Low frequencies are good for smoothness and controlling sidelobes.
    # Let's try a few different sets of bins.
    bin_sets = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 3, 5, 7, 9], # Odd only?
        [1, 2, 4, 8, 16], # Powers of 2?
    ]
    
    rng = np.random.default_rng(42)
    
    for bins in bin_sets:
        if time.time() > deadline - 5:
            break
            
        num_params = 2 * len(bins)
        
        # Multiple random restarts for this bin set
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            # Initialize params
            # Small random values to start near DC=1, 0,0...
            # If coeffs are 0, h is a delta function (if DC=1, h[0]=1, others 0? No.)
            # If X[0]=1 and others 0, h = ifft([1,0,...]) = [1/N, 1/N, ...]?
            # Wait. ifft([1,0,...]) = [1/N, 1/N, ..., 1/N].
            # So h is constant.
            # conv of constant is N * h^2 * (N-k).
            # max is at k=0: N * h^2 * N = N^2 h^2.
            # h = 1/N.
            # max = N^2 * (1/N)^2 = 1.
            # val = 2 * N * 1 = 2N = 800.
            # Seed val is 2.0.
            # So constant is bad.
            
            # We need non-trivial coefficients.
            # Let's initialize with random complex values.
            init_real = rng.standard_normal(len(bins)) * 0.5
            init_imag = rng.standard_normal(len(bins)) * 0.5
            params0 = np.concatenate([init_real, init_imag])
            
            try:
                res = minimize(
                    _objective_and_grad,
                    params0,
                    args=(bins,),
                    method='L-BFGS-B',
                    jac=lambda p, b: _numerical_grad(p, b), # Using numerical grad
                    options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000}
                )
                
                # Reconstruct h from best params
                best_params = res.x
                coeffs = best_params[:len(bins)] + 1j * best_params[len(bins):]
                h_candidate = _h_from_coeffs(coeffs, bins)
                
                # Ensure sum is not zero (should be 1 by construction, but check)
                s = np.sum(h_candidate)
                if abs(s) < 1e-6:
                    continue
                    
                # Evaluate
                val_candidate = _evaluate(h_candidate)
                
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
                    
            except Exception:
                continue
                
    # Final check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
        
    return best_h

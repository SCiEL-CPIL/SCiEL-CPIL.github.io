"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))

def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _soft_max_abs(conv, beta=100.0):
    """Smooth approximation of max |conv| using log-sum-exp."""
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    if max_val == 0:
        return 0.0
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta

def _spectral_gradient_descent(x_init, deadline, rng):
    """
    Optimize the magnitude spectrum |X(k)| directly using gradient descent.
    
    Insight: The autoconvolution is IFFT(|X|^2). Minimizing max|IFFT(|X|^2)|
    is equivalent to making the power spectrum |X|^2 have a flat inverse FFT.
    This is the "PAPR reduction" problem for the power spectrum.
    
    We parameterize x by its magnitude spectrum M = |X| and phase P.
    Since the autoconvolution depends only on M (not P), we can fix P to zero 
    (or optimize it, but zero phase is a good start for real symmetric signals) 
    and optimize M.
    
    Actually, fixing phase to zero makes x real and even. This restricts the 
    solution space but is a strong constraint that often yields good results 
    for autocorrelation problems. We will also allow phase optimization later 
    if needed, but let's start with magnitude only.
    
    To allow more flexibility, we can optimize the full complex X, but 
    ensure x is real by enforcing Hermitian symmetry.
    
    Strategy:
    1. Start with x = ones (M is delta, P is zero).
    2. Compute X = FFT(x).
    3. Iterate:
       a. Compute conv = IFFT(|X|^2).
       b. Compute gradient of soft_max(|conv|) w.r.t. Re(X) and Im(X).
       c. Update X.
       d. Enforce Hermitian symmetry to keep x real.
       e. x = IFFT(X).
    
    Gradient derivation:
    Let J = soft_max(|conv|) = (1/beta) log(sum exp(beta |conv_k|)).
    dJ/dconv_k = exp(beta |conv_k|) / sum exp(beta |conv_k|) * sign(conv_k) (if we used |conv|, 
    but soft_max is on abs, so derivative wrt conv_k is w_k * sign(conv_k) where 
    w_k = exp(beta |conv_k|) / sum... wait.
    
    Let f = sum_j exp(beta |conv_j|).
    J = (1/beta) log f.
    dJ/dconv_k = (1/f) * exp(beta |conv_k|) * beta * sign(conv_k) = w_k * sign(conv_k).
    where w_k = exp(beta |conv_k|) / sum_j exp(beta |conv_j|).
    
    conv = IFFT(|X|^2).
    Let Y = |X|^2. conv = IFFT(Y).
    d conv_k / d Y_m = (1/L) * exp(i 2 pi k m / L).
    
    Y_m = |X_m|^2 = X_m * conj(X_m).
    d Y_m / d Re(X_m) = 2 Re(X_m).
    d Y_m / d Im(X_m) = -2 Im(X_m) (if X = a+ib, |X|^2 = a^2+b^2, d/da=2a, d/db=2b. 
    Wait, X_m = a+ib. conj(X_m) = a-ib. |X_m|^2 = a^2+b^2.
    d|X_m|^2 / da = 2a. d|X_m|^2 / db = 2b.
    
    So dJ/d Re(X_m) = sum_k (dJ/dconv_k) * (dconv_k/dY_m) * (dY_m/dRe(X_m))
    = sum_k (w_k sign(conv_k)) * (1/L) exp(i 2 pi k m / L) * 2 Re(X_m)
    
    This is a convolution-like operation.
    Let G_k = w_k sign(conv_k).
    dJ/dY_m = (1/L) FFT(G)_m? No.
    conv = IFFT(Y) => Y = FFT(conv).
    dJ/dY_m = sum_k (dJ/dconv_k) * (dconv_k/dY_m).
    Since conv = IFFT(Y), dconv_k/dY_m = (1/L) exp(i 2 pi k m / L).
    So dJ/dY_m = (1/L) sum_k (dJ/dconv_k) exp(i 2 pi k m / L) = (1/L) FFT(dJ/dconv)_m.
    
    Let H = FFT(G) where G_k = w_k sign(conv_k).
    Then dJ/dY_m = H_m / L.
    
    Then dJ/d Re(X_m) = (dJ/dY_m) * 2 Re(X_m) = (2 Re(X_m) / L) * H_m.
    dJ/d Im(X_m) = (dJ/dY_m) * 2 Im(X_m) = (2 Im(X_m) / L) * H_m.
    
    We want to minimize J / (sum x)^2.
    Let S = sum x = Re(X_0).
    Metric M = J / S^2.
    dM/dX = (S^2 dJ/dX - J * 2 S dS/dX) / S^4 = (1/S^2) (dJ/dX - (J/S) dS/dX).
    
    dS/d Re(X_0) = 1, dS/d other = 0.
    So the gradient has a correction term only for the DC component (m=0).
    For m > 0: grad_m = dJ/dX_m / S^2.
    For m = 0: grad_0 = (dJ/dX_0 - (J/S)) / S^2.
    
    We will perform gradient descent on X (complex), then project to real x.
    """
    x = x_init.copy()
    best_x = x.copy()
    best_val = _metric(x)
    
    # Initialize X
    X = np.fft.rfft(x, _FFT_SIZE)
    
    # We optimize the full complex FFT, but we must maintain Hermitian symmetry.
    # It is easier to optimize the rfft coefficients and reconstruct.
    # However, the gradient derivation above assumed full FFT.
    # Let's stick to rfft.
    # L = _FFT_SIZE.
    # The autoconv is linear, so we use zero-padding.
    
    beta = 50.0
    step_size = 0.05
    lr_decay = 0.99
    
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        # Compute conv and metric
        Y = X * np.conj(X)
        conv = np.fft.irfft(Y, _FFT_SIZE)[:2 * N - 1]
        abs_conv = np.abs(conv)
        max_val = np.max(abs_conv)
        
        if max_val < 1e-9:
            break
            
        # Compute weights w_k
        shifted = abs_conv - max_val
        exp_shifted = np.exp(beta * shifted)
        w = exp_shifted / np.sum(exp_shifted)
        
        # G_k = w_k * sign(conv_k)
        G = w * np.sign(conv)
        
        # H = FFT(G)
        # G is real, length 2N-1. We need to FFT it to length _FFT_SIZE.
        # Note: G is defined on lags 0..2N-2.
        # The gradient dJ/dY_m corresponds to the frequency domain representation of G.
        # Y is the power spectrum.
        # dJ/dY_m = (1/L) FFT(G)_m.
        # We need FFT of G padded to _FFT_SIZE.
        G_padded = np.zeros(_FFT_SIZE)
        G_padded[:2 * N - 1] = G
        H = np.fft.rfft(G_padded)
        
        # Gradient of J w.r.t. X
        # dJ/d Re(X_m) = (2 Re(X_m) / L) * H_m
        # dJ/d Im(X_m) = (2 Im(X_m) / L) * H_m
        # So dJ/dX_m (complex gradient, treating X as complex variable) is:
        # If we view X and conj(X) as independent, dJ/dX_m = (dJ/dRe + i dJ/dIm) / 2 ?
        # Actually, for real-valued optimization of complex variables, 
        # the update is X_new = X - step * (dJ/dRe + i dJ/dIm) ?
        # No, standard complex gradient descent uses:
        # X_new = X - step * (dJ/dX + dJ/dconj(X))
        # Here dJ/dX_m = (2 Re(X_m) / L) H_m / 2 ? 
        # Let's use the Wirtinger calculus or just treat Re/Im separately.
        # grad_re = 2 * np.real(X) * np.real(H) / _FFT_SIZE - 2 * np.imag(X) * np.imag(H) / _FFT_SIZE?
        # No.
        # dJ/dRe(X_m) = 2 Re(X_m) * Re(H_m) / L  (since H is real? No H is complex)
        # Wait, G is real, so H is complex.
        # dJ/dY_m is complex? No, J is real. Y is real.
        # G is real. FFT(G) is complex.
        # But dJ/dY_m must be real because Y is real.
        # Let's check: dJ/dY_m = sum_k G_k (1/L) exp(i 2 pi k m / L).
        # Since G is real, this sum is complex. But Y is real.
        # Ah, Y_m = |X_m|^2 is real.
        # The relationship conv = IFFT(Y) implies Y = FFT(conv).
        # If conv is real, Y is Hermitian.
        # G is real. FFT(G) is Hermitian.
        # So H_m is complex.
        # But dJ/dY_m should be real?
        # J is a function of real conv.
        # dJ/dconv_k is real.
        # dconv_k/dY_m is complex.
        # So dJ/dY_m is complex.
        # But Y_m is real.
        # This implies that we cannot simply take the complex gradient.
        # We must project the gradient to the real subspace of Y.
        
        # Let's step back.
        # We want to minimize J(Y) where Y is real non-negative and Hermitian.
        # Y = |X|^2.
        # Let's optimize X directly.
        # x = IFFT(X).
        # conv = IFFT(|X|^2).
        
        # Let's compute the gradient of J w.r.t. x, then map to X?
        # No, let's stick to the previous derivation but handle the real constraint.
        
        # Alternative: Use the gradient of J w.r.t. x, computed via FFT.
        # grad_x = 2 * IFFT(FFT(x) * conj(FFT(G_padded))) ?
        # Let's try this. It's simpler.
        # J = soft_max(|conv|).
        # dJ/dx = 2 * (x * G)_circular ?
        # conv_k = sum_i x_i x_{i-k}.
        # dconv_k/dx_j = x_{j-k} + x_{j+k}.
        # dJ/dx_j = sum_k w_k sign(conv_k) (x_{j-k} + x_{j+k}).
        # This is a convolution of x with a kernel K where K_m = w_m + w_{-m}.
        # But w is defined on 0..2N-2.
        # Let's compute this directly for N=400. It's O(N^2). 160k ops.
        # 1000 iterations -> 160M ops. Might be too slow in Python.
        
        # Let's use FFT.
        # The kernel is effectively the "error" signal.
        # Let E = G (the weights).
        # grad_x = 2 * convolve(x, E, mode='same')?
        # Let's test this logic.
        # If we use circular convolution, it works for circular autoconv.
        # For linear autoconv, it's an approximation.
        
        # Let's try the direct computation of the gradient in time domain.
        # It's safer.
        # grad_x = np.zeros(N)
        # for j in range(N):
        #    for k in range(2*N-1):
        #       # x_{j-k} and x_{j+k}
        #       # indices must be in [0, N-1]
        #       pass
        
        # Vectorized direct gradient:
        # We need sum_k w_k * (x_{j-k} + x_{j+k})
        # This is equivalent to:
        # term1 = sum_k w_k x_{j-k} -> convolve(x, w, mode='valid' or 'same'?)
        # Let's use np.convolve.
        # w is length 2N-1.
        # x is length N.
        # conv = np.convolve(x, w, mode='same') gives length N?
        # np.convolve(x, w, mode='same') returns length max(N, 2N-1) = 2N-1?
        # No, 'same' returns length max(M, N).
        # Let's just use the FFT approximation for speed, as N=400 is small enough 
        # that circular vs linear might not differ much if we pad correctly.
        
        # Let's use the FFT-based gradient for circular autoconvolution.
        # This is a common heuristic.
        fx = np.fft.rfft(x, _FFT_SIZE)
        # grad_x in freq domain:
        # grad_x_freq = 2 * fx * conj(FFT(G_padded))
        # G_padded is the weights in time domain (lags).
        # FFT(G_padded) is the freq domain representation of the kernel.
        grad_x_freq = 2 * fx * np.conj(np.fft.rfft(G_padded, _FFT_SIZE))
        grad_x = np.fft.irfft(grad_x_freq, _FFT_SIZE)[:N]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad_x)
        if grad_norm < 1e-9:
            break
            
        grad_x = grad_x / grad_norm
        
        # Update x
        # We want to reduce the metric.
        # Metric = J / S^2.
        # If S is large, we can take larger steps.
        # Let's use a fixed step size for now.
        x_new = x - step_size * grad_x
        
        # Optional: Normalize x to keep integral constant?
        # The metric is scale-invariant: J(cx) / (cS)^2 = J(x) / S^2.
        # So scaling doesn't change the metric.
        # But numerical stability might improve if we keep S ~ N.
        S_new = np.sum(x_new)
        if abs(S_new) > 1e-6:
            x_new = x_new * (N / S_new)
        
        val_new = _metric(x_new)
        if val_new < best_val:
            best_val = val_new
            best_x = x_new.copy()
            x = x_new
            # Decay step size
            step_size *= 0.99
        else:
            # Rejection: reduce step size
            step_size *= 0.9
            if step_size < 0.001:
                break
        
        # Update X for next iteration's potential use (not strictly needed if we recompute)
        # X = np.fft.rfft(x, _FFT_SIZE)
        
    return best_x, best_val

def _try_local_search_fast(x, deadline, rng, max_rounds=200000):
    """Fast local search with in-place modifications and early termination."""
    best_x = x.copy()
    best_val = _metric(x)
    no_improve = 0
    step_scale = 0.3
    total_steps = 0
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 200:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.0005:
                break
        
        k = int(rng.integers(1, 4))
        indices = rng.choice(N, size=k, replace=False)
        
        # Try multiple perturbations per round
        best_delta = 0
        best_idx = None
        for trial in range(3):
            x_test = best_x.copy()
            for idx in indices:
                x_test[idx] += rng.normal(0, step_scale)
            v = _metric(x_test)
            if v < best_val + best_delta:
                best_delta = v - best_val
                best_idx = x_test
        
        if best_idx is not None and best_delta < 0:
            best_val += best_delta
            best_x = best_idx
            no_improve = 0
            total_steps += 1
        else:
            no_improve += 1
    
    return best_x, best_val

def construct():
    deadline = time.time() + 145.0

    best_x = np.ones(N)
    best_val = _metric(best_x)

    rng = np.random.default_rng(2024)

    # Phase 1: Spectral Gradient Descent
    if time.time() < deadline - 50:
        x_spec, val_spec = _spectral_gradient_descent(best_x, deadline - 50, rng)
        if val_spec < best_val:
            best_val = val_spec
            best_x = x_spec

    # Phase 2: Fast local search
    if time.time() < deadline - 5:
        best_x, best_val = _try_local_search_fast(best_x, deadline - 5, rng, max_rounds=100000)

    return best_x
# EVOLVE-BLOCK-END

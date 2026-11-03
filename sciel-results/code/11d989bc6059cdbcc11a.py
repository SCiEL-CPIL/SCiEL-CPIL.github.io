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

def _soft_max_abs(conv, beta=500.0):
    """Smooth approximation of max |conv| using log-sum-exp."""
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    # log-sum-exp trick for numerical stability
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta

def _smooth_metric(x, beta=500.0):
    """Smooth surrogate: soft_max |autoconv| / (integral)^2."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)[:2 * N - 1]
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _soft_max_abs(conv, beta) / (s * s)

def _spectral_design_optimize(x_init, deadline, rng):
    """
    Optimize the magnitude spectrum |X(k)| directly.
    We parameterize the signal by its log-magnitude spectrum: x = IFFT(exp(log_mag + i*phase)).
    However, since we want to control the signal, we can optimize the signal directly 
    but using the spectral insight: 
    The autoconvolution is IFFT(|X|^2). 
    Minimizing max|IFFT(|X|^2)| is equivalent to making |X|^2 as "flat" as possible 
    in the time domain (i.e., minimizing the peak-to-average power ratio of |X|^2).
    
    This is a known problem: designing a sequence with low peak-to-average power ratio (PAPR) 
    for its power spectrum.
    
    We use a gradient descent approach on the signal x, but guided by the spectral flatness.
    Actually, a more direct approach is to optimize the phases of X(k) while keeping 
    magnitudes somewhat fixed, or vice versa.
    
    Let's try a different angle: 
    We know that for a constant signal, X is a delta at 0. |X|^2 is a delta. IFFT is constant.
    Max |conv| = N. Integral = N. Metric = N / N^2 = 1/N.
    
    If we spread out |X|^2 in frequency, the time-domain |X|^2 (which is the autoconv) 
    will have lower peaks.
    
    Strategy: 
    1. Start with x = ones.
    2. Compute X = FFT(x).
    3. We want to modify X such that IFFT(|X|^2) is flat.
    4. This is equivalent to making |X|^2 have a flat inverse FFT.
    5. This is the same as making the power spectral density flat? No, 
       IFFT(|X|^2) is the autocorrelation. We want the autocorrelation to be flat (except lag 0).
       This is the definition of a "perfect" or "low autocorrelation" sequence.
       
    We can use a gradient descent on x to minimize the soft-max of the autocorrelation.
    The gradient of the soft-max can be computed efficiently.
    """
    x = x_init.copy()
    best_x = x.copy()
    best_val = _metric(x)
    current_val = _smooth_metric(x)
    
    # Gradient descent with adaptive step size
    step_size = 0.1
    patience = 0
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Compute gradient of the smooth metric w.r.t. x
        fx = np.fft.rfft(x, _FFT_SIZE)
        conv = np.fft.irfft(fx * fx, _FFT_SIZE)[:2 * N - 1]
        abs_conv = np.abs(conv)
        max_val = np.max(abs_conv)
        
        # Soft max gradient: 
        # d(soft_max)/dx = sum_j (d(soft_max)/d|conv_j|) * d|conv_j|/dx
        # d(soft_max)/d|conv_j| = exp(beta * (|conv_j| - max_val)) / sum_k exp(beta * (|conv_k| - max_val))
        # Let w_j = exp(beta * (|conv_j| - max_val)) / sum_k ...
        
        beta = 500.0
        shifted = abs_conv - max_val
        exp_shifted = np.exp(beta * shifted)
        w = exp_shifted / np.sum(exp_shifted)
        
        # d|conv_j|/dx:
        # conv = IFFT(|X|^2) = IFFT(X * conj(X))
        # This is a quadratic form.
        # grad_x (x^T A x) = 2 A x
        # Here, conv_j = sum_i x_i x_{i-j}
        # d conv_j / dx_k = x_{k-j} + x_{k+j} (with appropriate bounds/wrapping for linear conv)
        
        # Efficient computation of gradient:
        # grad = 2 * sum_j w_j * (shift(x, j) + shift(x, -j))
        # Note: conv is linear autoconvolution, not circular.
        # We can compute the gradient via FFT as well, but linear conv gradient is trickier.
        # Let's use the fact that:
        # grad_k = 2 * sum_j w_j * (x_{k-j} + x_{k+j})
        # where x_i = 0 for i < 0 or i >= N.
        
        # We can compute this using FFT if we embed in a larger space, 
        # or just use direct computation if N is small enough.
        # N=400, so 2*N-1 = 799 lags.
        # Direct computation: O(N^2) per iteration. 400^2 = 160,000 operations. 
        # 500 iterations -> 80M operations. This might be slow in Python.
        
        # Let's try to compute the gradient more efficiently.
        # We can use the correlation theorem.
        # grad = 2 * IFFT(FFT(x) * conj(FFT(w_padded)))
        # where w_padded is w_j placed at the correct frequencies?
        # Actually, the gradient of the quadratic form x^T A x is 2 A x.
        # Here A is a Toeplitz matrix with rows being w_j.
        # A x = sum_j w_j * (shift(x, j) + shift(x, -j))
        # This is a convolution of x with a kernel k where k_j = w_j + w_{-j} (symmetric).
        # So grad = 2 * convolve(x, k, mode='same')
        
        # Construct the kernel k
        # w is defined for lags j = 0 to 2N-2.
        # The kernel for the gradient is h[m] = w[m] + w[-m]
        # But w is non-negative and defined on 0..2N-2.
        # The autoconvolution is linear, so the "Toeplitz" matrix is not circulant.
        # However, for the gradient of the sum of weighted autoconv lags, 
        # we can use:
        # grad = 2 * (convolve(x, w, mode='full') + convolve(x, w, mode='full')[::-1])
        # This is getting complicated. Let's just do the direct sum if it's fast enough.
        
        # Alternative: Use the FFT-based gradient for the *circular* autoconvolution, 
        # and hope it's a good approximation, or just use a simple finite difference 
        # for the gradient? No, finite difference is too slow.
        
        # Let's use a simpler gradient: 
        # We want to reduce the peaks in conv.
        # A simple heuristic: x_new = x - step * grad
        # where grad is the "error" signal that, when convolved with x, produces the peaks.
        # grad = x * (conv - mean(conv)) ? No.
        
        # Let's stick to a simpler optimization: 
        # Use the fact that we can optimize the phases of X.
        # x = IFFT(|X| * exp(i*phase))
        # We can fix |X| and optimize phase.
        
        # Initialize |X| from x_init
        fx_init = np.fft.rfft(x, _FFT_SIZE)
        mag = np.abs(fx_init)
        
        # Optimize phase
        phase = np.angle(fx_init)
        
        for iter2 in range(50):
            if time.time() > deadline:
                break
            
            # Reconstruct x from mag and phase
            fx_new = mag * np.exp(1j * phase)
            x_new = np.fft.irfft(fx_new, _FFT_SIZE)[:N]
            
            val_new = _metric(x_new)
            if val_new < best_val:
                best_val = val_new
                best_x = x_new.copy()
            
            # Update phase: 
            # We want to minimize max|IFFT(|X|^2)|
            # This is equivalent to minimizing the peaks of IFFT(mag^2)
            # The phase of X does not affect |X|^2, so this optimization is useless!
            # The autoconvolution depends only on the magnitude spectrum, not the phase.
            # So we must optimize the magnitude spectrum.
            
            # Let's optimize the magnitude spectrum directly.
            # x = IFFT(mag * exp(i*phase))
            # We can fix the phase (e.g., zero phase) and optimize mag.
            # If we use zero phase, x is real and even (symmetric).
            # This restricts the search space but might be enough.
            
            # Let's try optimizing mag with zero phase.
            # x = IFFT(mag)
            # We want to minimize max|IFFT(mag^2)| / (sum(x))^2
            
            # Gradient of mag:
            # This is a complex optimization.
            
            # Let's try a simpler approach: 
            # Use a random search in the magnitude spectrum.
            # Perturb mag randomly and accept if the metric improves.
            
            mag_perturbed = mag * (1 + 0.1 * rng.normal(0, 1, size=mag.shape))
            mag_perturbed = np.abs(mag_perturbed) # Ensure non-negative
            
            fx_perturbed = mag_perturbed # Zero phase
            x_perturbed = np.fft.irfft(fx_perturbed, _FFT_SIZE)[:N]
            
            val_perturbed = _metric(x_perturbed)
            if val_perturbed < best_val:
                best_val = val_perturbed
                best_x = x_perturbed.copy()
                mag = mag_perturbed.copy()
    
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

    # Phase 1: Spectral design optimization
    if time.time() < deadline - 50:
        x_spec, val_spec = _spectral_design_optimize(best_x, deadline - 50, rng)
        if val_spec < best_val:
            best_val = val_spec
            best_x = x_spec

    # Phase 2: Fast local search
    if time.time() < deadline - 5:
        best_x, best_val = _try_local_search_fast(best_x, deadline - 5, rng, max_rounds=100000)

    return best_x
# EVOLVE-BLOCK-END

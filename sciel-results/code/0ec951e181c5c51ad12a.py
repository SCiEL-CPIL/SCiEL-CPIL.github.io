"""Phase optimization with prolate spheroidal amplitude spectrum."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
START_TIME = time.time()
TIME_LIMIT = 115

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 5)

def _autoconv_fft(x):
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _prolate_amplitude(w):
    """
    Generate prolate spheroidal sequence amplitudes.
    w is the normalized bandwidth (0 < w < 0.5).
    Returns real amplitudes for DFT bins 0..N//2.
    """
    # Use the eigenfunctions of the time-limited band-limited operator
    # For simplicity, we'll use a Gaussian approximation to prolate sequences
    # which is computationally cheaper and often sufficient
    k = np.arange(N)
    kf = k / N
    # Prolate-like: concentrated in low frequencies
    # The parameter w controls the bandwidth
    # A good approximation is exp(-pi^2 * w^2 * (kf - 0.5)^2 / (w * (1-w)))
    # but let's use a simpler Gaussian that concentrates energy
    sigma = 1.0 / (2 * np.pi * w) if w > 0.01 else 1.0
    mag = np.exp(-0.5 * (kf / sigma) ** 2)
    return mag

def _phase_to_signal(mag, phase):
    """Convert amplitude and phase to real signal."""
    # mag is for bins 0..N//2, phase should be same length
    # Build complex spectrum
    X = np.zeros(N, dtype=complex)
    n_half = N // 2
    X[:n_half+1] = mag[:n_half+1] * np.exp(1j * phase[:n_half+1])
    # Enforce Hermitian symmetry for real signal
    for k in range(1, n_half):
        X[N-k] = np.conj(X[k])
    if N % 2 == 0:
        X[N//2] = np.real(X[N//2])  # Nyquist bin must be real
    x = np.fft.ifft(X).real
    return x

def _smooth_obj_and_grad_phase(mag, phase, beta=30.0):
    """
    Smooth objective and gradient w.r.t. phase only.
    mag is fixed.
    """
    x = _phase_to_signal(mag, phase)
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(phase)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    L = len(ac)
    
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        smooth_max = max_val
        weights = np.ones(L) / L
    else:
        shifted = abs_ac - max_val
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        smooth_max = (max_val + np.log(sum_exp)) / beta
        weights = exp_shifted / sum_exp
    
    val = smooth_max / (s * s)
    
    # Gradient w.r.t. phase
    # d(ac[k])/d(phase[m]) involves chain rule through x
    # This is complex, so let's use finite differences or a simpler approach
    # For efficiency, let's use the gradient w.r.t. x first, then chain rule
    
    # Gradient of smooth_max w.r.t. x
    c = weights * np.sign(ac)
    c = np.concatenate([c, np.zeros(N-1)])  # pad to length 2N-1 for convolution
    
    # grad_smooth_max[j] = sum_k c[k] * d(ac[k])/dx[j]
    # d(ac[k])/dx[j] = x[k-j] + x[j] (with appropriate bounds)
    
    # More efficient: use FFT-based gradient
    # ac = x * x (linear convolution)
    # d(ac)/dx = 2 * (x * x) but with proper handling
    # Actually, for linear autoconvolution:
    # ac[k] = sum_j x[j] * x[k-j]
    # d(ac[k])/dx[m] = x[k-m] + x[m] (if indices valid)
    
    # Let's compute grad_smooth_max using the adjoint
    # grad_x = 2 * (c * x) where * is appropriate convolution
    # But we need to be careful with the linear convolution
    
    # Simpler approach: use the fact that ac = ifft(fft(x)^2)
    # d(ac)/dx = 2 * ifft(fft(x) * fft(c_padded))
    
    c_padded = np.zeros(2*N-1)
    c_padded[:L] = c
    c_fft = np.fft.rfft(c_padded, n=2*N-1)
    x_fft = np.fft.rfft(x, n=2*N-1)
    grad_smooth_max_fft = 2 * x_fft * c_fft
    grad_smooth_max = np.fft.irfft(grad_smooth_max_fft, n=2*N-1)[:N]
    
    # Now chain rule: d(val)/d(phase[m]) = sum_j (d(val)/dx[j]) * (dx[j]/d(phase[m]))
    # x = ifft(X), X[k] = mag[k] * exp(i*phase[k])
    # dx[j]/d(phase[m]) = (1/N) * sum_k X[k] * i * delta_{k,m} * exp(i*phase[k]) * ... 
    # Actually: x[j] = (1/N) sum_k mag[k] exp(i phase[k]) exp(-2pi i jk/N)
    # dx[j]/d(phase[m]) = (1/N) * mag[m] * i * exp(i phase[m]) * exp(-2pi i jm/N)
    
    # This is still expensive. Let's use a different approach:
    # Optimize in the time domain but with amplitude constraint
    
    # For now, return a placeholder gradient and use a different method
    return val, np.zeros_like(phase)

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Try different prolate bandwidths
    bandwidths = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    
    for w in bandwidths:
        if _check_deadline():
            break
        
        mag = _prolate_amplitude(w)
        
        # Initialize phase: try several initializations
        rng = np.random.default_rng(42 + int(w * 1000))
        
        for init_idx in range(3):
            if _check_deadline():
                break
            
            if init_idx == 0:
                phase = np.zeros(N // 2 + 1)
            elif init_idx == 1:
                phase = rng.uniform(-np.pi, np.pi, N // 2 + 1)
            else:
                # Linear phase (time shift)
                phase = -2 * np.pi * rng.uniform(0, 1) * np.arange(N // 2 + 1) / N
            
            x = _phase_to_signal(mag, phase)
            val = _objective(x)
            
            if val < best_val:
                best_val = val
                best_x = x.copy()
            
            # Local refinement using time-domain gradient descent
            # with amplitude constraint projected back
            current_x = x.copy()
            current_val = val
            
            for step in range(200):
                if _check_deadline():
                    break
                
                # Compute gradient using finite differences for simplicity
                # (slow but reliable)
                eps = 1e-4
                grad = np.zeros(N)
                for j in range(N):
                    x_plus = current_x.copy()
                    x_plus[j] += eps
                    val_plus = _objective(x_plus)
                    grad[j] = (val_plus - current_val) / eps
                
                # Gradient descent step
                lr = 0.01
                x_new = current_x - lr * grad
                val_new = _objective(x_new)
                
                if val_new < current_val:
                    current_x = x_new
                    current_val = val_new
                    if current_val < best_val:
                        best_val = current_val
                        best_x = current_x.copy()
                else:
                    # Backtrack
                    lr *= 0.5
                    if lr < 1e-6:
                        break
    
    return best_x

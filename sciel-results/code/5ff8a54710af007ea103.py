"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve


def _autoconv(f):
    """Compute full linear autoconvolution via FFT."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(f, size)
    ac = np.fft.irfft(F * F, size)[:2 * n - 1]
    return ac


def _score(f):
    """Compute max|autoconv| / (sum f)^2."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv(f)
    return np.max(np.abs(ac)) / (s * s)


def _lowpass_filter(x, sigma):
    """Apply Gaussian low-pass filter."""
    n = len(x)
    ksize = int(6 * sigma) | 1  # odd size
    if ksize < 3:
        ksize = 3
    k = np.arange(-ksize // 2, ksize // 2 + 1)
    kernel = np.exp(-k**2 / (2 * sigma**2))
    kernel /= np.sum(kernel)
    # Use same padding to maintain length
    return fftconvolve(x, kernel, mode='same')


def _spectral_objective(x, sigma):
    """
    Objective: maximize the minimum of |F(omega)|^2 / (sum x)^2
    Equivalently, minimize the peak of |F(omega)|^2 / (sum x)^2
    after low-pass filtering.
    """
    y = _lowpass_filter(x, sigma)
    s = np.sum(y)
    if abs(s) < 1e-12:
        return np.inf
    F = np.fft.rfft(y)
    power = np.abs(F)**2
    return np.max(power) / (s * s)


def _spectral_objective_grad(x, sigma, beta=10.0):
    """
    Compute gradient of spectral objective w.r.t. x.
    Uses smooth approximation for max.
    """
    y = _lowpass_filter(x, sigma)
    s = np.sum(y)
    if abs(s) < 1e-12:
        return np.inf, np.zeros_like(x)
    
    F = np.fft.rfft(y)
    power = np.abs(F)**2
    # Smooth max approximation
    m = np.max(power)
    log_sum = np.log(np.sum(np.exp(power - m))) + m
    weights = np.exp(power - m)
    weights /= np.sum(weights)
    
    # d(power[k])/dy = 2*Re(F[k]*conj(F[k])/dy) ... this is complex
    # Actually, power[k] = |F[k]|^2 = F[k]*conj(F[k])
    # d(power[k])/dy[j] = 2*Re(F[k] * conj(dF[k]/dy[j]))
    # dF[k]/dy[j] = exp(-2*pi*i*k*j/n) * (1/n) ... for DFT
    # This gets complicated. Let's use numerical gradient or a simpler approach.
    
    # Alternative: use the fact that we want to minimize peak |F|^2
    # which is equivalent to making |F| as flat as possible
    # Gradient descent on y to make |F| flat, then backprop through filter
    
    # For now, use a simpler gradient: push down the peaks of |F|
    # d(obj)/dy = (2/s^2) * Re(conj(F) * grad_power) - 2*obj/s * 1
    # where grad_power[k] = weights[k] * F[k] (from smooth max)
    
    # Actually, let's just do coordinate descent on the filtered signal
    return log_sum / (s * s), None


def _optimize_smooth(deadline, max_iters=500, lr=0.1, sigma=5.0):
    """
    Optimize a smooth waveform by minimizing peak autoconvolution.
    Start from a structured initial condition and use gradient-like steps.
    """
    # Initial: use a chirp or structured pattern
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.cos(2 * t) + 0.5 * np.cos(4 * t)  # multi-frequency
    x = x - np.mean(x)  # zero mean to avoid DC dominance
    
    best_x = x.copy()
    best_score = _score(np.sign(x))
    
    for it in range(max_iters):
        if time.time() > deadline - 20.0:
            break
        
        # Compute current score
        s = np.sum(x)
        if abs(s) < 1e-12:
            x = x + 0.1 * np.random.randn(N)
            continue
        
        ac = _autoconv(x)
        peak_idx = np.argmax(np.abs(ac))
        peak_val = np.abs(ac[peak_idx])
        
        # Gradient: reduce the peak by adjusting x
        # d(ac[k])/dx[j] = x[k-j] + x[j-k] (with wraparound considerations)
        # We want to reduce ac[peak_idx], so move x in direction that reduces it
        
        # Compute gradient of ac[peak_idx] w.r.t. x
        k = peak_idx
        grad = np.zeros(N)
        for j in range(N):
            # ac[k] = sum_i x[i]*x[k-i]
            # d(ac[k])/dx[j] = x[k-j] + x[j-k] (mod N for circular, but we use linear)
            # For linear convolution, only valid indices contribute
            if 0 <= k - j < N:
                grad[j] += x[k - j]
            if 0 <= j - k < N:
                grad[j] += x[j - k]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad /= grad_norm
        
        # Step: reduce the peak
        x_new = x - lr * grad
        
        # Apply low-pass to smooth
        x_new = _lowpass_filter(x_new, sigma)
        
        # Check score
        sc = _score(np.sign(x_new))
        if sc < best_score:
            best_score = sc
            best_x = x_new.copy()
            x = x_new
        else:
            # Try smaller step
            x_new = x - lr * 0.5 * grad
            x_new = _lowpass_filter(x_new, sigma)
            sc = _score(np.sign(x_new))
            if sc < best_score:
                best_score = sc
                best_x = x_new.copy()
                x = x_new
                lr *= 0.8
            else:
                lr *= 0.7
    
    return best_x, best_score


def _local_search_ones(f, deadline, max_iters=15000):
    """
    Fast coordinate descent on ±1 sequences.
    """
    current = np.sign(f).astype(float)
    current[current == 0] = 1.0
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    np.random.seed(123)
    
    for it in range(max_iters):
        if time.time() > deadline - 15.0:
            break
        
        idx = np.random.randint(N)
        old_val = current[idx]
        new_val = -old_val
        
        current[idx] = new_val
        s = np.sum(current)
        if abs(s) < 1e-12:
            current[idx] = old_val
            continue
        
        ac = _autoconv(current)
        sc = np.max(np.abs(ac)) / (s * s)
        
        if sc < current_score:
            current_score = sc
            if sc < best_score:
                best_score = sc
                best = current.copy()
        else:
            current[idx] = old_val
    
    return best, best_score


def _structured_initial(deadline):
    """Generate structured initial patterns."""
    patterns = []
    
    # Cosine patterns
    for freq in [1, 2, 3, 5, 8]:
        t = np.linspace(0, 2 * np.pi, N, endpoint=False)
        patterns.append(np.cos(freq * t))
    
    # Sine patterns
    for freq in [1, 2, 3, 5]:
        t = np.linspace(0, 2 * np.pi, N, endpoint=False)
        patterns.append(np.sin(freq * t))
    
    # Multi-frequency
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    patterns.append(np.cos(t) + 0.5 * np.cos(2 * t) + 0.3 * np.cos(3 * t))
    patterns.append(np.sin(t) + 0.5 * np.sin(2 * t) - 0.3 * np.sin(3 * t))
    
    # Block patterns
    for block_size in [10, 20, 40, 50]:
        n_blocks = N // block_size
        pat = np.zeros(N)
        for i in range(n_blocks):
            pat[i*block_size:(i+1)*block_size] = 1.0 if i % 2 == 0 else -1.0
        patterns.append(pat)
    
    # Alternating
    patterns.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    return patterns


def construct():
    deadline = time.time() + 135.0
    
    best = np.ones(N)
    best_score = _score(best)
    
    # Phase 1: Try structured initial patterns
    patterns = _structured_initial(deadline)
    for pat in patterns:
        if time.time() > deadline - 30.0:
            break
        rounded = np.sign(pat)
        rounded[rounded == 0] = 1.0
        sc = _score(rounded)
        if sc < best_score:
            best_score = sc
            best = rounded.copy()
    
    # Phase 2: Smooth optimization from best structured pattern
    if time.time() < deadline - 40.0:
        smooth_x, smooth_score = _optimize_smooth(deadline, max_iters=300, lr=0.5, sigma=3.0)
        if smooth_score < best_score:
            best_score = smooth_score
            best = np.sign(smooth_x)
            best[best == 0] = 1.0
    
    # Phase 3: Local search on ±1
    f_ls, ls_score = _local_search_ones(best, deadline, max_iters=12000)
    if ls_score < best_score:
        best_score = ls_score
        best = f_ls.copy()
    
    # Phase 4: Try different sigma values for smooth optimization
    if time.time() < deadline - 25.0:
        for sigma in [1.0, 2.0, 5.0, 10.0]:
            if time.time() > deadline - 15.0:
                break
            smooth_x2, smooth_score2 = _optimize_smooth(deadline, max_iters=150, lr=0.3, sigma=sigma)
            if smooth_score2 < best_score:
                best_score = smooth_score2
                best = np.sign(smooth_x2)
                best[best == 0] = 1.0
        
        # Final local search
        if time.time() < deadline - 10.0:
            f_ls2, ls_score2 = _local_search_ones(best, deadline, max_iters=5000)
            if ls_score2 < best_score:
                best_score = ls_score2
                best = f_ls2.copy()
    
    # Ensure output is ±1
    result = np.sign(best)
    result[result == 0] = 1.0
    
    return result
# EVOLVE-BLOCK-END

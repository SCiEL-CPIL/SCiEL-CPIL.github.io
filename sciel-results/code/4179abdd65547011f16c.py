"""Frequency-domain peak suppression with structured low-frequency initialization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _eval_value(h):
    """Compute the exact evaluation metric."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _autoconvolve_fft(h):
    """Compute autoconvolution using FFT for speed."""
    n_conv = 2 * N - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = np.fft.fft(h, n_fft)
    conv = np.fft.ifft(H * H).real[:n_conv]
    return conv

def _objective_soft(h, beta):
    """
    Soft-max surrogate objective. Normalizes sum(h)=N internally.
    """
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    h_norm = h * (N / s)
    conv = _autoconvolve_fft(h_norm)
    m = np.max(conv)
    if not np.isfinite(m):
        return 1e10
    exp_vals = np.exp(beta * (conv - m))
    sm = m + np.log(np.sum(exp_vals)) / beta
    return abs(2.0 * N * sm / (N * N))

def _make_structured_h(frequencies, amplitudes, phases, bg=0.0):
    """
    Construct h from sum of sinusoids plus background.
    frequencies: array of integer frequencies (1..N//2)
    amplitudes: array of amplitudes
    phases: array of phases in radians
    bg: background level
    """
    h = np.full(N, bg)
    t = np.arange(N)
    for f, a, p in zip(frequencies, amplitudes, phases):
        h += a * np.cos(2 * np.pi * f * t / N + p)
    return h

def _local_peak_suppression(h, n_iters=5, window=5):
    """
    Identify peaks in autoconvolution and locally adjust h to reduce them.
    Uses the fact that conv[k] = sum_i h[i]*h[i+k], so to reduce conv[k],
    we can adjust h at positions that contribute most to that peak.
    """
    h_work = h.copy()
    s = np.sum(h_work)
    if abs(s) < 1e-10:
        return h_work
    
    for it in range(n_iters):
        conv = _autoconvolve_fft(h_work)
        # Find the position of the maximum (signed)
        peak_idx = np.argmax(conv)
        peak_val = conv[peak_idx]
        
        if peak_val <= 0:
            break
        
        # The peak at index k in full convolution corresponds to:
        # conv_full[k] = sum_{i=max(0,k-N+1)}^{min(N-1,k)} h[i]*h[i-(k-N+1) if k>=N else 0]
        # Actually for full conv of length N with itself, output length 2N-1
        # conv_full[k] = sum_i h[i] * h[i - k] where valid i
        # For k in [0, N-1]: conv_full[k] = sum_{i=k}^{N-1} h[i]*h[i-k]
        # For k in [N, 2N-2]: conv_full[k] = sum_{i=0}^{2N-2-k} h[i]*h[i+k-N+1]
        
        # To reduce the peak, we want to reduce the correlation at that lag.
        # Simple approach: find the two most correlated pairs and reduce one of them.
        
        # For lag k (0-indexed in full conv):
        k = peak_idx
        if k < N:
            # conv[k] = sum_{i=k}^{N-1} h[i] * h[i-k]
            i_start = k
            i_end = N
        else:
            # k >= N: let lag = k - (N-1), conv[k] = sum_{i=0}^{N-1-lag} h[i] * h[i+lag]
            lag = k - (N - 1)
            i_start = 0
            i_end = N - lag
        
        # Find the pair with maximum product
        if i_end > i_start:
            if k < N:
                products = h_work[i_start:i_end] * h_work[i_start - k:i_end - k]
            else:
                lag = k - (N - 1)
                products = h_work[i_start:i_end] * h_work[i_start + lag:i_end + lag]
            
            max_prod_idx = np.argmax(products)
            
            # Determine which element to adjust
            if k < N:
                i1 = i_start + max_prod_idx
                i2 = i_start - k + max_prod_idx
            else:
                lag = k - (N - 1)
                i1 = i_start + max_prod_idx
                i2 = i_start + lag + max_prod_idx
            
            # Reduce the larger-magnitude element slightly
            if abs(h_work[i1]) >= abs(h_work[i2]):
                h_work[i1] -= 0.01 * np.sign(h_work[i1])
            else:
                h_work[i2] -= 0.01 * np.sign(h_work[i2])
    
    # Renormalize
    s = np.sum(h_work)
    if abs(s) > 1e-10:
        h_work = h_work * (N / s)
    
    return h_work

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Strategy: 
    1. Initialize with low-frequency structured patterns (sinusoid combos)
    2. Gradient descent with FFT-accelerated autoconvolution
    3. Local peak suppression
    4. Coordinate descent polish
    """
    deadline = time.time() + 143.0
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # ---- Phase 1: Try structured low-frequency initializations ----
    rng = np.random.RandomState(42)
    
    # Try various combinations of low frequencies
    freq_sets = [
        [1],
        [2],
        [3],
        [1, 2],
        [1, 3],
        [2, 3],
        [1, 2, 3],
        [1, 2, 4],
        [2, 3, 4],
        [1, 3, 5],
        [2, 4, 6],
        [1, 2, 3, 4],
    ]
    
    for freqs in freq_sets:
        if time.time() > deadline - 80:
            break
        
        for trial in range(5):
            if time.time() > deadline - 80:
                break
            
            n_freqs = len(freqs)
            amplitudes = rng.uniform(-1.5, 1.5, size=n_freqs)
            phases = rng.uniform(0, 2 * np.pi, size=n_freqs)
            bg = rng.uniform(-0.5, 0.5)
            
            h_candidate = _make_structured_h(freqs, amplitudes, phases, bg)
            s = np.sum(h_candidate)
            if abs(s) < 1e-10:
                continue
            h_candidate = h_candidate * (N / s)
            
            v = _eval_value(h_candidate)
            if v < best_val:
                best_val = v
                best_h = h_candidate.copy()
    
    # ---- Phase 2: Gradient descent with FFT-accelerated objective ----
    # Use simple projected gradient descent on the soft-max objective
    for beta in [20.0, 50.0, 100.0]:
        if time.time() > deadline - 40:
            break
        
        h_work = best_h.copy()
        s = np.sum(h_work)
        if abs(s) < 1e-10:
            continue
        h_work = h_work * (N / s)
        
        lr = 0.01
        n_grad_iters = 200
        
        for it in range(n_grad_iters):
            if time.time() > deadline - 40:
                break
            
            # Compute gradient numerically (central difference)
            # For efficiency, use forward difference with small step
            eps = 1e-4
            grad = np.zeros(N)
            
            for i in range(0, N, 2):  # Sample every other index for speed
                if time.time() > deadline - 40:
                    break
                
                h_plus = h_work.copy()
                h_plus[i] += eps
                s_plus = np.sum(h_plus)
                if abs(s_plus) < 1e-10:
                    continue
                h_plus = h_plus * (N / s_plus)
                f_plus = _objective_soft(h_plus, beta)
                
                h_minus = h_work.copy()
                h_minus[i] -= eps
                s_minus = np.sum(h_minus)
                if abs(s_minus) < 1e-10:
                    continue
                h_minus = h_minus * (N / s_minus)
                f_minus = _objective_soft(h_minus, beta)
                
                grad[i] = (f_plus - f_minus) / (2 * eps)
                # Also set the adjacent element's gradient approximately
                if i + 1 < N:
                    grad[i + 1] = grad[i] * 0.5  # Crude approximation
            
            # Gradient step
            h_work -= lr * grad
            
            # Renormalize
            s = np.sum(h_work)
            if abs(s) < 1e-10:
                h_work = best_h.copy()
                break
            h_work = h_work * (N / s)
            
            v = _eval_value(h_work)
            if v < best_val:
                best_val = v
                best_h = h_work.copy()
            
            # Adaptive learning rate: reduce if no improvement
            if it % 50 == 49:
                lr *= 0.5
        
        # Local peak suppression
        h_suppressed = _local_peak_suppression(h_work, n_iters=10, window=5)
        v_sup = _eval_value(h_suppressed)
        if v_sup < best_val:
            best_val = v_sup
            best_h = h_suppressed.copy()
    
    # ---- Phase 3: Coordinate descent polish ----
    if time.time() < deadline - 10:
        h_work = best_h.copy()
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        for step in step_sizes:
            if time.time() > deadline - 10:
                break
            improved_any = False
            for i in range(N):
                if time.time() > deadline - 10:
                    break
                old_val = h_work[i]
                for delta in [step, -step]:
                    h_work[i] = old_val + delta
                    s = np.sum(h_work)
                    if abs(s) < 1e-10:
                        h_work[i] = old_val
                        continue
                    v = _eval_value(h_work)
                    if v < best_val:
                        best_val = v
                        best_h = h_work.copy()
                        improved_any = True
                    else:
                        h_work[i] = old_val
            if not improved_any:
                break
    
    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END

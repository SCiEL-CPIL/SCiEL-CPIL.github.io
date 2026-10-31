import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def time_left():
        return deadline - time.time()

    def compute_metric(h):
        """Compute normalized max of autoconvolution: max(f*f) / (sum(h))^2"""
        h = np.asarray(h, dtype=np.float64)
        total = np.sum(h)
        if total < 1e-15:
            return float('inf')
        # Autoconvolution via FFT
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        acorr = np.fft.irfft(H * np.conj(H), n=size)
        peak = np.max(acorr)
        return peak / (total * total)

    def spectral_to_heights(spectrum):
        """
        Convert a nonnegative, even spectrum to step heights.
        spectrum: array of length N+1 (rfft bins)
        """
        H = np.sqrt(np.maximum(spectrum, 0.0))
        h = np.fft.irfft(H, n=N)
        # Project to nonnegative
        h = np.maximum(h, 0.0)
        total = np.sum(h)
        if total > 1e-15:
            h = h / total
        else:
            h = np.ones(N) / N
        return h

    def log_sum_exp_penalty(values, temperature):
        """Smooth max surrogate using log-sum-exp."""
        if temperature <= 0:
            return np.max(values)
        shifted = values - np.max(values)
        return np.max(values) + temperature * np.log(np.sum(np.exp(shifted / temperature)))

    def smooth_objective(spectrum, temperature):
        """
        Objective: smooth max of autoconvolution spectrum.
        The autoconvolution in time domain has FFT equal to |H|^2 = spectrum.
        So the autoconvolution values are the inverse FFT of the spectrum.
        We want to minimize the max of the inverse FFT of the spectrum.
        
        But wait - let me reconsider. The autoconvolution of h is:
        (h*h)[k] = sum_i h[i] * h[k-i]
        Its FFT is |FFT(h)|^2.
        
        If we parameterize h via its FFT (spectrum), then:
        FFT(h) = sqrt(spectrum) (taking real, nonnegative square root for even spectrum)
        The autoconvolution in frequency domain is |FFT(h)|^2 = spectrum.
        The autoconvolution in time domain is IFFT(spectrum).
        
        So we want to minimize max(IFFT(spectrum)) subject to:
        - spectrum >= 0
        - IFFT(sqrt(spectrum)) >= 0 (nonnegativity of h)
        - sum(h) = 1 (normalization)
        
        Actually, the constraint is on h, not on the autoconvolution directly.
        Let me re-derive:
        
        Let h be our step function. We want to minimize max(autoconv(h)) / (sum(h))^2.
        
        If we set sum(h) = 1, we just minimize max(autoconv(h)).
        
        In frequency domain: autoconv(h) has DFT = |DFT(h)|^2.
        So max(autoconv(h)) = max(IFFT(|DFT(h)|^2)).
        
        If we parameterize by S = |DFT(h)|^2 (the power spectrum), then:
        - S >= 0
        - h = IFFT(sqrt(S)) must be >= 0
        - sum(h) = 1, which means S[0] = 1 (since sum(h) = S[0] for DFT)
        
        So the problem becomes: minimize max(IFFT(S)) subject to S >= 0, S[0]=1, IFFT(sqrt(S)) >= 0.
        
        This is a convex optimization problem! The objective max(IFFT(S)) is convex in S
        (it's the max of linear functions), and the constraints are convex.
        """
        # Compute autoconvolution in time domain
        h = np.fft.irfft(np.sqrt(np.maximum(spectrum, 0.0)), n=N)
        acorr = np.fft.irfft(spectrum, n=N)
        
        # Use log-sum-exp for smoothness
        return log_sum_exp_penalty(acorr, temperature)

    def projected_gradient_descent(spectrum_init, max_iters, lr, temperature):
        """
        Projected gradient descent on the power spectrum.
        Constraints:
        - S >= 0
        - S[0] = 1 (normalization)
        - IFFT(sqrt(S)) >= 0 (nonnegativity of h)
        """
        S = spectrum_init.copy().astype(np.float64)
        S = np.maximum(S, 0.0)
        S[0] = 1.0  # Ensure sum(h) = 1
        
        best_S = S.copy()
        best_val = float('inf')
        
        for it in range(max_iters):
            if time_left() < 5:
                break
            
            # Compute gradient of log-sum-exp penalty
            acorr = np.fft.irfft(S, n=N)
            max_val = np.max(acorr)
            shifted = (acorr - max_val) / temperature
            weights = np.exp(shifted)
            weights /= np.sum(weights)
            
            # Gradient of log-sum-exp wrt S
            # d/dS [max + T*log(sum(exp((acorr-max)/T))]
            # = T * d/dS [log(sum(exp((acorr-max)/T))]
            # acorr = IFFT(S), so d acorr / dS = IFFT identity (linear)
            # Gradient = T * IFFT(weights)
            grad = temperature * np.fft.irfft(weights, n=N)
            
            # Update S
            S_new = S - lr * grad
            
            # Project onto feasible set
            S_new = np.maximum(S_new, 0.0)
            S_new[0] = 1.0  # Enforce normalization
            
            # Check nonnegativity of h = IFFT(sqrt(S_new))
            h_new = np.fft.irfft(np.sqrt(S_new), n=N)
            if np.min(h_new) < -1e-10:
                # Project back: reduce S to make h nonnegative
                # Simple approach: scale down the problematic frequencies
                # For now, just keep the old S if projection fails
                S_new = S.copy()
            
            S = S_new
            
            # Evaluate
            h = np.fft.irfft(np.sqrt(S), n=N)
            h = np.maximum(h, 0.0)
            total = np.sum(h)
            if total > 1e-15:
                h = h / total
            else:
                h = np.ones(N) / N
            
            val = compute_metric(h)
            if val < best_val:
                best_val = val
                best_S = S.copy()
            
            # Adaptive learning rate
            if it % 50 == 0 and it > 0:
                # Reduce learning rate slowly
                lr *= 0.95
        
        return best_S, best_val

    def initialize_spectrum(shape='gaussian', bandwidth=0.3):
        """Initialize the power spectrum."""
        k = np.arange(N + 1, dtype=float)
        
        if shape == 'gaussian':
            S = np.exp(-(k / (bandwidth * N)) ** 2)
        elif shape == 'uniform':
            S = np.ones(N + 1)
        elif shape == 'lorentzian':
            gamma = 0.1
            S = gamma / (k**2 + gamma**2 * N**2)
        else:
            S = np.exp(-(k / (0.3 * N)) ** 2)
        
        # Normalize so S[0] = 1
        S[0] = 1.0
        
        return S

    # Main optimization loop
    best_h = None
    best_m = float('inf')
    
    # Try different initializations
    initializations = [
        ('gaussian', 0.2),
        ('gaussian', 0.3),
        ('gaussian', 0.5),
        ('lorentzian', None),
        ('uniform', None),
    ]
    
    for shape, bw in initializations:
        if time_left() < 30:
            break
        
        S_init = initialize_spectrum(shape, bw if bw is not None else 0.3)
        
        # Run projected gradient descent with different temperatures
        for temperature in [0.01, 0.05, 0.1]:
            if time_left() < 20:
                break
            
            S_best, val = projected_gradient_descent(
                S_init,
                max_iters=2000,
                lr=0.01,
                temperature=temperature
            )
            
            if val < best_m:
                best_m = val
                h = spectral_to_heights(S_best)
                best_h = h
    
    # Local refinement in height space
    if best_h is not None and time_left() > 30:
        h = best_h.copy()
        np.random.seed(42)
        
        for it in range(5000):
            if time_left() < 10:
                break
            
            # Random perturbation
            idx = np.random.randint(0, N)
            delta = np.random.normal(0, 0.005)
            
            new_h = h.copy()
            new_h[idx] += delta
            new_h = np.maximum(new_h, 0)
            
            total = np.sum(new_h)
            if total > 1e-15:
                new_h = new_h / total
            
            m_new = compute_metric(new_h)
            if m_new < best_m:
                best_m = m_new
                best_h = new_h
                h = new_h
    
    # Ensure valid result
    if best_h is None:
        best_h = np.ones(N) / N
        best_m = compute_metric(best_h)
    
    return list(best_h)

# EVOLVE-BLOCK-END

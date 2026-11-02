"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import rfft, irfft

def _autocorrelation_max_abs(s):
    """Compute max absolute off-peak autocorrelation for a +/-1 sequence."""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    
    s_float = s.astype(np.float64)
    f = np.fft.fft(s_float, n=n_fft)
    acf = np.fft.ifft(f * np.conj(f)).real[:N]
    acf /= N  # normalize
    
    # Off-peak: exclude lag 0
    off_peak = np.abs(acf[1:])
    if len(off_peak) == 0:
        return 0.0
    return float(np.max(off_peak))

def _score_sequence(s):
    """The score is the upper bound on C3 - we want to minimize this."""
    return _autocorrelation_max_abs(s)

def _l2_autocorrelation(s):
    """Compute L2 norm of off-peak autocorrelation for gradient-based optimization."""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    
    s_float = s.astype(np.float64)
    f = np.fft.fft(s_float, n=n_fft)
    acf = np.fft.ifft(f * np.conj(f)).real[:N]
    acf /= N
    
    # Off-peak: exclude lag 0
    off_peak = acf[1:]
    if len(off_peak) == 0:
        return 0.0
    return float(np.sum(off_peak ** 2))

def _gradient_descent_continuous():
    """
    Use continuous gradient descent on L2 norm of autocorrelation to find
    a good continuous signal, then binarize it to get a strong initial binary seed.
    """
    rng = np.random.default_rng(123)
    
    # Start with random continuous values in [-1, 1]
    x = rng.uniform(-1, 1, size=N)
    
    lr = 0.05
    n_iter = 2000
    
    for i in range(n_iter):
        # Compute autocorrelation
        n_fft = 1
        while n_fft < 2 * N:
            n_fft *= 2
        
        f = np.fft.fft(x, n=n_fft)
        acf = np.fft.ifft(f * np.conj(f)).real[:N]
        acf /= N
        
        # Gradient of L2 norm of off-peak autocorrelation
        # The autocorrelation is a quadratic form, so gradient is linear in x
        # d/dx_k sum_{lag=1}^{N-1} acf[lag]^2 = 2 * sum_{lag=1}^{N-1} acf[lag] * d(acf[lag])/dx_k
        # d(acf[lag])/dx_k = 1/N * (x[k+lag] + x[k-lag]) (with wraparound for circular)
        # But we want linear autocorrelation, so let's use a simpler approach
        
        # Actually, let's just use numerical gradient or a simpler heuristic
        # For now, use a simple approach: project towards zero autocorrelation
        
        # Compute the autocorrelation power spectrum
        power = np.abs(f) ** 2
        # We want to flatten the spectrum (make it white noise-like)
        # Gradient of sum of squared autocorrelation off-peak
        
        # Use a simpler approach: minimize variance of the power spectrum
        # which is equivalent to making the autocorrelation as flat as possible
        
        # For a +/-1 sequence, the optimal is related to Barker sequences
        # Let's just do a simple iterative method
        
        # Compute current L2
        off_peak = acf[1:]
        current_l2 = np.sum(off_peak ** 2)
        
        # Simple gradient step: move each x[k] to reduce autocorrelation
        # This is approximate but should work
        grad = np.zeros(N)
        for lag in range(1, N):
            weight = 2 * acf[lag] / N
            for k in range(N - lag):
                grad[k] += weight * x[k + lag]
                grad[k + lag] += weight * x[k]
        
        x -= lr * grad
        # Clip to [-1, 1]
        x = np.clip(x, -1, 1)
        
        if i % 500 == 0:
            # Binarize and check
            s = np.sign(x)
            s[s == 0] = 1
            score = _score_sequence(s)
    
    # Final binarization
    s = np.sign(x)
    s[s == 0] = 1
    return s

def _local_search(s, max_time=10.0):
    """
    Perform local search by flipping bits to minimize the peak sidelobe.
    """
    start_time = time.time()
    best_s = s.copy()
    best_score = _score_sequence(best_s)
    
    # Precompute FFT size
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    
    def _quick_score(s_arr):
        """Quick score computation using precomputed FFT size."""
        s_float = s_arr.astype(np.float64)
        f = np.fft.fft(s_float, n=n_fft)
        acf = np.fft.ifft(f * np.conj(f)).real[:N]
        acf /= N
        off_peak = np.abs(acf[1:])
        return float(np.max(off_peak)) if len(off_peak) > 0 else 0.0
    
    best_score = _quick_score(best_s)
    improved = True
    pass_count = 0
    
    while improved and (time.time() - start_time) < max_time:
        improved = False
        pass_count += 1
        
        # Try flipping each bit
        for i in range(N):
            if (time.time() - start_time) >= max_time:
                break
            
            s_old = best_s[i]
            best_s[i] = -s_old
            new_score = _quick_score(best_s)
            
            if new_score < best_score - 1e-12:
                best_score = new_score
                improved = True
            else:
                best_s[i] = s_old
    
    return best_s, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    # Step 1: Get a good initial seed using continuous gradient descent
    best_s = _gradient_descent_continuous()
    best_score = _score_sequence(best_s)
    
    # Step 2: Try some structured patterns
    # Alternating
    alt = np.ones(N)
    alt[1::2] = -1
    score_alt = _score_sequence(alt)
    if score_alt < best_score:
        best_s = alt.copy()
        best_score = score_alt
    
    # Check time
    if time.time() > deadline:
        return best_s
    
    # Step 3: Local search from the best seed
    best_s, best_score = _local_search(best_s, max_time=30.0)
    
    if time.time() > deadline:
        return best_s
    
    # Step 4: Try random restarts with local search
    rng = np.random.default_rng(42)
    
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        s = rng.choice([-1, 1], size=N)
        score = _score_sequence(s)
        
        if score < best_score:
            best_s = s.copy()
            best_score = score
        
        # Local search with limited time
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        s_local, score_local = _local_search(s, max_time=min(remaining - 5.0, 10.0))
        
        if score_local < best_score:
            best_s = s_local.copy()
            best_score = score_local
    
    return best_s
# EVOLVE-BLOCK-END

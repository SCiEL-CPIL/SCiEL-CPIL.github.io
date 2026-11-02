"""Program for the suite task; uses continuous optimization then binary local search."""
N = 400
import numpy as np
import time

# EVOLVE-BLOCK-START

def _autocorr_max_abs(s):
    """Compute max |autocorrelation| over non-zero lags for a length-N sequence."""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    if N <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:])))

def _autocorr_l2_sq(s):
    """Compute sum of squared autocorrelation values over non-zero lags."""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    if N <= 1:
        return 0.0
    return float(np.sum(acf[1:]**2))

def _continuous_to_binary(x):
    """Convert continuous vector to binary ±1 vector by sign."""
    return np.sign(x).astype(np.float64)
    # Ensure no zeros become 0; replace with +1
    s = np.where(x >= 0, 1.0, -1.0)
    return s

def _continuous_optimize(max_time=30.0):
    """
    Optimize a continuous vector x to minimize L2 norm of autocorrelation.
    This gives a good initial binary seed via sign(x).
    Uses gradient descent with adaptive step size.
    """
    np.random.seed(12345)
    x = np.random.randn(N)
    x = x / np.linalg.norm(x) * np.sqrt(N)  # normalize to have unit energy per element
    
    best_x = x.copy()
    best_l2 = _autocorr_l2_sq(_continuous_to_binary(x))
    
    start_time = time.time()
    lr = 0.1
    patience = 0
    max_patience = 50
    
    while time.time() - start_time < max_time:
        # Compute gradient of L2^2 autocorrelation w.r.t. x
        # autocorr[k] = sum_i x[i] x[i+k]
        # d/dx[j] (sum_k autocorr[k]^2) = 2 * sum_k autocorr[k] * d(autocorr[k])/dx[j]
        # d(autocorr[k])/dx[j] = x[j+k] + x[j-k] (for valid indices)
        
        n_fft = 1
        while n_fft < 2 * N:
            n_fft *= 2
        F = np.fft.rfft(x, n=n_fft)
        acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
        
        # Gradient via FFT-based approach
        # grad[j] = 2 * sum_{k=1}^{N-1} acf[k] * (x[j+k] + x[j-k]) with boundary handling
        # Use circular correlation for simplicity (approximation for large N)
        acf_circ = np.fft.irfft(F * np.conj(F), n=n_fft)[:n_fft]
        # grad = 2 * ifft(fft(acf_circ) * conj(fft(x)))... actually let's use direct formula
        # For circular: grad[j] = 2 * sum_k acf[k] * (x[(j+k)%n_fft] + x[(j-k)%n_fft])
        # This is 2 * circular_convolution(acf, x) + 2 * circular_convolution(acf_reversed, x)
        
        # Simpler: use the fact that d/dx sum_k (sum_i x[i]x[i+k])^2 
        # = 4 * sum_k acf[k] * circ_corr(x, shift k)
        
        # Let's compute gradient directly via FFT
        # autocorr[k] = sum_i x[i] x[i+k] (circular for simplicity)
        # Using FFT: acf = ifft(|fft(x)|^2)
        # grad = 4 * ifft(fft(acf) * conj(fft(x)))
        
        acf_full = np.fft.irfft(F * np.conj(F), n=n_fft)
        # We only care about lags 1..N-1
        acf_lags = np.zeros(n_fft)
        acf_lags[1:N] = acf[1:N]
        acf_lags[n_fft - (N-1):n_fft] = acf[1:N]  # negative lags
        
        F_acf = np.fft.rfft(acf_lags, n=n_fft)
        grad = 4.0 * np.fft.irfft(F_acf * np.conj(F), n=n_fft)[:N]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad = grad / grad_norm
        
        # Adaptive step size
        x_new = x - lr * grad
        
        # Evaluate
        s_new = _continuous_to_binary(x_new)
        l2_new = _autocorr_l2_sq(s_new)
        
        if l2_new < best_l2:
            best_l2 = l2_new
            best_x = x_new.copy()
            patience = 0
            lr = min(lr * 1.05, 1.0)
        else:
            patience += 1
            lr *= 0.95
            if patience > max_patience:
                break
        
        x = x_new
    
    return _continuous_to_binary(best_x)

def _local_search(s, max_time=5.0):
    """Greedy local search: flip individual elements to reduce max |autocorr|."""
    s = s.copy()
    current_val = _autocorr_max_abs(s)
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        improved = False
        for i in range(N):
            s[i] = -s[i]
            new_val = _autocorr_max_abs(s)
            if new_val < current_val:
                current_val = new_val
                improved = True
            else:
                s[i] = -s[i]  # revert
        if not improved:
            break
    return s, current_val

def _pairwise_local_search(s, max_time=10.0):
    """Local search trying pairwise flips to escape local minima."""
    s = s.copy()
    current_val = _autocorr_max_abs(s)
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        improved = False
        # Try pairwise flips on a random subset
        for _ in range(200):
            i, j = np.random.choice(N, 2, replace=False)
            s[i] = -s[i]
            s[j] = -s[j]
            new_val = _autocorr_max_abs(s)
            if new_val < current_val:
                current_val = new_val
                improved = True
            else:
                s[i] = -s[i]
                s[j] = -s[j]
        if not improved:
            break
    return s, current_val

def _simulated_annealing(s, max_time=40.0):
    """Simulated annealing to minimize max |autocorr|."""
    s = s.copy()
    current_val = _autocorr_max_abs(s)
    best_s = s.copy()
    best_val = current_val
    start_time = time.time()
    T = 5.0
    cooling = 0.9995
    
    while time.time() - start_time < max_time:
        # Flip a random element
        i = np.random.randint(N)
        s[i] = -s[i]
        new_val = _autocorr_max_abs(s)
        
        delta = new_val - current_val
        if delta < 0 or (T > 1e-10 and np.random.rand() < np.exp(-delta / T)):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_s = s.copy()
        else:
            s[i] = -s[i]  # revert
        
        T *= cooling
    
    return best_s, best_val

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Step 1: Continuous optimization to get a good initial binary seed
    s = _continuous_optimize(max_time=30.0)
    val = _autocorr_max_abs(s)
    
    # Step 2: Local search to refine
    remaining = deadline - time.time()
    if remaining > 5:
        s, val = _local_search(s, max_time=min(remaining - 2, 5.0))
    
    # Step 3: Pairwise local search to escape local minima
    remaining = deadline - time.time()
    if remaining > 10:
        s, val = _pairwise_local_search(s, max_time=min(remaining - 5, 10.0))
    
    # Step 4: Simulated annealing with remaining time
    remaining = deadline - time.time()
    if remaining > 15:
        s, val = _simulated_annealing(s, max_time=min(remaining - 5, 40.0))
    
    # Step 5: Final local search
    remaining = deadline - time.time()
    if remaining > 2:
        s, val = _local_search(s, max_time=min(remaining - 1, 5.0))
    
    # Ensure output is ±1
    s = np.where(s >= 0, 1.0, -1.0)
    
    return s

# EVOLVE-BLOCK-END

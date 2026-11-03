"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Improved approach: FFT-accelerated Subgradient Descent.
We minimize the peak of the autoconvolution by iteratively reducing the
values at the lags where the convolution is highest.
Using FFT reduces convolution cost from O(N^2) to O(N log N), allowing
for many more iterations within the time limit.
"""

import numpy as np
import time

N = 600
TIME_LIMIT = 115  # safety margin before 120s kill

def fft_convolve(f):
    """Compute full autoconvolution using FFT."""
    # We need size at least 2*N - 1
    size = 2 * N
    F = np.fft.rfft(f, n=size)
    C = np.fft.irfft(F * F, n=size)
    return C[:2*N-1]

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = fft_convolve(f)
    return np.max(conv) / (s * s)

def initial_guess_parabolic():
    """Create a parabolic (radial basis) initial guess."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f = np.maximum(1.0 - (x / 0.25) ** 2, 0.0)
    return f

def initial_guess_cosine():
    """Create a cosine bump initial guess."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f = 0.5 * (1.0 + np.cos(2 * np.pi * x))
    return f

def subgradient_optimization(f_init, start_time, max_time, rng=None):
    """
    Optimize f using subgradient descent on the peak of the autoconvolution.
    
    Objective: Minimize max_k conv[k] subject to sum(f) = N, f >= 0.
    
    We use a smooth approximation: Minimize sum_{k in ActiveSet} conv[k].
    The gradient of conv[k] w.r.t f[j] is f[k-j] (with appropriate bounds).
    """
    if rng is None:
        rng = np.random.RandomState(42)
        
    best_f = f_init.copy()
    best_val = compute_score(f_init)
    current_f = f_init.copy()
    
    # Initial step size
    step_size = 0.1
    cooling = 0.99
    active_fraction = 0.1  # Consider top 10% of convolution lags
    
    # Precompute FFT size
    fft_size = 2 * N
    
    for iteration in range(100000):
        if time.time() - start_time > max_time:
            break
            
        # Normalize to sum = N
        s = np.sum(current_f)
        if s < 1e-12:
            break
        current_f = current_f * (N / s)
        
        # Compute convolution
        conv = fft_convolve(current_f)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_f = current_f.copy()
            
        # Determine active set: indices where conv is close to max
        # We want to reduce the peak, so we focus on high values
        threshold = max_c * (1.0 - active_fraction)
        active_indices = np.where(conv >= threshold)[0]
        
        if len(active_indices) == 0:
            break
            
        # Compute gradient of sum_{k in active} conv[k]
        # grad[j] = sum_{k in active} d(conv[k])/d(f[j])
        # d(conv[k])/d(f[j]) = f[k-j] if 0 <= k-j < N else 0
        # Note: conv[k] = sum_{j} f[j] f[k-j]
        # This is equivalent to correlating the indicator of active set with f
        # Let A be the indicator vector of active indices (length 2N-1)
        # Then grad[j] = sum_{k} A[k] f[k-j]
        # This is the cross-correlation of A and f.
        
        # Construct A
        A = np.zeros(2*N-1)
        A[active_indices] = 1.0
        
        # Compute cross-correlation using FFT
        # corr[j] = sum_k A[k] f[k-j]
        # This is equivalent to convolution of A and f reversed?
        # Let's use direct correlation formula via FFT:
        # corr = ifft(fft(A) * conj(fft(f))) ? No, that's circular.
        # Let's just do the linear correlation.
        # A has length 2N-1, f has length N.
        # We want g[j] = sum_{k=0}^{2N-2} A[k] f[k-j] where f is 0 outside [0, N-1].
        
        # Efficient way:
        # Pad f to length 2N-1
        f_pad = np.zeros(2*N-1)
        f_pad[:N] = current_f
        
        # Cross correlation: sum_k A[k] f_pad[k-j]
        # This is the convolution of A and reverse(f_pad)?
        # Let's use np.correlate if it's fast enough? No, N=600, 2N-1=1199.
        # np.correlate is O(N^2) ~ 1e6 ops. 100k iterations -> 1e11 ops. Too slow.
        # Must use FFT.
        
        # corr[j] = sum_k A[k] f[k-j]
        # Let A_rev = A[::-1]. Then corr is convolution of A_rev and f?
        # (A_rev * f)[j] = sum_k A_rev[k] f[j-k] = sum_k A[2N-2-k] f[j-k]
        # Let m = 2N-2-k. k = 2N-2-m.
        # sum_m A[m] f[j - (2N-2-m)] = sum_m A[m] f[m + j - (2N-2)]
        # This looks like a shift.
        
        # Actually, the gradient of sum_{k in A} (f*f)[k] with respect to f[j] is:
        # sum_{k in A} f[k-j]
        # This is the cross-correlation of the indicator vector of A (padded to length N? No, A is longer)
        
        # Let's just compute the gradient by summing the shifted f's.
        # Since active set is sparse-ish (10% of 1199 ~ 120 indices), we can sum 120 shifted arrays.
        # 120 * 600 = 72000 ops per iteration. 100k iterations -> 7e9 ops. Might be okay in numpy if vectorized?
        # Let's try to vectorize the sum.
        
        # grad = sum_{k in active_indices} shift(f, k)
        # We can do this by zero-padding f to length 2N-1, then for each k, adding f[k:k+N] to grad[0:N].
        
        grad = np.zeros(N)
        for k in active_indices:
            # f[k-j] is non-zero when 0 <= k-j < N => k-N < j <= k
            # j ranges from max(0, k-N+1) to min(N-1, k)
            # Actually, f is defined on [0, N-1].
            # Term is f[k-j]. Let i = k-j. i in [0, N-1] => j = k-i.
            # j must be in [0, N-1].
            # So i must be in [max(0, k-N+1), min(N-1, k)]
            i_min = max(0, k - N + 1)
            i_max = min(N - 1, k)
            if i_min <= i_max:
                # grad[j] += f[i] where j = k-i
                # j ranges from k-i_max to k-i_min
                j_min = k - i_max
                j_max = k - i_min
                grad[j_min:j_max+1] += current_f[i_min:i_max+1][::-1]
                
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
            
        # Update
        f_new = current_f - step_size * grad / grad_norm
        
        # Project to non-negative
        f_new = np.maximum(f_new, 0.0)
        
        # Normalize sum to N
        s_new = np.sum(f_new)
        if s_new < 1e-12:
            f_new = current_f
        else:
            f_new = f_new * (N / s_new)
            
        # Check improvement
        conv_new = fft_convolve(f_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_f = f_new
            step_size *= 0.95  # Slight reduction if good
        else:
            step_size *= 0.8  # Reduce step if bad
            # Optional: Revert
            # current_f = current_f # Keep old
            
        # Cool down
        step_size *= cooling
        if step_size < 1e-6:
            break
            
    return best_f, best_val

def construct():
    start_time = time.time()
    rng = np.random.RandomState(123)
    
    initial_guesses = [
        initial_guess_parabolic(),
        initial_guess_cosine(),
        np.ones(N),
    ]
    
    best_f = None
    best_score = np.inf
    
    for guess in initial_guesses:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        refined_f, refined_score = subgradient_optimization(
            guess, start_time, TIME_LIMIT - 5, rng
        )
        
        if refined_score < best_score:
            best_score = refined_score
            best_f = refined_f
            
    if best_f is None:
        best_f = np.ones(N)
        
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")

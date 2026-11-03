"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
import scipy.signal as sig

def _autoconv_max_abs(h):
    """Compute the maximum absolute value of the autoconvolution of h."""
    # For a discrete sequence h, autoconvolution is h * h (full convolution)
    # But the task mentions "autoconvolution" which for step functions might mean
    # the continuous autoconvolution. Let's interpret it as the discrete convolution
    # of the step heights with themselves, evaluated at all lags.
    #
    # Actually, for a step function defined on [0,1] with N steps, the autoconvolution
    # at lag k is sum_{i=0}^{N-k-1} h[i] * h[i+k] for k >= 0, and by symmetry for k < 0.
    # The "maximum absolute autoconvolution" is max_k |sum_i h[i] h[i+k]|.
    #
    # However, the task says "divided by squared integral". The integral of a step function
    # with heights h[i] and width 1/N is (1/N) * sum h[i]. So squared integral is ((1/N)*sum(h))^2.
    #
    # Let's compute this efficiently using FFT.
    
    # Pad to avoid circular convolution effects
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    
    # The linear autoconvolution of length 2N-1 is in conv[0:2*N-1]
    # For real signals, the autoconvolution is symmetric: conv[k] = conv[2*N-1-k]
    # We want the maximum absolute value over all lags
    return np.max(np.abs(conv[:2*N-1]))

def _score(h):
    """Compute the score: max_abs_autoconv / (integral^2). Lower is better."""
    integral = np.sum(h) / N
    if abs(integral) < 1e-15:
        return np.inf
    max_abs_conv = _autoconv_max_abs(h)
    return max_abs_conv / (integral * integral)

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin within 150s limit
    
    best_h = None
    best_score = np.inf
    
    # Try some simple structured initial guesses
    # 1. All ones (baseline)
    h = np.ones(N)
    s = _score(h)
    if s < best_score:
        best_score = s
        best_h = h.copy()
    
    # 2. Alternating signs
    h = np.ones(N)
    h[::2] = -1
    s = _score(h)
    if s < best_score:
        best_score = s
        best_h = h.copy()
    
    # 3. Block structure: +1 for first half, -1 for second half
    h = np.ones(N)
    h[N//2:] = -1
    s = _score(h)
    if s < best_score:
        best_score = s
        best_h = h.copy()
    
    # 4. Try to optimize using gradient-based methods with a smooth surrogate
    # The objective is max_k |conv[k]| / (sum(h)/N)^2
    # We can use a soft-max approximation: sum_k exp(beta * |conv[k]|) / (N*beta)
    # to make it differentiable, then minimize this.
    
    # Use L-BFGS-B with bounds on h (e.g., |h[i]| <= some max)
    # Actually, the problem is scale-invariant in a sense: if we scale h by c,
    # autoconv scales by c^2 and integral^2 scales by c^2, so the ratio is invariant.
    # So we can fix the norm of h, e.g., sum(h^2) = N (unit RMS), and minimize max_abs_conv.
    
    # Let's reformulate: minimize max_k |conv[k]| subject to sum(h) = N (so integral = 1)
    # or just minimize max_k |conv[k]| / (sum(h)/N)^2 directly with a soft-max.
    
    def objective(h):
        # Soft-max of |conv| with beta
        beta = 100.0
        conv = _get_autoconv(h)
        abs_conv = np.abs(conv)
        # Soft max: (1/beta) * log(sum(exp(beta * x)))
        # For numerical stability:
        m = np.max(abs_conv)
        if m < 1e-15:
            return 0.0
        soft_max = m + (1.0 / beta) * np.log(np.sum(np.exp(beta * (abs_conv - m))))
        integral = np.sum(h) / N
        if abs(integral) < 1e-15:
            return 1e10
        return soft_max / (integral * integral)
    
    def _get_autoconv(h):
        n_fft = 1 << (2 * N - 1).bit_length()
        H = np.fft.rfft(h, n=n_fft)
        conv = np.fft.irfft(H * H, n=n_fft)
        return conv[:2*N-1]
    
    # Gradient via finite differences or analytical?
    # The autoconvolution is quadratic in h, so the gradient of conv[k] w.r.t. h[i]
    # is: for k >= 0, conv[k] = sum_j h[j] h[j+k], so d(conv[k])/dh[i] = h[i+k] (if i+k < N) + h[i-k] (if i-k >= 0)
    # But for soft-max, the gradient is more complex. Let's use finite differences for simplicity,
    # but that's O(N) evaluations per gradient step, which is too slow for N=400.
    
    # Alternative: use scipy.optimize with a penalty method or just try random perturbations
    # and accept improvements (simulated annealing style).
    
    # Simulated annealing / random search
    current_h = best_h.copy().astype(float)
    current_score = best_score
    
    T = 0.1  # Initial temperature
    cooling = 0.999
    
    # Number of iterations limited by time
    max_iter = 50000
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Random perturbation: flip a random subset of signs or perturb magnitudes
        # Since the problem is about signed step functions, let's try flipping signs of random elements
        idx = np.random.randint(0, N, size=np.random.randint(1, 10))
        new_h = current_h.copy()
        new_h[idx] *= -1  # Flip signs
        
        new_score = _score(new_h)
        
        if new_score < current_score or np.random.rand() < np.exp((current_score - new_score) / T):
            current_h = new_h
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
        
        T *= cooling
        if T < 1e-6:
            T = 0.01  # Restart with higher temperature
    
    # Final polish: try local search around best_h
    if best_h is not None and time.time() < deadline - 10:
        # Try flipping each element individually and keep if it improves
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(N):
                if time.time() > deadline - 10:
                    break
                h_trial = best_h.copy()
                h_trial[i] *= -1
                s_trial = _score(h_trial)
                if s_trial < best_score:
                    best_score = s_trial
                    best_h = h_trial
                    improved = True
    
    # Ensure output is a sequence of signed step heights
    # The task says "signed step heights", so return an array of floats
    return best_h if best_h is not None else np.ones(N)
# EVOLVE-BLOCK-END

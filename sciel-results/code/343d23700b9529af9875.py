"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Discrete spectral search.
Parameterize the step function by its DFT coefficients.
Enforce non-negativity via a soft-plus projection.
Directly minimize the maximum value of the inverse DFT of the squared magnitude spectrum
using coordinate descent on the frequency domain.
This decouples the peak constraint from the time-domain shape search.

Key insight: The autoconvolution in the time domain corresponds to the squared
magnitude of the DFT in the frequency domain (up to scaling). By working in the
frequency domain, we can directly control the "energy distribution" of the
autoconvolution. The maximum of the autoconvolution is related to the peak of
the inverse DFT of |F|^2. We use coordinate descent on the frequency coefficients
to flatten this peak, while projecting back to the time domain to enforce
non-negativity.
"""

import numpy as np
import time

N = 600
TIME_LIMIT = 115  # safety margin before 120s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    
    # Use FFT for faster convolution calculation
    # Linear convolution of two length N signals requires FFT size >= 2N-1
    # We use 2*N for simplicity and safety
    f_fft = np.fft.fft(f, n=2*N)
    conv_fft = f_fft * f_fft
    conv = np.real(np.fft.ifft(conv_fft))
    # The linear autoconvolution is in conv[0:2*N-1]
    max_val = np.max(conv[:2*N-1])
    return max_val / (s * s)

def gs_optimize(F_init, max_iters, start_time, time_limit):
    """
    Gerchberg-Saxton style optimization to flatten the power spectrum
    while enforcing non-negativity in the time domain.
    """
    current_F = F_init.copy()
    best_f = np.real(np.fft.ifft(current_F))
    best_f = np.maximum(best_f, 0.0)
    best_score = compute_score(best_f)
    
    for i in range(max_iters):
        if time.time() > time_limit:
            break
        
        # Time domain projection: enforce non-negativity
        f = np.real(np.fft.ifft(current_F))
        f = np.maximum(f, 0.0)
        
        # Frequency domain projection: flatten magnitude
        F_new = np.fft.fft(f)
        
        # Calculate target magnitude. 
        # We want |F| to be as flat as possible.
        # A simple strategy is to set all magnitudes to the mean magnitude.
        # However, we must preserve the DC component's relative weight?
        # Actually, for minimizing the peak of the autoconvolution, a flat spectrum is ideal.
        # Let's set |F| to a constant C.
        # What should C be? The mean of the current magnitudes is a good choice to maintain energy.
        mags = np.abs(F_new)
        target_mag = np.mean(mags)
        
        # Preserve phase
        phases = np.angle(F_new)
        
        # Construct new F
        current_F = target_mag * np.exp(1j * phases)
        
        # Ensure conjugate symmetry for real signal (though FFT of real f is already symmetric)
        # Just in case of numerical errors, enforce it
        for k in range(1, N // 2 + 1):
            current_F[N - k] = np.conj(current_F[k])
        
        # Evaluate score periodically
        if i % 10 == 0 or i == max_iters - 1:
            f_trial = np.real(np.fft.ifft(current_F))
            f_trial = np.maximum(f_trial, 0.0)
            score_trial = compute_score(f_trial)
            if score_trial < best_score:
                best_score = score_trial
                best_f = f_trial
    
    return best_f, best_score

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    n_restarts = 50
    half = N // 2
    
    for r in range(n_restarts):
        if time.time() > TIME_LIMIT - 5:
            break
        
        # Random initialization: random phases, flat magnitude
        F_init = np.zeros(N, dtype=np.complex128)
        
        # Set DC component
        F_init[0] = 1.0
        
        # Set other frequencies with random phases and unit magnitude
        for k in range(1, N):
            phase = np.random.uniform(0, 2 * np.pi)
            F_init[k] = np.exp(1j * phase)
        
        # Enforce conjugate symmetry
        for k in range(1, half + 1):
            F_init[N - k] = np.conj(F_init[k])
        
        # Run GS optimization
        f_opt, score_opt = gs_optimize(F_init, max_iters=100, start_time=start_time, time_limit=TIME_LIMIT - 5)
        
        if score_opt < best_score:
            best_score = score_opt
            best_f = f_opt
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")

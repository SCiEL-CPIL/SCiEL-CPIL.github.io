"""
Strategy: Spectral Minimization via Alternating Projections.
We want to minimize max_k |autocorr(k)| / (sum f)^2.
Note that autocorr(k) is the inverse FFT of |FFT(f)|^2.
The max absolute value of a signal is related to the peak of its time-domain representation.
By flattening the magnitude spectrum |FFT(f)|, we are minimizing the peak sidelobe of the autocorrelation.
We use an alternating projection algorithm:
1. Compute FFT of current f.
2. Force magnitude to a target level (flat spectrum) while keeping phase.
3. Inverse FFT to get time domain.
4. Project time domain onto the feasible set (signed step functions, i.e., values in {-1, 1}).
   This projection is simply f_new = sign(f_time).
5. Repeat.
"""

import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
_DEADLINE_S = 110.0

def _objective(f):
    """max_k |autocorr(k)| / (sum f)^2."""
    f = np.asarray(f, dtype=float)
    S = float(np.sum(f))
    if not np.isfinite(S) or abs(S) < 1e-9:
        return float('inf')
    
    n = len(f)
    nfft = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(f, nfft)
    ac = np.fft.irfft(F * np.conj(F), nfft)[:2 * n - 1]
    
    if not np.all(np.isfinite(ac)):
        return float('inf')
    
    max_abs = float(np.max(np.abs(ac)))
    if not np.isfinite(max_abs) or max_abs < 0.0:
        return float('inf')
    
    return max_abs / (S * S)

def construct():
    t0 = time.time()
    
    # Initial guess: All ones (or random)
    # All ones has autocorr N at lag 0, 0 elsewhere (if N is such that no wrap-around issues, but linear autocorr of ones is triangular).
    # Actually, for f = ones(N), autocorr is N - |k|. Max is N. Sum is N. Score = N / N^2 = 1/N.
    # Let's start with a random binary sequence to avoid symmetry traps.
    rng = np.random.default_rng(42)
    f = 2.0 * (rng.integers(0, 2, size=N).astype(float) - 1.0)
    
    # Ensure sum is not zero
    while abs(np.sum(f)) < 1.0:
        f = 2.0 * (rng.integers(0, 2, size=N).astype(float) - 1.0)
        
    state = {
        'best_f': f.copy(),
        'best_val': _objective(f)
    }
    
    # Alternating Projections Loop
    # We iterate to flatten the spectrum.
    # The target magnitude is determined by the current average power or a fixed level.
    # Let's target a flat magnitude equal to the root mean square of the current magnitude.
    
    for i in range(2000):
        if time.time() - t0 > _DEADLINE_S:
            break
            
        # 1. Compute FFT
        F = np.fft.rfft(f)
        
        # 2. Flatten Magnitude
        # Calculate target magnitude. 
        # We want |F| to be constant. 
        # The energy is conserved in the projection if we scale carefully, but here we are just shaping.
        # Let's set the magnitude to the geometric mean or just a constant level.
        # A common approach in "flat spectrum" design is to set |F| = sqrt( sum |F|^2 / N_freqs ).
        # This preserves total power.
        
        power = np.sum(np.abs(F)**2)
        n_freqs = len(F)
        target_mag = np.sqrt(power / n_freqs)
        
        # Apply new magnitude, keep phase
        phase = np.angle(F)
        F_new = target_mag * np.exp(1j * phase)
        
        # 3. Inverse FFT
        f_time = np.fft.irfft(F_new, n=N)
        
        # 4. Project to {-1, 1}
        # The projection of a real number x onto {-1, 1} is sign(x).
        # However, sign(0) is 0, which is not in {-1, 1}. 
        # We can handle 0 by assigning a random sign or keeping the previous value, but sign(x) for x!=0 is robust.
        # To avoid all zeros (unlikely), we can use np.where(f_time >= 0, 1.0, -1.0).
        f_new = np.where(f_time >= 0, 1.0, -1.0)
        
        # Check if we improved
        val = _objective(f_new)
        if val < state['best_val']:
            state['best_val'] = val
            state['best_f'] = f_new.copy()
            
        f = f_new
        
    # Final check: maybe local search from the best spectral result?
    # The spectral method might get stuck in a local minimum of the "flat spectrum" objective which might not be the global min of the max autocorr objective, though they are closely related.
    # Given the time limit, we return the best found.
    
    return state['best_f']

# EVOLVE-BLOCK-END

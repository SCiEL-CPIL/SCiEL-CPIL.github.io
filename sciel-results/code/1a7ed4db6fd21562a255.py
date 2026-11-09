import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_fft(f):
    """Compute full autocorrelation via FFT. Returns array of length 2N-1."""
    N = len(f)
    # Zero-pad to avoid circular convolution
    size = 1
    while size < 2 * N - 1:
        size *= 2
    F = np.fft.rfft(f, n=size)
    ac = np.fft.irfft(F * np.conj(F), n=size)[:2*N-1]
    return ac

def _compute_objective(f):
    """Compute max|autocorr| / (sum f)^2. Return (objective, max_abs_ac, sum_f)."""
    ac = _autocorr_fft(f)
    max_abs_ac = np.max(np.abs(ac))
    sum_f = np.sum(f)
    if abs(sum_f) < 1e-10:
        return float('inf'), max_abs_ac, sum_f
    obj = max_abs_ac / (sum_f * sum_f)
    return obj, max_abs_ac, sum_f

def _spectral_init(N, seed=42):
    """Generate initial ±1 sequence from inverse FFT of phase-only mask."""
    rng = np.random.RandomState(seed)
    # Create phase mask with random phases
    size = 1
    while size < 2 * N - 1:
        size *= 2
    phases = rng.uniform(0, 2*np.pi, size//2 + 1)
    # Ensure Hermitian symmetry for real output
    mask = np.exp(1j * phases)
    # Make it conjugate symmetric
    full_mask = np.zeros(size, dtype=complex)
    full_mask[:size//2+1] = mask
    for i in range(1, size//2):
        full_mask[size - i] = np.conj(mask[i])
    
    sig = np.fft.irfft(full_mask, n=size)[:N]
    # Threshold to ±1
    f = np.sign(sig)
    f[f == 0] = 1.0
    return f.astype(np.float32)

def _greedy_peak_shave(f, max_iters=5000, deadline=None):
    """
    Greedy algorithm: iteratively flip the sign of the element most correlated
    with the current maximum autocorrelation peak.
    """
    N = len(f)
    best_f = f.copy()
    best_obj, _, _ = _compute_objective(f)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        ac = _autocorr_fft(f)
        max_abs_ac = np.max(np.abs(ac))
        sum_f = np.sum(f)
        if abs(sum_f) < 1e-10:
            break
        obj = max_abs_ac / (sum_f * sum_f)
        
        if obj < best_obj:
            best_obj = obj
            best_f = f.copy()
        
        # Find the lag with maximum |autocorr|
        lag_idx = np.argmax(np.abs(ac))
        # The autocorrelation at lag k is sum_i f[i] * f[i+k]
        # To reduce this peak, we want to flip elements that contribute most
        # to this specific autocorrelation term.
        
        # Compute gradient-like signal: for each position i, how much does
        # flipping f[i] change the autocorrelation at the peak lag?
        # ac[lag] = sum_{i} f[i] * f[i+lag] (with appropriate bounds)
        # If we flip f[j], it affects ac[lag] by -2*f[j]*f[j+lag] (if j+lag in range)
        # or -2*f[j]*f[j-lag] (if j-lag in range)
        
        # More precisely, for lag L:
        # ac[L] = sum_{i=0}^{N-1-L} f[i] * f[i+L]
        # Flipping f[j] changes ac[L] by:
        #   -2*f[j]*f[j+L] if 0 <= j+L < N  (from term i=j)
        #   -2*f[j]*f[j-L] if 0 <= j-L < N  (from term i=j-L)
        #   (plus symmetric contributions from negative lags if we consider full autocorr)
        
        # For simplicity, compute the "influence" of each position on the peak lag
        L = lag_idx
        influence = np.zeros(N)
        
        # Contribution from i=j: f[j]*f[j+L] for j in [0, N-1-L]
        if L < N:
            idx1 = np.arange(0, N - L)
            influence[idx1] -= 2.0 * f[idx1] * f[idx1 + L]
        
        # Contribution from i=j-L: f[j-L]*f[j] for j in [L, N-1]
        if L < N:
            idx2 = np.arange(L, N)
            influence[idx2] -= 2.0 * f[idx2 - L] * f[idx2]
        
        # Also consider the negative lag -L (same as L for real autocorr)
        # ac[-L] = ac[L] for real signals, so same influence
        
        # Find the element with maximum |influence| (flipping it changes ac most)
        # We want to flip the one that most reduces |ac[L]|
        # If ac[L] > 0, we want to decrease it: flip j where influence[j] < 0 (most negative)
        # If ac[L] < 0, we want to increase it: flip j where influence[j] > 0 (most positive)
        
        if ac[L] > 0:
            # Want to decrease ac[L], so flip where influence is most negative
            flip_idx = np.argmin(influence)
        else:
            # Want to increase ac[L], so flip where influence is most positive
            flip_idx = np.argmax(influence)
        
        # Only flip if it actually helps
        if abs(influence[flip_idx]) < 1e-10:
            break
        
        # Test the flip
        f[flip_idx] *= -1
        new_obj, _, _ = _compute_objective(f)
        
        if new_obj < obj:
            # Keep the flip
            continue
        else:
            # Revert
            f[flip_idx] *= -1
            break
    
    return best_f, best_obj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    N = 400  # Fixed length based on task context
    
    # Start with spectral initialization
    best_f = None
    best_obj = float('inf')
    
    # Try multiple random seeds for spectral init
    for seed in range(20):
        if time.time() > deadline:
            break
        f_init = _spectral_init(N, seed=seed)
        obj, _, _ = _compute_objective(f_init)
        if obj < best_obj:
            best_obj = obj
            best_f = f_init.copy()
    
    if best_f is None:
        best_f = np.ones(N, dtype=np.float32)
        best_obj, _, _ = _compute_objective(best_f)
    
    # Apply greedy peak shaving
    if time.time() < deadline - 5.0:
        f_shaved, obj_shaved = _greedy_peak_shave(best_f, max_iters=3000, deadline=deadline - 2.0)
        if obj_shaved < best_obj:
            best_f = f_shaved
            best_obj = obj_shaved
    
    return best_f
# EVOLVE-BLOCK-END

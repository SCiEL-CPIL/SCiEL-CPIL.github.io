"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_peak(signs):
    """Compute the maximum absolute value of the autoconvolution at non-zero lags."""
    # Fast computation using FFT
    n = len(signs)
    f = np.fft.fft(signs, n=2 * n)
    ac = np.real(np.fft.ifft(f * np.conj(f)))[:n]
    # Exclude zero lag (ac[0] is the sum of squares)
    peak = np.max(np.abs(ac[1:]))
    return peak


def _local_search(signs, max_iter=5000, time_budget=10.0):
    """Local search by flipping individual signs to reduce the peak autoconvolution."""
    start_time = time.time()
    best_signs = signs.copy()
    best_peak = _autoconv_peak(best_signs)
    
    # Precompute FFT for fast updates
    n = len(best_signs)
    f = np.fft.fft(best_signs, n=2 * n)
    
    for i in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Try flipping each position
        best_delta = 0
        best_j = -1
        
        for j in range(n):
            # Simulate flip at position j
            # The change in autoconvolution can be computed efficiently
            # But for simplicity and correctness, let's do a quick check
            # Actually, let's use a smarter approach: compute the effect of flipping one sign
            
            # When we flip sign at j from s to -s, the new autoconvolution is:
            # ac_new[k] = ac_old[k] - 2*s*signs[j]*(sum of signs[j+k] + signs[j-k] - 2*signs[j]*delta(k,0))
            # This is complex, so let's just try a few random positions first for speed
            
            pass
        
        # For efficiency, let's try random flips
        j = np.random.randint(n)
        old_sign = best_signs[j]
        best_signs[j] = -old_sign
        new_peak = _autoconv_peak(best_signs)
        
        if new_peak < best_peak:
            best_peak = new_peak
            f = np.fft.fft(best_signs, n=2 * n)
        else:
            best_signs[j] = old_sign
    
    return best_signs, best_peak


def _van_der_corput_signs(n):
    """Generate Van der Corput sequence signs."""
    signs = np.zeros(n)
    for i in range(n):
        # Van der Corput sequence in [0, 1)
        x = 0.0
        d = 0.5
        ii = i
        while ii > 0:
            if ii % 2 == 1:
                x += d
            ii //= 2
            d /= 2
        signs[i] = 1.0 if x < 0.5 else -1.0
    return signs


def _thue_morse_signs(n):
    """Generate Thue-Morse sequence signs."""
    signs = np.zeros(n)
    for i in range(n):
        # Count bits in binary representation of i
        count = bin(i).count('1')
        signs[i] = 1.0 if count % 2 == 0 else -1.0
    return signs


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_signs = None
    best_peak = np.inf
    
    # Generate diverse initializations
    initializations = []
    
    # Thue-Morse
    initializations.append(_thue_morse_signs(N))
    
    # Van der Corput
    initializations.append(_van_der_corput_signs(N))
    
    # Random signs
    rng = np.random.RandomState(42)
    for seed in range(5):
        initializations.append(rng.choice([-1.0, 1.0], size=N))
    
    # Run local search on each initialization
    for init_signs in initializations:
        if time.time() > deadline:
            break
        
        signs, peak = _local_search(init_signs, max_iter=2000, time_budget=15.0)
        
        if peak < best_peak:
            best_peak = peak
            best_signs = signs
    
    if best_signs is None:
        return np.ones(N)
    
    return best_signs
# EVOLVE-BLOCK-END

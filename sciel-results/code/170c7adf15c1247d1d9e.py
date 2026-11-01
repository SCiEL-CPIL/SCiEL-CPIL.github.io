"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorrelation(x):
    """Compute full autocorrelation of sequence x (length N)."""
    # Use FFT-based method for speed
    n = len(x)
    # Zero-pad to avoid circular correlation
    size = 1
    while size < 2 * n:
        size *= 2
    X = np.fft.fft(x, size)
    acf = np.fft.ifft(X * np.conj(X)).real
    return acf[:n]  # lags 0 to n-1

def _compute_c3(x):
    """Compute the C3 upper bound from the autocorrelation."""
    acf = _autocorrelation(x)
    # Sidelobes: lags 1 to N-1
    sidelobes = acf[1:]
    # The third autocorrelation inequality bound
    # Using max absolute sidelobe as the metric
    return np.max(np.abs(sidelobes))

def _spectral_init(rng):
    """Initialize sequence from inverse FFT of target low-peak magnitude spectrum."""
    # Target: flat magnitude spectrum (all equal)
    # This gives a sequence with flat autocorrelation (ideal)
    size = 1
    while size < 2 * N:
        size *= 2
    
    # Create target magnitude: flat
    target_mag = np.ones(size)
    # Zero out DC and Nyquist for symmetry
    target_mag[0] = 1.0
    target_mag[size // 2] = 1.0
    
    # Random phases
    phases = rng.uniform(0, 2 * np.pi, size)
    spectrum = target_mag * np.exp(1j * phases)
    
    # Make spectrum conjugate symmetric for real output
    spectrum[size // 2 + 1:] = np.conj(spectrum[1:size // 2][::-1])
    
    # Inverse FFT
    x = np.fft.ifft(spectrum).real
    
    # Threshold to ±1
    seq = np.sign(x[:N])
    # Ensure no zeros
    seq[seq == 0] = 1
    return seq.astype(np.float64)

def _local_search(x, rng, time_limit=10.0):
    """Local search: try flipping bits to reduce C3."""
    best_x = x.copy()
    best_c3 = _compute_c3(best_x)
    start_time = time.time()
    
    # Precompute autocorrelation
    best_acf = _autocorrelation(best_x)
    
    # Try all positions, keep best improvement
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        best_improve = 0.0
        best_pos = -1
        
        for pos in range(N):
            # Try flipping position pos
            x_trial = best_x.copy()
            x_trial[pos] *= -1.0
            
            # Compute new C3 efficiently
            c3_trial = _compute_c3(x_trial)
            
            if c3_trial < best_c3 - 1e-10:
                best_c3 = c3_trial
                best_pos = pos
                best_x = x_trial
                improved = True
                break  # Restart scan after improvement
        
        if not improved:
            # No single flip helps; try random flips
            for _ in range(100):
                pos = rng.integers(0, N)
                x_trial = best_x.copy()
                x_trial[pos] *= -1.0
                c3_trial = _compute_c3(x_trial)
                if c3_trial < best_c3 - 1e-10:
                    best_c3 = c3_trial
                    best_x = x_trial
                    improved = True
                    break
    
    return best_x, best_c3

def _simulated_annealing(x, rng, time_limit=60.0):
    """Simulated annealing to escape local minima."""
    best_x = x.copy()
    best_c3 = _compute_c3(best_x)
    current_x = x.copy()
    current_c3 = best_c3
    
    start_time = time.time()
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    while T > min_T and (time.time() - start_time) < time_limit:
        # Propose: flip random bit
        pos = rng.integers(0, N)
        x_trial = current_x.copy()
        x_trial[pos] *= -1.0
        
        c3_trial = _compute_c3(x_trial)
        delta = c3_trial - current_c3
        
        if delta < 0 or rng.uniform() < np.exp(-delta / T):
            current_x = x_trial
            current_c3 = c3_trial
            
            if current_c3 < best_c3 - 1e-10:
                best_c3 = current_c3
                best_x = current_x.copy()
        
        T *= cooling_rate
    
    return best_x, best_c3

def construct():
    rng = np.random.default_rng(42)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s
    
    # Phase 1: Spectral initialization
    x = _spectral_init(rng)
    c3 = _compute_c3(x)
    
    # Phase 2: Local search to polish
    if time.time() < deadline - 30:
        x, c3 = _local_search(x, rng, time_limit=min(20.0, deadline - time.time() - 30))
    
    # Phase 3: Simulated annealing for global improvement
    if time.time() < deadline - 10:
        x, c3 = _simulated_annealing(x, rng, time_limit=min(50.0, deadline - time.time() - 10))
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 5:
        x, c3 = _local_search(x, rng, time_limit=min(5.0, deadline - time.time() - 5))
    
    # Ensure we return a valid ±1 sequence
    x = np.sign(x)
    x[x == 0] = 1
    
    return x
# EVOLVE-BLOCK-END

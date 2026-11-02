"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _compute_max_abs_autocorr(s):
    """
    Compute the maximum absolute value of the linear autocorrelation
    for lags 1 to N-1 for a sequence s of length N.
    """
    n = len(s)
    # Next power of 2 for FFT efficiency
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    
    # FFT of the sequence
    S = np.fft.fft(s, n=n_fft)
    # Power spectrum
    power = S * np.conj(S)
    # Inverse FFT gives circular autocorrelation. 
    # Since we zero-padded to 2N, the first N values are the linear autocorrelation.
    acf = np.fft.ifft(power).real[:n]
    
    # We want max abs of acf[1:] (lags 1 to N-1)
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:])))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Pre-compute FFT size
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    
    def fast_score(s):
        """Optimized scoring function using pre-allocated logic if possible, 
        but np.fft is already fast. We just ensure we use the right size."""
        S = np.fft.fft(s, n=n_fft)
        power = S * np.conj(S)
        acf = np.fft.ifft(power).real[:N]
        return float(np.max(np.abs(acf[1:])))

    # 1. Generate diverse initializations
    initializations = []
    
    # All ones (Seed)
    initializations.append(np.ones(N, dtype=np.float64))
    
    # Alternating
    alt = np.ones(N, dtype=np.float64)
    alt[1::2] = -1.0
    initializations.append(alt)
    
    # Blocks of various sizes
    for block_size in [2, 3, 4, 5, 7, 8, 10, 16, 20]:
        init = np.ones(N, dtype=np.float64)
        for i in range(N):
            if (i // block_size) % 2 == 1:
                init[i] = -1.0
        initializations.append(init)
        
    # Random initializations
    rng = np.random.default_rng(42)
    for _ in range(20):
        initializations.append(rng.choice([-1.0, 1.0], size=N))
        
    # 2. Evaluate initializations and find the best starting point
    best_s = None
    best_score = float('inf')
    
    for init in initializations:
        score = fast_score(init)
        if score < best_score:
            best_score = score
            best_s = init.copy()
            
    if best_s is None:
        return np.ones(N)

    # 3. Simulated Annealing / Local Search
    # We will perform a search that tries to flip bits to reduce the max autocorr.
    # To handle the non-smooth max, we use SA.
    
    current_s = best_s.copy()
    current_score = best_score
    
    T_start = 10.0  # Initial temperature (scale of score differences)
    T_min = 0.01
    cooling = 0.999
    
    # We iterate until time limit
    # Each iteration: pick a random index, flip it, compute new score.
    # If better, accept. If worse, accept with prob exp(-delta/T).
    
    # To speed up, we can try multiple flips per iteration or just single flips.
    # Single flips are safer for convergence.
    
    # Estimate iterations possible: 
    # FFT of 1024 floats is ~1-2 microseconds? No, likely 10-50us in Python/NumPy.
    # Let's assume 1000 evaluations per second is conservative.
    # 145 seconds * 1000 = 145,000 iterations.
    
    it = 0
    T = T_start
    
    while time.time() < deadline:
        # Check time more frequently in tight loops
        if time.time() > deadline - 1.0:
            break
            
        # Pick a random index to flip
        idx = rng.integers(0, N)
        
        # Flip
        old_val = current_s[idx]
        current_s[idx] = -old_val
        
        # Compute new score
        new_score = fast_score(current_s)
        
        # Decision
        delta = new_score - current_score
        if delta < 0:
            # Improvement
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_s = current_s.copy()
        else:
            # Worse or equal
            if delta == 0:
                # Keep it, no harm
                current_score = new_score
            else:
                # Accept with probability
                if rng.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    # Revert
                    current_s[idx] = old_val
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            
        it += 1
        
        # Optional: Periodic restarts or restart if stuck?
        # For now, let it run. The cooling schedule should help escape.
        
    return best_s
# EVOLVE-BLOCK-END

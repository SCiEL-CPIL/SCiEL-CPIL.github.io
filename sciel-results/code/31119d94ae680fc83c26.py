"""Rewritten program for third autocorrelation inequality optimization."""
import numpy as np
import time

N = 400
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds before deadline to stop

# Precompute indices for fast autocorrelation computation
# For a step function with N intervals, autocorrelation at lag k is:
# sum_{i=0}^{N-1-k} h[i] * h[i+k]  (for k >= 0)
# The "upper bound on C3" relates to the maximum sidelobe of the autocorrelation.

def _autocorr_sidelobe_max(h):
    """
    Compute the maximum absolute value of the autocorrelation sidelobes.
    h: array of signed step heights, length N.
    Returns the max |R(k)| for k=1..N-1, where R(k) = sum h[i]*h[i+k].
    """
    # Use FFT for fast computation
    n_fft = 1 << (2 * N - 1).bit_length()
    f = np.zeros(n_fft, dtype=np.float64)
    f[:N] = h
    F = np.fft.rfft(f)
    # Autocorrelation via inverse FFT of |F|^2
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    # acf[0] is the energy, acf[1:] are the sidelobes
    return np.max(np.abs(acf[1:]))


def construct():
    """
    Construct a signed step function with N intervals to minimize the
    upper bound on C3 (maximum autocorrelation sidelobe).
    
    Uses simulated annealing with block moves to escape local minima.
    Returns an array of signed step heights (values in {-1, 1} or similar).
    """
    start_time = time.time()
    
    # Initialize with random signs
    np.random.seed(42)
    h = np.random.choice([-1.0, 1.0], size=N)
    
    best_h = h.copy()
    best_val = _autocorr_sidelobe_max(h)
    
    # Simulated annealing parameters
    T_start = 1.0
    T_end = 0.001
    n_iters = 200000  # will be limited by deadline
    block_sizes = [1, 2, 4, 8, 16]  # different block move sizes
    
    # Precompute some helper data
    # We'll use a simple cooling schedule
    
    current_val = best_val
    current_h = h.copy()
    
    iter_count = 0
    while iter_count < n_iters:
        # Check deadline
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Cooling schedule: geometric
        progress = iter_count / n_iters
        T = T_start * (T_end / T_start) ** progress
        
        # Choose a random block move
        block_size = np.random.choice(block_sizes, p=[0.3, 0.25, 0.2, 0.15, 0.1])
        start_idx = np.random.randint(0, N - block_size + 1)
        
        # Create a perturbed copy
        h_new = current_h.copy()
        # Flip signs in the block
        h_new[start_idx:start_idx + block_size] *= -1.0
        
        # Evaluate
        new_val = _autocorr_sidelobe_max(h_new)
        
        # Acceptance criterion
        delta = new_val - current_val
        if delta < 0 or (T > 0 and np.random.random() < np.exp(-delta / T)):
            current_h = h_new
            current_val = new_val
            
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
        
        iter_count += 1
        
        # Batch check every 1000 iterations for speed
        if iter_count % 1000 == 0 and time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
    
    return best_h


# For testing/evaluation, we can also try a few different initializations
def _try_multiple_seeds():
    """Try multiple random initializations and return the best."""
    start_time = time.time()
    best_h = None
    best_val = np.inf
    
    for seed in range(50):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        np.random.seed(seed)
        h = np.random.choice([-1.0, 1.0], size=N)
        
        # Quick local search with greedy improvement
        improved = True
        while improved and time.time() - start_time < DEADLINE - SAFETY_MARGIN:
            improved = False
            current_val = _autocorr_sidelobe_max(h)
            
            # Try flipping individual bits
            for i in range(0, N, 4):  # sample positions for speed
                if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                    break
                h_old = h[i]
                h[i] = -h[i]
                new_val = _autocorr_sidelobe_max(h)
                if new_val < current_val:
                    current_val = new_val
                    improved = True
                else:
                    h[i] = h_old
        
        val = _autocorr_sidelobe_max(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    return best_h if best_h is not None else np.ones(N)


def construct():
    """
    Main entry point. Uses simulated annealing to find a good signed step function.
    Returns an array of N signed step heights.
    """
    start_time = time.time()
    
    # Try a combination of approaches within the time budget
    
    # First, try multiple random initializations with local search
    best_h = None
    best_val = np.inf
    
    # Phase 1: Multiple random starts with greedy local search
    n_starts = 20
    for seed in range(n_starts):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        np.random.seed(seed + 100)
        h = np.random.choice([-1.0, 1.0], size=N)
        
        # Greedy local search: flip bits that improve the objective
        current_val = _autocorr_sidelobe_max(h)
        improved = True
        while improved and time.time() - start_time < DEADLINE - SAFETY_MARGIN:
            improved = False
            # Try flipping each bit
            for i in range(N):
                if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                    break
                old_val = h[i]
                h[i] = -old_val
                new_val = _autocorr_sidelobe_max(h)
                if new_val < current_val - 1e-12:
                    current_val = new_val
                    improved = True
                else:
                    h[i] = old_val
        
        val = _autocorr_sidelobe_max(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    if best_h is None:
        return np.ones(N)
    
    # Phase 2: Simulated annealing starting from the best solution
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        current_h = best_h.copy()
        current_val = best_val
        T_start = 0.5
        T_end = 0.0005
        n_sa_iters = 100000
        block_sizes = [1, 2, 4, 8]
        
        for iter_count in range(n_sa_iters):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            progress = iter_count / n_sa_iters
            T = T_start * (T_end / T_start) ** progress
            
            block_size = np.random.choice(block_sizes, p=[0.35, 0.3, 0.2, 0.15])
            start_idx = np.random.randint(0, N - block_size + 1)
            
            h_new = current_h.copy()
            h_new[start_idx:start_idx + block_size] *= -1.0
            
            new_val = _autocorr_sidelobe_max(h_new)
            delta = new_val - current_val
            
            if delta < 0 or (T > 0 and np.random.random() < np.exp(-delta / T)):
                current_h = h_new
                current_val = new_val
                
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
    
    return best_h

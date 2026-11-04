"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_peak(h):
    """Compute max absolute autocorrelation at non-zero lags."""
    n = len(h)
    if n <= 1:
        return 0.0
    # Use FFT for efficient autocorrelation
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # acf[k] for k in 1..n-1 gives positive lags
    # For real signals, acf is symmetric, so max over 1..n-1 suffices
    return float(np.max(np.abs(acf[1:n])))

def _score(h):
    """Upper bound on C3: max absolute autocorrelation at non-zero lags."""
    return _autocorr_peak(h)

def _fast_greedy(h, time_limit=5.0):
    """
    Fast greedy local search.
    1. Flip signs of individual bits.
    2. Zero out bits.
    3. Activate zero bits with +/- 1.
    Uses random ordering to escape local minima.
    """
    n = len(h)
    best_h = h.copy()
    best_score = _score(best_h)
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        
        # Identify active indices
        nonzero_idx = np.nonzero(best_h)[0]
        if len(nonzero_idx) == 0:
            break
            
        # 1. Try flipping signs
        order = np.random.permutation(len(nonzero_idx))
        for idx_pos in order:
            i = nonzero_idx[idx_pos]
            trial = best_h.copy()
            trial[i] = -trial[i]
            s = _score(trial)
            if s < best_score - 1e-12:
                best_h = trial
                best_score = s
                improved = True
                break
        
        # 2. Try zeroing out
        if not improved:
            nonzero_idx = np.nonzero(best_h)[0]
            if len(nonzero_idx) > 0:
                order = np.random.permutation(len(nonzero_idx))
                for idx_pos in order:
                    i = nonzero_idx[idx_pos]
                    trial = best_h.copy()
                    trial[i] = 0.0
                    s = _score(trial)
                    if s < best_score - 1e-12:
                        best_h = trial
                        best_score = s
                        improved = True
                        break
        
        # 3. Try activating zeros
        if not improved:
            zero_idx = np.nonzero(best_h == 0)[0]
            if len(zero_idx) > 0:
                # Limit activations per iteration to keep it fast
                k = min(20, len(zero_idx))
                order = np.random.choice(len(zero_idx), size=k, replace=False)
                for idx_pos in order:
                    i = zero_idx[idx_pos]
                    for sign in [1.0, -1.0]:
                        trial = best_h.copy()
                        trial[i] = sign
                        s = _score(trial)
                        if s < best_score - 1e-12:
                            best_h = trial
                            best_score = s
                            improved = True
                            break
                    if improved:
                        break
        
        if not improved:
            break
            
    return best_h, best_score

def _simulated_annealing(h, time_limit=10.0):
    """
    Simulated annealing to escape local minima.
    """
    n = len(h)
    current_h = h.copy()
    current_score = _score(current_h)
    best_h = current_h.copy()
    best_score = current_score
    
    start = time.time()
    T = 5.0
    cooling = 0.995
    min_T = 0.1
    
    while T > min_T and time.time() - start < time_limit:
        # Pick a random move
        i = np.random.randint(n)
        trial = current_h.copy()
        
        # Randomly choose between flip, zero, or activate
        r = np.random.random()
        if r < 0.6:
            # Flip sign (or activate if 0)
            if trial[i] == 0:
                trial[i] = 1.0 if np.random.random() < 0.5 else -1.0
            else:
                trial[i] = -trial[i]
        elif r < 0.8:
            # Zero out
            trial[i] = 0.0
        else:
            # Activate (if zero) or flip (if non-zero)
            if trial[i] == 0:
                trial[i] = 1.0 if np.random.random() < 0.5 else -1.0
            else:
                trial[i] = -trial[i]
        
        new_score = _score(trial)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_h = trial
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = current_h.copy()
        
        T *= cooling
        
    return best_h, best_score

def _quasiperiodic_init(frequencies, phases, amplitudes, threshold=0.0):
    """
    Generate a signed step function based on a sum of complex exponentials.
    h[i] = sign( sum_j A_j * cos(2*pi*f_j*i + phi_j) )
    If the sum is below threshold, set to 0.
    """
    n = N
    t = np.arange(n)
    signal = np.zeros(n)
    
    for f, p, a in zip(frequencies, phases, amplitudes):
        signal += a * np.cos(2 * np.pi * f * t + p)
    
    h = np.zeros(n)
    h[signal > threshold] = 1.0
    h[signal < -threshold] = -1.0
    # Values between -threshold and threshold are 0
    
    return h

def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound.
    
    Strategy:
    1. Initialize with a structured sparse pattern (e.g., based on prime gaps or random sparse sets).
    2. Use fast greedy local search to refine.
    3. Use simulated annealing to escape local minima.
    4. Return the best solution found within the time budget.
    """
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # 1. Quasi-periodic Initialization (Structural Change)
    # Use irrational frequencies to avoid periodicity and create low-autocorrelation patterns.
    # sqrt(2), sqrt(3), sqrt(5) are irrational.
    freq_sets = [
        [1/np.sqrt(2), 1/np.sqrt(3), 1/np.sqrt(5)],
        [1/np.sqrt(2), 1/np.sqrt(7), 1/np.sqrt(11)],
        [1/np.sqrt(3), 1/np.sqrt(5), 1/np.sqrt(13)],
    ]
    
    for freqs in freq_sets:
        if time.time() > deadline - 10:
            break
        
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            
            # Random phases and amplitudes
            phases = np.random.uniform(0, 2*np.pi, size=len(freqs))
            amplitudes = np.random.uniform(0.5, 1.5, size=len(freqs))
            
            # Vary threshold to control sparsity
            # Lower threshold -> denser signal (more +/- 1)
            # Higher threshold -> sparser signal (more 0)
            thresholds = [0.0, 0.2, 0.5]
            
            for thresh in thresholds:
                if time.time() > deadline - 10:
                    break
                
                h = _quasiperiodic_init(freqs, phases, amplitudes, thresh)
                
                # Refine with local search
                h_opt, score = _fast_greedy(h, time_limit=3.0)
                
                # Refine with simulated annealing if time permits
                if time.time() < deadline - 5:
                    h_opt, score = _simulated_annealing(h_opt, time_limit=5.0)
                
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
    
    # 2. Fallback: Sparse Random Initialization (from previous best approach)
    # If quasi-periodic didn't yield a good result, try sparse random.
    # We only do this if we haven't found something very good yet or if time remains.
    if best_h is None or best_score > 10.0:
        for k in [10, 20, 30, 40, 50, 60, 80, 100, 150]:
            if time.time() > deadline - 10:
                break
            
            for trial in range(3):
                if time.time() > deadline - 10:
                    break
                
                indices = np.random.choice(N, size=k, replace=False)
                h = np.zeros(N)
                h[indices] = 1.0
                
                h_opt, score = _fast_greedy(h, time_limit=2.0)
                
                if time.time() < deadline - 5:
                    h_opt, score = _simulated_annealing(h_opt, time_limit=3.0)
                
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
    
    # 3. Final Polish
    if best_h is not None and time.time() < deadline - 5:
        best_h, best_score = _fast_greedy(best_h, time_limit=5.0)
        if time.time() < deadline - 2:
            best_h, best_score = _simulated_annealing(best_h, time_limit=5.0)
    
    # Fallback: if no solution found
    if best_h is None:
        best_h = np.ones(N)
        best_score = _score(best_h)
    
    return best_h
# EVOLVE-BLOCK-END

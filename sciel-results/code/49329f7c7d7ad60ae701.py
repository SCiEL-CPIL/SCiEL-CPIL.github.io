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

def _thue_morse_sequence(n):
    """
    Generate the Thue-Morse sequence of length n.
    The Thue-Morse sequence is defined by t(0)=0, t(n) = t(floor(n/2)) XOR (n mod 2).
    Equivalently, t(n) is the parity of the number of 1s in the binary representation of n.
    We map 0 -> +1, 1 -> -1 for the signed step function.
    """
    t = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # Count set bits in i
        count = 0
        x = i
        while x:
            count += x & 1
            x >>= 1
        # Map even parity to +1, odd parity to -1
        if count % 2 == 0:
            t[i] = 1.0
        else:
            t[i] = -1.0
    return t

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

def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound.
    
    Strategy:
    1. Initialize with Thue-Morse sequence (deterministic, low complexity, flat autocorrelation).
    2. Also try random full sequences.
    3. Use fast greedy local search to refine.
    4. Use simulated annealing to escape local minima.
    5. Return the best solution found within the time budget.
    """
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Thue-Morse initialization (deterministic)
    h_tm = _thue_morse_sequence(N)
    h_opt, score = _fast_greedy(h_tm, time_limit=5.0)
    if time.time() < deadline - 5:
        h_opt, score = _simulated_annealing(h_opt, time_limit=10.0)
    
    if score < best_score:
        best_score = score
        best_h = h_opt.copy()
    
    # Phase 2: Random full sequences
    for trial in range(5):
        if time.time() > deadline - 10:
            break
        h_rand = np.random.choice([-1, 1], size=N).astype(np.float64)
        h_opt, score = _fast_greedy(h_rand, time_limit=3.0)
        if time.time() < deadline - 5:
            h_opt, score = _simulated_annealing(h_opt, time_limit=5.0)
        
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Phase 3: Sparse sequences
    for k in [50, 100, 150]:
        if time.time() > deadline - 10:
            break
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            indices = np.random.choice(N, size=k, replace=False)
            h = np.zeros(N)
            h[indices] = 1.0
            
            h_opt, score = _fast_greedy(h, time_limit=3.0)
            if time.time() < deadline - 5:
                h_opt, score = _simulated_annealing(h_opt, time_limit=5.0)
            
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
    
    if best_h is None:
        return _thue_morse_sequence(N)
    
    return best_h
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

# Precompute FFT parameters for speed
_fft_size = 1
while _fft_size < 2 * N:
    _fft_size *= 2

def _autocorr_peak(h):
    """Compute max absolute autocorrelation at non-zero lags."""
    n = len(h)
    if n <= 1:
        return 0.0
    H = np.fft.rfft(h, n=_fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=_fft_size)
    # acf[1:n] contains the positive lags
    # We take the max absolute value
    return float(np.max(np.abs(acf[1:n])))

def _score(h):
    """Upper bound on C3: max absolute autocorrelation at non-zero lags."""
    return _autocorr_peak(h)

def _init_thue_morse(n):
    """Generate Thue-Morse sequence mapped to +/- 1."""
    t = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # Count set bits
        count = 0
        x = i
        while x:
            count += x & 1
            x >>= 1
        t[i] = 1.0 if count % 2 == 0 else -1.0
    return t

def _init_chirp(n):
    """Generate a chirp-like sequence mapped to +/- 1."""
    i = np.arange(n)
    # Quadratic phase
    phase = 2 * np.pi * (i ** 2) / n
    t = np.sign(np.cos(phase))
    # Ensure no zeros (sign(0) is 0)
    t[t == 0] = 1.0
    return t

def _init_random(n):
    """Generate random +/- 1 sequence."""
    return np.random.choice([-1.0, 1.0], size=n)

def _simulated_annealing_block(h_init, time_limit):
    """
    Simulated Annealing with block flips.
    Structural change: Uses block reversals/flips to escape local minima
    that single-bit flips cannot.
    """
    n = len(h_init)
    current_h = h_init.copy()
    current_score = _score(current_h)
    best_h = current_h.copy()
    best_score = current_score
    
    start_time = time.time()
    
    # Parameters
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.999
    
    # We will run until time limit or T < T_min
    while time.time() - start_time < time_limit and T > T_min:
        # Generate a move
        # 50% chance: flip single bit
        # 50% chance: flip a random block of length 1 to 15
        if random.random() < 0.5:
            i = random.randint(0, n - 1)
            trial_h = current_h.copy()
            trial_h[i] = -trial_h[i]
        else:
            i = random.randint(0, n - 1)
            block_len = random.randint(1, 15)
            # Handle wrap-around or just clamp? Clamping is safer for step functions
            # But for autocorrelation, position matters.
            # Let's just pick a start and end within bounds
            end = min(n, i + block_len)
            trial_h = current_h.copy()
            trial_h[i:end] = -trial_h[i:end]
        
        new_score = _score(trial_h)
        delta = new_score - current_score
        
        # Metropolis criterion
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_h = trial_h
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = current_h.copy()
        
        T *= cooling_rate
        
    return best_h, best_score

def _greedy_polish(h, time_limit):
    """
    Fast greedy local search to polish the result from SA.
    """
    n = len(h)
    current_h = h.copy()
    current_score = _score(current_h)
    best_h = current_h.copy()
    best_score = current_score
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try flipping each bit in random order
        indices = np.random.permutation(n)
        for i in indices:
            trial_h = current_h.copy()
            trial_h[i] = -trial_h[i]
            new_score = _score(trial_h)
            if new_score < current_score - 1e-12:
                current_h = trial_h
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_h = current_h.copy()
                improved = True
                break # Restart permutation to find more improvements
        
        if not improved:
            break
            
    return best_h, best_score

def construct():
    """
    Construct a signed step function with N=400 intervals.
    Strategy:
    1. Multiple diverse initializations (Thue-Morse, Chirp, Random).
    2. Structural Optimization: Simulated Annealing with Block Flips.
    3. Final Polish: Greedy Local Search.
    4. Return the best solution found within the time budget.
    """
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # List of initializers
    initializers = [
        _init_thue_morse,
        _init_chirp,
        _init_random,
        _init_random,
        _init_random
    ]
    
    # Time allocation:
    # ~10s per SA run.
    # ~5s per Greedy polish.
    # Total ~15s per restart.
    # 140s / 15s ~ 9 restarts.
    
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        # Select initializer
        init_func = initializers[restarts % len(initializers)]
        h_init = init_func(N)
        
        # Phase 1: Simulated Annealing with Block Flips
        # Allocate time based on remaining budget
        time_remaining = deadline - time.time()
        sa_time = min(10.0, max(2.0, time_remaining * 0.6))
        
        h_sa, score_sa = _simulated_annealing_block(h_init, sa_time)
        
        # Phase 2: Greedy Polish
        # Allocate remaining time for this restart
        time_remaining = deadline - time.time()
        greedy_time = min(5.0, max(1.0, time_remaining * 0.3))
        
        h_final, score_final = _greedy_polish(h_sa, greedy_time)
        
        if score_final < best_score:
            best_score = score_final
            best_h = h_final.copy()
            
    # Fallback
    if best_h is None:
        best_h = _init_thue_morse(N)
        best_score = _score(best_h)
        
    return best_h
# EVOLVE-BLOCK-END

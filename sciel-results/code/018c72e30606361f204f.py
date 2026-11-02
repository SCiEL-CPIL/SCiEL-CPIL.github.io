"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorrelation_max_abs(s):
    """Compute max absolute off-peak autocorrelation for a +/-1 sequence."""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    
    s_float = s.astype(np.float64)
    f = np.fft.fft(s_float, n=n_fft)
    acf = np.fft.ifft(f * np.conj(f)).real[:N]
    acf /= N  # normalize
    
    # Off-peak: exclude lag 0
    off_peak = np.abs(acf[1:])
    if len(off_peak) == 0:
        return 0.0
    return float(np.max(off_peak))

def _score_sequence(s):
    """The score is the upper bound on C3 - we want to minimize this."""
    return _autocorrelation_max_abs(s)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    # Start with all ones (baseline)
    best_s = np.ones(N)
    best_score = _score_sequence(best_s)
    
    # Try some structured patterns
    # Alternating
    alt = np.ones(N)
    alt[1::2] = -1
    score_alt = _score_sequence(alt)
    if score_alt < best_score:
        best_s = alt.copy()
        best_score = score_alt
    
    # Check time
    if time.time() > deadline:
        return best_s
    
    # Try random initializations with local search
    rng = np.random.default_rng(42)
    
    for _ in range(50):  # Up to 50 random restarts
        if time.time() > deadline:
            break
        
        # Random initialization
        s = rng.choice([-1, 1], size=N)
        score = _score_sequence(s)
        
        if score < best_score:
            best_s = s.copy()
            best_score = score
        
        # Local search: flip individual bits
        improved = True
        while improved:
            if time.time() > deadline:
                return best_s
            
            improved = False
            # Try flipping each bit
            for i in range(N):
                s_old = s[i]
                s[i] = -s_old
                new_score = _score_sequence(s)
                if new_score < score:
                    score = new_score
                    improved = True
                    if score < best_score:
                        best_s = s.copy()
                        best_score = score
                else:
                    s[i] = s_old
    
    return best_s
# EVOLVE-BLOCK-END

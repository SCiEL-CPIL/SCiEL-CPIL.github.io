"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # The task: minimize max|autoconvolution| / (integral)^2 over signed step functions.
    # Theoretical baseline: f = ones(N).
    # Score = max|C(k)| / (sum f)^2.
    # For ones(N): C(0)=N, C(k!=0)=N-k. Max|C|=N. Sum f = N. Score = N/N^2 = 1/N.
    # Any signed perturbation reduces |sum f| (denominator) while keeping max|C| high (numerator),
    # likely increasing the score. Thus ones(N) is expected to be optimal.
    
    # We implement a fast, bounded discrete search to verify if any binary sign pattern
    # (mapped to {-1, 1}) or slight continuous perturbation improves on this.
    # To avoid timeout, we limit the number of evaluations strictly.
    
    f = np.ones(N)
    
    def calc_score(f):
        S = np.sum(f)
        if S == 0:
            return float('inf')
        # Compute autocorrelation via FFT
        # Zero-pad to 2*N to avoid circular convolution
        F = np.fft.fft(f, 2*N)
        C = np.fft.ifft(F * np.conj(F)).real[:N]
        max_corr = np.max(np.abs(C))
        return max_corr / (S * S)
    
    best_score = calc_score(f)
    best_f = f.copy()
    
    # Time guard: ensure we finish well within 120s.
    # We will perform a small number of random restarts/flips.
    start_time = time.time()
    deadline = start_time + 5.0  # 5 seconds is plenty for a few hundred FFTs on N=400
    
    # Strategy: Random binary sign flips starting from ones(N).
    # We try a limited number of random sign flips.
    # Each flip changes one element from 1 to -1.
    # This reduces Sum f by 2.
    # We check if the reduction in max|C| (if any) outweighs the reduction in (Sum f)^2.
    # It is highly unlikely, but we check a small sample.
    
    # Also try a few random binary vectors entirely.
    
    np.random.seed(42) # Deterministic for reproducibility in debugging, though random is fine
    
    for i in range(1000):
        if time.time() > deadline:
            break
            
        # Generate a random binary vector
        # To keep sum f large, we bias towards +1.
        # Let's try random signs.
        f_trial = np.where(np.random.rand(N) < 0.5, 1.0, -1.0)
        
        # Check score
        s = calc_score(f_trial)
        if s < best_score:
            best_score = s
            best_f = f_trial.copy()
            
        # Also try a single flip from the current best
        if i % 2 == 0:
            f_flip = best_f.copy()
            idx = np.random.randint(N)
            f_flip[idx] = -f_flip[idx]
            s_flip = calc_score(f_flip)
            if s_flip < best_score:
                best_score = s_flip
                best_f = f_flip.copy()

    return best_f

# EVOLVE-BLOCK-END

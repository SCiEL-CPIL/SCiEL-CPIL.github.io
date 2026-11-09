import numpy as np
import time

N = 400

def compute_score(x):
    """
    Compute the upper bound metric: peak normalized autocorrelation.
    x is a numpy array of shape (N,) with values in {-1, 1}.
    """
    # Compute autocorrelation via FFT for speed
    # Zero-pad to avoid circular convolution artifacts
    f = np.fft.fft(x, n=2 * N)
    acf = np.fft.ifft(f * np.conjugate(f)).real[:N]
    
    # Normalize by R[0] which is sum(x^2) = N for binary +/-1
    acf_norm = acf / acf[0]
    
    # We are interested in the peak off-diagonal autocorrelation
    # Exclude the lag 0 component
    peak = np.max(np.abs(acf_norm[1:]))
    return peak

def thue_morse_sequence(n):
    """
    Generate the first n terms of the Thue-Morse sequence mapped to +/- 1.
    t_k = parity of sum of bits of k.
    """
    # Efficient generation using recurrence or bit counting
    # t_k = t_{k//2} XOR (k%2)
    # Or simply: popcount(k) % 2
    # For n=400, simple loop is fine.
    seq = np.zeros(n, dtype=np.int8)
    for k in range(n):
        # Count bits
        # Python's bin(k).count('1') is slow for large n, but 400 is small.
        # Better: use np.bitwise_count if available (numpy 2.0+?) or a lookup table.
        # Let's use a simple recursive/iterative property or just bin count for 400 items.
        seq[k] = (-1) ** (bin(k).count('1') % 2)
    return seq

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # Safety margin for 120s limit
    
    # 1. Generate Initial Candidates
    candidates = []
    
    # Candidate 1: Thue-Morse
    tm = thue_morse_sequence(N)
    candidates.append(tm)
    
    # Candidate 2: Random (Fixed Seed)
    rng = np.random.default_rng(42)
    rand_seq = rng.choice([-1, 1], size=N)
    candidates.append(rand_seq)
    
    # Candidate 3: Alternating blocks (e.g., +++++-----+++++-----...)
    # This might have high correlation at specific lags, but let's check.
    # Actually, simple periodic patterns usually have high peaks.
    # Let's try a "randomized" Hadamard-like structure if possible.
    # For N=400, we can't use a perfect Hadamard matrix (order 400 is not a power of 2, 
    # and 400 is not 4k+1 or 4k+9 etc necessarily in a simple way).
    # But we can use a segment of a larger Hadamard matrix.
    # Let's stick to Thue-Morse and Random for now, and add a simple local search.
    
    # Evaluate candidates
    best_seq = None
    best_score = float('inf')
    
    for seq in candidates:
        s = compute_score(seq)
        if s < best_score:
            best_score = s
            best_seq = seq.copy()
            
    # 2. Local Search (Simulated Annealing)
    # We will try to improve best_seq by flipping bits.
    # Cost function: peak normalized autocorrelation.
    # This is a non-smooth function, so SA is appropriate.
    
    current_seq = best_seq
    current_score = best_score
    best_seq_sa = current_seq.copy()
    best_score_sa = current_score
    
    T = 0.1 # Initial temperature
    T_min = 0.001
    alpha = 0.995 # Cooling rate
    
    # Estimate max steps based on time
    # Each step involves a flip and score calculation.
    # Score calculation is O(N log N) due to FFT. N=400.
    # Let's benchmark one step roughly.
    # If 1000 steps take 1s, we can do ~100k steps in 100s.
    # Let's run until time limit.
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick a random index to flip
        idx = np.random.randint(N)
        
        # Flip
        old_val = current_seq[idx]
        current_seq[idx] = -old_val
        
        new_score = compute_score(current_seq)
        
        delta = new_score - current_score
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score_sa:
                best_score_sa = new_score
                best_seq_sa = current_seq.copy()
        else:
            # Revert
            current_seq[idx] = old_val
            
        T *= alpha
        
    # Return the best sequence found
    # The task asks for a sequence of signed step heights.
    # The seed returned np.ones(N). The evaluator expects a sequence.
    # The score is the upper bound C3. Lower is better.
    # We return the sequence that minimizes the score.
    
    return best_seq_sa

# EVOLVE-BLOCK-END

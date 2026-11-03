"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import random
import math

def construct():
    """
    Construct a signed step function with N equal intervals on [-1/4, 1/4].
    Uses a hybrid of low-discrepancy (Sobol-like) and chaotic initializations,
    followed by an aggressive Simulated Annealing with pair-flip moves.
    """
    
    def compute_c3_bound(s):
        """
        Compute the upper bound on C3 (max absolute value of 3rd order autocorrelation).
        """
        # Circular convolution of 3 sequences.
        # Zero padding to avoid wrap-around artifacts for linear convolution interpretation?
        # The task says "autoconvolution". Usually implies linear or circular?
        # "Third autocorrelation inequality" in papers often refers to circular or linear.
        # The seed code uses zero-padding, implying linear convolution.
        # We stick to linear convolution via zero-padding.
        
        # Max length of linear convolution of three N-length sequences is 3N-2.
        # Next power of 2 >= 3N-2.
        N_pad = 1 << (3 * N - 1).bit_length()
        s_pad = np.zeros(N_pad)
        s_pad[:N] = s
        S = np.fft.fft(s_pad)
        C3 = np.fft.ifft(S ** 3).real
        return np.max(np.abs(C3))

    # Generate diverse initial candidates
    candidates = []
    
    # 1. Chaotic Logistic Map seeds (as in original, but more diverse)
    seeds = [
        np.sqrt(2) % 1.0,
        np.sqrt(3) % 1.0,
        np.pi % 1.0,
        (1 + np.sqrt(5)) / 2 % 1.0,
        np.e % 1.0,
        0.123456789,
        0.987654321,
        0.5 + (1/np.sqrt(2)) % 1.0,
        0.333333333,
        0.666666666
    ]
    
    for seed in seeds:
        x = seed
        signs = np.zeros(N)
        for i in range(N):
            x = 4.0 * x * (1.0 - x)
            signs[i] = 1.0 if x > 0.5 else -1.0
        candidates.append(signs)

    # 2. Low-discrepancy-like initialization using Halton sequences or simple irrational rotations
    # Using irrational rotation on circle to determine sign
    irrationals = [
        1/np.sqrt(2),
        1/np.sqrt(3),
        1/np.sqrt(5),
        (np.pi - 3)/2,
        (np.e - 2)/2
    ]
    for alpha in irrationals:
        signs = np.zeros(N)
        for i in range(N):
            val = (i * alpha) % 1.0
            # Map to sign. Using a threshold that might balance +/-
            # To ensure low discrepancy, we can use the fractional part directly.
            signs[i] = 1.0 if val < 0.5 else -1.0
        candidates.append(signs)
        
    # 3. Random initializations
    random.seed(12345)
    for _ in range(5):
        signs = np.random.choice([-1.0, 1.0], N)
        candidates.append(signs)

    # Evaluate all candidates and pick the best few for SA
    scored_candidates = []
    for s in candidates:
        bound = compute_c3_bound(s)
        scored_candidates.append((bound, s))
        
    scored_candidates.sort(key=lambda x: x[0])
    
    # Run SA on the top 3 candidates
    best_signs = scored_candidates[0][1].copy()
    best_bound = scored_candidates[0][0]
    
    # Simulated Annealing Parameters
    # We need to be fast. 120s limit.
    # FFT of size 2048 is ~microseconds. 100k evals is ~0.5s.
    # We can afford 500k evals.
    
    def run_sa(start_signs, start_bound, max_evals=200000, T_start=1.0, T_end=0.01):
        signs = start_signs.copy()
        current_bound = start_bound
        best_local_bound = start_bound
        best_local_signs = signs.copy()
        
        T = T_start
        # Cooling rate to reach T_end in max_evals
        # T_end = T_start * alpha^max_evals
        alpha = (T_end / T_start) ** (1.0 / max_evals)
        
        # Pre-compute random indices for speed? No, just use random.
        
        for _ in range(max_evals):
            # Move: Single flip or Pair flip
            if random.random() < 0.5:
                i = random.randint(0, N - 1)
                signs[i] *= -1.0
                new_bound = compute_c3_bound(signs)
                delta = new_bound - current_bound
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_bound = new_bound
                    if current_bound < best_local_bound:
                        best_local_bound = current_bound
                        best_local_signs = signs.copy()
                else:
                    signs[i] *= -1.0
            else:
                i = random.randint(0, N - 1)
                j = random.randint(0, N - 1)
                if i != j:
                    signs[i] *= -1.0
                    signs[j] *= -1.0
                    new_bound = compute_c3_bound(signs)
                    delta = new_bound - current_bound
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        current_bound = new_bound
                        if current_bound < best_local_bound:
                            best_local_bound = current_bound
                            best_local_signs = signs.copy()
                    else:
                        signs[i] *= -1.0
                        signs[j] *= -1.0
            T *= alpha
            
        return best_local_signs, best_local_bound

    # Run SA on top candidates
    for i in range(min(3, len(scored_candidates))):
        bound, signs = scored_candidates[i]
        # Vary SA parameters slightly for diversity
        sa_signs, sa_bound = run_sa(signs, bound, max_evals=150000, T_start=1.5, T_end=0.005)
        if sa_bound < best_bound:
            best_bound = sa_bound
            best_signs = sa_signs
            
    # Final Polish: Greedy Hill Climbing
    # Try flipping each bit, if it improves, keep it. Repeat until no improvement.
    improved = True
    while improved:
        improved = False
        for i in range(N):
            best_signs[i] *= -1.0
            new_bound = compute_c3_bound(best_signs)
            if new_bound < best_bound:
                best_bound = new_bound
                improved = True
            else:
                best_signs[i] *= -1.0
                
    return best_signs

# EVOLVE-BLOCK-END

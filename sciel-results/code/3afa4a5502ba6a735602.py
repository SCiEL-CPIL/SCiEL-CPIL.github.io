"""
C3-prime: Signed step functions minimizing max absolute autoconvolution / (sum)^2.
We search for a sequence of +/-1 that balances the sum S close to zero (but not zero, to avoid division by zero in the metric) while keeping the maximum absolute autocorrelation low.
"""
import numpy as np
import time

N = 400

def compute_score(x):
    """
    Computes the score: max(|R(k)|) / (sum(x))^2
    R(k) is the full linear autocorrelation: R[k] = sum_i x[i] * x[i+k]
    """
    if len(x) == 0:
        return float('inf')
    S = np.sum(x)
    if S == 0:
        # If S is 0, the denominator is 0. 
        # Usually this is undefined or infinity. 
        # In the context of 'divided by squared integral', if integral is 0, the ratio is unbounded.
        # We treat it as a very large penalty.
        return float('inf')
    
    # Compute autocorrelation efficiently using FFT
    # Zero-pad to 2*N to avoid circular effects (linear convolution)
    size = 2 * N
    fft_size = size
    X = np.fft.fft(x, n=fft_size)
    R = np.fft.ifft(X * np.conj(X)).real
    # The linear autocorrelation lags are 0 to N-1 in R[0:N] and -(N-1) to -1 in R[N:2N-1]
    # Max absolute value over all lags
    max_abs_R = np.max(np.abs(R))
    
    return max_abs_R / (S * S)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Initial population: random +/-1 sequences
    # We want to minimize max_abs_R / S^2.
    # S^2 is small if S is close to 0. This makes the score large if max_abs_R is not also extremely small.
    # However, if S is large, S^2 is large, which helps the score, but typically max_abs_R is also large.
    # For random +/-1, S ~ sqrt(N), max_abs_R ~ sqrt(N log N). Score ~ 1/sqrt(N).
    # If we can find a sequence with S=0, it's invalid (inf).
    # If we find S=2 (min non-zero even sum for N=400), S^2=4.
    # If max_abs_R is small, say 2, score is 0.5.
    # If we use all 1s, S=400, S^2=160000, max_abs_R=400. Score = 400/160000 = 0.0025.
    # Let's check if we can do better than 0.0025.
    # 1/sqrt(N) is approx 1/20 = 0.05. So random is worse.
    # Can we get max_abs_R < S^2?
    # If S=2, S^2=4. We need max_abs_R < 4.
    # For N=400, it is very hard to keep max autocorrelation < 4 with S=2.
    # Typically min max autocorrelation for length N is ~sqrt(N).
    # So for S=2, score ~ sqrt(400)/4 = 20/4 = 5.
    # For S=20 (S^2=400), if max_abs_R ~ 20, score ~ 0.05.
    # For S=400, score 0.0025.
    
    # It seems keeping S large is better for this metric specifically because S^2 grows quadratically while max_abs_R grows roughly linearly or sub-linearly with N for simple structures like all-ones.
    # However, "signed" might imply we need to mix signs?
    # If we use all 1s, it is a signed step function (sign is +1).
    # Let's try to optimize around the all-ones solution or find a better trade-off.
    
    # Strategy:
    # 1. Start with all 1s. Score = 1/N.
    # 2. Try to flip some bits to reduce max_abs_R while maintaining a large S.
    # Actually, flipping bits reduces S. 
    # If we flip k bits, S becomes N-2k.
    # S^2 = (N-2k)^2.
    # If we flip random bits, max_abs_R increases towards sqrt(N log N).
    # Let's simulate:
    # N=400. All 1s: R[0]=400, R[k]=400-k. Max=400. Score=400/160000=0.0025.
    # Flip 1 bit: S=398. S^2=158404.
    # R[0]=400.
    # R[k] changes.
    # Let's just run a search to see if we can beat 0.0025.
    
    best_x = np.ones(N)
    best_score = compute_score(best_x)
    
    # Random search with local refinement
    num_evals = 0
    max_evals = 50000 # Estimate: compute_score is fast (FFT). 50k evals should be < 10s.
    
    # We will perform a local search starting from all-ones and random starts.
    
    def local_search(x0, max_iters=200):
        x = x0.copy()
        current_score = compute_score(x)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            # Try flipping a random position
            i = np.random.randint(N)
            x_new = x.copy()
            x_new[i] *= -1
            new_score = compute_score(x_new)
            if new_score < current_score:
                x = x_new
                current_score = new_score
            num_evals += 1
            if num_evals > max_evals:
                break
        return x, current_score

    # Start 1: All ones
    x, score = local_search(np.ones(N), max_iters=1000)
    if score < best_score:
        best_score = score
        best_x = x
    
    # Start 2: Random
    for _ in range(5):
        x0 = np.random.choice([-1, 1], size=N)
        x, score = local_search(x0, max_iters=500)
        if score < best_score:
            best_score = score
            best_x = x
        if time.time() > deadline:
            break
            
    # Convert to list of floats as requested by potential interface needs
    return best_x.tolist()

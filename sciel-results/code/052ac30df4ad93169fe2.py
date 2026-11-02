"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_metric(x):
    """
    Compute the metric: max|autoconvolution| / (integral)^2.
    For signed step functions with values in {-1, 1}:
    - autoconvolution is circular.
    - max|autoconv| is the maximum absolute value of the circular autocorrelation.
    - "squared integral" is interpreted as (sum(x^2)) = N for binary signs, 
      or potentially (sum(x))^2. 
    Given the seed score is 1.0 for np.ones(N):
    If metric = max|acf| / N:
       acf[0] = N, max|acf| = N. Score = N/N = 1.
    If metric = max|acf| / (sum(x))^2:
       sum(x) = N, (N)^2 = N^2. Score = N/N^2 = 1/N. (Unlikely to be 1.0 seed)
    If metric = max|acf| / (||x||_2)^2:
       ||x||_2^2 = N. Score = N/N = 1.
    
    We assume the metric is max|acf| / N, which is equivalent to max|acf| / sum(x^2) for binary signs.
    """
    X = np.fft.rfft(x)
    power = X * np.conj(X)
    acf = np.fft.irfft(power, N)
    max_abs_acf = np.max(np.abs(acf))
    norm_sq = np.sum(x * x)
    if norm_sq < 1e-12:
        return np.inf
    return max_abs_acf / norm_sq

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initialize with random signs
    rng = np.random.RandomState(42)
    x = rng.choice([-1, 1], size=N)
    current_score = _compute_metric(x)
    best_x = x.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    # We want to minimize the max absolute autocorrelation.
    # The "temperature" controls the probability of accepting worse solutions.
    # Initial temperature should be comparable to the score scale.
    T = 1.0
    cooling_rate = 0.999
    
    # Precompute index for random flips
    idx = np.arange(N)
    
    # We will perform a large number of steps.
    # Each step involves:
    # 1. Flip a random sign
    # 2. Compute new metric (expensive if done from scratch, but FFT is O(N log N))
    #    To speed up, we could use the property that flipping x[i] changes acf by 
    #    -2*x[i]*x[j] for all j? No, that's for linear. For circular, it's complex.
    #    Actually, flipping x[i] to -x[i] changes the autocorrelation R[k] = sum x[n]x[n+k].
    #    Delta R[k] = -2 x[i] (x[i+k] + x[i-k] ... )? 
    #    Let's stick to full FFT computation for simplicity and robustness, 
    #    but optimize the loop.
    
    # Optimization: Instead of full FFT every step, we can try to estimate the change.
    # However, N=400 is small. FFT of 400 points is very fast (< 100us).
    # We can do ~10000 steps in 1 second. 145 seconds allows ~100k-200k steps.
    
    steps = 0
    while time.time() < deadline:
        # Generate a random index to flip
        flip_idx = rng.randint(N)
        
        # Perform flip
        x[flip_idx] *= -1
        
        # Compute new score
        new_score = _compute_metric(x)
        
        # Acceptance criterion
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
        else:
            # Accept with probability exp(-delta / T)
            delta = new_score - current_score
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_score = new_score
                else:
                    # Reject: undo flip
                    x[flip_idx] *= -1
            else:
                # Should not happen if new_score >= current_score, but just in case
                current_score = new_score
                
        # Cool down
        T *= cooling_rate
        if T < 1e-4:
            # Restart with higher temperature if stuck? 
            # Or just stop if T is too low.
            # Let's restart the annealing schedule if we haven't improved in a long time.
            # For now, just keep cooling. If T gets too small, it becomes greedy.
            # Greedy is fine for local search.
            pass
            
        steps += 1
        
        # Periodic check for improvement to reset temperature?
        # Let's add a simple restart strategy: if no improvement for 1000 steps,
        # and T is low, bump T up slightly to escape local minima.
        if steps % 1000 == 0:
             # Check if we are stuck
             if current_score > best_score * 1.01: # 1% worse than best
                 T = max(T, 0.5)
    
    return best_x
# EVOLVE-BLOCK-END

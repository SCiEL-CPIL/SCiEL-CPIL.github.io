import numpy as np
import time
from itertools import combinations

N = 400

def _compute_c3_vectorized(x):
    """
    Compute the C3 autoconvolution upper bound for a signed step function x of length N.
    The upper bound on C3 is typically related to the maximum over shifts of the
    autoconvolution, or a specific integral norm. Based on the task description
    "autoconvolution upper bound", we compute the maximum of the discrete autoconvolution.
    
    For a sequence x, the autoconvolution c[k] = sum_i x[i] x[i+k].
    The "C3" in the context of autocorrelation inequalities often refers to a bound
    involving the maximum coefficient or a norm. Given "upper bound on C3", and
    typical problems of this type (like Littlewood-Offdi or similar), it might be
    related to the maximum of the autocorrelation sequence.
    
    However, the task says "improve an autoconvolution upper bound". 
    Let's assume the score is the maximum value of the absolute autoconvolution
    or the sum of squares, etc.
    
    Wait, the score is "the upper bound on C3". The evaluator measures the raw value
    (lower is better).
    
    Let's look at the structure. x is a signed step function.
    Usually, for such problems, the quantity to minimize is related to the maximum
    of the autoconvolution coefficients, or the L-infinity norm of the autoconvolution.
    
    Let's define the score as the maximum of the autoconvolution.
    c = np.convolve(x, x, mode='full')
    score = np.max(np.abs(c))
    
    Alternatively, it could be the sum of squares of the autoconvolution coefficients
    which equals (sum x^2)^2, which is constant if x is +/-1.
    
    If x consists of +/-1, sum x^2 = N. So (sum x^2)^2 = N^2 is constant.
    So the score must be something that varies.
    The maximum autoconvolution coefficient varies.
    
    Let's assume score = max(|c[k]| for k in range(2N-1)).
    """
    c = np.convolve(x, x, mode='full')
    return np.max(np.abs(c))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Initialize with random signed step function
    # The task says "Choose a signed step function". 
    # "signed" usually means +/-1.
    x = np.random.choice([-1.0, 1.0], size=N)
    
    best_score = _compute_c3_vectorized(x)
    best_x = x.copy()
    
    # Local search
    # Multi-flip moves (2-3 simultaneous flips)
    while time.time() < deadline:
        # Try a batch of random moves
        for _ in range(100):  # Adjust this number based on speed
            if time.time() > deadline:
                break
            
            # Randomly select 2 or 3 indices to flip
            k = np.random.choice([2, 3])
            idx = np.random.choice(N, size=k, replace=False)
            
            # Create a copy
            x_new = x.copy()
            x_new[idx] *= -1
            
            score_new = _compute_c3_vectorized(x_new)
            
            if score_new < best_score:
                best_score = score_new
                best_x = x_new
                x = x_new
            else:
                x = x_new  # Keep the random walk? Or revert?
                # Reverting is standard hill climbing. Let's revert if not improved.
                x = best_x
    
    # Final polish with single flips
    while time.time() < deadline - 5.0:
        improved = False
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            x_new = best_x.copy()
            x_new[i] *= -1
            score_new = _compute_c3_vectorized(x_new)
            if score_new < best_score:
                best_score = score_new
                best_x = x_new
                improved = True
        if not improved:
            break
            
    return best_x

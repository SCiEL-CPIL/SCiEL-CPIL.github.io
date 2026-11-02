"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
def _compute_score(x):
    """Compute the maximum absolute autoconvolution divided by squared integral."""
    # x is a 1D array of length N
    # Autoconvolution (linear, full)
    ac = np.convolve(x, x, mode='full')
    # Integral is the sum of x
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    squared_integral = integral * integral
    max_abs_ac = np.max(np.abs(ac))
    return max_abs_ac / squared_integral


def _compute_score_and_grad(x):
    """Compute score and gradient using FFT for efficiency."""
    # For efficiency with N=400, direct convolution is fast enough
    # ac[k] = sum_i x[i] * x[k-i]
    # We want to minimize max(abs(ac)) / (sum(x))^2
    
    # Let's use a different approach for gradient.
    # The objective is f(x) = max_k |ac[k]| / (sum(x))^2
    # This is not differentiable at the max or when ac[k]=0.
    # We can use subgradient or a smooth approximation, or just use finite differences
    # if analytic gradient is too complex. Given the time budget, let's try finite differences
    # or a simple heuristic.
    
    # Actually, let's stick to a simpler, more robust method:
    # 1. Try random sign flips
    # 2. Try local search
    
    ac = np.convolve(x, x, mode='full')
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    squared_integral = integral * integral
    max_abs_ac = np.max(np.abs(ac))
    return max_abs_ac / squared_integral


def _local_search(x, max_iter=10000, time_limit=5.0):
    """Local search by flipping signs of individual elements."""
    start_time = time.time()
    current_score = _compute_score(x)
    best_x = x.copy()
    best_score = current_score
    
    # Precompute autoconvolution for efficiency
    # ac = np.convolve(x, x, mode='full')
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping each element
        for j in range(N):
            old_val = x[j]
            x[j] = -x[j]
            new_score = _compute_score(x)
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
            else:
                x[j] = old_val
    
    return best_x, best_score


def _random_search(n_trials=1000, time_limit=5.0):
    """Random search with sign flips."""
    start_time = time.time()
    best_x = None
    best_score = float('inf')
    
    for trial in range(n_trials):
        if time.time() - start_time > time_limit:
            break
        
        # Start with random signs
        x = np.random.choice([-1.0, 1.0], size=N)
        score = _compute_score(x)
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
            
            # Try local search from this point
            if time.time() - start_time < time_limit - 2.0:
                improved_x, improved_score = _local_search(x, max_iter=1000, time_limit=2.0)
                if improved_score < best_score:
                    best_score = improved_score
                    best_x = improved_x
    
    return best_x, best_score


def _structured_search(time_limit=10.0):
    """Try structured patterns."""
    start_time = time.time()
    best_x = None
    best_score = float('inf')
    
    # Try various structured patterns
    patterns = []
    
    # Alternating
    patterns.append(np.tile([1.0, -1.0], N // 2))
    
    # Blocks
    for block_size in [2, 4, 8, 16, 32]:
        if N % block_size == 0:
            n_blocks = N // block_size
            for pattern in [[1.0], [-1.0], [1.0, -1.0], [1.0, 1.0, -1.0, -1.0]]:
                if len(pattern) <= n_blocks:
                    x = np.tile(pattern, N // len(pattern))[:N]
                    patterns.append(x)
    
    # Random structured
    for _ in range(50):
        if time.time() - start_time > time_limit:
            break
        x = np.random.choice([-1.0, 1.0], size=N)
        # Smooth it a bit
        if N > 10:
            x_smooth = x.copy()
            for i in range(1, N-1):
                if x[i-1] == x[i+1] and x[i] != x[i-1]:
                    x_smooth[i] = x[i-1]
            patterns.append(x_smooth)
    
    for x in patterns:
        if len(x) != N:
            continue
        score = _compute_score(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
            
            # Local search from structured pattern
            if time.time() - start_time < time_limit - 2.0:
                improved_x, improved_score = _local_search(x, max_iter=2000, time_limit=3.0)
                if improved_score < best_score:
                    best_score = improved_score
                    best_x = improved_x
    
    return best_x, best_score


def construct():
    """Construct a signed step function minimizing the maximum absolute autoconvolution 
    divided by squared integral."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_x = None
    best_score = float('inf')
    
    # Phase 1: Try all-ones baseline
    x = np.ones(N)
    score = _compute_score(x)
    best_x = x.copy()
    best_score = score
    
    # Phase 2: Structured search
    if time.time() < deadline - 30:
        x_struct, score_struct = _structured_search(time_limit=15.0)
        if x_struct is not None and score_struct < best_score:
            best_x = x_struct
            best_score = score_struct
    
    # Phase 3: Random search with local refinement
    if time.time() < deadline - 20:
        x_rand, score_rand = _random_search(n_trials=500, time_limit=25.0)
        if x_rand is not None and score_rand < best_score:
            best_x = x_rand
            best_score = score_rand
    
    # Phase 4: More intensive local search from best found
    if time.time() < deadline - 5 and best_x is not None:
        x_improved, score_improved = _local_search(best_x, max_iter=50000, 
                                                   time_limit=min(15.0, deadline - time.time() - 2.0))
        if score_improved < best_score:
            best_x = x_improved
            best_score = score_improved
    
    return best_x
# EVOLVE-BLOCK-END

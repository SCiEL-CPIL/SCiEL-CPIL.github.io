import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    def compute_value(h):
        """
        Computes the objective function value.
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
        """
        s = np.sum(h)
        if s == 0 or not np.isfinite(s):
            return float('inf')
        
        # Normalize h to have sum 1 to simplify and stabilize
        # Let h_norm = h / s. sum(h_norm) = 1.
        # conv(h, h) = s^2 * conv(h_norm, h_norm)
        # value = abs(2 * N * s^2 * max(conv(h_norm, h_norm)) / s^2)
        # value = abs(2 * N * max(conv(h_norm, h_norm)))
        h_norm = h / s
        # Use FFT for faster convolution if N is large, but N=400 is small enough 
        # that direct convolution is often faster due to overhead. 
        # However, for safety and speed in optimization loops, FFT is good.
        # np.convolve is O(N^2) = 160k ops. FFT is O(N log N) ~ 3600 ops.
        # Let's use FFT for speedup in the inner loop.
        
        # Zero-pad to 2*N for linear convolution via FFT
        n_fft = 2 * N
        H = np.fft.fft(h_norm, n=n_fft)
        conv = np.fft.ifft(H * H).real
        max_conv = np.max(conv)
        
        return abs(2 * N * max_conv)

    def generate_signal(r, initial_x=0.5):
        """
        Generate a sequence of N points from the logistic map x_{n+1} = r * x_n * (1 - x_n).
        Returns a vector of heights.
        We map the chaotic values to a zero-mean (or sum-normalized) vector.
        """
        h = np.zeros(N)
        x = initial_x
        for i in range(N):
            h[i] = x
            x = r * x * (1.0 - x)
        return h

    def objective(r):
        """
        Objective function for the optimizer.
        r is the bifurcation parameter.
        """
        # Ensure r is within valid range for chaos (roughly 3.57 to 4.0)
        # The logistic map is chaotic for r in approx [3.5699, 4.0]
        # We will search in [3.57, 4.0]
        
        # Check time
        if time.time() > deadline:
            return float('inf')
            
        try:
            h = generate_signal(r)
            # To avoid issues with the sum being zero or very small, 
            # we can center the signal or just let the normalization handle it.
            # The chaotic signal has a non-zero mean (approx 0.5 for r=4).
            # compute_value handles normalization.
            val = compute_value(h)
            return val
        except Exception:
            return float('inf')

    best_score = float('inf')
    best_h = np.ones(N)
    
    # We will search for the best r in the chaotic regime.
    # The landscape might be rugged, so we might do a few coarse scans 
    # or rely on the bounded optimizer which is robust for 1D.
    
    # Define the search bounds for r
    r_min = 3.57
    r_max = 4.0
    
    # Option 1: Single bounded optimization
    # Option 2: Grid search + local refinement (more robust for rugged 1D landscapes)
    
    # Let's do a coarse grid search first to identify promising regions, 
    # then refine with minimize_scalar.
    
    # Time budget allocation:
    # 1. Coarse grid: 100 points. ~100 evaluations.
    # 2. Local refine: 3 best regions. ~3 * 50 evals.
    
    num_grid = 100
    r_values = np.linspace(r_min, r_max, num_grid)
    
    grid_scores = []
    for r in r_values:
        if time.time() > deadline - 10:
            break
        score = objective(r)
        grid_scores.append(score)
        
        if score < best_score:
            best_score = score
            best_h = generate_signal(r)

    if not grid_scores:
        return best_h

    # Find the top 3 candidates from the grid
    grid_scores = np.array(grid_scores)
    # Handle infinities
    valid_indices = np.where(np.isfinite(grid_scores))[0]
    if len(valid_indices) == 0:
        return best_h
        
    sorted_idx = valid_indices[np.argsort(grid_scores[valid_indices])]
    top_k = min(3, len(sorted_idx))
    
    for i in range(top_k):
        if time.time() > deadline - 5:
            break
            
        idx = sorted_idx[i]
        r_center = r_values[idx]
        
        # Define a local search window around the best grid point
        # The step size in the grid is (r_max - r_min) / num_grid
        step = (r_max - r_min) / num_grid
        local_min = max(r_min, r_center - step)
        local_max = min(r_max, r_center + step)
        
        # Use bounded minimizer
        try:
            res = minimize_scalar(
                objective, 
                bounds=(local_min, local_max), 
                method='bounded',
                options={'xatol': 1e-8, 'maxiter': 50}
            )
            
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_h = generate_signal(res.x)
        except Exception:
            pass

    return best_h

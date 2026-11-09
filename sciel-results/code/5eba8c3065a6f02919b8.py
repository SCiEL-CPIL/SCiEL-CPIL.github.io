# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Uses Keich binary construction as initial guess, then optimizes 1-D positions
    to minimize the union volume. Since the union is a product of two 1-D unions
    (x-union * y-union * height), we minimize each 1-D union length independently.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin before 630s kill
    
    N = 8
    
    # Keich binary construction for initial positions
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Function to compute 1-D union length for N intervals
    # Interval k: [p[k] + c*k/N, p[k] + c*k/N + 1/N] for c in [0,1]
    # Union over c of these intervals, projected onto 1-D
    def union_length_1d(p, N=N):
        """Compute the length of the union of N sheared intervals in 1-D."""
        # For each c in [0,1], interval k is [p[k] + c*k/N, p[k] + c*k/N + 1/N]
        # The union at height c has some length; we want the integral over c
        # Actually, we want the volume in 2-D (x, c) space
        # This is the measure of {(x, c) : exists k, x in [p[k] + c*k/N, p[k] + c*k/N + 1/N]}
        
        # Use Monte Carlo for speed during optimization
        # But for accuracy, use a more direct approach
        # 
        # The set in (x, c) space: for each k, it's a parallelogram
        # (x, c) where x - c*k/N in [p[k], p[k] + 1/N], c in [0, 1]
        # 
        # Let's use a grid-based approach for accuracy
        c_grid = np.linspace(0, 1, 200)
        total_length = 0.0
        for c in c_grid:
            intervals = []
            for k in range(N):
                left = p[k] + c * k / N
                right = left + 1.0 / N
                intervals.append((left, right))
            # Merge intervals
            intervals.sort()
            merged = [intervals[0]]
            for left, right in intervals[1:]:
                if left <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], right))
                else:
                    merged.append((left, right))
            length = sum(r - l for l, r in merged)
            total_length += length
        return total_length / len(c_grid)
    
    # Better: use a vectorized Monte Carlo estimator for speed
    def union_length_mc(p, n_samples=50000, seed=42):
        """Monte Carlo estimate of 2-D measure of union in (x, c) space."""
        rng = np.random.default_rng(seed)
        c = rng.uniform(0, 1, n_samples)
        
        # For each sample c, find the union length of intervals at that height
        # Vectorize: for each k, check if point is in interval
        # x range: min(p[k] + c*k/N) to max(p[k] + c*k/N + 1/N)
        k_idx = np.arange(N)
        # left[k] = p[k] + c * k / N, right[k] = left[k] + 1/N
        # For a given c, the union length can be computed by merging intervals
        
        # Use a different approach: estimate the measure directly
        # Sample x from the bounding box and check if covered
        x_min = np.min(p)
        x_max = np.max(p + 1.0/N + 1.0)  # max shift is (N-1)/N
        x_samples = rng.uniform(x_min, x_max, n_samples)
        
        # Check coverage: for each (x, c) pair, is there a k such that
        # p[k] + c*k/N <= x <= p[k] + c*k/N + 1/N
        # i.e., p[k] <= x - c*k/N <= p[k] + 1/N
        
        covered = np.zeros(n_samples, dtype=bool)
        for k in range(N):
            shifted = x_samples - c * k / N
            in_interval = (shifted >= p[k]) & (shifted <= p[k] + 1.0/N)
            covered |= in_interval
        
        box_width = x_max - x_min
        return box_width * np.mean(covered)
    
    # Optimize 1-D positions to minimize union length
    def optimize_1d(initial_p, time_budget):
        """Optimize 1-D positions using scipy minimize."""
        p0 = initial_p.copy()
        
        def objective(p):
            return union_length_mc(p, n_samples=30000, seed=123)
        
        best_p = p0.copy()
        best_val = objective(p0)
        
        # Try multiple local optimizations
        n_attempts = 0
        while time.time() < time_budget:
            # Add small perturbation for diversity
            if n_attempts > 0:
                p_init = p0 + np.random.default_rng(n_attempts).normal(0, 0.05, N)
            else:
                p_init = p0.copy()
            
            try:
                result = minimize(
                    objective, p_init, method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-7}
                )
                if result.fun < best_val:
                    best_val = result.fun
                    best_p = result.x.copy()
            except:
                pass
            n_attempts += 1
            if n_attempts >= 5:
                break
        
        # Final polish with tighter tolerance
        try:
            result = minimize(
                objective, best_p, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-9}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_p = result.x.copy()
        except:
            pass
        
        return best_p, best_val
    
    # Optimize x and y independently
    time_budget_x = time.time() + 120.0
    time_budget_y = time.time() + 240.0
    
    p_x, val_x = optimize_1d(keich, time_budget_x)
    p_y, val_y = optimize_1d(keich, time_budget_y)
    
    # Build the 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = p_x[i_flat]
    base_y = p_y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    # Verify validity
    X0 = np.min(positions[:, 0])
    X1 = np.max(positions[:, 0] + (i_flat + 1) / N)
    Y0 = np.min(positions[:, 1])
    Y1 = np.max(positions[:, 1] + (j_flat + 1) / N)
    
    if not np.all(np.isfinite(positions)) or (X1 - X0) * (Y1 - Y0) <= 1e-9:
        # Fallback to original Keich
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END

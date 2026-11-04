import numpy as np
import time
from scipy.optimize import minimize

N = 100
TARGET_SUM = N / 2.0

def _compute_overlaps(h):
    """
    Compute C_k = sum_i h_i (1 - h_{i+k}) for k=1..N-1.
    """
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
    # c[N+k] corresponds to shift k.
    # c[N-k] corresponds to shift -k.
    # We want lags 1 to N-1 (and -1 to -(N-1)).
    c_nonzero = np.concatenate([c[:N], c[N+1:]])
    return c_nonzero

def _max_overlap(h):
    """Return the maximum overlap value."""
    c = _compute_overlaps(h)
    return np.max(c)

def _objective_and_grad(h_flat, beta, lambda_sum):
    """
    Objective: LogSumExp of overlaps + penalty for sum constraint.
    """
    h = h_flat
    c = _compute_overlaps(h)
    
    # Log-sum-exp for numerical stability
    c_max = np.max(c)
    lse = c_max + np.log(np.sum(np.exp(beta * (c - c_max)))) / beta
    
    sum_diff = np.sum(h) - TARGET_SUM
    penalty = lambda_sum * sum_diff**2
    
    obj = lse + penalty
    
    # Gradient
    weights = np.exp(beta * (c - c_max))
    weights /= np.sum(weights)
    
    grad = np.zeros(N)
    lags = np.concatenate([np.arange(-(N-1), 0), np.arange(1, N)])
    
    for k_idx, lag in enumerate(lags):
        w = weights[k_idx]
        if w < 1e-10: continue
        
        # Contribution to grad from this lag
        # d c_lag / d h_j = 1 - (h_{j+lag} + h_{j-lag})
        
        j = np.arange(N)
        
        # Indices for h[j+lag]
        idx_pos = j + lag
        valid_pos = (idx_pos >= 0) & (idx_pos < N)
        h_pos = np.zeros(N)
        h_pos[valid_pos] = h[idx_pos[valid_pos]]
        
        # Indices for h[j-lag]
        idx_neg = j - lag
        valid_neg = (idx_neg >= 0) & (idx_neg < N)
        h_neg = np.zeros(N)
        h_neg[valid_neg] = h[idx_neg[valid_neg]]
        
        grad += w * (1.0 - h_pos - h_neg)
        
    grad += 2.0 * lambda_sum * sum_diff
    
    return obj, grad

def _project_to_feasible(h):
    """
    Project h onto the feasible set: 0 <= h_i <= 1 and sum(h) = N/2.
    Uses iterative water-filling style projection.
    """
    h = np.clip(h, 0, 1).copy()
    target = N / 2.0
    current = np.sum(h)
    
    if abs(current - target) < 1e-12:
        return h
    
    for _ in range(100):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        
        if diff > 0:
            mask = h < 1.0
            if not np.any(mask):
                break
            step = diff / np.sum(mask)
            h[mask] += step
            h = np.clip(h, 0, 1)
        else:
            mask = h > 0.0
            if not np.any(mask):
                break
            step = (-diff) / np.sum(mask)
            h[mask] -= step
            h = np.clip(h, 0, 1)
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = None
    best_score = float('inf')
    
    # Initial candidates
    candidates = []
    
    # 1. Alternating
    h_alt = np.full(N, 0.5)
    candidates.append(h_alt)
    
    # 2. Sine waves
    t = np.arange(N)
    for freq in [1, 2, 3, 5, 10]:
        for amp in [0.3, 0.4, 0.49]:
            h_sin = 0.5 + amp * np.sin(2 * np.pi * freq * t / N)
            candidates.append(np.clip(h_sin, 0, 1))
            
    # 3. Random Binary
    rng = np.random.default_rng(42)
    for _ in range(5):
        h_bin = rng.choice([0, 1], size=N)
        # Adjust sum to be exactly 50
        # If sum is not 50, swap some values
        s = np.sum(h_bin)
        if s > 50:
            idx_ones = np.where(h_bin == 1)[0]
            to_swap = s - 50
            swap_idx = rng.choice(idx_ones, size=to_swap, replace=False)
            h_bin[swap_idx] = 0
        elif s < 50:
            idx_zeros = np.where(h_bin == 0)[0]
            to_swap = 50 - s
            swap_idx = rng.choice(idx_zeros, size=to_swap, replace=False)
            h_bin[swap_idx] = 1
        candidates.append(h_bin.astype(float))
        
    # 4. Random Uniform
    for _ in range(5):
        h_rand = rng.uniform(0, 1, size=N)
        # Project to sum 50 roughly
        h_rand = h_rand - np.mean(h_rand) + 0.5
        h_rand = np.clip(h_rand, 0, 1)
        # Fine tune sum
        diff = 50 - np.sum(h_rand)
        if diff != 0:
            # Distribute diff
            step = diff / N
            h_rand += step
            h_rand = np.clip(h_rand, 0, 1)
        candidates.append(h_rand)
        
    # 5. Spectral initialization (inspired by graph coloring/spectral partitioning)
    # Construct a circulant graph Laplacian-like matrix or just use FFT-based initialization
    # The idea is to find a binary pattern that minimizes overlap.
    # A good initial guess for binary patterns is often found via spectral methods on the adjacency matrix.
    # Here we simulate a spectral heuristic by trying patterns derived from eigenvectors of a simple circulant.
    # For now, we just add a few more structured patterns.
    
    # 6. "Block" patterns
    for block_size in [2, 4, 5, 10]:
        h_block = np.zeros(N)
        h_block[::block_size] = 1.0
        # Adjust to sum 50
        h_block = _project_to_feasible(h_block)
        candidates.append(h_block)

    # Optimize each candidate
    for i, h_init in enumerate(candidates):
        if time.time() > deadline - 10.0:
            break
            
        h_init = h_init.copy()
        
        # Ensure sum is close to 50 initially
        h_init = _project_to_feasible(h_init)
        
        # Run optimization with annealed beta
        # Start with low beta, increase
        current_h = h_init.copy()
        current_obj = float('inf')
        
        betas = [10.0, 50.0, 100.0]
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
                
            # Define objective for this beta
            lambda_sum = 1000.0 # Strong penalty for sum constraint
            
            def obj_func(x):
                val, _ = _objective_and_grad(x, beta, lambda_sum)
                return val
                
            def grad_func(x):
                _, g = _objective_and_grad(x, beta, lambda_sum)
                return g
                
            try:
                res = minimize(obj_func, current_h, method='L-BFGS-B', jac=grad_func,
                               bounds=[(0, 1)] * N,
                               options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
                current_h = res.x
                current_obj = res.fun
            except Exception as e:
                # If optimization fails, keep previous best
                pass
                
        # Evaluate true max overlap
        c_final = _compute_overlaps(current_h)
        true_max = np.max(c_final)
        
        # Check sum constraint
        if abs(np.sum(current_h) - 50) > 1e-5:
            # Final adjustment
            current_h = _project_to_feasible(current_h)
            c_final = _compute_overlaps(current_h)
            true_max = np.max(c_final)
            
        if true_max < best_score:
            best_score = true_max
            best_h = current_h.copy()
            
    # If no valid h found, return a default
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    # Final polish: Local search with small perturbations
    if time.time() < deadline - 2.0:
        rng_local = np.random.default_rng(123)
        h_best = best_h.copy()
        best_val = _max_overlap(h_best)
        
        for _ in range(50):
            if time.time() > deadline - 2.0: break
            # Random small perturbation
            idx = rng_local.choice(N, size=2, replace=False)
            delta = rng_local.uniform(-0.01, 0.01)
            h_try = h_best.copy()
            h_try[idx[0]] += delta
            h_try[idx[1]] -= delta
            
            # Project to [0,1] and sum 50
            h_try = _project_to_feasible(h_try)
            
            val = _max_overlap(h_try)
            if val < best_val:
                best_val = val
                h_best = h_try.copy()
                
        best_h = h_best
        
    return best_h

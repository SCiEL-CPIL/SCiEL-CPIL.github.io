import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 145.0 # 150s limit, 5s margin

def compute_ratio(h):
    h = np.array(h, dtype=np.float64)
    if np.any(h < -1e-9):
        return -1.0
    h = np.clip(h, 0, None)
    
    l1 = np.sum(h)
    if l1 < 1e-12:
        return -1.0
        
    l2_sq = np.sum(h**2)
    
    # Linear convolution using FFT for speed
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)  # h is non-negative, so all values are non-negative
    
    if max_conv < 1e-12:
        return -1.0
        
    return (l2_sq / (l1 * max_conv))

def is_sidon(arr):
    """Check if a set of integers is a Sidon set (B2 set)."""
    arr = sorted(set(arr))
    n = len(arr)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def construct():
    start_time = time.time()
    
    # Initial candidates
    candidates = []
    
    # 1. Uniform
    h_uni = np.ones(N) / N
    candidates.append(h_uni)
    
    # 2. Spike
    h_spike = np.zeros(N)
    h_spike[N//2] = 1.0
    candidates.append(h_spike)
    
    # 3. Gaussian-like with different widths
    x = np.linspace(-1, 1, N)
    for width in [3, 5, 8]:
        h_gauss = np.exp(-width * x**2)
        h_gauss /= np.sum(h_gauss)
        candidates.append(h_gauss)
    
    # 4. Cosine bump
    h_cos = 0.5 * (1 + np.cos(np.pi * x))
    h_cos /= np.sum(h_cos)
    candidates.append(h_cos)
    
    # 5. Sidon set based initialization
    # Find a Sidon set of size as large as possible within [0, N-1]
    sidon_indices = []
    for i in range(N):
        candidate = sidon_indices + [i]
        if is_sidon(candidate):
            sidon_indices.append(i)
    
    # Create h based on Sidon indices with equal weights
    h_sidon = np.zeros(N)
    if sidon_indices:
        h_sidon[sidon_indices] = 1.0 / np.sqrt(len(sidon_indices))
    candidates.append(h_sidon)
    
    # 6. Sparse random initializations with various sparsity levels
    rng = np.random.RandomState(42)
    for K in [5, 8, 10, 15, 20, 25, 30]:
        h_sparse = np.zeros(N)
        indices = rng.choice(N, K, replace=False)
        h_sparse[indices] = 1.0 / np.sqrt(K)
        candidates.append(h_sparse)
    
    # 7. Smooth sparse: combine Sidon structure with smooth decay
    if sidon_indices:
        h_smooth_sparse = np.zeros(N)
        for idx in sidon_indices:
            h_smooth_sparse[idx] = 1.0
        # Apply smooth decay based on position
        decay = np.exp(-0.5 * np.abs(np.arange(N) - N/2) / (N/4))
        h_smooth_sparse *= decay
        h_smooth_sparse /= np.sum(h_smooth_sparse)
        candidates.append(h_smooth_sparse)
    
    # Evaluate all candidates
    best_h = candidates[0].copy()
    best_val = compute_ratio(best_h)
    
    for h in candidates[1:]:
        val = compute_ratio(h)
        if val > best_val:
            best_val = val
            best_h = h.copy()
    
    # Optimization using direct parameterization (h_i >= 0, sum h_i = 1)
    # Use softmax parameterization for better conditioning
    
    def neg_log_ratio(params):
        # params are unnormalized log-weights
        # Use stable softmax
        params_shifted = params - np.max(params)
        h = np.exp(params_shifted)
        h = h / np.sum(h)
        r = compute_ratio(h)
        if r <= 0:
            return 1000.0
        return -np.log(r)
    
    # Random restarts with more diverse initializations
    n_restarts = 30
    for i in range(n_restarts):
        if time.time() > DEADLINE - 10:
            break
            
        if i == 0:
            # Start from best initial candidate
            x0 = np.log(best_h + 1e-10)
        elif i == 1 and sidon_indices:
            # Start from Sidon
            x0 = np.log(h_sidon + 1e-10)
        elif i < 5:
            # Start from other good candidates
            x0 = np.log(candidates[i] + 1e-10)
        else:
            # Random sparse start with varying sparsity
            K = rng.randint(3, 25)
            h_start = np.zeros(N)
            indices = rng.choice(N, K, replace=False)
            h_start[indices] = 1.0 / np.sqrt(K)
            x0 = np.log(h_start + 1e-10)
            
        try:
            # Use L-BFGS-B with more iterations for better convergence
            res = minimize(neg_log_ratio, x0, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            h_opt = np.exp(res.x - np.max(res.x))
            h_opt = h_opt / np.sum(h_opt)
            val_opt = compute_ratio(h_opt)
            
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final polish: try a few more iterations from the best solution
    if time.time() < DEADLINE - 15:
        try:
            x0 = np.log(best_h + 1e-10)
            res = minimize(neg_log_ratio, x0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10})
            
            h_opt = np.exp(res.x - np.max(res.x))
            h_opt = h_opt / np.sum(h_opt)
            val_opt = compute_ratio(h_opt)
            
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Ensure non-negative and valid
    best_h = np.clip(best_h, 0, None)
    total = np.sum(best_h)
    if total < 1e-12:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / total
        
    return list(best_h)

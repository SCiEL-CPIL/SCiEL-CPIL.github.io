import numpy as np
import time
import scipy.optimize as opt

N = 50

def compute_ratio(h):
    """
    Compute the ratio (L2^2) / (L1 * SupConv).
    Uses FFT for speed.
    The ratio is scale invariant, so we can assume any normalization.
    However, for numerical stability in optimization, we often fix L1=1.
    This function computes the raw ratio for any h.
    """
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s < 1e-15:
        return 0.0
    
    # Scale invariant: ratio(h) = ratio(h/s)
    # Let's compute with h as is, but ensure we handle the scale correctly if needed.
    # Actually, R = (sum h^2) / (sum h * max conv)
    # If we scale h by c, numerator scales by c^2, denominator by c*c = c^2. Ratio is invariant.
    
    l2_sq = np.sum(h**2)
    
    # FFT convolution
    size = 2 * len(h)
    H = np.zeros(size)
    H[:len(h)] = h
    H_fft = np.fft.fft(H)
    Conv_fft = H_fft * H_fft
    Conv = np.real(np.fft.ifft(Conv_fft))
    sup_conv = np.max(Conv)
    
    if sup_conv < 1e-15:
        return 0.0
    
    ratio = l2_sq / (s * sup_conv)
    return ratio

def optimize_sparse(h0, indices, max_iter=2000, time_limit=5.0):
    """
    Optimize the values at 'indices' while keeping others at 0.
    We fix the L1 norm to 1 to remove scale invariance, which simplifies the landscape.
    Objective: maximize (L2^2) / SupConv subject to sum(h[indices])=1, h[i]>=0.
    """
    if len(indices) == 0:
        return 0.0, h0
    
    # Initial values for active indices
    x0 = h0[indices].copy()
    s0 = np.sum(x0)
    if s0 < 1e-10:
        x0 = np.ones(len(indices)) / len(indices)
    else:
        x0 = x0 / s0
        
    x0 = np.maximum(x0, 1e-12)
    
    def objective(x):
        # x are the active values, sum(x) should be 1, but optimizer might drift.
        # We enforce sum=1 by normalizing inside? No, better to use a parameterization or just trust L-BFGS-B with bounds.
        # However, L-BFGS-B doesn't enforce equality constraints easily.
        # Alternative: Maximize log(L2^2) - log(SupConv) subject to sum(x)=1?
        # Let's stick to the previous approach but with FFT.
        # We will normalize x to sum 1 inside the objective to handle scale invariance, 
        # but this makes the gradient tricky. 
        # Better: Just optimize the unnormalized h and rely on the fact that the ratio is scale invariant.
        # But L-BFGS-B might drift in scale. 
        # Let's use the standard trick: fix sum(x)=1 by optimizing K-1 variables and setting the last one?
        # Or simpler: Optimize x, and in the objective, normalize x to sum 1.
        # Since the function is scale invariant, the gradient with respect to scale is 0.
        # So we can just compute the ratio of the normalized vector.
        
        h = np.zeros(N)
        h[indices] = x
        s = np.sum(h)
        if s < 1e-15:
            return 0.0
        h_norm = h / s
        l2_sq = np.sum(h_norm**2)
        
        # FFT convolution
        size = 2 * N
        H = np.zeros(size)
        H[:N] = h_norm
        H_fft = np.fft.fft(H)
        Conv_fft = H_fft * H_fft
        Conv = np.real(np.fft.ifft(Conv_fft))
        sup_conv = np.max(Conv)
        
        if sup_conv < 1e-15:
            return 0.0
        
        ratio = l2_sq / sup_conv
        return -ratio

    bounds = [(0, 1)] * len(indices)
    
    start_time = time.time()
    try:
        # Use L-BFGS-B
        # maxiter is a hard limit, but we also check time
        result = opt.minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                              options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12})
        best_x = result.x
    except Exception:
        best_x = x0
        
    h_opt = np.zeros(N)
    h_opt[indices] = best_x
    
    s = np.sum(h_opt)
    if s > 0:
        h_opt = h_opt / s
    else:
        h_opt = np.ones(N) / N
        
    r = compute_ratio(h_opt)
    return r, h_opt

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_ratio = 0.0
    best_h = np.ones(N) / N
    
    # Initial best
    best_ratio = compute_ratio(best_h)
    
    # Search strategy:
    # 1. Try various sparsity levels K
    # 2. For each K, try multiple random initializations
    # 3. Use FFT-accelerated optimization
    
    # We allocate time based on K. Smaller K is faster per iteration.
    # Let's try K from 2 to 20 first, as high sparsity might yield better local optima.
    # Then expand to higher K if time permits.
    
    k_values = list(range(2, 16)) + list(range(16, 31, 2)) + list(range(31, N+1, 5))
    # Ensure unique and sorted
    k_values = sorted(set(k_values))
    
    num_trials_per_k = 20
    
    for K in k_values:
        if time.time() > deadline:
            break
            
        for trial in range(num_trials_per_k):
            if time.time() > deadline:
                break
            
            # Randomly select K indices
            indices = np.random.choice(N, size=K, replace=False)
            indices.sort()
            
            # Random initial values
            h0 = np.zeros(N)
            # Strategy 1: Random uniform
            # Strategy 2: Geometric decay
            # Strategy 3: Gaussian bump
            
            if trial % 3 == 0:
                h0[indices] = np.random.rand(K)
            elif trial % 3 == 1:
                # Geometric
                r = np.random.uniform(0.3, 0.9)
                vals = r ** np.arange(K)
                # Randomly reverse or shift?
                if np.random.rand() > 0.5:
                    vals = vals[::-1]
                h0[indices] = vals
            else:
                # Gaussian bump on indices
                x = np.linspace(0, 1, K)
                center = np.random.uniform(0.2, 0.8)
                width = np.random.uniform(0.1, 0.5)
                h0[indices] = np.exp(-((x - center) ** 2) / (2 * width ** 2))
            
            # Optimize
            # Adjust max_iter based on K. Smaller K allows more iterations.
            max_iter = 500 if K > 10 else 1000
            
            r, h_opt = optimize_sparse(h0, indices, max_iter=max_iter)
            
            if r > best_ratio:
                best_ratio = r
                best_h = h_opt.copy()
                
    # Final polish on the best solution found
    # Identify the active set of the best solution
    active_indices = np.where(best_h > 1e-8)[0]
    if len(active_indices) > 0:
        if time.time() < deadline - 5:
            # Polish with higher precision
            r_polish, h_polish = optimize_sparse(best_h, active_indices, max_iter=2000)
            if r_polish > best_ratio:
                best_ratio = r_polish
                best_h = h_polish.copy()
                
    return list(best_h)

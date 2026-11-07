import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_value(h):
    """Compute the exact value metric for given heights."""
    h = np.asarray(h, dtype=float)
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    
    # Enforce non-negativity for the final evaluation
    h = np.maximum(h, 0)
    
    if np.all(h == 0):
        return 0.0
        
    g = np.convolve(h, h)
    m = len(g)
    
    # Construct y: [0, g[0], ..., g[m-1], 0]
    # Length m+2
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w / 3) * (y[i]^2 + y[i] * y[i+1] + y[i+1]^2)
    # y has indices 0 to m+1. The sum is over i=0 to m.
    # y[i] and y[i+1] are involved.
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    value = l2 / (norm_1 * norm_inf)
    return value

def soft_max(x, k):
    """Smooth approximation of max(x)."""
    # Handle large k to avoid overflow
    max_x = np.max(x)
    if k == 0:
        return max_x
    # exp(k * (x - max_x)) is stable
    shifted = k * (x - max_x)
    # Clip to prevent underflow issues in exp, though exp(-large) is 0
    shifted = np.clip(shifted, -50, 50) 
    exp_sum = np.sum(np.exp(shifted))
    return max_x + (1.0 / k) * np.log(exp_sum)

def objective_smoothed(h, k):
    """
    Smoothed objective function for L-BFGS-B.
    Minimizes -value_smoothed.
    """
    h = np.maximum(h, 0)
    if np.all(h == 0):
        return 0.0
        
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    
    # Use soft max for norm_inf
    norm_inf = soft_max(g, k)
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    value = l2 / (norm_1 * norm_inf)
    return -value

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Generate diverse initial points
    initial_points = []
    
    # 1. Uniform
    initial_points.append(np.ones(N))
    
    # 2. Random Uniform
    for _ in range(10):
        initial_points.append(np.random.rand(N))
        
    # 3. Sparse Spikes
    for _ in range(10):
        h = np.zeros(N)
        idx = np.random.choice(N, size=np.random.randint(3, 10), replace=False)
        h[idx] = 1.0
        initial_points.append(h)
        
    # 4. Blocky Patterns
    for _ in range(10):
        h = np.zeros(N)
        start = np.random.randint(0, N - 5)
        length = np.random.randint(5, 20)
        h[start:start+length] = 1.0
        initial_points.append(h)
        
    # 5. Gaussian Bumps
    for _ in range(10):
        x = np.linspace(-1, 1, N)
        center = np.random.uniform(-0.8, 0.8)
        width = np.random.uniform(0.1, 0.6)
        h = np.exp(-((x - center)**2) / (2 * width**2))
        initial_points.append(h)
        
    # 6. Cosine Waves
    for _ in range(5):
        x = np.linspace(0, 1, N)
        freq = np.random.uniform(1, 5)
        phase = np.random.uniform(0, 2*np.pi)
        h = np.abs(np.cos(2*np.pi*freq*x + phase))
        initial_points.append(h)

    np.random.shuffle(initial_points)
    
    # Optimization parameters
    # We use L-BFGS-B for smooth optimization
    # Bounds: h >= 0. Upper bound can be large, but scale invariance suggests
    # we should constrain the scale to prevent numerical issues.
    # Let's constrain sum(h^2) <= 100 or max(h) <= 10.
    # Actually, L-BFGS-B handles unbounded well, but normalization helps.
    # Let's just use bounds (0, 100).
    
    bounds = [(0, 100)] * N
    
    # k values for soft max annealing
    k_values = [5, 20, 100, 500]
    
    for i, h0 in enumerate(initial_points):
        if time.time() > deadline - 10:
            break
            
        h0 = np.maximum(h0, 0)
        
        # Normalize h0 to have reasonable magnitude
        if np.max(h0) > 0:
            h0 = h0 / np.max(h0) * 1.0
            
        current_h = h0.copy()
        
        # Anneal k: start smooth, get sharper
        for k in k_values:
            if time.time() > deadline - 5:
                break
                
            try:
                res = minimize(
                    objective_smoothed,
                    current_h,
                    args=(k,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-5}
                )
                if res.success or res.fun < 0:
                    current_h = res.x
            except Exception:
                pass
        
        # Polish with Nelder-Mead on the exact objective
        # NM is good for the final non-smooth steps
        if time.time() < deadline - 2:
            try:
                res_nm = minimize(
                    lambda x: -compute_value(x),
                    current_h,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5}
                )
                if res_nm.success or res_nm.fun < 0:
                    current_h = res_nm.x
            except Exception:
                pass
        
        # Final check
        current_h = np.maximum(current_h, 0)
        val = compute_value(current_h)
        if val > best_val:
            best_val = val
            best_h = current_h
            
    return best_h.tolist()

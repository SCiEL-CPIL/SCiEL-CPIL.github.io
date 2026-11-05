import numpy as np
import time

N = 400

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin
    
    best_value = np.inf
    best_h = np.ones(N).astype(float)
    
    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid

    val_seed, _ = evaluate(best_h)
    best_value = val_seed
    
    def normalize(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            h += np.random.normal(0, 1e-6, N)
            s = np.sum(h)
        return h / s

    def smooth_grad(h, beta):
        """
        Compute gradient of log-sum-exp(beta * conv(h,h))
        """
        conv = np.convolve(h, h)
        # Log-sum-exp for stability
        max_c = np.max(conv)
        exp_conv = np.exp(beta * (conv - max_c))
        sum_exp = np.sum(exp_conv)
        w = exp_conv / sum_exp
        
        # g_m = 2 * sum_k w_k h_{k-m}
        g = np.zeros(N)
        # Vectorized calculation
        # k ranges 0 to 2N-2
        # m ranges 0 to N-1
        # We need h_{k-m} where 0 <= k-m < N
        # Let's iterate over m for clarity, or use advanced indexing
        # For N=400, a loop is fine
        for m in range(N):
            # k-m must be in [0, N-1]
            # So k in [m, m+N-1]
            k_start = m
            k_end = m + N
            # conv indices are 0 to 2N-2
            # So k_end might exceed 2N-2
            k_end = min(k_end, 2*N - 1)
            if k_start <= k_end:
                h_indices = np.arange(k_start, k_end) - m
                # w indices are k_start to k_end-1
                w_segment = w[k_start:k_end]
                h_segment = h[h_indices]
                g[m] = 2.0 * np.dot(w_segment, h_segment)
                
        return g

    def optimize_from(h_init, max_iter=2000):
        h = h_init.copy().astype(float)
        h = normalize(h)
        
        eta = 0.01
        beta_start = 10.0
        beta_end = 1000.0
        
        best_local_val = np.inf
        best_local_h = h.copy()
        
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE:
                break
            
            # Anneal beta
            progress = it / max_iter
            beta = beta_start * (beta_end / beta_start) ** progress
            
            # Compute gradient
            g = smooth_grad(h, beta)
            
            # Project gradient to be orthogonal to 1s?
            # We project h after update, so we can just update
            # However, the component of g along 1s will be removed by normalization
            # To be efficient, we can remove the mean of g
            g = g - np.mean(g)
            
            h = h - eta * g
            h = normalize(h)
            
            # Evaluate
            val, valid = evaluate(h)
            if valid and val < best_local_val:
                best_local_val = val
                best_local_h = h.copy()
                
            # Adaptive step size
            # If val increased, reduce eta
            # Simple heuristic: if beta is high, use smaller eta
            if it % 100 == 0:
                eta = 0.01 * (1.0 / (1.0 + progress))
                
        return best_local_h, best_local_val

    # Initializations
    rng = np.random.RandomState(42)
    inits = []
    
    # Random
    for _ in range(10):
        inits.append(rng.uniform(-1, 1, N))
        
    # Bimodal
    for _ in range(5):
        h = np.zeros(N)
        h[:N//2] = 1
        h[N//2:] = -1
        h += rng.normal(0, 0.5, N)
        inits.append(h)
        
    # Chirp
    t = np.arange(N)
    for chirp_rate in [1, 5, 10]:
        h = np.cos(2 * np.pi * chirp_rate * t**2 / N)
        inits.append(h)
        
    # Run optimization
    for h_init in inits:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        h_opt, val_opt = optimize_from(h_init, max_iter=2000)
        
        val_final, valid = evaluate(h_opt)
        if valid and val_final < best_value:
            best_value = val_final
            best_h = h_opt.copy()
            
    # Final check
    val_final, valid = evaluate(best_h)
    if not valid:
        best_h = np.ones(N)
        best_value = 2.0
    else:
        best_value = val_final
        
    return best_h

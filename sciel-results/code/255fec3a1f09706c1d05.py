import numpy as np

N = 400

def construct():
    # Use a deterministic seed for reproducibility if needed, 
    # but random is fine.
    np.random.seed(42)
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # Initial random guess
    h = np.random.randn(N)
    s = np.sum(h)
    if s == 0:
        h = np.ones(N)
    else:
        h = h / s
    
    # Local search
    import time
    start_time = time.time()
    timeout = 100  # 100 seconds budget
    
    for _ in range(100000): # Max iterations
        if time.time() - start_time > timeout:
            break
            
        # Compute current value
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2 * N * max_conv / (np.sum(h)**2))
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
        # Perturb
        idx = np.random.randint(N)
        h[idx] += np.random.randn() * 0.1
        
        # Re-normalize
        s = np.sum(h)
        if s == 0:
            h = best_h.copy()
            continue
        h = h / s
        
    return best_h

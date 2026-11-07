import numpy as np
import time

N = 50

def compute_value(h):
    h = np.array(h)
    if np.any(h < 0) or np.any(~np.isfinite(h)):
        return 0.0
    
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    
    # w is spacing
    # Points are linspace(-0.5, 0.5, m+2)
    # m+2 points, so m+1 intervals.
    # Spacing w = 1.0 / (m+1)
    w = 1.0 / (m + 1)
    
    # l2 = sum (w/3) * (y[i]^2 + y[i]y[i+1] + y[i+1]^2) for i in 0..m
    # y has length m+2. Indices 0..m+1.
    # The sum is over i=0 to m.
    # Terms involve y[i] and y[i+1].
    
    # Vectorized calculation
    # y[0..m] and y[1..m+1]
    y1 = y[:m+1]
    y2 = y[1:m+2]
    
    terms = y1**2 + y1*y2 + y2**2
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf == 0:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def construct():
    candidates = []
    
    # 1. Seed
    candidates.append(np.ones(N))
    
    # 2. Single spike in center
    h = np.zeros(N)
    h[N//2] = 1.0
    candidates.append(h)
    
    # 3. Two spikes in center
    h = np.zeros(N)
    h[N//2 - 1] = 1.0
    h[N//2] = 1.0
    candidates.append(h)
    
    # 4. Cosine window
    x = np.linspace(-np.pi/2, np.pi/2, N)
    candidates.append(np.cos(x))
    
    # 5. Gaussian
    x = np.linspace(-2, 2, N)
    candidates.append(np.exp(-x**2))
    
    # 6. Triangular window
    x = np.linspace(-1, 1, N)
    candidates.append(np.maximum(0, 1 - np.abs(x)))
    
    # 7. Uniform but with edges cut?
    # 8. Random search with a few iterations?
    # Let's do a simple local improvement on the best candidate.
    
    best_h = candidates[0]
    best_val = compute_value(best_h)
    
    for c in candidates[1:]:
        v = compute_value(c)
        if v > best_val:
            best_val = v
            best_h = c
            
    # Simple coordinate ascent / local search
    # The function is non-smooth due to max, but we can use sub-gradients or just random perturbations.
    # Given time constraints, a few random perturbations on the best candidate might help.
    
    start_time = time.time()
    timeout = 100.0 # Safety margin
    
    current_h = best_h.copy()
    current_val = best_val
    
    # Try random perturbations
    rng = np.random.default_rng(50)
    
    # We can try to "sharpen" the peak or "spread" it.
    # Since we don't know the landscape, let's try a few structured variations.
    
    # Variation: Multiply by a power of a cosine window?
    for p in [0.5, 1.0, 1.5, 2.0, 3.0]:
        h_new = np.cos(np.linspace(-np.pi/2, np.pi/2, N))**p
        v = compute_value(h_new)
        if v > current_val:
            current_val = v
            current_h = h_new
            
    # Variation: Shifted peaks?
    # If the optimal solution is symmetric, it should be centered.
    
    # Let's try a simple gradient-free local search with random steps
    # Since N=50 is small, we can do a few steps.
    
    step_size = 0.1
    for _ in range(5000):
        if time.time() - start_time > timeout:
            break
        h_new = current_h + rng.normal(0, step_size, N)
        h_new = np.maximum(0, h_new)
        v = compute_value(h_new)
        if v > current_val:
            current_val = v
            current_h = h_new
            step_size *= 0.9 # Shrink step if improved
        else:
            step_size = min(step_size * 1.1, 0.5) # Expand if not
            
    return list(current_h)

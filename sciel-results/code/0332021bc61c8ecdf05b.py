import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def _compute_value(h):
    """Compute the objective value for given step heights."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return -1.0
    h = np.clip(h, 0.0, None)
    if np.all(h == 0):
        return -1.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(terms)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    if norm_1 * norm_inf == 0:
        return -1.0
    return l2 / (norm_1 * norm_inf)

def _local_search(h, deadline, max_iters=50000):
    """Local search with random perturbations and coordinate descent."""
    h = np.array(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    best_val = _compute_value(h)
    best_h = h.copy()
    
    rng = np.random.default_rng(42)
    step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try different perturbation strategies
        strategy = rng.random()
        
        if strategy < 0.3:
            # Coordinate descent: adjust one element
            i = rng.integers(N)
            delta = rng.normal(0, 0.1)
            h_new = h.copy()
            h_new[i] = max(0, h[i] + delta)
        elif strategy < 0.5:
            # Gaussian perturbation on random subset
            mask = rng.random(N) < 0.3
            h_new = h.copy()
            h_new[mask] += rng.normal(0, 0.1, mask.sum())
            h_new = np.clip(h_new, 0, None)
        elif strategy < 0.7:
            # Scale random subset
            mask = rng.random(N) < 0.2
            h_new = h.copy()
            h_new[mask] *= rng.uniform(0.5, 1.5, mask.sum())
            h_new = np.clip(h_new, 0, None)
        else:
            # Larger random step
            h_new = h + rng.normal(0, 0.3, N)
            h_new = np.clip(h_new, 0, None)
        
        val_new = _compute_value(h_new)
        if val_new > best_val + 1e-12:
            best_val = val_new
            best_h = h_new.copy()
            h = h_new
        else:
            # Occasionally accept slightly worse to escape local optima
            if rng.random() < 0.01 and val_new > -1.0:
                h = h_new
    
    return best_h, best_val

def _init_concentrated(center, width):
    """Create a concentrated bump."""
    h = np.zeros(N)
    idx = np.arange(N)
    x = (idx - center) / width
    h = np.exp(-0.5 * x**2)
    h = np.clip(h, 0, None)
    return h

def _init_hump():
    """Create a hump-shaped initial guess."""
    x = np.linspace(-1, 1, N)
    h = np.maximum(0, 1 - np.abs(x))
    return h

def _init_spiky():
    """Create a spiky initial guess with few nonzeros."""
    h = np.zeros(N)
    n_spikes = rng.integers(3, 8)
    positions = rng.choice(N, size=n_spikes, replace=False)
    h[positions] = rng.uniform(0.5, 1.0, n_spikes)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(50)
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    # Generate diverse initializations
    inits = []
    
    # Hump
    inits.append(_init_hump())
    
    # Concentrated bumps at different positions and widths
    for center in [N//4, N//2, 3*N//4]:
        for width in [3, 5, 8, 12]:
            inits.append(_init_concentrated(center, width))
    
    # Spiky
    for _ in range(5):
        inits.append(_init_spiky())
    
    # Uniform (seed)
    inits.append(np.ones(N))
    
    # Random smooth
    for _ in range(5):
        h_rand = rng.uniform(0, 1, N)
        # Smooth it
        kernel = np.ones(5) / 5
        h_smooth = np.convolve(h_rand, kernel, mode='same')
        inits.append(np.clip(h_smooth, 0, None))
    
    # Local search from each initialization
    for h_init in inits:
        if time.time() > deadline:
            break
        
        h_res, val_res = _local_search(h_init, deadline)
        if val_res > best_val:
            best_val = val_res
            best_h = h_res
    
    # Final polish with more iterations on the best
    if time.time() < deadline - 5:
        h_res, val_res = _local_search(best_h, deadline, max_iters=100000)
        if val_res > best_val:
            best_val = val_res
            best_h = h_res
    
    # Ensure nonnegative and finite
    best_h = np.clip(best_h, 0, None)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1.0, neginf=0.0)
    
    return list(best_h)
# EVOLVE-BLOCK-END


"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def compute_value(h):
    """Compute the value metric for given heights."""
    h = np.array(h, dtype=float)
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    value = l2 / (norm_1 * norm_inf)
    return value

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Start with seed
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Precompute indices for Gaussian
    x = np.linspace(-1, 1, N)
    
    # Multi-start local search over a smooth parameterization
    # Parameters: center c, width s, type t (0=gaussian, 1=cosine)
    # We'll do a grid of starts and then local refinement
    
    starts = []
    # Gaussian centers and widths
    for c in np.linspace(-0.8, 0.8, 15):
        for s in np.linspace(0.2, 1.2, 10):
            starts.append(('gauss', c, s))
    # Cosine bumps
    for c in np.linspace(-0.8, 0.8, 10):
        for s in np.linspace(0.3, 1.0, 8):
            starts.append(('cos', c, s))
    
    def eval_params(p):
        kind, c, s = p
        if kind == 'gauss':
            h = np.exp(-((x - c)**2) / (2 * s**2))
        else:
            # Cosine bump centered at c, width s
            # Support where |x-c| < s
            h = np.where(np.abs(x - c) < s, 0.5 * (1 + np.cos(np.pi * (x - c) / s)), 0.0)
        h = np.maximum(0, h)
        if np.all(h == 0):
            return 0.0, h
        return compute_value(h), h
    
    # Evaluate all starts
    for p in starts:
        if time.time() > deadline:
            break
        val, h = eval_params(p)
        if val > best_val:
            best_val = val
            best_h = h
    
    # Local refinement on the best found so far
    # Use a simple coordinate descent on the parameters of the best type
    # Or just do a local random search around the best h
    
    if time.time() < deadline - 5:
        # Local search around best_h
        step = 0.05
        for it in range(5000):
            if time.time() > deadline:
                break
            # Random direction
            d = np.random.randn(N)
            d /= np.linalg.norm(d)
            new_h = best_h + step * d
            new_h = np.maximum(0, new_h)
            val = compute_value(new_h)
            if val > best_val:
                best_val = val
                best_h = new_h
                step *= 1.05
            else:
                step *= 0.95
    
    return best_h.tolist()
# EVOLVE-BLOCK-END

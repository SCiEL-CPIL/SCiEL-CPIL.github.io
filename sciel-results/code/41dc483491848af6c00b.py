import numpy as np
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave margin before 150s kill

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for step function on [-1/4, 1/4]."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15):
        return 0.0
    h = np.maximum(h, 0.0)
    dx = 1.0 / N  # interval width
    l1 = dx * np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = dx * np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    conv_scaled = conv * dx
    conv_inf = np.max(conv_scaled)
    if conv_inf <= 0:
        return 0.0
    return l2_sq / (l1 * conv_inf)

def compute_ratio_normalized(h):
    """Compute ratio with l1 normalized to 1, for easier optimization."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    conv_inf = np.max(conv)
    if conv_inf <= 0:
        return 0.0
    # Ratio = (dx * l2_sq) / ((dx * l1) * (dx * conv_inf)) = l2_sq / (dx * l1 * conv_inf)
    # With dx = 1/N: ratio = l2_sq / ((1/N) * l1 * conv_inf) = N * l2_sq / (l1 * conv_inf)
    dx = 1.0 / N
    return l2_sq / (dx * l1 * conv_inf)

def random_sidon_like(n_ones, seed=None):
    """Generate a random binary vector with n_ones ones, trying to minimize autocorrelation."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    best_h = None
    best_ratio = -1
    
    for _ in range(200):
        h = np.zeros(N)
        positions = rng.choice(N, size=n_ones, replace=False)
        h[positions] = 1.0
        ratio = compute_ratio_normalized(h)
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = h.copy()
    
    return best_h, best_ratio

def optimize_heights(h0, max_iter=5000, lr=0.01):
    """Optimize heights using gradient-free local search (simulated annealing style)."""
    h = h0.copy().astype(float)
    best_h = h.copy()
    best_ratio = compute_ratio_normalized(h)
    current_ratio = best_ratio
    temp = 1.0
    
    for i in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        # Random perturbation
        h_new = h.copy()
        idx = np.random.randint(N)
        delta = np.random.uniform(-0.5, 0.5)
        h_new[idx] = max(0, h_new[idx] + delta)
        
        new_ratio = compute_ratio_normalized(h_new)
        
        if new_ratio > current_ratio or np.random.random() < np.exp((new_ratio - current_ratio) / temp):
            h = h_new
            current_ratio = new_ratio
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_h = h.copy()
        
        temp *= 0.999
    
    return best_h, best_ratio

def construct():
    """
    Maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf) using Sidon-set-inspired binary patterns
    followed by continuous optimization.
    """
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    # Try different numbers of nonzero entries
    for n_ones in range(1, 15):
        # Generate several random Sidon-like candidates
        for seed in range(10):
            h_cand, ratio = random_sidon_like(n_ones, seed=seed)
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h_cand.copy()
        
        if time.time() - START_TIME > TIME_LIMIT - 30:
            break
    
    # Refine the best candidate with continuous optimization
    if time.time() - START_TIME < TIME_LIMIT - 20:
        h_opt, ratio_opt = optimize_heights(best_h, max_iter=3000)
        if ratio_opt > best_ratio:
            best_ratio = ratio_opt
            best_h = h_opt
    
    # Also try a few more random starts with optimization
    for _ in range(5):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        n_ones = np.random.randint(3, 12)
        h_cand, _ = random_sidon_like(n_ones, seed=None)
        h_opt, ratio_opt = optimize_heights(h_cand, max_iter=2000)
        if ratio_opt > best_ratio:
            best_ratio = ratio_opt
            best_h = h_opt
    
    return [float(x) for x in best_h]

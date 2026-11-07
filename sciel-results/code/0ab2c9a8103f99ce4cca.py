"""Rewritten program: optimize h to minimize max autocorrelation with (1-h)."""
import time
import numpy as np

N = 100
TIME_LIMIT = 120.0
SAFETY_MARGIN = 15.0  # seconds to leave before kill

def _compute_value(h):
    """Compute the objective value: max(np.correlate(h, 1-h, mode='full')) / N * 2."""
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / N * 2.0

def _is_valid(h):
    """Check if h satisfies all constraints."""
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-12) or np.any(h > 1.0 + 1e-12):
        return False
    if abs(np.sum(h) - N / 2.0) > 1e-9 * N:
        return False
    return True

def _project_to_feasible(h):
    """Project h onto feasible region: sum(h)=N/2, 0<=h<=1."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        for _ in range(200):
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
            diff = target - np.sum(h)
            if abs(diff) < 1e-14:
                break
    return h

def _corr_grad(h):
    """
    Compute gradient of correlation vector w.r.t. h.
    Returns array of shape (N, 2*N-1) where grad[i,k] = d corr[k] / d h[i].
    
    For np.correlate(h, 1-h, mode='full'):
    corr[k] = sum_j h[j] * (1 - h[j - (k - (N-1))])
    where valid indices satisfy 0 <= j < N and 0 <= j - (k-(N-1)) < N.
    
    d corr[k] / d h[i] has two contributions:
    1. From h[i] term: (1 - h[i - offset]) if 0 <= i - offset < N, where offset = k - (N-1)
    2. From (1-h[i]) term: -h[j] if j - offset = i, i.e., j = i + offset, and 0 <= j < N
    """
    # offsets for each k in 0..2*N-2
    offsets = np.arange(2 * N - 1) - (N - 1)  # shape (2*N-1,)
    
    # For each i, compute d corr[k]/d h[i] for all k
    # Vectorized: for each i, we need to check which offsets make i-offset and i+offset valid
    # Let's build this efficiently
    
    grad = np.zeros((N, 2 * N - 1))
    
    for i in range(N):
        # Term 1: (1 - h[i - offset]) for valid offsets where 0 <= i - offset < N
        # i - offset >= 0 => offset <= i
        # i - offset < N => offset > i - N (always true since offset >= -(N-1) and i < N)
        # So valid offsets: -(N-1) <= offset <= i
        mask1 = offsets <= i
        # Also need i - offset >= 0, which is offset <= i (already checked)
        # And i - offset < N, which is offset > i - N
        mask1 = mask1 & (offsets > i - N)
        
        if np.any(mask1):
            valid_offsets = offsets[mask1]
            indices = i - valid_offsets
            grad[i, mask1] = (1.0 - h[indices])
        
        # Term 2: -h[i + offset] for valid offsets where 0 <= i + offset < N
        # i + offset >= 0 => offset >= -i
        # i + offset < N => offset < N - i
        mask2 = (offsets >= -i) & (offsets < N - i)
        
        if np.any(mask2):
            valid_offsets = offsets[mask2]
            indices = i + valid_offsets
            grad[i, mask2] += (-1.0) * h[indices]
    
    return grad

def _smooth_max_grad(corr, T):
    """Compute softmax weights for smooth max."""
    m = np.max(corr)
    shifted = corr - m
    weights = np.exp(shifted / T)
    weights /= np.sum(weights)
    return weights

def _gradient_descent(h0, lr=0.05, max_iter=2000, T=0.05, tol=1e-12):
    """
    Gradient descent with projection to minimize max correlation.
    Uses smooth max via softmax for differentiable objective.
    """
    h = h0.copy()
    best_h = h.copy()
    best_val = _compute_value(h)
    
    # Precompute correlation gradient structure
    # Actually, we'll compute it on the fly since h changes
    
    for it in range(max_iter):
        # Compute correlation
        corr = np.correlate(h, 1.0 - h, mode="full")
        
        # Smooth max weights
        weights = _smooth_max_grad(corr, T)
        
        # Compute gradient: grad_value[i] = sum_k weights[k] * d corr[k] / d h[i] / N * 2
        # Vectorized computation
        # d corr[k] / d h[i] for all i,k
        # Use the analytical formula
        
        offsets = np.arange(2 * N - 1) - (N - 1)
        
        grad_val = np.zeros(N)
        
        # Term 1: sum_k weights[k] * (1 - h[i - offsets[k]]) for valid k
        # Term 2: sum_k weights[k] * (-h[i + offsets[k]]) for valid k
        
        for i in range(N):
            # Term 1
            mask1 = (offsets <= i) & (offsets > i - N)
            if np.any(mask1):
                idx1 = i - offsets[mask1]
                grad_val[i] += np.sum(weights[mask1] * (1.0 - h[idx1]))
            
            # Term 2
            mask2 = (offsets >= -i) & (offsets < N - i)
            if np.any(mask2):
                idx2 = i + offsets[mask2]
                grad_val[i] += np.sum(weights[mask2] * (-1.0) * h[idx2])
        
        grad_val *= 2.0 / N
        
        # Take gradient step
        h_new = h - lr * grad_val
        
        # Project to feasible region
        h_new = _project_to_feasible(h_new)
        
        val = _compute_value(h_new)
        if val < best_val - tol:
            best_val = val
            best_h = h_new.copy()
            h = h_new
        else:
            lr *= 0.98
            if lr < 1e-12:
                break
    
    return best_h, best_val

def _sinusoidal_start(freq, phase):
    """Generate a sinusoidal starting point."""
    t = np.linspace(0, 2 * np.pi * freq, N)
    h = 0.5 + 0.3 * np.sin(t + phase)
    return _project_to_feasible(h)

def construct():
    """Construct 100 step heights minimizing the correlation objective."""
    start_time = time.time()
    
    # Initial seed
    h_seed = np.full(N, 0.5)
    best_h = h_seed.copy()
    best_val = _compute_value(h_seed)  # Should be 0.5
    
    # Try multiple sinusoidal starting points with different frequencies and phases
    start_configs = []
    for freq in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
        for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
            start_configs.append((freq, phase))
    
    # Also add some random starts
    rng = np.random.RandomState(42)
    for _ in range(10):
        h_rand = rng.uniform(0, 1, N)
        h_rand = _project_to_feasible(h_rand)
        start_configs.append(('random', h_rand))
    
    for config in start_configs:
        # Check time
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        if isinstance(config[0], str) and config[0] == 'random':
            h0 = config[1]
        else:
            freq, phase = config
            h0 = _sinusoidal_start(freq, phase)
        
        # Run gradient descent with multiple temperatures
        for T in [0.1, 0.05, 0.02]:
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            h_opt, val_opt = _gradient_descent(h0, lr=0.05, max_iter=1500, T=T)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt
        
        # Check time
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
    
    # Final polish: run with very small learning rate and small T
    if time.time() - start_time < TIME_LIMIT - SAFETY_MARGIN:
        h_polish, val_polish = _gradient_descent(best_h, lr=0.005, max_iter=1000, T=0.01)
        if val_polish < best_val:
            best_val = val_polish
            best_h = h_polish
    
    # Ensure validity
    best_h = _project_to_feasible(best_h)
    
    return best_h

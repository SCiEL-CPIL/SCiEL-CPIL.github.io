import numpy as np

N = 400

def compute_value(h):
    """Compute the objective value for a given height vector."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    c = np.convolve(h, h)
    return abs(2.0 * N * np.max(c) / (s ** 2))

def compute_gradient_smooth(h, beta):
    """Compute gradient of smooth max (logsumexp) of autoconvolution."""
    c = np.convolve(h, h)
    max_c = np.max(c)
    shifted = c - max_c
    # Clip for numerical stability
    shifted = np.clip(shifted, -50, 50)
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    
    # Gradient: d/dh_i sum_k w_k * c_k = 2 * sum_k w_k * h_{k-i} (for valid k)
    # This is 2 * cross-correlation of weights and h
    L = 2 * N
    W = np.fft.fft(weights, n=L)
    H = np.fft.fft(h, n=L)
    grad = 2.0 * np.fft.ifft(W * np.conj(H), n=L).real[:N]
    
    return grad, c

def line_search(h, grad, eta, max_iter=10):
    """Backtracking line search to ensure descent."""
    h_current = h.copy()
    val_current = compute_value(h_current)
    c_current = np.convolve(h_current, h_current)
    max_c_current = np.max(c_current)
    
    for k in range(max_iter):
        h_try = h_current - eta * grad
        s_try = np.sum(h_try)
        if abs(s_try) < 1e-12:
            eta *= 0.5
            continue
        h_try = h_try / s_try
        
        if np.any(np.abs(h_try) > 50.0):
            eta *= 0.5
            continue
        
        val_try = compute_value(h_try)
        if val_try < val_current - 1e-10:
            return h_try, eta
        eta *= 0.5
    
    return h_current, eta

def gradient_descent(h_init, max_iter=500, verbose=False):
    """Projected gradient descent with line search and adaptive beta."""
    h = h_init.copy()
    s = np.sum(h)
    h = h / s
    
    best_h = h.copy()
    best_val = compute_value(h)
    
    eta = 0.1
    momentum = np.zeros(N)
    mu = 0.8
    
    for i in range(max_iter):
        # Adaptive beta: start low, increase over time
        beta = 3.0 + 12.0 * (i / max_iter) ** 0.5
        
        grad, c = compute_gradient_smooth(h, beta)
        
        # Project gradient to maintain sum constraint
        grad -= np.mean(grad)
        
        # Momentum step
        momentum = mu * momentum - eta * grad
        h_new = h + momentum
        
        # Line search
        h_new, eta = line_search(h, momentum, eta)
        
        # Project to sum=1
        s_new = np.sum(h_new)
        if abs(s_new) < 1e-12:
            break
        h_new = h_new / s_new
        
        # Check for extreme values
        if np.any(np.abs(h_new) > 50.0):
            eta *= 0.5
            continue
        
        h = h_new
        val = compute_value(h)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Early stopping if converged
        if i > 100 and best_val < 1.01:
            break
        
        # Adaptive learning rate adjustment
        if i % 100 == 0 and i > 0:
            if best_val > 1.05:
                eta = min(0.2, eta * 1.2)
            else:
                eta = max(0.01, eta * 0.8)
        
        if verbose and i % 100 == 0:
            print(f"Iter {i}: val={best_val:.6f}, eta={eta:.4f}, beta={beta:.2f}")
    
    return best_h, best_val

def spectral_init(frequency, amplitude, phase=0):
    """Create initialization based on spectral design."""
    k = np.arange(N)
    h = np.ones(N)
    h += amplitude * np.cos(2 * np.pi * frequency * k / N + phase)
    h = h / np.sum(h)
    return h

def random_init(seed, noise_level=0.3):
    """Create random initialization with zero-mean perturbation."""
    rng = np.random.default_rng(seed)
    h = np.ones(N) + noise_level * rng.standard_normal(N)
    h -= np.mean(h)
    h = 1.0 + 0.5 * h / (np.std(h) + 1e-12)
    h = h / np.sum(h)
    return h

def construct():
    """Construct optimal step heights."""
    best_h = None
    best_val = np.inf
    
    # Multiple initializations
    inits = []
    
    # Spectral inits at different frequencies
    for freq in [1, 2, 3, 5, 7, 10]:
        for amp in [0.2, 0.4, 0.6]:
            for phase in [0, np.pi/2, np.pi]:
                inits.append(spectral_init(freq, amp, phase))
    
    # Random inits
    for seed in [42, 123, 456, 789, 1011]:
        for noise in [0.2, 0.4, 0.6]:
            inits.append(random_init(seed, noise))
    
    # Run gradient descent from each init, keep best
    for idx, h_init in enumerate(inits):
        h, val = gradient_descent(h_init, max_iter=300, verbose=False)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Early stop if we're doing well
        if best_val < 1.015:
            break
    
    # Final polish with more iterations
    if best_h is not None:
        h, val = gradient_descent(best_h, max_iter=200, verbose=False)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h

import numpy as np
import time

N = 400
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def _autoconv(x):
    """Compute full autoconvolution of x (length N) -> length 2N-1."""
    n = 2 * N
    X = np.fft.fft(x, n=n)
    C = np.fft.ifft(X * X).real[:2*N-1]
    return C

def _eval_ratio(x):
    """Compute the raw value: max|c_k| / (sum x)^2 for k != 0."""
    c = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    if len(c) > 1:
        max_sidelobe = np.max(np.abs(c[1:]))
    else:
        max_sidelobe = 0.0
    return max_sidelobe / (s * s)

def _soft_max_abs(c, k=20.0):
    """Smooth approximation of max|c_k| for k != 0."""
    abs_c = np.abs(c[1:])
    max_val = np.max(abs_c)
    if max_val < 1e-15:
        return 0.0
    # Shift to avoid overflow
    shifted = abs_c - max_val
    return max_val + np.log(np.sum(np.exp(k * shifted))) / k

def _objective_smooth(x, k=20.0):
    """Smooth objective: soft_max(|c_k|, k!=0) / S^2."""
    c = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    return _soft_max_abs(c, k) / (s * s)

def _gradient_descent_smooth(x0, max_iters=500, lr=0.001, deadline=None):
    """Gradient descent on smooth surrogate objective."""
    x = x0.copy()
    best_x = x.copy()
    best_val = _eval_ratio(x)
    
    for i in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient numerically (small perturbation)
        eps = 1e-4
        grad = np.zeros_like(x)
        for j in range(0, N, 4):  # Sample gradient at every 4th point for speed
            xp = x.copy()
            xm = x.copy()
            xp[j] += eps
            xm[j] -= eps
            fp = _objective_smooth(xp, k=10.0)
            fm = _objective_smooth(xm, k=10.0)
            grad[j] = (fp - fm) / (2 * eps)
            # Spread gradient to neighbors for smoother updates
            if j > 0:
                grad[j-1] = 0.5 * grad[j]
            if j < N-1:
                grad[j+1] = 0.5 * grad[j]
        
        # Update
        x -= lr * grad
        
        # Evaluate actual metric
        val = _eval_ratio(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        # Adaptive learning rate
        if i % 50 == 49 and i > 0:
            lr *= 0.8
    
    return best_x, best_val

def _local_polish(x, max_iters=200, deadline=None):
    """Local search: try flipping individual elements to improve metric."""
    x = x.copy()
    best_val = _eval_ratio(x)
    
    for i in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        # Try random subset of positions
        indices = np.random.choice(N, size=min(50, N), replace=False)
        for idx in indices:
            x_old = x[idx]
            # Try a few candidate values
            for candidate in [x_old * 0.5, x_old * 1.5, -x_old, x_old + 0.1, x_old - 0.1]:
                x[idx] = candidate
                val = _eval_ratio(x)
                if val < best_val:
                    best_val = val
                    improved = True
                    break
                x[idx] = x_old
            if improved:
                break
        
        if not improved:
            break
    
    return x, best_val

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Initialize with constant vector
    x = np.ones(N)
    best_val = _eval_ratio(x)
    best_result = x.copy()
    
    np.random.seed(42)
    
    # Phase 1: Continuous relaxation via gradient descent on smooth surrogate
    # Start from several different initializations
    initializations = [
        np.ones(N),
        np.cos(np.pi * np.arange(N) ** 2 / N),
        np.random.randn(N) + 1.0,
        np.sin(2 * np.pi * np.arange(N) / N) + 0.5,
    ]
    
    for x_init in initializations:
        if time.time() > deadline:
            break
        
        x_gd, val_gd = _gradient_descent_smooth(x_init, max_iters=300, lr=0.001, deadline=deadline)
        if val_gd < best_val:
            best_val = val_gd
            best_result = x_gd.copy()
        
        # Phase 2: Local polish on the result
        x_polished, val_polished = _local_polish(x_gd, max_iters=100, deadline=deadline)
        if val_polished < best_val:
            best_val = val_polished
            best_result = x_polished.copy()
    
    # Phase 3: Try structured sequences as fallback
    t = np.arange(N)
    structured = [
        np.cos(np.pi * t**2 / N),
        np.sin(2 * np.pi * t / N) + 0.5,
        np.random.choice([-1, 1], N) + 1.0,
    ]
    
    for x_struct in structured:
        if time.time() > deadline:
            break
        val = _eval_ratio(x_struct)
        if val < best_val:
            best_val = val
            best_result = x_struct.copy()
    
    return best_result

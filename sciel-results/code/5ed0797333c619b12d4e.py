import numpy as np
import time

N = 50

def compute_score(h):
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    c = np.correlate(h, h, mode='full')
    sup_c = np.max(c)
    if sup_c < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_c)

def _autoconv(h):
    """Compute full autoconvolution (correlation)."""
    return np.correlate(h, h, mode='full')

def _smooth_max(c, temp):
    """Smooth max using log-sum-exp."""
    c_max = np.max(c)
    if c_max < 1e-30:
        return 0.0, np.zeros_like(c)
    shifted = (c - c_max) / temp
    shifted = np.clip(shifted, -50, 0)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    smooth_val = c_max + temp * np.log(np.sum(weights))
    return smooth_val, weights

def _gradient_ratio(h, temp=0.01):
    """
    Compute gradient of log(R) w.r.t. log(h_i) where R = ||h||_2^2 / (||h||_1 * max_k c_k).
    Using smooth max surrogate.
    
    d log(R) / d log(h_i) = 2*h_i^2/||h||_2^2 - h_i/||h||_1 - (h_i/smooth_max) * sum_k w_k * (dc_k/dh_i)/h_i
    """
    h = np.maximum(h, 1e-10)
    c = _autoconv(h)
    smooth_max, weights = _smooth_max(c, temp)
    if smooth_max < 1e-30:
        return np.zeros(N)
    
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    
    # Compute dc_k/dh_i for all i, using the weights
    # c[m] = sum_j h[j] * h[j - m + N-1] for m in [0, 2N-2]
    # dc[m]/dh[i] = h[i - m + N-1] if 0<=i-m+N-1<N, plus h[i + m - (N-1)] if 0<=i+m-(N-1)<N
    # More carefully: for lag L = m - (N-1):
    #   dc/dh[i] = h[i-L] (if 0<=i-L<N) + h[i+L] (if 0<=i+L<N), with factor 2 if L=0
    
    pressure = np.zeros(N)
    num_lags = 2 * N - 1
    for m in range(num_lags):
        w = weights[m]
        if w < 1e-10:
            continue
        L = m - (N - 1)
        if L == 0:
            pressure += (2.0 * w / smooth_max) * h
        elif L > 0:
            # dc/dh[i] = h[i-L] if i>=L, plus h[i+L] if i<N-L
            for i in range(N):
                contrib = 0.0
                if i - L >= 0:
                    contrib += h[i - L]
                if i + L < N:
                    contrib += h[i + L]
                pressure[i] += w * contrib / smooth_max
        else:
            Lp = -L
            for i in range(N):
                contrib = 0.0
                if i + Lp < N:
                    contrib += h[i + Lp]
                if i - Lp >= 0:
                    contrib += h[i - Lp]
                pressure[i] += w * contrib / smooth_max
    
    # log-derivative
    log_deriv = 2.0 * h**2 / l2_sq - h / l1 - pressure
    return log_deriv

def _smooth_optimize(h_init, n_iter=2000, temp_start=0.1, temp_end=0.001, deadline=None):
    """
    Smooth optimization using multiplicative updates with annealed temperature.
    """
    h = np.asarray(h_init, dtype=float).copy()
    h = np.maximum(h, 1e-10)
    
    best_local = 0.0
    best_local_h = h.copy()
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        # Anneal temperature
        frac = it / max(1, n_iter - 1)
        temp = temp_start * (temp_end / temp_start) ** frac
        
        log_deriv = _gradient_ratio(h, temp)
        
        # Multiplicative update with adaptive step size
        eta = 0.3 * (1.0 - 0.5 * frac)  # decrease step size
        h = h * np.exp(eta * log_deriv)
        h = np.maximum(h, 1e-10)
        
        # Normalize to avoid overflow/underflow
        h_mean = np.mean(h)
        if h_mean > 0:
            h = h / h_mean * 1.0
        
        s = compute_score(h)
        if s > best_local:
            best_local = s
            best_local_h = h.copy()
    
    return best_local_h, best_local

def _coordinate_descent(h, max_iter=100, deadline=None):
    """
    Coordinate descent with golden-section search on each coordinate.
    """
    h = h.copy()
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
        improved = False
        for i in range(N):
            old = h[i]
            best_val = compute_score(h)
            best_x = old
            
            # Try a range of values around old
            # Scale-relative perturbations
            scale = max(old, 0.01)
            for factor in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 3.0, 5.0]:
                new_val = old * factor
                h[i] = new_val
                s = compute_score(h)
                if s > best_val:
                    best_val = s
                    best_x = new_val
            
            # Also try absolute perturbations
            for delta in [-0.5, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.5, 1.0]:
                new_val = max(0.0, old + delta)
                h[i] = new_val
                s = compute_score(h)
                if s > best_val:
                    best_val = s
                    best_x = new_val
            
            if best_x != old:
                h[i] = best_x
                improved = True
            else:
                h[i] = old
        
        if not improved:
            break
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    def try_candidate(h):
        nonlocal best_h, best_score
        h = np.asarray(h, dtype=float)
        if np.any(h < -1e-12):
            h = np.maximum(h, 0.0)
        s = compute_score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
        return s
    
    # Baseline: 2-element
    h2 = np.zeros(N)
    h2[0] = 2.0/3.0
    h2[1] = 1.0/3.0
    try_candidate(h2)
    
    # Baseline: uniform
    h_uni = np.ones(N) / N
    try_candidate(h_uni)
    
    # Generate diverse initializations
    structures = []
    
    # 1. Ramp functions
    for slope in [0.5, 1.0, 2.0]:
        h = np.linspace(0, slope, N)
        structures.append(h)
    
    # 2. Symmetric bump
    center = N // 2
    for width in [5, 10, 15, 20, 25]:
        h = np.zeros(N)
        for i in range(max(0, center-width), min(N, center+width+1)):
            h[i] = np.exp(-((i-center)**2) / (2*width**2))
        structures.append(h)
    
    # 3. Periodic patterns
    for period in [2, 3, 4, 5, 7, 10]:
        for phase in range(period):
            h = np.zeros(N)
            for i in range(N):
                if (i + phase) % period == 0:
                    h[i] = 1.0
            structures.append(h)
    
    # 4. Random sparse patterns
    np.random.seed(42)
    for sparsity in [0.1, 0.2, 0.3, 0.5]:
        for _ in range(5):
            h = np.zeros(N)
            n_nonzero = max(1, int(N * sparsity))
            idx = np.random.choice(N, n_nonzero, replace=False)
            h[idx] = np.random.uniform(0.1, 1.0, n_nonzero)
            structures.append(h)
    
    # 5. Power-law decay
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        h = np.zeros(N)
        for i in range(N):
            h[i] = (i+1)**(-alpha)
        structures.append(h)
    
    # 6. Two-bump structures
    for pos1 in [5, 10, 15, 20, 25]:
        for pos2 in [30, 35, 40, 45]:
            h = np.zeros(N)
            h[pos1] = 1.0
            h[pos2] = 1.0
            structures.append(h)
    
    # 7. Cosine windows
    for freq in [1, 2, 3, 4]:
        h = np.cos(np.pi * np.arange(N) * freq / N)
        h = np.maximum(h, 0)
        if np.sum(h) > 0:
            structures.append(h)
    
    # 8. Uniform with small noise
    np.random.seed(123)
    for _ in range(5):
        h = np.ones(N) * 0.5 + np.random.uniform(-0.2, 0.2, N)
        h = np.maximum(h, 0)
        structures.append(h)
    
    # 9. Sidon-like sequences (sparse with large gaps)
    for gap in [3, 4, 5]:
        h = np.zeros(N)
        idx = 0
        while idx < N:
            h[idx] = 1.0
            idx += gap
        structures.append(h)
    
    # Run smooth optimization on each structure
    for h_init in structures:
        if time.time() > deadline - 20:
            break
        h_init = np.asarray(h_init, dtype=float)
        if np.sum(h_init) < 1e-15:
            continue
        h_refined, s_refined = _smooth_optimize(h_init, n_iter=500, deadline=deadline)
        try_candidate(h_refined)
    
    # Polish the best found with coordinate descent
    if time.time() < deadline - 5:
        h_final = _coordinate_descent(best_h, max_iter=200, deadline=deadline)
        try_candidate(h_final)
    
    # Final intensive polish
    if time.time() < deadline - 2:
        h_final2 = _coordinate_descent(best_h, max_iter=100, deadline=deadline)
        try_candidate(h_final2)
    
    return list(best_h)

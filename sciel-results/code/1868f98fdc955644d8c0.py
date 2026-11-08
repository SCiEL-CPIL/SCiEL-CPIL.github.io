"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 115.0  # safety margin before the 150 s kill
_N = N


def _overlaps_fft(h):
    """S(k) = sum_i h[i] * (1 - h[(i+k) mod N]) for all shifts k using FFT."""
    n = len(h)
    one_minus = 1.0 - h
    # Circular cross-correlation via FFT
    H = np.fft.fft(h)
    O = np.fft.fft(one_minus)
    # Cross-correlation: ifft(conj(H) * O) gives sum_i h[i] * one_minus[(i+k) mod n]
    S = np.fft.ifft(np.conj(H) * O).real
    return S


def _softmax(S, beta):
    """Smooth max of S via log-sum-exp (scaled)."""
    m = np.max(S)
    return m + np.log(np.sum(np.exp(beta * (S - m)))) / beta


def _project(h, n):
    """Project h onto [0,1]^n with sum = n/2."""
    h = np.clip(h, 0.0, 1.0)
    h += (n / 2.0 - h.sum()) / n
    h = np.clip(h, 0.0, 1.0)
    h += (n / 2.0 - h.sum()) / n
    return h


def _gradient(h, S, beta, n):
    """Compute gradient of softmax(S, beta) w.r.t. h."""
    e = np.exp(beta * (S - np.max(S)))
    e = e / e.sum()
    g = np.zeros(n)
    for k in range(n):
        ek = e[k]
        if ek < 1e-12:
            continue
        g += ek * ((1.0 - np.roll(h, k)) - np.roll(h, -k))
    g -= g.mean()
    return g


def _line_search(h, g, lr, beta, n, max_iter=10):
    """Backtracking line search to find good step size."""
    S_curr = _overlaps_fft(h)
    val_curr = _softmax(S_curr, beta)
    for i in range(max_iter):
        lr_try = lr / (2 ** i)
        h_try = _project(h - lr_try * g, n)
        S_try = _overlaps_fft(h_try)
        val_try = _softmax(S_try, beta)
        if val_try < val_curr:
            return h_try, val_try
    return h, val_curr


def _structured_init(n, pattern_type, rng):
    """Generate structured initializations that may have low overlap."""
    if pattern_type == 0:
        # Alternating pattern
        h = np.full(n, 0.5)
        h[::2] = 0.7
        h[1::2] = 0.3
    elif pattern_type == 1:
        # Sine wave
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        h = 0.5 + 0.3 * np.sin(t)
    elif pattern_type == 2:
        # Sawtooth
        t = np.linspace(0, 1, n, endpoint=False)
        h = 0.5 + 0.4 * (2 * t - 1)
    elif pattern_type == 3:
        # Square wave with different period
        period = max(3, n // 4)
        h = np.zeros(n)
        for i in range(n):
            h[i] = 0.7 if (i % period) < period // 2 else 0.3
    elif pattern_type == 4:
        # Random with structure
        h = rng.uniform(0.2, 0.8, n)
    else:
        # Pure random
        h = rng.uniform(0, 1, n)
    
    h = _project(h, n)
    return h


def _optimize_from_start(h, n, beta, lr, steps, rng, time_limit):
    """Run gradient descent from a starting point."""
    best_h = h.copy()
    best_val = float('inf')
    
    for it in range(steps):
        if time.time() > time_limit:
            break
        
        S = _overlaps_fft(h)
        val = _softmax(S, beta)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Compute gradient
        g = _gradient(h, S, beta, n)
        
        # Line search
        h_new, val_new = _line_search(h, g, lr, beta, n)
        
        # Adaptive learning rate
        if val_new < val:
            lr = min(lr * 1.1, 0.2)
        else:
            lr = max(lr * 0.5, 0.001)
        
        # Occasional random perturbation to escape local minima
        if it % 500 == 499 and rng.random() < 0.3:
            d = rng.standard_normal(n)
            d -= d.mean()
            h = _project(h + 0.05 * d / (np.linalg.norm(d) + 1e-12), n)
        else:
            h = h_new
    
    return best_h, best_val


def construct():
    t0 = time.time()
    n = _N
    rng = np.random.default_rng(42)
    
    best_h = np.full(n, 0.5)
    best_val = float('inf')
    
    # Try multiple structured initializations
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > _DEADLINE - 15.0:
            break
        
        # Generate structured initial point
        pattern_type = restart % 5
        h = _structured_init(n, pattern_type, rng)
        
        # Phase 1: Coarse optimization
        beta = 20.0
        lr = 0.05
        steps = 2000
        
        h, val = _optimize_from_start(h, n, beta, lr, steps, rng, _DEADLINE - 10.0)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Phase 2: Medium precision
        if time.time() > _DEADLINE - 10.0:
            break
        
        beta2 = 50.0
        lr2 = 0.02
        steps2 = 1500
        
        h, val2 = _optimize_from_start(h, n, beta2, lr2, steps2, rng, _DEADLINE - 8.0)
        
        if val2 < best_val:
            best_val = val2
            best_h = h.copy()
        
        # Phase 3: Fine precision
        if time.time() > _DEADLINE - 8.0:
            break
        
        beta3 = 100.0
        lr3 = 0.005
        steps3 = 1000
        
        h, val3 = _optimize_from_start(h, n, beta3, lr3, steps3, rng, _DEADLINE - 5.0)
        
        if val3 < best_val:
            best_val = val3
            best_h = h.copy()
    
    # Final high-precision polish on the best solution
    if time.time() < _DEADLINE - 3.0:
        h = best_h.copy()
        beta_final = 200.0
        lr_final = 0.002
        steps_final = 500
        
        for it in range(steps_final):
            if time.time() > _DEADLINE - 1.0:
                break
            
            S = _overlaps_fft(h)
            val = _softmax(S, beta_final)
            
            if val < best_val:
                best_val = val
                best_h = h.copy()
            
            g = _gradient(h, S, beta_final, n)
            h_new, val_new = _line_search(h, g, lr_final, beta_final, n, max_iter=3)
            
            if val_new < val:
                lr_final = min(lr_final * 1.02, 0.01)
            else:
                lr_final = max(lr_final * 0.8, 0.0001)
            
            h = h_new
    
    # Enforce exact sum and bounds one last time
    best_h = _project(best_h, n)
    
    return best_h
# EVOLVE-BLOCK-END

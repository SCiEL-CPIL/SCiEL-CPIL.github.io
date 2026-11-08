import numpy as np
import time

N = 50
_start_time = time.time()
_TIMEOUT = 140.0  # safety margin before 150s kill

def _score(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-15):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    conv = np.convolve(h, h, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_inf)

def _score_fast(h):
    """Faster score computation using FFT for large vectors."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-15):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    # Use FFT for convolution when beneficial
    n = len(h)
    if n >= 16:
        size = 1
        while size < 2 * n - 1:
            size *= 2
        H = np.fft.rfft(h, size)
        conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    else:
        conv = np.convolve(h, h, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_inf)

def _coordinate_ascent(h0, max_iters=5000, tol=1e-12):
    """Coordinate ascent: optimize each coordinate one at a time."""
    h = h0.copy()
    h = np.maximum(h, 0.0)
    best_score = _score(h)
    best_h = h.copy()
    
    for it in range(max_iters):
        if time.time() - _start_time > _TIMEOUT:
            break
        
        improved = False
        for i in range(N):
            # Try optimizing coordinate i
            # The score as a function of h[i] is:
            # (h[i]^2 + S) / (L + h[i]) * max_conv(h[i])
            # where S = sum_{j!=i} h[j]^2, L = sum_{j!=i} h[j]
            # This is complex due to the max in conv, so use line search
            
            old_val = h[i]
            # Try a few values around old_val
            candidates = [0.0, old_val * 0.5, old_val * 0.75, old_val, old_val * 1.25, old_val * 1.5, old_val * 2.0]
            # Also try some absolute values
            if old_val > 0:
                candidates.extend([old_val + 0.1, old_val - 0.1, old_val + 0.5, max(0, old_val - 0.5)])
            
            best_local = old_val
            best_local_score = _score(h)
            
            for c in candidates:
                if c < 0:
                    continue
                h[i] = c
                s = _score(h)
                if s > best_local_score:
                    best_local_score = s
                    best_local = c
            
            h[i] = best_local
            
            if best_local != old_val:
                improved = True
        
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
        
        if not improved:
            break
    
    return best_h, best_score

def _random_perturbation(h0, max_iters=3000):
    """Random perturbation with adaptive step size."""
    h = h0.copy()
    h = np.maximum(h, 0.0)
    best_score = _score(h)
    best_h = h.copy()
    step_size = 0.1
    
    for i in range(max_iters):
        if time.time() - _start_time > _TIMEOUT:
            break
        
        h_trial = h + np.random.normal(0, step_size, size=N)
        h_trial = np.maximum(h_trial, 0.0)
        s_trial = _score(h_trial)
        
        if s_trial > best_score:
            h = h_trial
            best_score = s_trial
            best_h = h.copy()
            step_size = min(step_size * 1.1, 1.0)
        else:
            step_size = max(step_size * 0.95, 0.001)
    
    return best_h, best_score

def _try_structural_patterns():
    """Try known good structural patterns, focusing on sparse configurations."""
    candidates = []
    
    # 2-point clusters with various ratios
    for r in np.linspace(0.1, 10.0, 200):
        a = r / (1 + r)
        b = 1 / (1 + r)
        h = np.zeros(N)
        h[0] = a
        h[1] = b
        candidates.append((h, _score(h)))
    
    # 2-point clusters at various positions
    for pos in range(N - 1):
        for r in [1.0, 1.5, 2.0, 3.0, 0.5]:
            a = r / (1 + r)
            b = 1 / (1 + r)
            h = np.zeros(N)
            h[pos] = a
            h[pos + 1] = b
            candidates.append((h, _score(h)))
    
    # 3-point clusters
    for a in np.linspace(0.1, 0.8, 20):
        for b in np.linspace(0.1, 0.8, 20):
            c = 1 - a - b
            if c < 0.01:
                continue
            h = np.zeros(N)
            h[0] = a
            h[1] = b
            h[2] = c
            candidates.append((h, _score(h)))
    
    # Spread out: every k-th point
    for k in [2, 3, 4, 5, 7, 10]:
        h = np.zeros(N)
        h[::k] = 1.0 / (N // k)
        candidates.append((h, _score(h)))
    
    # Single point
    h = np.zeros(N)
    h[0] = 1.0
    candidates.append((h, _score(h)))
    
    return candidates

def construct():
    """Construct 50 nonnegative heights maximizing ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    global _start_time
    _start_time = time.time()
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Try structural patterns
    patterns = _try_structural_patterns()
    for h, s in patterns:
        if time.time() - _start_time > _TIMEOUT:
            break
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Coordinate ascent from best structural pattern
    if time.time() - _start_time < _TIMEOUT - 20:
        h_opt, s_opt = _coordinate_ascent(best_h, max_iters=3000)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
    
    # Random perturbation refinement
    if time.time() - _start_time < _TIMEOUT - 10:
        h_opt, s_opt = _random_perturbation(best_h, max_iters=2000)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
    
    # Multiple random restarts
    n_restarts = 10
    for i in range(n_restarts):
        if time.time() - _start_time > _TIMEOUT - 15:
            break
        
        # Random sparse initialization
        if np.random.random() < 0.5:
            k = np.random.randint(2, 6)
            idx = np.random.choice(N, size=k, replace=False)
            h_rand = np.zeros(N)
            h_rand[idx] = np.random.exponential(1.0, size=k)
        else:
            h_rand = np.random.exponential(1.0, size=N)
            if np.random.random() < 0.5:
                x = np.linspace(-1, 1, N)
                sigma = np.random.uniform(0.3, 1.5)
                h_rand *= np.exp(-x**2 / (2 * sigma**2))
        
        # Coordinate ascent from random start
        h_opt, s_opt = _coordinate_ascent(h_rand, max_iters=2000)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
        
        # Random perturbation from there
        if time.time() - _start_time < _TIMEOUT - 10:
            h_opt2, s_opt2 = _random_perturbation(h_opt, max_iters=1000)
            if s_opt2 > best_score:
                best_score = s_opt2
                best_h = h_opt2
    
    # Final coordinate ascent polish
    if time.time() - _start_time < _TIMEOUT - 5:
        h_final, s_final = _coordinate_ascent(best_h, max_iters=2000)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final
    
    return list(best_h)

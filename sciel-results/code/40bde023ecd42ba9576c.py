"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_score(h):
    """Compute the maximum overlap of h with shifted 1-h.
    
    Overlap at shift k is sum_i h[i] * (1-h[(i+k) % N])
    We want to minimize the maximum over all non-zero shifts k.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    # Compute all circular cross-correlations of h with (1-h)
    H = np.fft.fft(h)
    H1m = np.fft.fft(1.0 - h)
    corr = np.fft.ifft(H * np.conj(H1m)).real
    max_overlap = np.max(corr[1:])
    return max_overlap

def _overlaps(h):
    """Compute all circular cross-correlations of h with (1-h)."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    H = np.fft.fft(h)
    H1m = np.fft.fft(1.0 - h)
    corr = np.fft.ifft(H * np.conj(H1m)).real
    return corr

def _objective(h, beta=20.0):
    """Smooth approximation of max overlap using log-sum-exp."""
    h = np.asarray(h, dtype=np.float64)
    corr = _overlaps(h)
    c = corr[1:]
    c = np.clip(c, 1e-12, None)
    log_sum_exp = np.log(np.sum(np.exp(beta * c))) / beta
    return log_sum_exp

def _project(h):
    """Project h to satisfy sum = N/2 and h in [0,1]."""
    h = np.asarray(h, dtype=np.float64)
    for _ in range(100):
        h = np.clip(h, 0.0, 1.0)
        h = h + (N/2.0 - np.sum(h)) / N
    h = np.clip(h, 0.0, 1.0)
    diff = N/2.0 - np.sum(h)
    if abs(diff) > 1e-10:
        mask = (h > 1e-6) & (h < 1.0 - 1e-6)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
        else:
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
    return h

def _construct_with_optimization():
    """Try to find a good h via optimization."""
    start_time = time.time()
    deadline = start_time + 100.0
    
    best_h = np.full(N, 0.5)
    best_score = _compute_score(best_h)
    
    np.random.seed(42)
    n_starts = 50
    
    for start_idx in range(n_starts):
        if time.time() > deadline:
            break
        
        if start_idx == 0:
            h0 = np.full(N, 0.5)
        else:
            h0 = np.random.rand(N)
            h0 = h0 - np.mean(h0) + 0.5
            h0 = np.clip(h0, 0.0, 1.0)
            for _ in range(100):
                h0 = np.clip(h0, 0.0, 1.0)
                h0 = h0 + (N/2.0 - np.sum(h0)) / N
            h0 = np.clip(h0, 0.0, 1.0)
            diff = N/2.0 - np.sum(h0)
            if abs(diff) > 1e-10:
                mask = (h0 > 1e-6) & (h0 < 1.0 - 1e-6)
                if np.any(mask):
                    h0[mask] += diff / np.sum(mask)
                else:
                    h0 += diff / N
                    h0 = np.clip(h0, 0.0, 1.0)
        
        try:
            result = minimize(
                lambda x: _objective(x, beta=20.0),
                h0,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500}
            )
            
            h_opt = result.x
            h_opt = _project(h_opt)
            
            score = _compute_score(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
                
        except Exception:
            continue
    
    return best_h

def _construct_binary():
    """Try a binary sequence approach."""
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_h = np.full(N, 0.5)
    best_score = _compute_score(best_h)
    
    np.random.seed(123)
    n_attempts = 200
    
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        indices = np.random.choice(N, size=N//2, replace=False)
        h = np.zeros(N)
        h[indices] = 1.0
        
        score = _compute_score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
        
        for _ in range(20):
            if time.time() > deadline:
                break
            
            ones = np.where(h == 1.0)[0]
            zeros = np.where(h == 0.0)[0]
            
            if len(ones) == 0 or len(zeros) == 0:
                break
            
            i = ones[np.random.randint(len(ones))]
            j = zeros[np.random.randint(len(zeros))]
            
            h[i] = 0.0
            h[j] = 1.0
            
            new_score = _compute_score(h)
            if new_score < score:
                score = new_score
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
            else:
                h[i] = 1.0
                h[j] = 0.0
    
    return best_h

def _refine(h):
    """Refine a given h using local optimization."""
    start_time = time.time()
    deadline = start_time + 30.0
    
    h = np.array(h, dtype=np.float64)
    
    try:
        result = minimize(
            lambda x: _objective(x, beta=30.0),
            h,
            method='L-BFGS-B',
            bounds=[(0.0, 1.0)] * N,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
        )
        
        h_opt = _project(result.x)
        
        if _compute_score(h_opt) <= _compute_score(h) + 1e-9:
            return h_opt
    except Exception:
        pass
    
    return h

def construct():
    """Main entry point."""
    start_time = time.time()
    total_deadline = start_time + 140.0
    
    h_opt = _construct_with_optimization()
    score_opt = _compute_score(h_opt)
    
    if time.time() > total_deadline:
        return h_opt
    
    h_bin = _construct_binary()
    score_bin = _compute_score(h_bin)
    
    if time.time() > total_deadline:
        return h_opt if score_opt <= score_bin else h_bin
    
    if score_opt <= score_bin:
        h_best = h_opt
    else:
        h_best = h_bin
    
    if time.time() < total_deadline - 5.0:
        h_best = _refine(h_best)
    
    h_best = np.clip(h_best, 0.0, 1.0)
    h_best = _project(h_best)
    
    return h_best
# EVOLVE-BLOCK-END

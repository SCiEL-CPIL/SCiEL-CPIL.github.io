"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _overlaps_fft(h):
    """Compute all circular cross-correlations of h with (1-h) using FFT."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    H = np.fft.fft(h)
    H1m = np.fft.fft(1.0 - h)
    corr = np.fft.ifft(H * np.conj(H1m)).real
    return corr

def _compute_score(h):
    """Compute the maximum overlap of h with shifted 1-h at non-zero shifts."""
    corr = _overlaps_fft(h)
    return float(np.max(corr[1:]))

def _objective_and_grad(h, beta=50.0):
    """Smooth approximation of max overlap using log-sum-exp, with gradient."""
    h = np.asarray(h, dtype=np.float64)
    corr = _overlaps_fft(h)
    c = corr[1:]
    c_clipped = np.clip(c, 1e-15, None)
    max_c = np.max(c_clipped)
    exp_vals = np.exp(beta * (c_clipped - max_c))
    sum_exp = np.sum(exp_vals)
    log_sum_exp = max_c + np.log(sum_exp) / beta
    
    # Gradient: d/dh of log-sum-exp
    # c[k] = sum_i h[i] * (1 - h[(i+k)%N])
    # dc[k]/dh[j] = (1 - h[(j+k)%N]) - h[j-k] (for the second term, careful with indexing)
    # Actually: c[k] = sum_i h[i]*(1-h[(i+k)%N])
    # dc[k]/dh[j] = (1 - h[(j+k)%N]) - h[(j-k)%N] * ... 
    # Let me be more careful:
    # c[k] = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
    # dc[k]/dh[j] = 1 - h[(j+k)%N] - h[(j-k)%N]
    # (the second term comes from differentiating sum_i h[i]*h[(i+k)%N] w.r.t. h[j])
    
    weights = exp_vals / sum_exp
    # grad[j] = sum_k weights[k] * dc[k]/dh[j]
    # dc[k]/dh[j] = 1 - h[(j+k)%N] - h[(j-k)%N]
    
    # Build gradient efficiently
    grad = np.zeros(N)
    for k in range(1, N):
        wk = weights[k-1]  # c[k] corresponds to index k-1 in c array
        # dc[k]/dh[j] = 1 - h[(j+k)%N] - h[(j-k)%N]
        if wk > 1e-15:
            grad += wk * (1.0 - np.roll(h, -k) - np.roll(h, k))
    
    return log_sum_exp, grad

def _project(h):
    """Project h to satisfy sum = N/2 and h in [0,1]."""
    h = np.asarray(h, dtype=np.float64).copy()
    for _ in range(200):
        h = np.clip(h, 0.0, 1.0)
        diff = N / 2.0 - np.sum(h)
        if abs(diff) < 1e-12:
            break
        h = h + diff / N
    h = np.clip(h, 0.0, 1.0)
    # Final correction
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-10:
        mask = (h > 1e-8) & (h < 1.0 - 1e-8)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
        else:
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
    return h

def _construct_continuous():
    """Continuous optimization with multiple restarts."""
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_h = np.full(N, 0.5)
    best_score = _compute_score(best_h)
    
    np.random.seed(42)
    
    # Different initializations
    initializations = []
    initializations.append(np.full(N, 0.5))
    
    # Random initializations
    for i in range(10):
        h0 = np.random.rand(N)
        h0 = _project(h0)
        initializations.append(h0)
    
    # Near-binary initializations
    for i in range(5):
        indices = np.random.choice(N, size=N//2, replace=False)
        h0 = np.zeros(N)
        h0[indices] = 1.0
        # Add small perturbation
        h0 += np.random.randn(N) * 0.05
        h0 = _project(h0)
        initializations.append(h0)
    
    for idx, h0 in enumerate(initializations):
        if time.time() > deadline:
            break
        
        try:
            # Use L-BFGS-B with custom objective
            def obj(x):
                val, _ = _objective_and_grad(x, beta=30.0)
                return val
            
            result = minimize(
                obj,
                h0,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
            )
            
            h_opt = _project(result.x)
            score = _compute_score(h_opt)
            if score < best_score - 1e-10:
                best_score = score
                best_h = h_opt.copy()
                
        except Exception:
            continue
    
    return best_h, best_score

def _construct_binary_search():
    """Search for good binary sequences using simulated annealing / local search."""
    start_time = time.time()
    deadline = start_time + 50.0
    
    best_h = np.full(N, 0.5)
    best_score = _compute_score(best_h)
    
    np.random.seed(123)
    
    # Try many random binary sequences
    n_attempts = 2000
    
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
        
        # Local search: try swapping
        if score <= best_score + 0.5:  # Only refine promising candidates
            for _ in range(30):
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
    
    return best_h, best_score

def _refine(h, remaining_time):
    """Final high-precision polish."""
    if remaining_time < 5.0:
        return h
    
    start_time = time.time()
    deadline = start_time + remaining_time - 2.0
    
    h = np.array(h, dtype=np.float64)
    current_score = _compute_score(h)
    
    # Try multiple beta values for the smooth max
    for beta in [50.0, 100.0, 200.0]:
        if time.time() > deadline:
            break
        
        try:
            def obj(x):
                val, _ = _objective_and_grad(x, beta=beta)
                return val
            
            result = minimize(
                obj,
                h,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 2000}
            )
            
            h_opt = _project(result.x)
            score = _compute_score(h_opt)
            if score < current_score - 1e-10:
                current_score = score
                h = h_opt.copy()
                
        except Exception:
            continue
    
    return h

def construct():
    """Main entry point."""
    start_time = time.time()
    total_deadline = start_time + 140.0
    
    # Phase 1: Continuous optimization
    h_cont, score_cont = _construct_continuous()
    
    elapsed = time.time() - start_time
    remaining = total_deadline - start_time - time.time()
    
    if remaining < 10.0:
        h_best = h_cont
    else:
        # Phase 2: Binary search
        h_bin, score_bin = _construct_binary_search()
        
        if score_bin < score_cont:
            h_best = h_bin
        else:
            h_best = h_cont
        
        # Phase 3: Refine
        elapsed = time.time() - start_time
        remaining = total_deadline - start_time - time.time()
        if remaining > 5.0:
            h_best = _refine(h_best, remaining)
    
    h_best = np.clip(h_best, 0.0, 1.0)
    h_best = _project(h_best)
    
    return h_best
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
import numpy as np

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

def _construct_binary_sa():
    """Construct a binary sequence using Simulated Annealing."""
    start_time = time.time()
    deadline = start_time + 130.0 # Leave 20s margin
    
    # Initialize with random binary sequence
    # Ensure exactly N/2 ones
    indices = np.random.choice(N, size=N//2, replace=False)
    h = np.zeros(N)
    h[indices] = 1.0
    
    current_score = _compute_score(h)
    best_h = h.copy()
    best_score = current_score
    
    # Precompute gradients for current h? 
    # Gradients change with h.
    
    T = 0.1
    cooling_rate = 0.99
    min_T = 1e-5
    
    np.random.seed(42)
    
    while time.time() < deadline and T > min_T:
        # Find the shift with maximum overlap
        corr = _overlaps_fft(h)
        c = corr[1:]
        k_max_idx = np.argmax(c)
        k_max = k_max_idx + 1
        
        # Calculate gradients for C_k_max
        # dC_k/dh_j = 1 - h[(j+k)%N] - h[(j-k)%N]
        grad = 1.0 - np.roll(h, -k_max) - np.roll(h, k_max)
        
        # We want to swap i (1->0) and j (0->1)
        # Delta C_k = -grad[i] + grad[j]
        # We want Delta C_k < 0 => grad[j] < grad[i]
        
        ones = np.where(h == 1.0)[0]
        zeros = np.where(h == 0.0)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            break
            
        # Select i from ones, j from zeros
        # To be efficient, we can sample a few candidates
        # Or just pick random ones and check
        
        # Try a few random swaps
        improved = False
        for _ in range(10):
            i = ones[np.random.randint(len(ones))]
            j = zeros[np.random.randint(len(zeros))]
            
            # Estimate delta
            delta = -grad[i] + grad[j]
            
            # Calculate new score exactly? 
            # It's O(N log N). 10 times per step might be slow if steps are many.
            # But N=100 is small.
            
            h[i] = 0.0
            h[j] = 1.0
            
            new_score = _compute_score(h)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
            else:
                # Accept with probability exp(-delta_score / T)
                # delta_score = new_score - current_score
                if new_score > current_score:
                    prob = np.exp(-(new_score - current_score) / T)
                    if np.random.rand() < prob:
                        current_score = new_score
                    else:
                        # Reject
                        h[i] = 1.0
                        h[j] = 0.0
                else:
                    # Should not happen if logic above is correct, but just in case
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_h = h.copy()
            
            if improved:
                break # Restart loop to recompute gradients? 
                # Actually, gradients are only for k_max. If we accept, k_max might change.
                # So we break and recompute.
        
        T *= cooling_rate
        
    return best_h, best_score

def _refine_continuous(h, remaining_time):
    """Refine the binary solution using continuous optimization."""
    if remaining_time < 10.0:
        return h
    
    start_time = time.time()
    deadline = start_time + remaining_time - 2.0
    
    h = np.array(h, dtype=np.float64)
    
    # Use L-BFGS-B on the smooth surrogate
    from scipy.optimize import minimize
    
    def _objective_and_grad(h, beta=50.0):
        h = np.asarray(h, dtype=np.float64)
        corr = _overlaps_fft(h)
        c = corr[1:]
        c_clipped = np.clip(c, 1e-15, None)
        max_c = np.max(c_clipped)
        exp_vals = np.exp(beta * (c_clipped - max_c))
        sum_exp = np.sum(exp_vals)
        log_sum_exp = max_c + np.log(sum_exp) / beta
        
        weights = exp_vals / sum_exp
        grad = np.zeros(N)
        for k in range(1, N):
            wk = weights[k-1]
            if wk > 1e-15:
                grad += wk * (1.0 - np.roll(h, -k) - np.roll(h, k))
        
        return log_sum_exp, grad

    def _project(h):
        h = np.asarray(h, dtype=np.float64).copy()
        for _ in range(200):
            h = np.clip(h, 0.0, 1.0)
            diff = N / 2.0 - np.sum(h)
            if abs(diff) < 1e-12:
                break
            h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        diff = N / 2.0 - np.sum(h)
        if abs(diff) > 1e-10:
            mask = (h > 1e-8) & (h < 1.0 - 1e-8)
            if np.any(mask):
                h[mask] += diff / np.sum(mask)
            else:
                h += diff / N
                h = np.clip(h, 0.0, 1.0)
        return h

    current_score = _compute_score(h)
    
    for beta in [20.0, 50.0, 100.0]:
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
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500}
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
    
    # Phase 1: Binary Simulated Annealing
    h_bin, score_bin = _construct_binary_sa()
    
    elapsed = time.time() - start_time
    remaining = total_deadline - start_time - time.time()
    
    h_best = h_bin
    
    # Phase 2: Continuous Refinement
    if remaining > 15.0:
        h_best = _refine_continuous(h_best, remaining)
    
    h_best = np.clip(h_best, 0.0, 1.0)
    # Final projection to ensure sum constraint is met exactly
    # The refine function projects, but let's be safe
    diff = N / 2.0 - np.sum(h_best)
    if abs(diff) > 1e-12:
        h_best = h_best + diff / N
        h_best = np.clip(h_best, 0.0, 1.0)
    
    return h_best
# EVOLVE-BLOCK-END

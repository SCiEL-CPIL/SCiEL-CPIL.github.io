import time
import numpy as np
from scipy.optimize import minimize

N = 100
TOTAL_TIME_BUDGET = 115  # 5s safety margin from 120s limit

def _fft_max_overlap(h):
    """Fast computation of max overlap using FFT."""
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    C = autocorr[(-np.arange(N)) % N]
    overlaps = N/2.0 - C
    return np.max(overlaps)

def _fft_overlaps(h):
    """Return all overlaps (not just max)."""
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    C = autocorr[(-np.arange(N)) % N]
    return N/2.0 - C

def _project(h):
    """Project h to valid space: [0,1] and sum N/2 exactly."""
    h = np.asarray(h, dtype=np.float64).copy()
    # Iterative water-filling projection
    for _ in range(10):
        h = np.clip(h, 0, 1)
        diff = N/2.0 - np.sum(h)
        if abs(diff) < 1e-15:
            break
        # Only adjust elements that are not at bounds
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
        else:
            # All at bounds, distribute evenly
            h += diff / N
    # Final exact correction
    diff = N/2.0 - np.sum(h)
    if abs(diff) > 1e-15:
        # Add to element closest to 0.5
        idx = np.argmin(np.abs(h - 0.5))
        h[idx] += diff
    return h

def _generate_candidates():
    """Generate diverse initial candidates."""
    candidates = []
    np.random.seed(42)
    
    # 1. Thue-Morse
    tm = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=float)
    if np.sum(tm) != N/2:
        idx_1 = np.where(tm == 1)[0]
        idx_0 = np.where(tm == 0)[0]
        curr_sum = np.sum(tm)
        target = N/2.0
        if curr_sum > target:
            to_flip = int(curr_sum - target)
            tm[idx_1[:to_flip]] = 0
        else:
            to_flip = int(target - curr_sum)
            tm[idx_0[:to_flip]] = 1
    candidates.append(tm.copy())
    
    # 2. Alternating 0011
    alt = np.array([0 if (i % 4) < 2 else 1 for i in range(N)], dtype=float)
    if np.sum(alt) != N/2:
        idx_1 = np.where(alt == 1)[0]
        idx_0 = np.where(alt == 0)[0]
        curr_sum = np.sum(alt)
        target = N/2.0
        if curr_sum > target:
            to_flip = int(curr_sum - target)
            alt[idx_1[:to_flip]] = 0
        else:
            to_flip = int(target - curr_sum)
            alt[idx_0[:to_flip]] = 1
    candidates.append(alt.copy())
    
    # 3. Random Binary Sequences
    for _ in range(30):
        h_rand = np.zeros(N)
        ones_idx = np.random.choice(N, N//2, replace=False)
        h_rand[ones_idx] = 1.0
        candidates.append(h_rand)
        
    # 4. Thresholded sine waves
    for freq in [1, 2, 3, 4, 5]:
        h_sine = np.sin(2 * np.pi * freq * np.arange(N) / N)
        h_bin = (h_sine > 0).astype(float)
        if np.sum(h_bin) != N/2:
            idx_1 = np.where(h_bin == 1)[0]
            idx_0 = np.where(h_bin == 0)[0]
            curr_sum = np.sum(h_bin)
            target = N/2.0
            if curr_sum > target:
                to_flip = int(curr_sum - target)
                h_bin[idx_1[:to_flip]] = 0
            else:
                to_flip = int(target - curr_sum)
                h_bin[idx_0[:to_flip]] = 1
        candidates.append(h_bin)
    
    # 5. Block patterns (runs of 1s and 0s)
    for block_size in [2, 3, 4, 5]:
        pattern = []
        for i in range(N):
            cycle_pos = i % (2 * block_size)
            pattern.append(1.0 if cycle_pos < block_size else 0.0)
        h_block = np.array(pattern)
        if np.sum(h_block) != N/2:
            idx_1 = np.where(h_block == 1)[0]
            idx_0 = np.where(h_block == 0)[0]
            curr_sum = np.sum(h_block)
            target = N/2.0
            if curr_sum > target:
                to_flip = int(curr_sum - target)
                h_block[idx_1[:to_flip]] = 0
            else:
                to_flip = int(target - curr_sum)
                h_block[idx_0[:to_flip]] = 1
        candidates.append(h_block)
    
    return candidates

def _fast_local_search(h_start, time_limit):
    """Fast local search using incremental overlap updates."""
    best_h = h_start.copy()
    best_val = _fft_max_overlap(best_h)
    best_overlaps = _fft_overlaps(best_h)
    deadline = time.time() + time_limit
    
    # Precompute for fast updates
    # When flipping i (1->0) and j (0->1):
    # New autocorr[s] = old_autocorr[s] - h[(i-s)%N] + h[(j-s)%N] 
    #   - h[(i-j+s)%N] + h[(j-i+s)%N] ... wait, this is getting complex.
    # Let me just use full FFT but with early termination on time.
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        idx_1 = np.where(best_h > 0.5)[0]
        idx_0 = np.where(best_h < 0.5)[0]
        
        if len(idx_1) == 0 or len(idx_0) == 0:
            break
        
        improved = False
        # Try up to 200 random flips
        for _ in range(200):
            i = idx_1[np.random.randint(len(idx_1))]
            j = idx_0[np.random.randint(len(idx_0))]
            
            h_new = best_h.copy()
            h_new[i] = 0
            h_new[j] = 1
            
            val_new = _fft_max_overlap(h_new)
            
            if val_new < best_val - 1e-12:
                best_val = val_new
                best_h = h_new
                best_overlaps = _fft_overlaps(best_h)
                improved = True
                no_improve_count = 0
                break
            elif time.time() > deadline:
                break
        
        if not improved:
            no_improve_count += 200
    
    return best_h, best_val

def _smooth_optimize(h_start, time_limit):
    """Smooth optimization using log-sum-exp surrogate."""
    deadline = time.time() + time_limit
    best_h = h_start.copy()
    best_val = _fft_max_overlap(best_h)
    
    # Adaptive beta
    betas = [5.0, 10.0, 20.0, 50.0]
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        def objective(h_flat):
            h = np.clip(h_flat, 0, 1)
            H = np.fft.fft(h)
            autocorr = np.real(np.fft.ifft(H * np.conj(H)))
            C = autocorr[(-np.arange(N)) % N]
            overlaps = N/2.0 - C
            
            max_ov = np.max(overlaps)
            scaled = beta * (overlaps - max_ov)
            sum_exp = np.sum(np.exp(scaled))
            return max_ov + np.log(sum_exp) / beta
        
        def gradient(h_flat):
            h = np.clip(h_flat, 0, 1)
            H = np.fft.fft(h)
            autocorr = np.real(np.fft.ifft(H * np.conj(H)))
            C = autocorr[(-np.arange(N)) % N]
            overlaps = N/2.0 - C
            
            max_ov = np.max(overlaps)
            scaled = beta * (overlaps - max_ov)
            weights = np.exp(scaled) / np.sum(np.exp(scaled))
            
            # Gradient of log-sum-exp wrt h
            # d/dh_k of (1/beta) log(sum_s exp(beta * O_s))
            # = (1/beta) * sum_s weights_s * dO_s/dh_k
            # O_s = N/2 - C_s, C_s = sum_i h[i] h[(i-s)%N]
            # dO_s/dh_k = -h[(k-s)%N] - h[(k+s)%N] + 2*h[k]*h[k] ... no
            # C_s = sum_i h[i]*h[(i-s)%N]
            # dC_s/dh_k = h[(k-s)%N] + h[(k+s)%N] - h[k]*h[k] (when i=k and i-s=k)
            # Actually: dC_s/dh_k = h[(k-s)%N] + sum over i where i-s=k, i.e., i=k+s: h[k+s]
            # But we count h[k]*h[(k-s)%N] and h[k+s]*h[k], so:
            # dC_s/dh_k = h[(k-s)%N] + h[(k+s)%N] - 2*h[k]*h[k] if s=0, else h[(k-s)%N] + h[(k+s)%N]
            
            # This is complex. Let's use numerical gradient or just SLSQP without gradient.
            # For now, skip gradient and let SLSQP use numerical.
            return None
        
        constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2}]
        bounds = [(0, 1)] * N
        
        try:
            res = minimize(objective, best_h, method='SLSQP', bounds=bounds,
                           constraints=constraints, options={'maxiter': 100, 'ftol': 1e-8})
            
            if res.success or res.fun < best_val:
                h_res = _project(res.x)
                val_res = _fft_max_overlap(h_res)
                if val_res < best_val:
                    best_h = h_res
                    best_val = val_res
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + TOTAL_TIME_BUDGET
    
    # Phase 1: Generate and evaluate initial candidates
    candidates = _generate_candidates()
    
    best_h = None
    best_val = float('inf')
    
    for h in candidates:
        val = _fft_max_overlap(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Phase 2: Fast local search
    remaining = deadline - time.time()
    if remaining > 5:
        local_time = min(30, remaining - 5)
        best_h, best_val = _fast_local_search(best_h, local_time)
    
    # Phase 3: Smooth optimization
    remaining = deadline - time.time()
    if remaining > 5:
        opt_time = min(60, remaining - 5)
        best_h, best_val = _smooth_optimize(best_h, opt_time)
    
    # Final projection
    best_h = _project(best_h)
    
    # Final time check
    if time.time() > deadline - 2:
        # Emergency return
        pass
    
    return best_h

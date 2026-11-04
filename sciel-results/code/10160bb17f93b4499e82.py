import numpy as np
import time
from scipy.optimize import minimize

N = 100
TARGET_SUM = N / 2.0

def _compute_overlaps(h):
    """
    Compute C_k = sum_i h_i (1 - h_{i+k}) for k=1..N-1.
    Returns array of length 2*(N-1) containing overlaps for lags -N+1..-1 and 1..N-1.
    """
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
    # c has length 2N-1. Index N is lag 0.
    # c[0:N] are lags -(N-1)..0
    # c[N:2N-1] are lags 0..N-1
    # We want lags != 0.
    # lags_neg = c[0:N-1]  (lags -N+1 to -1)
    # lags_pos = c[N+1:2N-1] (lags 1 to N-1)
    c_neg = c[:N-1]
    c_pos = c[N+1:]
    return np.concatenate([c_neg, c_pos])

def _max_overlap(h):
    c = _compute_overlaps(h)
    return np.max(c)

def _project_to_feasible(h):
    """
    Project h to [0, 1]^N with sum = TARGET_SUM.
    Simple iterative projection.
    """
    h = np.clip(h, 0, 1)
    for _ in range(50):
        diff = TARGET_SUM - np.sum(h)
        if abs(diff) < 1e-9:
            break
        if diff > 0:
            # Increase values that are not at 1
            mask = h < 1.0
            if not np.any(mask):
                break
            step = diff / np.sum(mask)
            h[mask] += step
        else:
            # Decrease values that are not at 0
            mask = h > 0.0
            if not np.any(mask):
                break
            step = diff / np.sum(mask)
            h[mask] += step
        h = np.clip(h, 0, 1)
    return h

def _objective_lse(h_flat, beta):
    """
    Objective: LSE of overlaps + penalty for sum constraint.
    """
    h = h_flat
    c = _compute_overlaps(h)
    
    # LSE for stability
    c_max = np.max(c)
    # Avoid log(0) if beta is huge and exp underflows, but c_max shift handles it.
    # exp(beta * (c - c_max))
    exp_vals = np.exp(beta * (c - c_max))
    lse = c_max + np.log(np.sum(exp_vals)) / beta
    
    sum_diff = np.sum(h) - TARGET_SUM
    penalty = 1000.0 * sum_diff**2
    
    obj = lse + penalty
    
    # Gradient
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient of c_k wrt h_j
    # c_k = sum_i h_i (1 - h_{i+k})
    # d c_k / d h_j = (1 - h_{j+k}) [if j is first term] - h_{j-k} [if j is second term]
    # Wait, c_k = sum_i h_i - sum_i h_i h_{i+k}
    # d/dc_k/dh_j = 1 - (h_{j+k} + h_{j-k}) with boundary 0.
    
    # We need to sum over k with weights.
    # grad_h[j] = sum_k w_k * (1 - h_{j+k} - h_{j-k})
    
    grad = np.zeros(N)
    lags = np.concatenate([np.arange(-(N-1), 0), np.arange(1, N)])
    
    for k_idx, lag in enumerate(lags):
        w = weights[k_idx]
        if w < 1e-12:
            continue
        
        # Vectorized computation of h_{j+lag} and h_{j-lag}
        j = np.arange(N)
        
        idx_pos = j + lag
        valid_pos = (idx_pos >= 0) & (idx_pos < N)
        h_pos = np.zeros(N)
        h_pos[valid_pos] = h[idx_pos[valid_pos]]
        
        idx_neg = j - lag
        valid_neg = (idx_neg >= 0) & (idx_neg < N)
        h_neg = np.zeros(N)
        h_neg[valid_neg] = h[idx_neg[valid_neg]]
        
        grad += w * (1.0 - h_pos - h_neg)
        
    grad += 2.0 * 1000.0 * sum_diff
    
    return obj, grad

def _local_search_binary(h_bin, time_limit=5.0):
    """
    Local search on binary sequence by swapping 0s and 1s to reduce max overlap.
    """
    start_time = time.time()
    h = h_bin.copy().astype(float)
    current_max = _max_overlap(h)
    
    ones = np.where(h == 1)[0]
    zeros = np.where(h == 0)[0]
    
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try a few random swaps
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
            i = rng.choice(ones)
            j = rng.choice(zeros)
            if i == j:
                continue
            
            # Swap
            h[i], h[j] = h[j], h[i]
            
            new_max = _max_overlap(h)
            if new_max < current_max:
                current_max = new_max
                # Update lists
                # i was 1, now 0. j was 0, now 1.
                # Find indices in lists
                idx_i_in_ones = np.where(ones == i)[0][0]
                idx_j_in_zeros = np.where(zeros == j)[0][0]
                
                ones[idx_i_in_ones] = j
                zeros[idx_j_in_zeros] = i
                improved = True
            else:
                # Revert
                h[i], h[j] = h[j], h[i]
                
        if not improved:
            # Try larger perturbations if stuck? Or just break if no improvement in many steps?
            # For now, break if no improvement in a batch.
            pass
            
    return h, current_max

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_h = None
    best_score = float('inf')
    
    # 1. Generate good binary initial candidates
    candidates = []
    
    # Random binary
    rng = np.random.default_rng(42)
    for _ in range(10):
        h_bin = rng.choice([0, 1], size=N)
        # Adjust sum to 50
        s = np.sum(h_bin)
        if s > 50:
            idx_ones = np.where(h_bin == 1)[0]
            to_swap = s - 50
            swap_idx = rng.choice(idx_ones, size=to_swap, replace=False)
            h_bin[swap_idx] = 0
        elif s < 50:
            idx_zeros = np.where(h_bin == 0)[0]
            to_swap = 50 - s
            swap_idx = rng.choice(idx_zeros, size=to_swap, replace=False)
            h_bin[swap_idx] = 1
        candidates.append(h_bin.astype(float))
        
    # Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 1
    candidates.append(h_alt)
    
    # Sine-based binary
    t = np.arange(N)
    for freq in [1, 2, 3]:
        h_sin = 0.5 + 0.5 * np.sin(2 * np.pi * freq * t / N)
        h_bin = (h_sin >= 0.5).astype(float)
        # Adjust sum
        s = np.sum(h_bin)
        if s != 50:
            # Simple adjustment
            if s > 50:
                idx_ones = np.where(h_bin == 1)[0]
                to_swap = s - 50
                swap_idx = rng.choice(idx_ones, size=to_swap, replace=False)
                h_bin[swap_idx] = 0
            else:
                idx_zeros = np.where(h_bin == 0)[0]
                to_swap = 50 - s
                swap_idx = rng.choice(idx_zeros, size=to_swap, replace=False)
                h_bin[swap_idx] = 1
        candidates.append(h_bin)
        
    # Optimize each binary candidate locally
    optimized_candidates = []
    for h_init in candidates:
        if time.time() > deadline - 20.0:
            break
        h_opt, score = _local_search_binary(h_init, time_limit=2.0)
        if score < best_score:
            best_score = score
            best_h = h_opt
        optimized_candidates.append(h_opt)
        
    # 2. Continuous optimization from best binary candidates
    # Pick top 3 candidates
    scored_candidates = [( _max_overlap(c), c) for c in optimized_candidates]
    scored_candidates.sort(key=lambda x: x[0])
    top_candidates = [c for _, c in scored_candidates[:3]]
    
    for h_init in top_candidates:
        if time.time() > deadline - 10.0:
            break
            
        current_h = h_init.copy()
        
        # Anneal beta
        betas = [20.0, 50.0, 100.0, 200.0]
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
                
            def obj_func(x):
                val, _ = _objective_lse(x, beta)
                return val
                
            def grad_func(x):
                _, g = _objective_lse(x, beta)
                return g
                
            try:
                res = minimize(obj_func, current_h, method='L-BFGS-B', jac=grad_func,
                               bounds=[(0, 1)] * N,
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                current_h = res.x
            except Exception:
                pass
                
        # Project to feasible set
        current_h = _project_to_feasible(current_h)
        
        # Evaluate true max overlap
        true_max = _max_overlap(current_h)
        
        # Snap to binary if close?
        # If the continuous solution is very close to binary, snapping might help.
        # But the continuous solution might be better.
        # Let's try a "soft snap": round values > 0.9 to 1, < 0.1 to 0, keep others?
        # No, let's just evaluate the continuous one and also a snapped version.
        
        h_snapped = np.round(current_h)
        # Adjust sum for snapped
        s = np.sum(h_snapped)
        if s != 50:
             # Re-adjust snapped
             if s > 50:
                 idx_ones = np.where(h_snapped == 1)[0]
                 # Sort by current_h value to remove those closest to 0.5?
                 # Actually, just random or based on value
                 vals = current_h[idx_ones]
                 order = np.argsort(vals)
                 to_swap = s - 50
                 swap_idx = idx_ones[order[:to_swap]]
                 h_snapped[swap_idx] = 0
             else:
                 idx_zeros = np.where(h_snapped == 0)[0]
                 vals = current_h[idx_zeros]
                 order = np.argsort(vals, descending=True)
                 to_swap = 50 - s
                 swap_idx = idx_zeros[order[:to_swap]]
                 h_snapped[swap_idx] = 1
                 
        snapped_max = _max_overlap(h_snapped)
        
        # Also do local search on the snapped version
        h_ls, ls_score = _local_search_binary(h_snapped, time_limit=1.0)
        
        # Compare
        if true_max < best_score:
            best_score = true_max
            best_h = current_h.copy()
            
        if snapped_max < best_score:
            best_score = snapped_max
            best_h = h_snapped.copy()
            
        if ls_score < best_score:
            best_score = ls_score
            best_h = h_ls.copy()
            
    # 3. Final polish on the best continuous solution
    if best_h is not None and time.time() < deadline - 2.0:
        # Try small perturbations on the best found
        h_best = best_h.copy()
        best_val = _max_overlap(h_best)
        
        rng_local = np.random.default_rng(123)
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            # Random small perturbation on 2 indices
            idx = rng_local.choice(N, size=2, replace=False)
            delta = rng_local.uniform(-0.05, 0.05)
            h_try = h_best.copy()
            h_try[idx[0]] += delta
            h_try[idx[1]] -= delta
            
            # Project
            h_try = _project_to_feasible(h_try)
            
            val = _max_overlap(h_try)
            if val < best_val:
                best_val = val
                h_best = h_try.copy()
                
        if best_val < best_score:
            best_score = best_val
            best_h = h_best.copy()
            
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    return best_h

if __name__ == "__main__":
    h = construct()
    print(_max_overlap(h))

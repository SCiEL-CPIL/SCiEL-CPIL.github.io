import numpy as np
import time

# Precompute bin indices for fast entropy calculation
_N = 10
_y_idx, _x_idx = np.indices((_N, _N))
_x_flat = _x_idx.flatten()
_y_flat = _y_idx.flatten()

# x - y ranges from -9 to 9, shift by 9 to get 0..18
_diff_shifted = _x_flat - _y_flat + 9  # 0..18
# x ranges from 0 to 9
_h0_vals = _x_flat  # 0..9
# x + y ranges from 0 to 18
_h1_vals = _x_flat + _y_flat  # 0..18
# x + 2y ranges from 0 to 27
_h2_vals = _x_flat + 2 * _y_flat  # 0..27

_N_BINS_DIFF = 19
_N_BINS_H0 = 10
_N_BINS_H1 = 19
_N_BINS_H2 = 28

def _entropy_from_bincounts(bin_counts, total_weight):
    """Compute Shannon entropy from bin counts, assuming total_weight is the sum of all weights."""
    probs = bin_counts / total_weight
    mask = probs > 1e-15
    if not np.any(mask):
        return 0.0
    p = probs[mask]
    return -np.sum(p * np.log2(p))

def _compute_ratio_fast(w):
    """Fast ratio computation using precomputed bin indices."""
    total = w.sum()
    if total <= 1e-15:
        return 0.0
    
    # Compute bin counts using bincount with precomputed indices
    diff_counts = np.bincount(_diff_shifted, weights=w, minlength=_N_BINS_DIFF)
    h0_counts = np.bincount(_h0_vals, weights=w, minlength=_N_BINS_H0)
    h1_counts = np.bincount(_h1_vals, weights=w, minlength=_N_BINS_H1)
    h2_counts = np.bincount(_h2_vals, weights=w, minlength=_N_BINS_H2)
    
    D = _entropy_from_bincounts(diff_counts, total)
    H0 = _entropy_from_bincounts(h0_counts, total)
    H1 = _entropy_from_bincounts(h1_counts, total)
    H2 = _entropy_from_bincounts(h2_counts, total)
    
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 0.0
    return D / M

def _compute_ratio(P):
    """Compute the ratio D/M for a given 10x10 weight matrix P."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0
    P = P / s
    return _compute_ratio_fast(P.flatten())

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    timeout = 600
    safety_margin = 30
    deadline = start_time + timeout - safety_margin
    
    # Generate diverse initial candidates
    candidates = []
    
    # Candidate 1: uniform
    candidates.append(np.ones((n, n), dtype=np.float64) / 100.0)
    
    # Candidate 2: concentrate on main diagonal (x = y), where x-y = 0
    w2 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        w2[i, i] = 1.0
    w2 /= w2.sum()
    candidates.append(w2)
    
    # Candidate 3: weight cells to make x-y as uniform as possible
    w3 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            d = j - i
            count_d = min(n - abs(d), n)
            w3[i, j] = 1.0 / np.sqrt(count_d)
    w3 /= w3.sum()
    candidates.append(w3)
    
    # Candidate 4: focus on cells where x is in a narrow range but x-y varies
    w4 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(2, 8):
            w4[i, j] = 1.0
    w4 /= w4.sum()
    candidates.append(w4)
    
    # Candidate 5: anti-diagonal focus (x + y = constant)
    w5 = np.zeros((n, n), dtype=np.float64)
    target_sum = 9
    for i in range(n):
        j = target_sum - i
        if 0 <= j < n:
            w5[i, j] = 1.0
    w5 /= w5.sum()
    candidates.append(w5)
    
    # Candidate 6: combine strategies
    w6 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            d = j - i
            s_xy = i + j
            score = np.exp(-0.1 * np.abs(d)) * np.exp(-0.05 * np.abs(s_xy - 9))
            w6[i, j] = score
    w6 /= w6.sum()
    candidates.append(w6)
    
    # Candidate 7: For each diagonal (x-y=d), put weight only on cells where x is as "central" as possible
    # This reduces H0 while keeping D high
    w7 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            d = j - i
            # For this diagonal, prefer x values near the center
            w7[i, j] = np.exp(-0.3 * (j - 4.5)**2)
    w7 /= w7.sum()
    candidates.append(w7)
    
    # Candidate 8: Weight proportional to 1/(x+0.5) to reduce H0, combined with diagonal uniformity
    w8 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            d = j - i
            count_d = min(n - abs(d), n)
            w8[i, j] = (1.0 / np.sqrt(count_d)) * (1.0 / (j + 0.5))
    w8 /= w8.sum()
    candidates.append(w8)
    
    # Evaluate all candidates
    best_score = -1
    best_P = candidates[0]
    for c in candidates:
        sc = _compute_ratio_fast(c.flatten())
        if sc > best_score:
            best_score = sc
            best_P = c.copy()
    
    # Phase 1: Local search with random perturbations
    np.random.seed(42)
    current_P = best_P.copy()
    current_score = best_score
    
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 200
    phase1_end = start_time + 120
    
    while time.time() < phase1_end:
        if no_improve_count > max_no_improve:
            if current_score < best_score * 0.98:
                noise = np.random.normal(0, 0.1, (n, n))
                current_P = best_P + noise
                current_P = np.clip(current_P, 0, None)
                s = current_P.sum()
                if s > 1e-9:
                    current_P /= s
                else:
                    current_P = best_P.copy()
                current_score = _compute_ratio_fast(current_P.flatten())
            no_improve_count = 0
            step_size = 0.2
            max_no_improve = 100
            continue
        
        i, j = np.random.randint(0, n, 2)
        delta = np.random.uniform(-step_size, step_size)
        
        new_P = current_P.copy()
        new_P[i, j] += delta
        new_P = np.clip(new_P, 0, None)
        s = new_P.sum()
        if s <= 1e-9:
            continue
        new_P = new_P / s
        
        new_score = _compute_ratio_fast(new_P.flatten())
        
        if new_score > current_score:
            current_P = new_P
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_P = current_P.copy()
            step_size *= 0.99
        else:
            no_improve_count += 1
            step_size = max(0.01, step_size * 1.02)
    
    # Phase 2: More aggressive multi-start local search
    # Try many different random starting points and do short local searches
    phase2_end = start_time + timeout - safety_margin - 60
    n_restarts = 20
    restart_time_budget = (phase2_end - start_time - 120) / max(n_restarts, 1)
    
    for r in range(n_restarts):
        if time.time() > phase2_end:
            break
        
        # Create a new random starting point
        if r == 0:
            start_P = best_P.copy()
        else:
            start_P = best_P + np.random.normal(0, 0.15, (n, n))
            start_P = np.clip(start_P, 0, None)
            s = start_P.sum()
            if s > 1e-9:
                start_P /= s
            else:
                start_P = best_P.copy()
        
        cur_P = start_P.copy()
        cur_score = _compute_ratio_fast(cur_P.flatten())
        
        if cur_score <= best_score:
            continue
        
        step_sz = 0.05
        no_imp = 0
        max_no_imp = 50
        restart_deadline = time.time() + restart_time_budget
        
        while time.time() < restart_deadline:
            if no_imp > max_no_imp:
                break
            
            i, j = np.random.randint(0, n, 2)
            delta = np.random.uniform(-step_sz, step_sz)
            
            new_P = cur_P.copy()
            new_P[i, j] += delta
            new_P = np.clip(new_P, 0, None)
            s = new_P.sum()
            if s <= 1e-9:
                continue
            new_P = new_P / s
            
            new_score = _compute_ratio_fast(new_P.flatten())
            
            if new_score > cur_score:
                cur_P = new_P
                cur_score = new_score
                no_imp = 0
                step_sz *= 0.97
            else:
                no_imp += 1
                step_sz = max(0.005, step_sz * 1.03)
        
        if cur_score > best_score:
            best_score = cur_score
            best_P = cur_P.copy()
    
    # Phase 3: Final high-precision polish with very small steps
    current_P = best_P.copy()
    current_score = best_score
    step_size = 0.005
    no_improve_count = 0
    max_no_improve = 80
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
        
        i, j = np.random.randint(0, n, 2)
        delta = np.random.uniform(-step_size, step_size)
        
        new_P = current_P.copy()
        new_P[i, j] += delta
        new_P = np.clip(new_P, 0, None)
        s = new_P.sum()
        if s <= 1e-9:
            continue
        new_P = new_P / s
        
        new_score = _compute_ratio_fast(new_P.flatten())
        
        if new_score > current_score:
            current_P = new_P
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_P = current_P.copy()
            step_size *= 0.95
        else:
            no_improve_count += 1
            step_size = max(0.001, step_size * 1.05)
    
    # Final validation and normalization
    result = np.clip(best_P, 0, None)
    s = result.sum()
    if s <= 1e-9:
        result = np.ones((n, n)) / 100.0
    else:
        result = result / s
    
    return result

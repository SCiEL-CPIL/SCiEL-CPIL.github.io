import numpy as np
import time
from scipy.optimize import minimize

def _compute_ratio(P):
    """Compute the ratio D/M for a given 10x10 weight matrix P."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0
    P = P / s
    
    y_idx, x_idx = np.indices((10, 10))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    w = P.flatten()
    
    def entropy_from_values(values, weights):
        min_v = np.min(values)
        shifted = values - min_v
        counts = np.bincount(shifted, weights=weights)
        probs = counts[counts > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    D = entropy_from_values(x_flat - y_flat, w)
    H0 = entropy_from_values(x_flat, w)
    H1 = entropy_from_values(x_flat + y_flat, w)
    H2 = entropy_from_values(x_flat + 2 * y_flat, w)
    
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 0.0
    return D / M

def _objective(neg_P_log, x_flat, y_flat):
    """Negative log-sum-exp based objective for smooth optimization."""
    # Convert from log-space to probability space
    P_log = -neg_P_log
    # Softmax to get normalized probabilities
    P_max = np.max(P_log)
    P = np.exp(P_log - P_max)
    P_sum = P.sum()
    if P_sum <= 1e-15:
        return 1e10
    P = P / P_sum
    
    w = P.flatten()
    
    def entropy_from_values(values, weights):
        min_v = np.min(values)
        shifted = values - min_v
        counts = np.bincount(shifted, weights=weights)
        probs = counts[counts > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    D = entropy_from_values(x_flat - y_flat, w)
    H0 = entropy_from_values(x_flat, w)
    H1 = entropy_from_values(x_flat + y_flat, w)
    H2 = entropy_from_values(x_flat + 2 * y_flat, w)
    
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 1e10
    
    # We want to maximize D/M, so minimize -D/M
    return -D / M

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    y_idx, x_idx = np.indices((n, n))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    
    start_time = time.time()
    timeout = 600
    safety_margin = 30
    
    # Precompute value arrays
    diff_vals = x_flat - y_flat
    h0_vals = x_flat
    h1_vals = x_flat + y_flat
    h2_vals = x_flat + 2 * y_flat
    
    def compute_entropies(w):
        """Compute all four entropies for a given weight vector."""
        def entropy_from_values(values, weights):
            min_v = np.min(values)
            shifted = values - min_v
            counts = np.bincount(shifted, weights=weights)
            probs = counts[counts > 1e-15]
            if len(probs) == 0:
                return 0.0
            return -np.sum(probs * np.log2(probs))
        
        D = entropy_from_values(diff_vals, w)
        H0 = entropy_from_values(h0_vals, w)
        H1 = entropy_from_values(h1_vals, w)
        H2 = entropy_from_values(h2_vals, w)
        return D, H0, H1, H2
    
    def compute_ratio_from_w(w):
        D, H0, H1, H2 = compute_entropies(w)
        M = max(H0, H1, H2)
        if M <= 1e-9:
            return 0.0
        return D / M
    
    # Generate diverse initial candidates
    candidates = []
    
    # Candidate 1: uniform
    candidates.append(np.ones((n, n), dtype=np.float64) / 100.0)
    
    # Candidate 2: concentrate on main diagonal (x = y), where x-y = 0
    # But spread across different x values to increase D
    w2 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        w2[i, i] = 1.0
    w2 /= w2.sum()
    candidates.append(w2)
    
    # Candidate 3: weight cells to make x-y as uniform as possible
    # For each x-y value d, there are cells (x,y) with x-y=d
    # Assign weight proportional to 1/sqrt(count) to spread more
    w3 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            d = j - i
            # Count cells with this d
            count_d = min(n - abs(d), n)
            w3[i, j] = 1.0 / np.sqrt(count_d)
    w3 /= w3.sum()
    candidates.append(w3)
    
    # Candidate 4: focus on cells where x is in a narrow range but x-y varies
    w4 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(2, 8):  # x in 2..7
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
    
    # Candidate 6: combine strategies - weight by product of indicators
    w6 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            d = j - i
            s_xy = i + j
            # Reward being on a diagonal with moderate |d| and central x+y
            score = np.exp(-0.1 * np.abs(d)) * np.exp(-0.05 * np.abs(s_xy - 9))
            w6[i, j] = score
    w6 /= w6.sum()
    candidates.append(w6)
    
    # Evaluate all candidates
    best_score = -1
    best_P = candidates[0]
    for c in candidates:
        sc = _compute_ratio(c)
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
    phase1_end = start_time + 120  # Spend 120s on local search
    
    while time.time() < phase1_end:
        if no_improve_count > max_no_improve:
            # Restart from best with larger step
            if current_score < best_score * 0.98:
                noise = np.random.normal(0, 0.1, (n, n))
                current_P = best_P + noise
                current_P = np.clip(current_P, 0, None)
                s = current_P.sum()
                if s > 1e-9:
                    current_P /= s
                else:
                    current_P = best_P.copy()
                current_score = _compute_ratio(current_P)
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
        
        new_score = _compute_ratio(new_P)
        
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
    
    # Phase 2: Gradient-based optimization using log-space parameterization
    # Convert best_P to log-space
    P_for_log = best_P.copy()
    P_for_log[P_for_log < 1e-12] = 1e-12
    log_P_init = np.log(P_for_log.flatten())
    
    # Define objective in terms of log-probabilities
    def neg_ratio_from_log(log_p_flat):
        log_p = log_p_flat.reshape(n, n)
        # Softmax normalization
        log_p_max = np.max(log_p)
        p = np.exp(log_p - log_p_max)
        p_sum = p.sum()
        if p_sum <= 1e-15:
            return 1e10
        p = p / p_sum
        w = p.flatten()
        D, H0, H1, H2 = compute_entropies(w)
        M = max(H0, H1, H2)
        if M <= 1e-9:
            return 1e10
        return -D / M
    
    # Try multiple starting points for gradient optimization
    phase2_end = start_time + timeout - safety_margin
    gradient_starts = [log_P_init]
    
    # Add some perturbed starts
    for _ in range(3):
        perturbed = log_P_init + np.random.normal(0, 0.5, n*n)
        gradient_starts.append(perturbed)
    
    best_gradient_score = best_score
    best_gradient_P = best_P.copy()
    
    for log_start in gradient_starts:
        if time.time() > phase2_end - 60:
            break
        
        try:
            result = minimize(
                neg_ratio_from_log,
                log_start,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.success or result.fun < -best_gradient_score:
                log_p_opt = result.x.reshape(n, n)
                log_p_max = np.max(log_p_opt)
                p_opt = np.exp(log_p_opt - log_p_max)
                p_opt = p_opt / p_opt.sum()
                
                score_opt = _compute_ratio(p_opt)
                if score_opt > best_gradient_score:
                    best_gradient_score = score_opt
                    best_gradient_P = p_opt.copy()
        except Exception:
            continue
    
    if best_gradient_score > best_score:
        best_score = best_gradient_score
        best_P = best_gradient_P.copy()
    
    # Phase 3: Final high-precision polish with very small steps
    current_P = best_P.copy()
    current_score = best_score
    step_size = 0.01
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < phase2_end - 10:
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
        
        new_score = _compute_ratio(new_P)
        
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

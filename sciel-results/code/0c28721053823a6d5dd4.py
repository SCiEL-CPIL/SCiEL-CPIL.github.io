# EVOLVE-BLOCK-START
import numpy as np
import time

def _clip_and_normalize(P):
    """Clip negative values to 0 and normalize to sum 1."""
    P = np.clip(P, 0.0, None)
    total = P.sum()
    if total <= 1e-9:
        return np.ones((10, 10)) / 100.0
    return P / total

def _shannon_entropy_from_counts(counts):
    """Compute Shannon entropy in bits from a count array."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 1e-15:
        return 0.0
    probs = counts / total
    mask = probs > 1e-15
    if not np.any(mask):
        return 0.0
    return float(-np.sum(probs[mask] * np.log2(probs[mask])))

def _compute_ratio(P):
    """Compute D/M ratio for a given 10x10 distribution P[y, x]."""
    n = 10
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)  # X[y,x] = x, Y[y,x] = y
    
    # D: H of x - y (slope -1)
    vals_d = X - Y  # values from -9 to 9
    counts_d = np.zeros(19)
    for v in range(19):
        mask = (vals_d == v - 9)
        counts_d[v] = P[mask].sum()
    H_d = _shannon_entropy_from_counts(counts_d)
    
    # M: max of H for slopes 0, 1, 2
    max_h = 0.0
    for r in [0, 1, 2]:
        vals = X + r * Y
        num_vals = n + r * (n - 1)
        counts = np.zeros(num_vals)
        for v in range(num_vals):
            mask = (vals == v)
            counts[v] = P[mask].sum()
        H_r = _shannon_entropy_from_counts(counts)
        if H_r > max_h:
            max_h = H_r
    
    if max_h <= 1e-9:
        return 0.0
    return H_d / max_h

def _parametric_P(params):
    """Generate P from parameters [alpha, beta, gamma, mu_x, mu_y]."""
    n = 10
    xs = np.arange(n, dtype=np.float64)
    ys = np.arange(n, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    alpha, beta, gamma, mu_x, mu_y = params
    log_w = -alpha * (X - mu_x)**2 - beta * (Y - mu_y)**2 - gamma * (X - mu_x) * (Y - mu_y)
    # For numerical stability
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    total = w.sum()
    if total <= 1e-15:
        return np.ones((n, n)) / (n * n)
    return w / total

def _parametric_ratio(params):
    """Compute ratio for parametric distribution."""
    P = _parametric_P(params)
    return _compute_ratio(P)

def _parametric_gradient(params):
    """Compute gradient of ratio w.r.t. params using finite differences."""
    P = _parametric_P(params)
    ratio = _compute_ratio(P)
    n = 10
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)
    
    # Precompute bin assignments
    vals_d = X - Y
    bin_d = vals_d + 9  # 0..18
    counts_d = np.zeros(19)
    for v in range(19):
        mask = (bin_d == v)
        counts_d[v] = P[mask].sum()
    H_d = _shannon_entropy_from_counts(counts_d)
    
    # Compute entropies and gradients for each slope
    slopes = [0, 1, 2]
    H_vals = []
    bin_ids = []
    counts_list = []
    for r in slopes:
        vals = X + r * Y
        num_vals = n + r * (n - 1)
        bin_id = vals
        counts = np.zeros(num_vals)
        for v in range(num_vals):
            mask = (bin_id == v)
            counts[v] = P[mask].sum()
        H_r = _shannon_entropy_from_counts(counts)
        H_vals.append(H_r)
        bin_ids.append(bin_id)
        counts_list.append(counts)
    
    r_star = int(np.argmax(H_vals))
    M = H_vals[r_star]
    if M <= 1e-9:
        return np.zeros(5)
    
    # Gradient of D w.r.t. P
    log2_factor = 1.0 / np.log(2.0)
    grad_d_P = -np.log2(np.maximum(counts_d[bin_d], 1e-15)) - log2_factor
    
    # Gradient of M w.r.t. P
    counts_r = counts_list[r_star]
    bin_r = bin_ids[r_star]
    grad_m_P = -np.log2(np.maximum(counts_r[bin_r], 1e-15)) - log2_factor
    
    # d(ratio)/dP = (grad_d_P * M - H_d * grad_m_P) / M^2
    grad_P = (grad_d_P * M - H_d * grad_m_P) / (M * M)
    
    # Now dP/dparams via finite differences (cheap since only 5 params)
    eps = 1e-5
    grad_params = np.zeros(5)
    for i in range(5):
        params_plus = params.copy()
        params_plus[i] += eps
        P_plus = _parametric_P(params_plus)
        dP = (P_plus - P) / eps
        grad_params[i] = np.sum(grad_P * dP)
    
    return grad_params

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    deadline = start_time + 590.0  # 590s safety margin
    
    # Best result tracking
    best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_ratio = _compute_ratio(best_P)
    
    # Parametric optimization with multiple random restarts
    np.random.seed(42)
    num_restarts = 50
    max_iter = 200
    step_size = 0.5
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization in parameter space
        if restart == 0:
            params = np.array([0.0, 0.0, 0.0, 4.5, 4.5])
        else:
            params = np.array([
                np.random.uniform(-2.0, 2.0),
                np.random.uniform(-2.0, 2.0),
                np.random.uniform(-2.0, 2.0),
                np.random.uniform(2.0, 7.0),
                np.random.uniform(2.0, 7.0)
            ])
        
        P = _parametric_P(params)
        ratio = _compute_ratio(P)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P.copy()
        
        # Gradient ascent in parameter space
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            grad = _parametric_gradient(params)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
            
            # Adaptive step size
            params_new = params + step_size * grad / grad_norm * min(grad_norm, 1.0)
            
            P_new = _parametric_P(params_new)
            ratio_new = _compute_ratio(P_new)
            
            if ratio_new > ratio:
                params = params_new
                ratio = ratio_new
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_P = P.copy()
                step_size = min(step_size * 1.05, 2.0)
            else:
                step_size *= 0.8
                if step_size < 1e-6:
                    break
        
        # After parametric optimization, do a few steps of full 100D gradient ascent
        if time.time() > deadline - 30:
            break
        
        P = best_P.copy()
        for it in range(20):
            if time.time() > deadline - 10:
                break
            
            # Compute gradient in 100D
            xs = np.arange(n)
            ys = np.arange(n)
            X, Y = np.meshgrid(xs, ys)
            log2_factor = 1.0 / np.log(2.0)
            
            vals_d = X - Y
            bin_d = vals_d + 9
            counts_d = np.zeros(19)
            for v in range(19):
                mask = (bin_d == v)
                counts_d[v] = P[mask].sum()
            H_d = _shannon_entropy_from_counts(counts_d)
            
            slopes = [0, 1, 2]
            H_vals = []
            bin_ids = []
            counts_list = []
            for r in slopes:
                vals = X + r * Y
                num_vals = n + r * (n - 1)
                bin_id = vals
                counts = np.zeros(num_vals)
                for v in range(num_vals):
                    mask = (bin_id == v)
                    counts[v] = P[mask].sum()
                H_r = _shannon_entropy_from_counts(counts)
                H_vals.append(H_r)
                bin_ids.append(bin_id)
                counts_list.append(counts)
            
            r_star = int(np.argmax(H_vals))
            M = H_vals[r_star]
            if M <= 1e-9:
                break
            
            grad_d_P = -np.log2(np.maximum(counts_d[bin_d], 1e-15)) - log2_factor
            counts_r = counts_list[r_star]
            bin_r = bin_ids[r_star]
            grad_m_P = -np.log2(np.maximum(counts_r[bin_r], 1e-15)) - log2_factor
            grad_P = (grad_d_P * M - H_d * grad_m_P) / (M * M)
            
            P_new = P + 0.01 * grad_P
            P_new = _clip_and_normalize(P_new)
            ratio_new = _compute_ratio(P_new)
            
            if ratio_new > ratio:
                P = P_new
                ratio = ratio_new
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_P = P.copy()
            else:
                break
    
    return _clip_and_normalize(best_P)

# EVOLVE-BLOCK-END

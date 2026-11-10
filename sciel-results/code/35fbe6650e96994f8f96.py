# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_ratio(P):
    """Compute D/M ratio for a given distribution P of shape (10,10)."""
    P = np.maximum(P, 0.0)
    s = P.sum()
    if s <= 1e-9:
        return 0.0
    P = P / s
    
    n = 10
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)  # X[y,x]=x, Y[y,x]=y
    
    # D = H(x - y)
    vals_d = (X - Y).astype(int)
    min_d, max_d = vals_d.min(), vals_d.max()
    dist_d = np.zeros(max_d - min_d + 1)
    for i in range(min_d, max_d + 1):
        mask = (vals_d == i)
        dist_d[i - min_d] = P[mask].sum()
    dist_d = dist_d[dist_d > 1e-15]
    H_d = -np.sum(dist_d * np.log2(dist_d)) if len(dist_d) > 0 else 0.0
    
    # M = max(H(x), H(x+y), H(x+2y))
    # H(x)
    dist_x = np.zeros(n)
    for xi in range(n):
        dist_x[xi] = P[:, xi].sum()
    dist_x = dist_x[dist_x > 1e-15]
    H_x = -np.sum(dist_x * np.log2(dist_x)) if len(dist_x) > 0 else 0.0
    
    # H(x+y)
    vals_s = (X + Y).astype(int)
    min_s, max_s = vals_s.min(), vals_s.max()
    dist_s = np.zeros(max_s - min_s + 1)
    for i in range(min_s, max_s + 1):
        mask = (vals_s == i)
        dist_s[i - min_s] = P[mask].sum()
    dist_s = dist_s[dist_s > 1e-15]
    H_s = -np.sum(dist_s * np.log2(dist_s)) if len(dist_s) > 0 else 0.0
    
    # H(x+2y)
    vals_2 = (X + 2 * Y).astype(int)
    min_2, max_2 = vals_2.min(), vals_2.max()
    dist_2 = np.zeros(max_2 - min_2 + 1)
    for i in range(min_2, max_2 + 1):
        mask = (vals_2 == i)
        dist_2[i - min_2] = P[mask].sum()
    dist_2 = dist_2[dist_2 > 1e-15]
    H_2 = -np.sum(dist_2 * np.log2(dist_2)) if len(dist_2) > 0 else 0.0
    
    M = max(H_x, H_s, H_2)
    if M <= 1e-9:
        return 0.0
    
    return H_d / M


def _compute_gradient_signal(P):
    """
    Compute a signal for multiplicative updates.
    We want to increase P where it helps D and decrease where it hurts M.
    Approximate: for each cell, estimate marginal contribution to D and M.
    """
    n = 10
    P = np.maximum(P, 0.0)
    s = P.sum()
    if s <= 1e-9:
        return np.ones((n, n))
    Pn = P / s
    
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)
    
    vals_d = (X - Y).astype(int)
    vals_s = (X + Y).astype(int)
    vals_2 = (X + 2 * Y).astype(int)
    
    # For each projection, compute the probability mass of each bucket
    def bucket_probs(vals, Pn):
        min_v, max_v = vals.min(), vals.max()
        probs = np.zeros(max_v - min_v + 1)
        for i in range(min_v, max_v + 1):
            mask = (vals == i)
            probs[i - min_v] = Pn[mask].sum()
        return probs, min_v
    
    probs_d, min_d = bucket_probs(vals_d, Pn)
    probs_s, min_s = bucket_probs(vals_s, Pn)
    probs_2, min_2 = bucket_probs(vals_2, Pn)
    
    # Entropy contribution: for a bucket with prob p, marginal effect of adding small mass
    # to a cell in that bucket: dH/dp = -log2(p) - 1/ln(2) ... approximately
    # For multiplicative updates, we use: score += alpha * (contribution to D) - beta * (contribution to M)
    
    # Signal for D: cells in rare x-y buckets should be boosted
    # Signal for M: cells in rare x, x+y, x+2y buckets should be penalized (to keep M low)
    
    signal = np.zeros((n, n))
    
    # Boost for D: -log2(p_bucket) for x-y bucket
    for i in range(n):
        for j in range(n):
            di = vals_d[i, j] - min_d
            si = vals_s[i, j] - min_s
            t2i = vals_2[i, j] - min_2
            xi = j
            
            p_d = probs_d[di]
            p_s = probs_s[si]
            p_2 = probs_2[t2i]
            p_x = Pn[:, xi].sum()
            
            # Contribution to D (want high): -log2(p_d)
            # Contribution to M (want low): penalize if this cell is in a rare bucket for x, s, or 2
            # Use max of the three M components' rareness
            
            if p_d > 1e-15:
                d_score = -np.log2(p_d)
            else:
                d_score = 10.0
            
            # For M, we want to keep the max entropy low. 
            # If a bucket is rare, adding mass to it increases entropy a lot.
            # So penalize cells in rare buckets for x, x+y, x+2y.
            m_penalties = []
            if p_x > 1e-15:
                m_penalties.append(-np.log2(p_x))
            if p_s > 1e-15:
                m_penalties.append(-np.log2(p_s))
            if p_2 > 1e-15:
                m_penalties.append(-np.log2(p_2))
            
            m_penalty = max(m_penalties) if m_penalties else 0.0
            
            signal[i, j] = d_score - m_penalty
    
    return signal


def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, keep safety margin
    
    # Start with uniform
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P.copy()
    best_ratio = _compute_ratio(P)
    
    # Multiplicative weight updates with exponential annealing
    alpha = 0.5  # step size
    lr_decay = 0.9995
    
    max_iters = 50000
    for it in range(max_iters):
        if time.time() > deadline - 30.0:
            break
        
        # Compute signal
        signal = _compute_gradient_signal(P)
        
        # Normalize signal
        sig_mean = signal.mean()
        sig_std = signal.std()
        if sig_std > 1e-15:
            signal = (signal - sig_mean) / sig_std
        
        # Multiplicative update
        P = P * np.exp(alpha * signal)
        
        # Clip and normalize
        P = np.maximum(P, 0.0)
        s = P.sum()
        if s > 1e-9:
            P = P / s
        
        # Compute ratio
        ratio = _compute_ratio(P)
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P.copy()
        
        # Decay learning rate
        alpha *= lr_decay
        if alpha < 1e-6:
            break
    
    # Also try some structured initializations and quick refinements
    # Try anti-diagonal concentrated distributions
    for c in range(0, 2 * n):  # x + y = c
        if time.time() > deadline - 30.0:
            break
        P_try = np.zeros((n, n))
        for xi in range(n):
            yi = c - xi
            if 0 <= yi < n:
                P_try[yi, xi] = 1.0
        s = P_try.sum()
        if s > 0:
            P_try = P_try / s
            r = _compute_ratio(P_try)
            if r > best_ratio:
                best_ratio = r
                best_P = P_try.copy()
    
    # Try diagonal concentrated distributions x - y = c
    for c in range(-(n-1), n):
        if time.time() > deadline - 30.0:
            break
        P_try = np.zeros((n, n))
        for xi in range(n):
            yi = xi - c
            if 0 <= yi < n:
                P_try[yi, xi] = 1.0
        s = P_try.sum()
        if s > 0:
            P_try = P_try / s
            r = _compute_ratio(P_try)
            if r > best_ratio:
                best_ratio = r
                best_P = P_try.copy()
    
    # Try mixture of anti-diagonals: put mass on x+y in a small set of values
    for c_min in range(0, n):
        if time.time() > deadline - 30.0:
            break
        for c_max in range(c_min, min(c_min + 3, 2*n-1)):
            P_try = np.zeros((n, n))
            count = 0
            for xi in range(n):
                for yi in range(n):
                    if c_min <= xi + yi <= c_max:
                        P_try[yi, xi] = 1.0
                        count += 1
            if count > 0:
                P_try = P_try / count
                r = _compute_ratio(P_try)
                if r > best_ratio:
                    best_ratio = r
                    best_P = P_try.copy()
    
    # Final: ensure valid output
    P = np.maximum(best_P, 0.0)
    s = P.sum()
    if s <= 1e-9:
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        P = P / s
    
    return P


# EVOLVE-BLOCK-END

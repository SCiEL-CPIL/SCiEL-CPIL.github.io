"""
Alternative Strategy: Structural Search + Dense Local Refinement.

The current program uses a smooth surrogate (log-sum-exp) for the entire 50-dimensional
space. This program instead hypothesizes that the optimal solution has a specific
"sparse" or "block" structure where many h_i are exactly zero. 

1. We enumerate candidate supports (sets of active indices) using random subsets 
   and structured patterns (e.g., contiguous blocks, symmetric patterns).
2. For each support, we optimize only the active variables using L-BFGS-B with a 
   very high smoothing parameter (effectively exact for the max term) or by 
   directly minimizing the ratio if possible.
3. We use a "peeling" heuristic: start with a broad support, optimize, then try 
   removing variables that contribute little to the numerator but much to the 
   denominator (peak).
"""

import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 149.0     # safety margin before 150s kill
RNG = np.random.RandomState(50)

def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y

def compute_value(h):
    """Exact evaluator metric for given heights (defensive)."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _exact_value_for_support(h_full, support_mask):
    """
    Compute value assuming h is zero outside support_mask.
    Uses exact max (no smoothing) because we are at the end of local refinement.
    """
    h = np.zeros(N)
    h[support_mask] = h_full[support_mask]
    return compute_value(h)

def _optimize_support(h_init, support_mask, time_budget):
    """
    Optimize the active variables in h_init constrained to support_mask.
    Uses L-BFGS-B with a high-beta smooth surrogate to handle the max term,
    but since the dimensionality is lower (only active vars), it converges faster.
    """
    active_indices = np.where(support_mask)[0]
    n_active = len(active_indices)
    
    if n_active == 0:
        return h_init, 0.0
    
    # Extract active variables
    x0 = h_init[active_indices].copy()
    
    # If only 1 active variable, we can solve analytically or just scale
    # Actually, for 1 var, g is scaled by x^2. Value is scale invariant?
    # l2 ~ x^4, norm_1 ~ x^2, norm_inf ~ x^2. Ratio is constant.
    # So for n_active=1, any non-zero value gives same score.
    # We just need to ensure it's non-zero.
    if n_active == 1:
        x0[0] = 1.0
        h_full = np.zeros(N)
        h_full[active_indices] = x0
        val = compute_value(h_full)
        return h_full, val

    def objective(x):
        h = np.zeros(N)
        h[active_indices] = np.maximum(0.0, x)
        g = np.convolve(h, h)
        m = len(g)
        y = _build_y(g)
        a = y[:m + 1]
        b = y[1:m + 2]
        l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
        norm_1 = np.sum(g) / (m + 1)
        
        # High beta smoothing for max
        g_max = np.max(g)
        if g_max <= 0:
            return 0.0
        beta = 1000.0
        lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
        
        if norm_1 <= 0 or lse <= 0:
            return 0.0
            
        val = l2 / (norm_1 * lse)
        return -val

    bounds = [(0, None)] * n_active
    
    try:
        res = opt.minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        if res.fun < 0:
            x_opt = np.maximum(0.0, res.x)
            h_full = np.zeros(N)
            h_full[active_indices] = x_opt
            val = compute_value(h_full)
            return h_full, val
    except Exception:
        pass
        
    return h_init, compute_value(h_init)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = 0.0
    best_h = np.ones(N)
    
    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
    
    # 1. Initial broad starts
    x = np.linspace(0, N - 1, N)
    starts = [
        np.ones(N),
        np.exp(-0.5 * ((x - N / 2) / (N / 4)) ** 2),
        np.sqrt(np.maximum(0.0, 1 - ((2 * x / (N - 1) - 1)) ** 2)),
        1 - 2 * np.abs(x - (N - 1) / 2) / (N - 1)
    ]
    
    for s in starts:
        track(s)
    
    # 2. Structural Search: Try different supports
    # We will try a series of support masks.
    # Strategy: Start with full support, optimize. Then try to "prune" variables
    # that are small or contribute to peak without contributing to bulk.
    
    # Generate a diverse set of support masks to explore
    support_candidates = []
    
    # Full support
    support_candidates.append(np.ones(N, dtype=bool))
    
    # Contiguous blocks centered
    for width in [10, 15, 20, 25, 30]:
        start_idx = (N - width) // 2
        mask = np.zeros(N, dtype=bool)
        mask[start_idx:start_idx+width] = True
        support_candidates.append(mask)
        
    # Symmetric pairs
    for k in range(5, 20, 2):
        mask = np.zeros(N, dtype=bool)
        center = N // 2
        for i in range(k):
            mask[center - i] = True
            mask[center + i] = True
        support_candidates.append(mask)
        
    # Random sparse supports
    for _ in range(50):
        k = RNG.randint(10, 30)
        indices = RNG.choice(N, k, replace=False)
        mask = np.zeros(N, dtype=bool)
        mask[indices] = True
        support_candidates.append(mask)
        
    # Random "blocky" supports
    for _ in range(50):
        mask = np.zeros(N, dtype=bool)
        i = 0
        while i < N:
            if RNG.rand() > 0.5:
                width = RNG.randint(2, 8)
                mask[i:i+width] = True
                i += width
            else:
                i += 1
        support_candidates.append(mask)

    # Sort candidates by complexity (number of active vars) to handle easier ones first?
    # Or just iterate.
    
    for mask in support_candidates:
        if time.time() > deadline - 20.0:
            break
            
        # Initialize active variables with best_h values, or ones
        h_init = best_h.copy()
        # Ensure non-zero initialization for active vars if they were zero
        active_idx = np.where(mask)[0]
        if np.all(h_init[active_idx] < 1e-6):
            h_init[active_idx] = 1.0
        
        # Optimize on this support
        h_opt, val_opt = _optimize_support(h_init, mask, time_budget=10.0)
        
        if val_opt > best_val:
            track(h_opt)
            
        # "Peeling" heuristic: If we found a good solution, try removing
        # variables with smallest magnitude to see if value increases (reduces peak)
        if val_opt > 0.9 * best_val:
            h_current = h_opt
            mask_current = mask.copy()
            
            for _ in range(5): # Try up to 5 peeling steps
                if time.time() > deadline - 10.0:
                    break
                    
                active_idx = np.where(mask_current)[0]
                if len(active_idx) <= 2:
                    break
                    
                # Find the smallest active variable
                mags = h_current[active_idx]
                min_idx_local = np.argmin(mags)
                min_idx_global = active_idx[min_idx_local]
                
                # Try removing it
                mask_test = mask_current.copy()
                mask_test[min_idx_global] = False
                
                h_test = h_current.copy()
                h_test[min_idx_global] = 0.0
                
                # Optimize again with reduced support
                h_new, val_new = _optimize_support(h_test, mask_test, time_budget=5.0)
                
                if val_new > best_val:
                    track(h_new)
                    h_current = h_new
                    mask_current = mask_test
                else:
                    break # If removing didn't help, stop peeling
    
    # 3. Final global polish with high beta
    if time.time() < deadline - 5.0:
        def obj_global(x):
            h = np.maximum(0.0, x)
            g = np.convolve(h, h)
            m = len(g)
            y = _build_y(g)
            a = y[:m + 1]
            b = y[1:m + 2]
            l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
            norm_1 = np.sum(g) / (m + 1)
            g_max = np.max(g)
            if g_max <= 0: return 0.0
            beta = 500.0
            lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
            if norm_1 <= 0 or lse <= 0: return 0.0
            return -(l2 / (norm_1 * lse))
            
        try:
            res = opt.minimize(
                obj_global,
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.fun < 0:
                track(res.x)
        except Exception:
            pass
            
    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))

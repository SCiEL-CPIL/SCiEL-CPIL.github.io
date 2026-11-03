import time
import numpy as np

N = 100
SCALE = 2.0 / N

def _compute_O(h):
    """
    Compute the overlap vector O for all shifts k=0..N-1.
    O_pos[k] = SCALE * sum_{j=k}^{N-1} h[j]*(1-h[j-k])
    O_neg[k] = SCALE * sum_{j=0}^{N-1-k} h[j]*(1-h[j+k])
    """
    S1 = np.cumsum(h[::-1])[::-1]
    S2 = np.array([np.dot(h[k:], h[:N - k]) for k in range(N)])
    O_pos = SCALE * (S1 - S2)
    
    csum = np.cumsum(h)
    S1n = csum[N - 1 - np.arange(N)]
    S2n = np.array([np.dot(h[:N - k], h[k:]) for k in range(N)])
    O_neg = SCALE * (S1n - S2n)
    
    return np.concatenate([O_pos, O_neg])

def _compute_grad_O(h):
    """
    Compute gradients of each O_k with respect to h.
    Returns G_pos (N x N) and G_neg (N x N) where G_pos[k, i] = dO_pos[k]/dh[i]
    """
    K = np.arange(N)[:, None]
    I = np.arange(N)[None, :]
    
    ge = (I >= K)
    lt = (I < (N - K))
    
    idx_p = np.clip(I - K, 0, N - 1)
    idx_m = np.clip(I + K, 0, N - 1)
    
    Gpos = SCALE * (ge.astype(float) - ge * h[idx_p] - lt * h[idx_m])
    Gneg = SCALE * (lt.astype(float) - lt * h[idx_m] - ge * h[idx_p])
    
    return Gpos, Gneg

def _project(z):
    """
    Project z onto {h in [0,1], sum h = N/2}
    """
    lo = float(z.min()) - 1.0
    hi = float(z.max()) + 1.0
    target = N / 2.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        s = np.clip(z - mid, 0.0, 1.0).sum()
        if s > target:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return np.clip(z - t, 0.0, 1.0)

def _fista(h_init, deadline, max_iter=3000):
    """
    FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) on the smooth surrogate.
    """
    h = _project(np.asarray(h_init, dtype=float).copy())
    y = h.copy()
    t_k = 1.0
    
    best_h = h.copy()
    best_val = float(_compute_O(h).max())
    
    # Adaptive step size with backtracking
    L = 100.0  # Initial Lipschitz estimate
    
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        # Compute gradient at y
        O = _compute_O(y)
        Omax = float(O.max())
        tau = 50.0 + 50.0 * it / max_iter  # Increasing tau for sharper max approximation
        z = tau * (O - Omax)
        ez = np.exp(z - z.max())  # Numerical stability
        w = ez / ez.sum()
        w_pos, w_neg = w[:N], w[N:]
        
        Gpos, Gneg = _compute_grad_O(y)
        grad = w_pos @ Gpos + w_neg @ Gneg
        
        if not np.all(np.isfinite(grad)):
            break
        
        # Backtracking line search
        step = 1.0 / L
        while True:
            h_new = _project(y - step * grad)
            # Check decrease condition (simplified)
            O_new = _compute_O(h_new)
            val_new = float(O_new.max())
            if val_new <= Omax + 1e-8 or step < 1e-10:
                break
            step *= 0.5
            L *= 2.0
        
        # FISTA update
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k))
        y = h_new + ((t_k - 1.0) / t_new) * (h_new - h)
        y = _project(y)
        h = h_new
        t_k = t_new
        
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
    
    return best_h, best_val

def _local_search_bb(h_init, deadline, max_iter=5000):
    """
    Local search on bang-bang (0/1) solutions.
    """
    h = _project(np.asarray(h_init, dtype=float).copy())
    # Snap to bang-bang
    h_bb = (h > 0.5).astype(float)
    # Ensure sum is N/2
    num_ones = int(h_bb.sum())
    if num_ones > N // 2:
        # Remove excess ones
        idxs = np.where(h_bb == 1.0)[0]
        to_remove = np.random.default_rng(42).choice(idxs, num_ones - N // 2, replace=False)
        h_bb[to_remove] = 0.0
    elif num_ones < N // 2:
        # Add needed ones
        idxs = np.where(h_bb == 0.0)[0]
        to_add = np.random.default_rng(42).choice(idxs, N // 2 - num_ones, replace=False)
        h_bb[to_add] = 1.0
    
    best_h = h_bb.copy()
    best_val = float(_compute_O(h_bb).max())
    
    rng = np.random.default_rng(123)
    
    for it in range(max_iter):
        if time.time() > deadline - 3.0:
            break
        
        # Pick a random pair to swap
        i, j = rng.choice(N, 2, replace=False)
        if h_bb[i] == h_bb[j]:
            continue
        
        h_bb[i], h_bb[j] = h_bb[j], h_bb[i]
        val = float(_compute_O(h_bb).max())
        
        if val < best_val - 1e-12:
            best_val = val
            best_h = h_bb.copy()
        else:
            h_bb[i], h_bb[j] = h_bb[j], h_bb[i]  # Revert
    
    return best_h, best_val

def _polish(h, best_val, deadline):
    """
    High-precision polish with FISTA and then local search.
    """
    h = h.copy()
    current_val = best_val
    
    # Phase 1: FISTA with high tau
    h_fista, val_fista = _fista(h, deadline - 5.0, max_iter=2000)
    if val_fista < current_val:
        current_val = val_fista
        h = h_fista
    
    # Phase 2: Bang-bang local search
    h_bb, val_bb = _local_search_bb(h, deadline - 2.0, max_iter=3000)
    if val_bb < current_val:
        current_val = val_bb
        h = h_bb
    
    # Phase 3: Final FISTA from bang-bang
    h_final, val_final = _fista(h, deadline, max_iter=1000)
    if val_final < current_val:
        current_val = val_final
        h = h_final
    
    return h, current_val

def _fixed_point_iteration(h_init, deadline, max_iter=2000):
    """
    Fixed-point iteration based on optimality conditions.
    This provides a new search direction that can escape local minima.
    """
    h = _project(np.asarray(h_init, dtype=float).copy())
    best_h = h.copy()
    best_val = float(_compute_O(h).max())
    
    # Use a soft max with increasing tau
    tau = 30.0
    step_size = 0.1
    
    for it in range(max_iter):
        if time.time() > deadline - 3.0:
            break
        
        O = _compute_O(h)
        Omax = float(O.max())
        
        # Soft max weights
        z = tau * (O - Omax)
        ez = np.exp(z - z.max())
        w = ez / ez.sum()
        w_pos, w_neg = w[:N], w[N:]
        
        Gpos, Gneg = _compute_grad_O(h)
        grad = w_pos @ Gpos + w_neg @ Gneg
        
        # Fixed point update: h = project(h - step * grad + lambda)
        # where lambda is chosen to maintain sum constraint
        # We use a simple gradient step with projection
        h_new = _project(h - step_size * grad)
        val_new = float(_compute_O(h_new).max())
        
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
            h = h_new
            # Decay step size slowly
            step_size *= 0.999
        else:
            # If stuck, try a random perturbation
            if it % 100 == 0:
                rng = np.random.default_rng(it)
                h = _project(h + 0.01 * rng.standard_normal(N))
        
        # Increase tau gradually
        if it % 200 == 0 and it > 0:
            tau *= 1.5
    
    return best_h, best_val

def construct():
    start = time.time()
    deadline = start + 145.0
    
    seed = np.full(N, 0.5)
    best_h = seed.copy()
    best_val = float(_compute_O(seed).max())
    
    idx = np.arange(N)
    inits = []
    
    rng = np.random.default_rng(42)
    
    # Bang-bang random inits
    for _ in range(12):
        bb = np.zeros(N)
        bb[rng.choice(N, N // 2, replace=False)] = 1.0
        inits.append(bb)
        
    # Structured continuous inits
    for f in (1, 2, 3, 5, 7, 11, 13):
        inits.append(0.5 + 0.4 * np.cos(2.0 * np.pi * f * idx / N))
        
    # Random continuous
    for _ in range(12):
        inits.append(rng.random(N))
        
    # Periodic bang-bang patterns
    for period in [2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 17, 19]:
        bb = np.zeros(N)
        ones = period // 2
        pattern = np.zeros(period)
        pattern[:ones] = 1.0
        bb = np.tile(pattern, N // period + 1)[:N]
        if bb.sum() > N/2:
             idxs = np.where(bb==1)[0]
             to_remove = rng.choice(idxs, int(bb.sum() - N/2), replace=False)
             bb[to_remove] = 0
        elif bb.sum() < N/2:
             idxs = np.where(bb==0)[0]
             to_add = rng.choice(idxs, int(N/2 - bb.sum()), replace=False)
             bb[to_add] = 1
        inits.append(bb)
    
    # Optimize each init with FISTA
    for init in inits:
        if time.time() > deadline - 20.0:
            break
        try:
            h, val = _fista(init, deadline)
            if np.all(np.isfinite(h)) and val < best_val:
                best_h = h
                best_val = val
        except Exception:
            continue
    
    # Apply fixed-point iteration to escape local minima
    if time.time() < deadline - 15.0:
        try:
            h_fp, val_fp = _fixed_point_iteration(best_h, deadline)
            if val_fp < best_val:
                best_h = h_fp
                best_val = val_fp
        except Exception:
            pass
    
    # Final polish
    best_h, best_val = _polish(best_h, best_val, deadline)
    
    # Ensure exact sum and bounds
    best_h = _project(best_h)
    for _ in range(10):
        d = N / 2.0 - best_h.sum()
        if abs(d) < 1e-12:
            break
        best_h = np.clip(best_h + d / N, 0.0, 1.0)
        best_h = _project(best_h)
        
    return best_h

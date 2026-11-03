import time
import numpy as np

N = 100
SCALE = 2.0 / N

def _compute_O_fft(h):
    """
    Compute overlap values using FFT for efficiency.
    O_pos[k] = SCALE * sum_{j=k}^{N-1} h[j]*(1-h[j-k])
    O_neg[k] = SCALE * sum_{j=0}^{N-1-k} h[j]*(1-h[j+k])
    
    We can rewrite:
    O_pos[k] = SCALE * (sum_{j=k}^{N-1} h[j] - sum_{j=k}^{N-1} h[j]*h[j-k])
    O_neg[k] = SCALE * (sum_{j=0}^{N-1-k} h[j] - sum_{j=0}^{N-1-k} h[j]*h[j+k])
    
    The cross-correlation terms can be computed via FFT.
    """
    # Pad to avoid circular convolution issues
    size = 2 * N
    h_pad = np.zeros(size)
    h_pad[:N] = h
    
    # Compute autocorrelation via FFT
    H = np.fft.rfft(h_pad)
    autocorr = np.fft.irfft(H * np.conj(H), n=size)
    
    # autocorr[k] = sum_j h_pad[j] * h_pad[j+k] for k >= 0
    # We need sum_{j=k}^{N-1} h[j]*h[j-k] which is autocorr[k] but shifted
    # Actually autocorr[k] = sum_{j=0}^{size-1-k} h_pad[j]*h_pad[j+k]
    # For k in 0..N-1: autocorr[k] includes terms where j+k can go up to N-1+k
    # We want sum_{j=k}^{N-1} h[j]*h[j-k] = sum_{i=0}^{N-1-k} h[i+k]*h[i] = autocorr[k]
    # But autocorr[k] from full size might include extra terms. Let's be careful.
    
    # Let me just use direct computation but vectorized better
    # Actually for N=100, O(N^2) is fine. Let me keep the original but optimize.
    
    S1 = np.cumsum(h[::-1])[::-1]
    
    # Compute all dot products h[k:] · h[:N-k] efficiently
    # h[k:] has length N-k, h[:N-k] has length N-k
    S2 = np.array([np.dot(h[k:], h[:N - k]) for k in range(N)])
    O_pos = SCALE * (S1 - S2)
    
    csum = np.cumsum(h)
    S1n = csum[N - 1 - np.arange(N)]
    S2n = np.array([np.dot(h[:N - k], h[k:]) for k in range(N)])
    O_neg = SCALE * (S1n - S2n)
    
    return np.concatenate([O_pos, O_neg])

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

def _simulated_annealing(h_init, deadline, max_iter=50000):
    """
    Simulated annealing on the continuous space [0,1]^N with sum constraint.
    Uses a cooling schedule and random perturbations.
    """
    h = _project(np.asarray(h_init, dtype=float).copy())
    best_h = h.copy()
    best_val = float(_compute_O_fft(h).max())
    
    T_start = 0.05
    T_end = 0.001
    rng = np.random.default_rng(42)
    
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        # Cooling schedule
        progress = it / max_iter
        T = T_start * (T_end / T_start) ** progress
        
        # Generate a random perturbation
        # Perturb a random subset of elements
        n_perturb = rng.integers(1, max(2, N // 10) + 1)
        indices = rng.choice(N, n_perturb, replace=False)
        deltas = rng.normal(0, 0.1, size=n_perturb)
        
        h_new = h.copy()
        h_new[indices] += deltas
        h_new = _project(h_new)
        
        val_new = float(_compute_O_fft(h_new).max())
        delta_val = val_new - best_val
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta_val < 0 or rng.random() < np.exp(-delta_val / max(T, 1e-10)):
            h = h_new
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
    
    return best_h, best_val

def _local_refine(h, deadline, max_iter=10000):
    """
    Local refinement using small random steps with acceptance criterion.
    """
    h = h.copy()
    best_h = h.copy()
    best_val = float(_compute_O_fft(h).max())
    
    rng = np.random.default_rng(123)
    
    for it in range(max_iter):
        if time.time() > deadline - 1.0:
            break
        
        # Small random perturbation
        idx = rng.integers(0, N)
        delta = rng.normal(0, 0.01)
        
        h_new = h.copy()
        h_new[idx] += delta
        h_new = _project(h_new)
        
        val_new = float(_compute_O_fft(h_new).max())
        
        if val_new < best_val - 1e-12:
            h = h_new
            best_val = val_new
            best_h = h_new.copy()
        elif val_new < best_val:
            # Accept if slightly better
            h = h_new
    
    return best_h, best_val

def _bang_bang_local(h_init, deadline, max_iter=20000):
    """
    Local search on bang-bang (0/1) solutions with pair swaps.
    """
    h = _project(np.asarray(h_init, dtype=float).copy())
    h_bb = (h > 0.5).astype(float)
    
    # Ensure sum is N/2
    num_ones = int(h_bb.sum())
    rng = np.random.default_rng(42)
    if num_ones > N // 2:
        idxs = np.where(h_bb == 1.0)[0]
        to_remove = rng.choice(idxs, num_ones - N // 2, replace=False)
        h_bb[to_remove] = 0.0
    elif num_ones < N // 2:
        idxs = np.where(h_bb == 0.0)[0]
        to_add = rng.choice(idxs, N // 2 - num_ones, replace=False)
        h_bb[to_add] = 1.0
    
    best_h = h_bb.copy()
    best_val = float(_compute_O_fft(h_bb).max())
    
    rng2 = np.random.default_rng(123)
    
    for it in range(max_iter):
        if time.time() > deadline - 1.0:
            break
        
        i, j = rng2.choice(N, 2, replace=False)
        if h_bb[i] == h_bb[j]:
            continue
        
        h_bb[i], h_bb[j] = h_bb[j], h_bb[i]
        val = float(_compute_O_fft(h_bb).max())
        
        if val < best_val - 1e-12:
            best_val = val
            best_h = h_bb.copy()
        else:
            h_bb[i], h_bb[j] = h_bb[j], h_bb[i]  # Revert
    
    return best_h, best_val

def construct():
    start = time.time()
    deadline = start + 145.0
    
    idx = np.arange(N)
    rng = np.random.default_rng(42)
    
    # Generate diverse initial solutions
    inits = []
    
    # Uniform
    inits.append(np.full(N, 0.5))
    
    # Structured continuous
    for f in (1, 2, 3, 5, 7, 11, 13, 17, 23, 31):
        inits.append(0.5 + 0.4 * np.cos(2.0 * np.pi * f * idx / N))
    
    # Random continuous
    for _ in range(20):
        inits.append(rng.random(N))
    
    # Bang-bang random
    for _ in range(20):
        bb = np.zeros(N)
        bb[rng.choice(N, N // 2, replace=False)] = 1.0
        inits.append(bb)
    
    # Periodic bang-bang patterns
    for period in [2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 17, 19, 23, 29]:
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
    
    # Run simulated annealing on each init
    best_h = inits[0].copy()
    best_val = float(_compute_O_fft(best_h).max())
    
    per_init_time = 8.0  # ~8 seconds per init
    for init in inits:
        if time.time() > deadline - 30.0:
            break
        try:
            h_sa, val_sa = _simulated_annealing(init, start + time.time() - start + per_init_time)
            if val_sa < best_val:
                best_val = val_sa
                best_h = h_sa
        except Exception:
            continue
    
    # Local refinement
    if time.time() < deadline - 15.0:
        h_ref, val_ref = _local_refine(best_h, deadline - 10.0)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref
    
    # Bang-bang local search
    if time.time() < deadline - 10.0:
        h_bb, val_bb = _bang_bang_local(best_h, deadline - 5.0)
        if val_bb < best_val:
            best_val = val_bb
            best_h = h_bb
    
    # Final local refinement
    if time.time() < deadline - 3.0:
        h_final, val_final = _local_refine(best_h, deadline - 1.0)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
    
    # Ensure exact sum and bounds
    best_h = _project(best_h)
    for _ in range(10):
        d = N / 2.0 - best_h.sum()
        if abs(d) < 1e-12:
            break
        best_h = np.clip(best_h + d / N, 0.0, 1.0)
        best_h = _project(best_h)
        
    return best_h

import time
import numpy as np

N = 100
SCALE = 2.0 / N

def _compute_O(h):
    """
    Compute the overlap vector O for all shifts k=0..N-1.
    O_pos[k] = SCALE * sum_{j=k}^{N-1} h[j]*(1-h[j-k])
    O_neg[k] = SCALE * sum_{j=0}^{N-1-k} h[j]*(1-h[j+k])
    Uses FFT for O(N log N) speed.
    """
    n_fft = 2 * N
    H = np.fft.rfft(h, n_fft)
    H1 = np.fft.rfft(1.0 - h, n_fft)
    
    # Cross-correlation: corr[m] = sum_i h[i+m] * (1-h[i]) for m >= 0
    corr_pos = np.fft.irfft(np.conj(H) * H1, n_fft)[:N]
    # Cross-correlation: corr[m] = sum_i h[i] * (1-h[i+m]) for m >= 0  
    corr_neg = np.fft.irfft(H * np.conj(H1), n_fft)[:N]
    
    O_pos = SCALE * corr_pos
    O_neg = SCALE * corr_neg
    
    return np.concatenate([O_pos, O_neg])

def _step(h, tau):
    """
    Compute the gradient of the softmax-smoothed max overlap.
    """
    O = _compute_O(h)
    Omax = float(O.max())
    z = tau * (O - Omax)
    ez = np.exp(z - z.max())  # numerical stability
    w = ez / ez.sum()
    w_pos, w_neg = w[:N], w[N:]
    
    # Gradient computation using vectorized operations
    K = np.arange(N)[:, None]  # (N, 1)
    I = np.arange(N)[None, :]  # (1, N)
    
    # For O_pos[k] = sum_{i=0}^{N-1-k} h[i+k]*(1-h[i])
    # d/dh[i] = sum_k w_pos[k] * [(1-h[i]) * (i>=k) - h[i+k] * (i<N-k)]
    
    ge = (I >= K)  # (N, N): I[i,k] = (i >= k)
    lt = (I < (N - K))  # (N, N): I[i,k] = (i < N-k)
    
    # (1-h[i]) for all i,k
    one_minus_h_i = (1.0 - h)[None, :]  # (1, N)
    # h[i+k] for valid i,k where i+k < N
    idx_p = np.clip(I + K, 0, N - 1)
    h_ipk = h[idx_p]  # (N, N)
    
    # Term 1: (1-h[i]) * (i>=k)
    term1 = one_minus_h_i * ge.astype(float)  # (N, N)
    # Term 2: h[i+k] * (i<N-k)
    term2 = h_ipk * lt.astype(float)  # (N, N)
    
    # grad_pos[i] = sum_k w_pos[k] * (term1[i,k] - term2[i,k])
    grad_pos = SCALE * (w_pos[None, :] * (term1 - term2)).sum(axis=1)
    
    # For O_neg[k] = sum_{i=0}^{N-1-k} h[i]*(1-h[i+k])
    # d/dh[i] = sum_k w_neg[k] * [(1-h[i+k]) * (i<N-k) - h[i] * (i>=k)]
    
    # (1-h[i+k]) for valid i,k
    h_ipk_1 = 1.0 - h_ipk  # (N, N)
    # h[i] for all i,k
    h_i = h[:, None]  # (N, 1) broadcast to (N, N)
    
    # Term 1: (1-h[i+k]) * (i<N-k)
    term1_neg = h_ipk_1 * lt.astype(float)  # (N, N)
    # Term 2: h[i] * (i>=k)
    term2_neg = h_i * ge.astype(float)  # (N, N)
    
    # grad_neg[i] = sum_k w_neg[k] * (term1_neg[i,k] - term2_neg[i,k])
    grad_neg = SCALE * (w_neg[None, :] * (term1_neg - term2_neg)).sum(axis=1)
    
    grad = grad_pos + grad_neg
    return Omax, grad

def _project(z):
    """
    Project z onto {h in [0,1], sum h = N/2}
    """
    lo = float(z.min()) - 1.0
    hi = float(z.max()) + 1.0
    target = N / 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        s = np.clip(z - mid, 0.0, 1.0).sum()
        if s > target:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return np.clip(z - t, 0.0, 1.0)

def _optimize(init, deadline):
    h = _project(np.asarray(init, dtype=float).copy())
    
    # Schedule of tau values (smoothness)
    tau_schedule = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    best_h_local = h.copy()
    best_val_local = float(_compute_O(h).max())
    
    for tau in tau_schedule:
        # Adaptive learning rate based on tau
        base_eta = 0.3 / (1.0 + 0.05 * tau)
        
        # More iterations for smaller tau, fewer for larger tau
        n_iter = max(50, int(300 / (1.0 + 0.02 * tau)))
        
        for _ in range(n_iter):
            if time.time() > deadline - 2.0:
                return best_h_local, best_val_local
                
            Omax, grad = _step(h, tau)
            if not np.all(np.isfinite(grad)):
                break
            
            eta = base_eta
            h_new = _project(h - eta * grad)
            O_new = _compute_O(h_new)
            val_new = float(O_new.max())
            
            if val_new < best_val_local:
                best_h_local = h_new.copy()
                best_val_local = val_new
                
            h = h_new

    return best_h_local, best_val_local

def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin

    seed = np.full(N, 0.5)
    best_h = seed.copy()
    best_val = float(_compute_O(seed).max())  # 0.5

    idx = np.arange(N)
    inits = []
    
    # 1. Bang-bang random inits
    rng = np.random.default_rng(42)
    for _ in range(15):
        bb = np.zeros(N)
        bb[rng.choice(N, N // 2, replace=False)] = 1.0
        inits.append(bb)
        
    # 2. Structured continuous inits
    for f in (1, 2, 3, 4, 5, 7, 11, 13):
        inits.append(0.5 + 0.4 * np.cos(2.0 * np.pi * f * idx / N))
        
    # 3. Random continuous
    for _ in range(15):
        inits.append(rng.random(N))
        
    # 4. "Sawtooth" or periodic bang-bang
    for period in [3, 4, 5, 6, 7, 8, 10, 11, 13, 17]:
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
        
    # 5. Block patterns
    for block_size in [2, 3, 4, 5, 7]:
        for offset in range(block_size):
            bb = np.zeros(N)
            for i in range(N):
                if (i - offset) % (2 * block_size) < block_size:
                    bb[i] = 1.0
            if bb.sum() > N/2:
                 idxs = np.where(bb==1)[0]
                 to_remove = rng.choice(idxs, int(bb.sum() - N/2), replace=False)
                 bb[to_remove] = 0
            elif bb.sum() < N/2:
                 idxs = np.where(bb==0)[0]
                 to_add = rng.choice(idxs, int(N/2 - bb.sum()), replace=False)
                 bb[to_add] = 1
            inits.append(bb)
    
    # 6. Low-rank structure inits: h[i] = clip(u[i]*v[i], 0, 1)
    for a in [0.3, 0.5, 0.7]:
        for b in [0.3, 0.5, 0.7]:
            u = a + (1-a) * np.cos(2.0 * np.pi * idx / N)
            v = b + (1-b) * np.cos(4.0 * np.pi * idx / N)
            inits.append(np.clip(u * v * 2.0, 0, 1))
    
    for init in inits:
        if time.time() > deadline - 10.0:
            break
        try:
            h, val = _optimize(init, deadline)
            if np.all(np.isfinite(h)) and val < best_val:
                best_h = h
                best_val = val
        except Exception:
            continue

    # Final polish with adaptive tau and eta - two phases
    h = best_h.copy()
    
    # Phase 1: moderate tau
    tau = 200.0
    eta = 0.05
    for _ in range(3000):
        if time.time() > deadline - 5.0:
            break
        Omax, grad = _step(h, tau)
        if not np.all(np.isfinite(grad)):
            break
        h_new = _project(h - eta * grad)
        val_new = float(_compute_O(h_new).max())
        if val_new < best_val:
            best_h = h_new
            best_val = val_new
            h = h_new
            eta = min(eta * 1.02, 0.1)
        else:
            eta *= 0.9
    
    # Phase 2: high tau for fine-tuning
    tau = 1000.0
    eta = 0.01
    for _ in range(2000):
        if time.time() > deadline - 1.0:
            break
        Omax, grad = _step(h, tau)
        if not np.all(np.isfinite(grad)):
            break
        h_new = _project(h - eta * grad)
        val_new = float(_compute_O(h_new).max())
        if val_new < best_val:
            best_h = h_new
            best_val = val_new
            h = h_new
            eta = min(eta * 1.01, 0.02)
        else:
            eta *= 0.95

    # Ensure exact sum and bounds
    best_h = _project(best_h)
    for _ in range(20):
        d = N / 2.0 - best_h.sum()
        if abs(d) < 1e-14:
            break
        best_h = np.clip(best_h + d / N, 0.0, 1.0)
        best_h = _project(best_h)
        
    # Verify final result
    final_val = float(_compute_O(best_h).max())
    if final_val < best_val:
        best_val = final_val
    
    return best_h

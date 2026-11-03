import time
import numpy as np

N = 100
SCALE = 2.0 / N

def _compute_O_binary(h):
    """
    Compute overlap for binary h (0/1 values).
    O[k] = SCALE * count of j where h[j]=1 and h[j-k mod N]=0
    Using FFT for speed.
    """
    # For binary h: h[j]*(1-h[j-k]) = 1 when h[j]=1 and h[j-k]=0
    # Sum over j: this is the cross-correlation
    
    # Use FFT: cross_corr[k] = sum_j h[j] * (1-h[j-k])
    # = sum_j h[j] - sum_j h[j]*h[j-k]
    # = N/2 - autocorr[k]
    
    # autocorr[k] = sum_j h[j]*h[j-k] (circular)
    n_fft = 2 * N
    H = np.fft.rfft(h, n_fft)
    autocorr = np.fft.irfft(np.conj(H) * H, n_fft)[:N]
    
    # O[k] = SCALE * (N/2 - autocorr[k]) for circular shifts
    # But we need non-circular: sum over valid j only
    # For simplicity in search, use circular version as proxy
    
    O = SCALE * (N / 2.0 - autocorr)
    return O

def _compute_O_exact(h):
    """Exact non-circular overlap computation."""
    n_fft = 2 * N
    H = np.fft.rfft(h, n_fft)
    H1 = np.fft.rfft(1.0 - h, n_fft)
    
    corr_pos = np.fft.irfft(np.conj(H) * H1, n_fft)[:N]
    corr_neg = np.fft.irfft(H * np.conj(H1), n_fft)[:N]
    
    O_pos = SCALE * corr_pos
    O_neg = SCALE * corr_neg
    
    return np.concatenate([O_pos, O_neg])

def _compute_O(h):
    """Compute overlap - use exact for final eval, fast for search."""
    return _compute_O_exact(h)

def _project(z):
    """Project z onto {h in [0,1], sum h = N/2}"""
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

def _binary_local_search(h_init, deadline, max_iters=50000):
    """
    Simulated annealing on binary configurations.
    Move: flip two bits (one 0->1, one 1->0) to maintain sum=N/2.
    """
    h = h_init.copy()
    # Ensure binary and correct sum
    h = (h > 0.5).astype(float)
    ones = np.where(h == 1)[0]
    zeros = np.where(h == 0)[0]
    if len(ones) > N // 2:
        to_flip = np.random.choice(ones, len(ones) - N//2, replace=False)
        h[to_flip] = 0
    elif len(ones) < N // 2:
        to_flip = np.random.choice(zeros, N//2 - len(ones), replace=False)
        h[to_flip] = 1
    
    current_O = _compute_O_binary(h)
    current_max = float(current_O.max())
    best_h = h.copy()
    best_max = current_max
    
    rng = np.random.default_rng(12345)
    
    T_start = 0.1
    T_end = 0.001
    cooling = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Pick a random 1 and a random 0 to swap
        ones = np.where(h == 1)[0]
        zeros = np.where(h == 0)[0]
        i = rng.choice(ones)
        j = rng.choice(zeros)
        
        if i == j:
            continue
        
        # Try the flip
        h[i] = 0
        h[j] = 1
        
        new_O = _compute_O_binary(h)
        new_max = float(new_O.max())
        
        delta = new_max - current_max
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_h = h.copy()
        else:
            # Reject
            h[i] = 1
            h[j] = 0
        
        T *= cooling
    
    return best_h, best_max

def construct():
    start = time.time()
    deadline = start + 140.0  # Safety margin

    # Start with a good initial binary solution
    # Use a structured pattern as seed
    idx = np.arange(N)
    
    # Initial: alternating pattern adjusted for N=100
    h0 = np.zeros(N)
    h0[::2] = 1.0  # 50 ones
    
    best_h = h0.copy()
    best_val = float(_compute_O(h0).max())
    
    # Multiple restarts with different initial patterns
    rng = np.random.default_rng(42)
    
    inits = []
    
    # 1. Alternating
    inits.append(h0.copy())
    
    # 2. Block patterns
    for block in [3, 4, 5, 7]:
        h = np.zeros(N)
        for i in range(N):
            if (i % (2*block)) < block:
                h[i] = 1.0
        # Adjust to have exactly N/2 ones
        ones = np.where(h == 1)[0]
        zeros = np.where(h == 0)[0]
        if len(ones) > N//2:
            to_flip = rng.choice(ones, len(ones) - N//2, replace=False)
            h[to_flip] = 0
        elif len(ones) < N//2:
            to_flip = rng.choice(zeros, N//2 - len(ones), replace=False)
            h[to_flip] = 1
        inits.append(h)
    
    # 3. Random binary
    for _ in range(5):
        h = np.zeros(N)
        h[rng.choice(N, N//2, replace=False)] = 1.0
        inits.append(h)
    
    # 4. Structured: low-discrepancy like
    for freq in [3, 5, 7, 11]:
        h = (np.sin(2 * np.pi * freq * idx / N) > 0).astype(float)
        ones = np.where(h == 1)[0]
        zeros = np.where(h == 0)[0]
        if len(ones) > N//2:
            to_flip = rng.choice(ones, len(ones) - N//2, replace=False)
            h[to_flip] = 0
        elif len(ones) < N//2:
            to_flip = rng.choice(zeros, N//2 - len(ones), replace=False)
            h[to_flip] = 1
        inits.append(h)
    
    # Run binary local search from each init
    for init in inits:
        if time.time() > deadline - 15.0:
            break
        try:
            h, val = _binary_local_search(init, deadline)
            # Convert to exact evaluation
            exact_val = float(_compute_O(h).max())
            if exact_val < best_val:
                best_val = exact_val
                best_h = h.copy()
        except Exception:
            continue
    
    # Final: polish with continuous optimization from best binary solution
    h = best_h.copy()
    
    # Softmax-smoothed max optimization for final polish
    tau = 500.0
    eta = 0.01
    for _ in range(2000):
        if time.time() > deadline - 3.0:
            break
        
        O = _compute_O(h)
        Omax = float(O.max())
        z = tau * (O - Omax)
        ez = np.exp(z)
        w = ez / ez.sum()
        w_pos, w_neg = w[:N], w[N:]
        
        K = np.arange(N)[:, None]
        I = np.arange(N)[None, :]
        
        ge = (I >= K)
        lt = (I < (N - K))
        
        idx_p = np.clip(I - K, 0, N - 1)
        idx_m = np.clip(I + K, 0, N - 1)
        
        Gpos = SCALE * (ge.astype(float) - ge * h[idx_p] - lt * h[idx_m])
        Gneg = SCALE * (lt.astype(float) - lt * h[idx_m] - ge * h[idx_p])
        
        grad = w_pos @ Gpos + w_neg @ Gneg
        
        if not np.all(np.isfinite(grad)):
            break
        
        h_new = _project(h - eta * grad)
        val_new = float(_compute_O(h_new).max())
        
        if val_new < best_val:
            best_val = val_new
            best_h = h_new
            h = h_new
            eta = min(eta * 1.01, 0.05)
        else:
            eta *= 0.95
    
    # Ensure exact constraints
    best_h = _project(best_h)
    for _ in range(20):
        d = N / 2.0 - best_h.sum()
        if abs(d) < 1e-14:
            break
        best_h = np.clip(best_h + d / N, 0.0, 1.0)
        best_h = _project(best_h)
    
    # Final verification
    final_val = float(_compute_O(best_h).max())
    if final_val < best_val:
        best_val = final_val
    
    return best_h

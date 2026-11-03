import time
import numpy as np

N = 100
SCALE = 2.0 / N
TARGET = N / 2.0

def _compute_O(h):
    """
    Compute the overlap vector O for all shifts k=0..N-1.
    O_pos[k] = SCALE * sum_{j=k}^{N-1} h[j]*(1-h[j-k])
    O_neg[k] = SCALE * sum_{j=0}^{N-1-k} h[j]*(1-h[j+k])
    """
    # O_pos[k] = SCALE * (sum_{j=k}^{N-1} h[j] - sum_{j=k}^{N-1} h[j]*h[j-k])
    # Let's compute this more efficiently using FFT or direct computation
    
    # For O_pos[k]: sum_{j=k}^{N-1} h[j] - sum_{j=k}^{N-1} h[j]*h[j-k]
    # The first term is the tail sum
    S1 = np.cumsum(h[::-1])[::-1]
    
    # For the dot product part, we can use a loop but it's O(N^2)
    # Alternative: use np.correlate or manual computation
    S2 = np.zeros(N)
    for k in range(N):
        if k == 0:
            S2[k] = np.dot(h, h)
        else:
            S2[k] = np.dot(h[k:], h[:N - k])
    O_pos = SCALE * (S1 - S2)
    
    # For O_neg[k]: sum_{j=0}^{N-1-k} h[j] - sum_{j=0}^{N-1-k} h[j]*h[j+k]
    csum = np.cumsum(h)
    S1n = csum[N - 1 - np.arange(N)]
    
    S2n = np.zeros(N)
    for k in range(N):
        if k == 0:
            S2n[k] = np.dot(h, h)
        else:
            S2n[k] = np.dot(h[:N - k], h[k:])
    O_neg = SCALE * (S1n - S2n)
    
    return np.concatenate([O_pos, O_neg])

def _step(h, tau):
    """
    Compute the gradient of the softmax-smoothed max overlap.
    """
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
    return Omax, grad

def _coordinate_descent_polish(h, deadline, max_iters=500):
    """
    Coordinate descent polish: try adjusting individual coordinates to reduce max overlap.
    This can escape local minima found by gradient-based methods.
    """
    best_h = h.copy()
    best_val = float(_compute_O(h).max())
    
    for _ in range(max_iters):
        if time.time() > deadline - 2.0:
            break
            
        improved = False
        for i in range(N):
            if time.time() > deadline - 2.0:
                break
                
            # Try small perturbations in each direction
            for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                h_trial = best_h.copy()
                h_trial[i] = np.clip(best_h[i] + delta, 0.0, 1.0)
                
                # Adjust to maintain sum constraint
                d = TARGET - h_trial.sum()
                if abs(d) > 1e-12:
                    # Distribute the correction to other elements
                    mask = np.ones(N, dtype=bool)
                    mask[i] = False
                    if h_trial[mask].sum() > 0:
                        # Scale down/up
                        factor = 1.0 - d / h_trial[mask].sum()
                        h_trial[mask] *= factor
                        h_trial = np.clip(h_trial, 0.0, 1.0)
                    
                    # If still off, use projection
                    h_trial = _project(h_trial)
                
                val_trial = float(_compute_O(h_trial).max())
                if val_trial < best_val - 1e-10:
                    best_h = h_trial.copy()
                    best_val = val_trial
                    improved = True
                    break
        
        if not improved:
            # Try larger steps
            for i in range(N):
                if time.time() > deadline - 2.0:
                    break
                for delta in [0.05, -0.05, 0.1, -0.1]:
                    h_trial = best_h.copy()
                    h_trial[i] = np.clip(best_h[i] + delta, 0.0, 1.0)
                    h_trial = _project(h_trial)
                    
                    val_trial = float(_compute_O(h_trial).max())
                    if val_trial < best_val - 1e-10:
                        best_h = h_trial.copy()
                        best_val = val_trial
                        improved = True
                        break
            
            if not improved:
                break
    
    return best_h, best_val

def _project(z):
    """
    Project z onto {h in [0,1], sum h = N/2}
    """
    lo = float(z.min()) - 1.0
    hi = float(z.max()) + 1.0
    target = TARGET
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        s = np.clip(z - mid, 0.0, 1.0).sum()
        if s > target:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return np.clip(z - t, 0.0, 1.0)

def _optimize(init, deadline, verbose=False):
    h = _project(np.asarray(init, dtype=float).copy())
    
    # Schedule of tau values (smoothness)
    tau_schedule = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    best_h_local = h.copy()
    best_val_local = float(_compute_O(h).max())
    
    giter = 0
    for tau in tau_schedule:
        base_eta = 0.5 / (1.0 + 0.1 * tau)
        
        for _ in range(200):
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
            giter += 1

    return best_h_local, best_val_local

def _polish_two_pass(h, deadline):
    """
    Two-pass adaptive polish: first pass with increasing tau, then refine.
    """
    best_h = h.copy()
    best_val = float(_compute_O(h).max())
    
    # Pass 1: geometrically increase tau
    tau = 100.0
    for i in range(8):
        if time.time() > deadline - 2.0:
            break
        eta = 0.5 / (1.0 + 0.1 * tau)
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            Omax, grad = _step(h, tau)
            if not np.all(np.isfinite(grad)):
                break
            h_new = _project(h - eta * grad)
            val_new = float(_compute_O(h_new).max())
            if val_new < best_val:
                best_h = h_new.copy()
                best_val = val_new
                h = h_new
            else:
                eta *= 0.95
        tau *= 2.0
    
    # Pass 2: refine with very high tau and small steps
    tau = 2000.0
    eta = 0.005
    for _ in range(200):
        if time.time() > deadline - 1.0:
            break
        Omax, grad = _step(h, tau)
        if not np.all(np.isfinite(grad)):
            break
        h_new = _project(h - eta * grad)
        val_new = float(_compute_O(h_new).max())
        if val_new < best_val:
            best_h = h_new.copy()
            best_val = val_new
            h = h_new
        else:
            eta *= 0.95
    
    # Pass 3: Coordinate descent polish
    if time.time() < deadline - 5.0:
        best_h, best_val = _coordinate_descent_polish(best_h, deadline, max_iters=200)
    
    return best_h, best_val

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
    for _ in range(5):
        bb = np.zeros(N)
        bb[rng.choice(N, N // 2, replace=False)] = 1.0
        inits.append(bb)
        
    # 2. Structured continuous inits
    for f in (1, 2, 3, 5):
        inits.append(0.5 + 0.4 * np.cos(2.0 * np.pi * f * idx / N))
        
    # 3. Random continuous
    for _ in range(5):
        inits.append(rng.random(N))
        
    # 4. "Sawtooth" or periodic bang-bang
    for period in [3, 4, 5, 7, 11]:
        bb = np.zeros(N)
        ones = period // 2
        pattern = np.zeros(period)
        pattern[:ones] = 1.0
        bb = np.tile(pattern, N // period + 1)[:N]
        if bb.sum() > TARGET:
            idxs = np.where(bb == 1)[0]
            to_remove = rng.choice(idxs, int(bb.sum() - TARGET), replace=False)
            bb[to_remove] = 0
        elif bb.sum() < TARGET:
            idxs = np.where(bb == 0)[0]
            to_add = rng.choice(idxs, int(TARGET - bb.sum()), replace=False)
            bb[to_add] = 1
        inits.append(bb)
    
    # 5. Product of two cosine sequences (separable structure)
    for f1 in [1, 2, 3]:
        for f2 in [1, 2, 3, 4]:
            a = 0.5 + 0.3 * np.cos(2.0 * np.pi * f1 * idx / N)
            b = 0.5 + 0.3 * np.cos(2.0 * np.pi * f2 * idx / N)
            prod = a * b
            prod = prod - prod.mean() + 0.5
            inits.append(prod)
    
    # 6. Additional diverse inits: sine waves, mixed frequencies
    for f in [1, 2, 3, 4, 5, 6]:
        inits.append(0.5 + 0.4 * np.sin(2.0 * np.pi * f * idx / N))
    
    # 7. Block structures with varying block sizes
    for block_size in [4, 5, 6, 8, 10]:
        n_blocks = N // block_size
        blocks = []
        for b in range(n_blocks):
            if b % 2 == 0:
                blocks.append(np.ones(block_size) * 0.8)
            else:
                blocks.append(np.ones(block_size) * 0.2)
        bb = np.concatenate(blocks)[:N]
        bb = _project(bb)
        inits.append(bb)

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

    # Final polish with two-pass adaptive strategy
    h = best_h.copy()
    best_h, best_val = _polish_two_pass(h, deadline)
    
    # Additional coordinate descent rounds if time permits
    if time.time() < deadline - 3.0:
        best_h, best_val = _coordinate_descent_polish(best_h, deadline, max_iters=300)

    # Ensure exact sum and bounds
    best_h = _project(best_h)
    for _ in range(10):
        d = TARGET - best_h.sum()
        if abs(d) < 1e-12:
            break
        best_h = np.clip(best_h + d / N, 0.0, 1.0)
        best_h = _project(best_h)
        
    return best_h

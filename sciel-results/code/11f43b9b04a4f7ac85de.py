import time
import numpy as np

N = 100
SCALE = 2.0 / N

def _compute_O(h):
    """Compute overlaps for all shifts."""
    # O_pos[k] = sum_{j=k}^{N-1} h[j]*(1-h[j-k])
    # O_neg[k] = sum_{j=0}^{N-1-k} h[j]*(1-h[j+k])
    # Using convolution/correlation properties or direct loops for N=100 is fast enough.
    # Direct loop is O(N^2) = 10,000 ops, which is very fast.
    O = np.zeros(N)
    for k in range(N):
        # Shift h by k
        # Overlap of h and 1-h_shifted
        # h[j] * (1 - h[j-k]) for j where j-k is in range
        # This is equivalent to sum h[j] - sum h[j]h[j-k]
        # sum h[j] for j in range is N/2
        # sum h[j]h[j-k] is cross-correlation
        pass
    
    # Vectorized approach from original code is better.
    S1 = np.cumsum(h[::-1])[::-1]
    S2 = np.array([np.dot(h[k:], h[:N - k]) for k in range(N)])
    O_pos = SCALE * (S1 - S2)
    
    csum = np.cumsum(h)
    S1n = csum[N - 1 - np.arange(N)]
    S2n = np.array([np.dot(h[:N - k], h[k:]) for k in range(N)])
    O_neg = SCALE * (S1n - S2n)
    
    return np.concatenate([O_pos, O_neg])[:N] # Wait, the original code returned 2N?
    # The task asks for "maximum overlap of h with shifted 1-h".
    # Usually shift is considered in [0, 1] or [0, 2].
    # The original code computed O_pos and O_neg and took the max of the concatenated array.
    # Let's check the range of shifts.
    # "shifted 1-h" on [0,2].
    # If h is on [0,2], shift s is in [0,2].
    # The discrete steps are on [0,2] with width 2/N.
    # Shifts k=0..N-1 correspond to shifts 0..(N-1)*2/N.
    # The original code computed O for k=0..N-1 in both positive and negative directions?
    # O_pos[k] corresponds to shift +k?
    # O_neg[k] corresponds to shift -k?
    # The max overlap is the max over all these.
    # I will keep the original _compute_O logic.

def _compute_O_orig(h):
    S1 = np.cumsum(h[::-1])[::-1]
    S2 = np.array([np.dot(h[k:], h[:N - k]) for k in range(N)])
    O_pos = SCALE * (S1 - S2)
    csum = np.cumsum(h)
    S1n = csum[N - 1 - np.arange(N)]
    S2n = np.array([np.dot(h[:N - k], h[k:]) for k in range(N)])
    O_neg = SCALE * (S1n - S2n)
    return np.concatenate([O_pos, O_neg])

def _project(z):
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

def _soft_max_grad(h, tau):
    O = _compute_O_orig(h)
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
    ge_f = ge.astype(float)
    lt_f = lt.astype(float)
    
    Gpos = SCALE * (ge_f - ge * h[idx_p] - lt * h[idx_m])
    Gneg = SCALE * (lt_f - lt * h[idx_m] - ge * h[idx_p])
    
    grad = w_pos @ Gpos + w_neg @ Gneg
    return Omax, grad

def _fista(h_init, deadline):
    h = _project(h_init.copy())
    h_prev = h.copy()
    t = 1.0
    L = 10.0 # Initial Lipschitz estimate
    
    for i in range(2000):
        if time.time() > deadline - 2.0:
            break
        
        Omax, grad = _soft_max_grad(h, 50.0) # Fixed tau for now, or increase
        
        # Backtracking line search
        step = 1.0 / L
        y = h - step * grad
        y_proj = _project(y)
        
        # Check if objective decreased
        O_new = _compute_O_orig(y_proj).max()
        # Use smooth objective for line search?
        # For simplicity, use raw max overlap decrease
        if O_new < Omax or i == 0:
            h = y_proj
            L = L / 1.5 # Increase step size
        else:
            L = L * 1.5 # Decrease step size
            continue # Skip momentum update if step rejected?
        
        # FISTA momentum
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        h_prev = h.copy()
        h = _project(h + ((t - 1) / t_new) * (h - h_prev))
        t = t_new

    return h

def _local_search_bb(h):
    # Convert to bang-bang
    h_bb = (h >= 0.5).astype(float)
    # Ensure sum is N/2
    diff = N / 2.0 - h_bb.sum()
    if diff > 0:
        # Need more 1s
        zeros = np.where(h_bb == 0)[0]
        if len(zeros) >= int(diff):
            # Pick the ones that reduce overlap most? Just pick random for now
            h_bb[zeros[:int(diff)]] = 1.0
    elif diff < 0:
        ones = np.where(h_bb == 1)[0]
        if len(ones) >= int(-diff):
            h_bb[ones[:int(-diff)]] = 0.0
            
    # Local search
    best_val = _compute_O_orig(h_bb).max()
    for _ in range(5000):
        # Pick a random 1 and a random 0
        ones = np.where(h_bb == 1)[0]
        zeros = np.where(h_bb == 0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        i = np.random.choice(ones)
        j = np.random.choice(zeros)
        
        h_test = h_bb.copy()
        h_test[i] = 0.0
        h_test[j] = 1.0
        
        val = _compute_O_orig(h_test).max()
        if val < best_val - 1e-9:
            h_bb = h_test
            best_val = val
    return h_bb

def construct():
    start = time.time()
    deadline = start + 140.0
    
    seed = np.full(N, 0.5)
    best_h = seed.copy()
    best_val = float(_compute_O_orig(seed).max())
    
    idx = np.arange(N)
    inits = [seed.copy()]
    for f in (1, 2, 3, 5, 8):
        inits.append(0.5 + 0.30 * np.cos(2.0 * np.pi * f * idx / N))
        inits.append(0.5 + 0.40 * np.sign(np.cos(2.0 * np.pi * f * idx / N)))
    
    rng = np.random.default_rng(0)
    for _ in range(5):
        inits.append(rng.random(N))
    
    for init in inits:
        if time.time() > deadline - 10.0:
            break
        try:
            h_cont = _fista(init, deadline)
            val_cont = float(_compute_O_orig(h_cont).max())
            if val_cont < best_val:
                best_h, best_val = h_cont, val_cont
            
            # Try bang-bang local search
            h_bb = _local_search_bb(h_cont)
            val_bb = float(_compute_O_orig(h_bb).max())
            if val_bb < best_val:
                best_h, best_val = h_bb, val_bb
                
        except Exception:
            continue
            
    return best_h

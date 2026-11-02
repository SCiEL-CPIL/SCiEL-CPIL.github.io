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
    # O_pos[k]
    # sum_{j=k}^{N-1} h[j] - sum_{j=k}^{N-1} h[j]h[j-k]
    S1 = np.cumsum(h[::-1])[::-1]
    S2 = np.array([np.dot(h[k:], h[:N - k]) for k in range(N)])
    O_pos = SCALE * (S1 - S2)
    
    # O_neg[k]
    # sum_{j=0}^{N-1-k} h[j] - sum_{j=0}^{N-1-k} h[j]h[j+k]
    csum = np.cumsum(h)
    S1n = csum[N - 1 - np.arange(N)]
    S2n = np.array([np.dot(h[:N - k], h[k:]) for k in range(N)])
    O_neg = SCALE * (S1n - S2n)
    
    return np.concatenate([O_pos, O_neg])

def _step(h, tau):
    """
    Compute the gradient of the softmax-smoothed max overlap.
    Objective: f(h) = (1/tau) * log( sum_{k} exp(tau * O_k(h)) )
    Grad: sum_k w_k * grad(O_k)
    """
    O = _compute_O(h)
    Omax = float(O.max())
    z = tau * (O - Omax)
    ez = np.exp(z)
    w = ez / ez.sum()
    w_pos, w_neg = w[:N], w[N:]
    
    # Precompute indices for efficiency
    K = np.arange(N)[:, None]
    I = np.arange(N)[None, :]
    
    # For O_pos[k] = sum_{j=k}^{N-1} h[j](1-h[j-k])
    # d/dh[i] O_pos[k]:
    # If i >= k: term h[i](1-h[i-k]) -> (1-h[i-k]) - h[i]
    # If i < k: term h[i-k](1-h[i]) is NOT in the sum for O_pos[k] (sum is j=k..N-1)
    # Wait, the sum is over j. The variables are h[j] and h[j-k].
    # Let's check the derivative properly.
    # O_pos[k] = sum_{j=k}^{N-1} h[j] - sum_{j=k}^{N-1} h[j]h[j-k]
    # d/dh[i] sum_{j=k}^{N-1} h[j] = 1 if i in [k, N-1] else 0
    # d/dh[i] sum_{j=k}^{N-1} h[j]h[j-k]:
    #   Term 1: h[i] appears as h[j] if j=i -> requires i in [k, N-1]. Deriv: h[i-k]
    #   Term 2: h[i] appears as h[j-k] if j-k=i -> j=i+k. Requires i+k in [k, N-1] -> i in [0, N-1-k]. Deriv: h[i+k]
    
    ge = (I >= K) # i >= k
    lt = (I < (N - K)) # i < N - k
    
    idx_p = np.clip(I - K, 0, N - 1) # h[i-k]
    idx_m = np.clip(I + K, 0, N - 1) # h[i+k]
    
    # Grad O_pos[k] w.r.t h[i]
    # 1_{i>=k} - 1_{i>=k}h[i-k] - 1_{i<N-k}h[i+k]
    Gpos = SCALE * (ge.astype(float) - ge * h[idx_p] - lt * h[idx_m])
    
    # For O_neg[k] = sum_{j=0}^{N-1-k} h[j] - sum_{j=0}^{N-1-k} h[j]h[j+k]
    # d/dh[i] sum_{j=0}^{N-1-k} h[j] = 1 if i in [0, N-1-k] else 0
    # d/dh[i] sum_{j=0}^{N-1-k} h[j]h[j+k]:
    #   Term 1: h[i] as h[j] if j=i -> i in [0, N-1-k]. Deriv: h[i+k]
    #   Term 2: h[i] as h[j+k] if j+k=i -> j=i-k. Requires i-k in [0, N-1-k] -> i in [k, N-1]. Deriv: h[i-k]
    
    # 1_{i<N-k} - 1_{i<N-k}h[i+k] - 1_{i>=k}h[i-k]
    Gneg = SCALE * (lt.astype(float) - lt * h[idx_m] - ge * h[idx_p])
    
    grad = w_pos @ Gpos + w_neg @ Gneg
    return Omax, grad

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

def _optimize(init, deadline):
    h = _project(np.asarray(init, dtype=float).copy())
    
    # Schedule of tau values (smoothness)
    # Start with low tau (smooth max, easier to navigate) and increase to sharp max
    tau_schedule = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    best_h_local = h.copy()
    best_val_local = float(_compute_O(h).max())
    
    for tau in tau_schedule:
        if time.time() > deadline - 5.0:
            break
            
        # FISTA-style accelerated gradient descent
        h_curr = h.copy()
        h_prev = h_curr.copy()
        t_k = 1.0
        
        # Lipschitz constant estimate for the smoothed objective
        # The gradient of O_k is bounded, so we can estimate L
        L = 1.0 / SCALE  # Conservative estimate
        
        for it in range(300):
            if time.time() > deadline - 2.0:
                break
                
            Omax, grad = _step(h_curr, tau)
            if not np.all(np.isfinite(grad)):
                break
            
            # Adaptive step size based on Lipschitz constant
            eta = 1.0 / (tau * L)
            
            # FISTA update
            h_next = _project(h_curr - eta * grad)
            
            # Update momentum
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k))
            h_curr = h_next + ((t_k - 1.0) / t_new) * (h_next - h_prev)
            h_curr = _project(h_curr)
            
            O_new = _compute_O(h_curr)
            val_new = float(O_new.max())
            
            if val_new < best_val_local:
                best_h_local = h_curr.copy()
                best_val_local = val_new
            
            h_prev = h_next
            h_curr = h_curr.copy()
            t_k = t_new
            
            # Early termination if convergence
            if np.linalg.norm(h_curr - h_prev) < 1e-10:
                break
        
        h = best_h_local.copy()

    return best_h_local, best_val_local

def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin

    seed = np.full(N, 0.5)
    best_h = seed.copy()
    best_val = float(_compute_O(seed).max())  # 0.5

    idx = np.arange(N)
    inits = []
    
    # 1. Bang-bang random inits (good for minimizing O_0)
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
        # Pattern: 1s for floor(period/2), 0s for rest
        ones = period // 2
        pattern = np.zeros(period)
        pattern[:ones] = 1.0
        bb = np.tile(pattern, N // period + 1)[:N]
        # Adjust to exactly N/2 ones
        if bb.sum() > N/2:
             # Remove excess ones
             idxs = np.where(bb==1)[0]
             to_remove = rng.choice(idxs, int(bb.sum() - N/2), replace=False)
             bb[to_remove] = 0
        elif bb.sum() < N/2:
             idxs = np.where(bb==0)[0]
             to_add = rng.choice(idxs, int(N/2 - bb.sum()), replace=False)
             bb[to_add] = 1
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

    # Final polish with adaptive step size and local search
    h = best_h.copy()
    tau = 1000.0
    eta = 0.05
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
            eta = min(eta * 1.1, 0.5)
        else:
            eta *= 0.7

    # Ensure exact sum and bounds
    best_h = _project(best_h)
    for _ in range(10):
        d = N / 2.0 - best_h.sum()
        if abs(d) < 1e-12:
            break
        best_h = np.clip(best_h + d / N, 0.0, 1.0)
        best_h = _project(best_h)
        
    return best_h

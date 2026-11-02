import time
import numpy as np
from scipy.optimize import minimize

N = 100
SCALE = 2.0 / N

def _compute_O(h):
    """Compute overlaps for all shifts using FFT-based convolution."""
    # O_pos[k] = SCALE * (sum_{j=k}^{N-1} h[j] - sum_{j=k}^{N-1} h[j]h[j-k])
    # O_neg[k] = SCALE * (sum_{j=0}^{N-1-k} h[j] - sum_{j=0}^{N-1-k} h[j]h[j+k])
    
    # For positive shifts:
    # S1_pos[k] = sum_{j=k}^{N-1} h[j]
    # S2_pos[k] = sum_{j=k}^{N-1} h[j]h[j-k]
    
    # S1_pos is just a reversed cumulative sum
    S1_pos = np.zeros(N)
    csum = np.cumsum(h)
    S1_pos = csum[N-1] - csum[:N-1]  # S1_pos[k] = sum_{j=k}^{N-1} h[j]
    S1_pos[N-1] = h[N-1]
    
    # S2_pos[k] = sum_{j=k}^{N-1} h[j]h[j-k]
    # This is a cross-correlation. Let's use FFT.
    # Actually, for each k, it's sum over j of h[j]*h[j-k] for j>=k
    # This is like the autocorrelation of h, but only for the upper triangle
    # Let's just compute it directly since N=100 is small
    S2_pos = np.zeros(N)
    for k in range(N):
        S2_pos[k] = np.dot(h[k:], h[:N-k])
    
    O_pos = SCALE * (S1_pos - S2_pos)
    
    # For negative shifts:
    # S1_neg[k] = sum_{j=0}^{N-1-k} h[j]
    # S2_neg[k] = sum_{j=0}^{N-1-k} h[j]h[j+k]
    
    S1_neg = csum[:N-1]  # S1_neg[k] = sum_{j=0}^{N-1-k} h[j]
    S1_neg[N-1] = 0
    
    S2_neg = np.zeros(N)
    for k in range(N):
        S2_neg[k] = np.dot(h[:N-k], h[k:])
    
    O_neg = SCALE * (S1_neg - S2_neg)
    
    return np.concatenate([O_pos, O_neg])

def _project(z):
    """Project onto {h: 0<=h<=1, sum(h)=N/2} via bisection on threshold."""
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

def _soft_max_obj_and_grad(h, beta=50.0):
    """
    Compute soft-max objective and gradient.
    f(h) = (1/beta) * log(sum_k exp(beta * O_k(h)))
    grad f / grad h = sum_k softmax_k * grad O_k / grad h
    """
    O = _compute_O(h)
    O_shifted = O - np.max(O)
    exp_O = np.exp(beta * O_shifted)
    weights = exp_O / np.sum(exp_O)
    
    val = (1.0 / beta) * np.log(np.sum(exp_O) + 1e-300)
    
    # Compute gradient
    grad = np.zeros(N)
    
    # Positive shifts
    for k in range(N):
        w = weights[k]
        if w > 1e-10:
            # d O_pos[k] / d h[j]
            # O_pos[k] = SCALE * (sum_{j=k}^{N-1} h[j] - sum_{j=k}^{N-1} h[j]h[j-k])
            # For j >= k: d/dh[j] = SCALE * (1 - h[j-k])
            # For j < k: d/dh[j] = SCALE * (-h[j+k])
            for j in range(k, N):
                grad[j] += w * SCALE * (1.0 - h[j-k])
            for j in range(k):
                grad[j] += w * SCALE * (-h[j+k])
    
    # Negative shifts
    for k in range(N):
        w = weights[N+k]
        if w > 1e-10:
            # d O_neg[k] / d h[j]
            # O_neg[k] = SCALE * (sum_{j=0}^{N-1-k} h[j] - sum_{j=0}^{N-1-k} h[j]h[j+k])
            # For j < N-k: d/dh[j] = SCALE * (1 - h[j+k])
            # For j >= N-k: d/dh[j] = SCALE * (-h[j-k])
            for j in range(N-k):
                grad[j] += w * SCALE * (1.0 - h[j+k])
            for j in range(N-k, N):
                grad[j] += w * SCALE * (-h[j-k])
    
    return val, grad

def _local_search_bb(h, max_iters=5000, rng=None):
    """Bang-bang local search."""
    if rng is None:
        rng = np.random.default_rng()
    
    h_bb = (h >= 0.5).astype(float)
    # Ensure sum is exactly N/2
    diff = N / 2.0 - h_bb.sum()
    if abs(diff) > 1e-9:
        if diff > 0:
            zeros = np.where(h_bb == 0)[0]
            if len(zeros) >= int(diff):
                idx = rng.choice(zeros, size=int(diff), replace=False)
                h_bb[idx] = 1.0
        else:
            ones = np.where(h_bb == 1)[0]
            if len(ones) >= int(-diff):
                idx = rng.choice(ones, size=int(-diff), replace=False)
                h_bb[idx] = 0.0
    
    best_val = float(_compute_O(h_bb).max())
    
    for _ in range(max_iters):
        ones = np.where(h_bb == 1)[0]
        zeros = np.where(h_bb == 0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        
        i = rng.choice(ones)
        j = rng.choice(zeros)
        
        h_test = h_bb.copy()
        h_test[i] = 0.0
        h_test[j] = 1.0
        
        val = float(_compute_O(h_test).max())
        if val < best_val - 1e-9:
            h_bb = h_test
            best_val = val
    
    return h_bb, best_val

def construct():
    start = time.time()
    deadline = start + 140.0
    
    best_h = np.full(N, 0.5)
    best_val = float(_compute_O(best_h).max())
    
    idx = np.arange(N)
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    inits = []
    inits.append(np.full(N, 0.5))
    
    # Periodic patterns
    for f in (1, 2, 3, 4, 5, 6, 7, 8):
        for amp in (0.2, 0.3, 0.4, 0.5):
            inits.append(0.5 + amp * np.cos(2.0 * np.pi * f * idx / N))
            inits.append(0.5 + amp * np.sin(2.0 * np.pi * f * idx / N))
    
    # Bang-bang patterns
    for f in (1, 2, 3, 5, 8):
        inits.append((0.5 + 0.45 * np.cos(2.0 * np.pi * f * idx / N) >= 0.5).astype(float))
        inits.append((0.5 + 0.45 * np.sin(2.0 * np.pi * f * idx / N) >= 0.5).astype(float))
    
    # Random
    for _ in range(10):
        inits.append(rng.random(N))
    
    # Block patterns
    for block_size in (2, 3, 4, 5, 7, 10):
        pattern = np.zeros(N)
        for i in range(0, N, block_size):
            pattern[i:i+block_size//2] = 1.0
        inits.append(pattern)
    
    # Sort inits by initial overlap to try promising ones first
    init_vals = []
    for init in inits:
        init_p = _project(init)
        val = float(_compute_O(init_p).max())
        init_vals.append((val, init_p))
    init_vals.sort(key=lambda x: x[0])
    
    for val_init, init in init_vals:
        if time.time() > deadline - 15.0:
            break
        
        try:
            # Try multiple beta values
            for beta in (30.0, 50.0, 80.0):
                if time.time() > deadline - 10.0:
                    break
                
                h0 = init.copy()
                
                def obj(h_flat):
                    val, _ = _soft_max_obj_and_grad(h_flat, beta)
                    return val
                
                def grad_fn(h_flat):
                    _, g = _soft_max_obj_and_grad(h_flat, beta)
                    return g
                
                res = minimize(
                    obj, h0, method='L-BFGS-B', jac=grad_fn,
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 300, 'maxfun': 3000, 'gtol': 1e-6}
                )
                
                h_opt = _project(res.x)
                val_opt = float(_compute_O(h_opt).max())
                
                if val_opt < best_val:
                    best_h = h_opt.copy()
                    best_val = val_opt
                
                # Bang-bang local search from continuous solution
                h_bb, val_bb = _local_search_bb(h_opt, max_iters=3000, rng=rng)
                if val_bb < best_val:
                    best_h = h_bb.copy()
                    best_val = val_bb
                    
        except Exception:
            continue
    
    # Final polish: try bang-bang local search from best_h
    if time.time() < deadline - 10.0:
        try:
            h_bb, val_bb = _local_search_bb(best_h, max_iters=10000, rng=rng)
            if val_bb < best_val:
                best_h = h_bb.copy()
                best_val = val_bb
        except Exception:
            pass
    
    # Ensure sum is exactly N/2
    best_h = _project(best_h)
    
    return best_h

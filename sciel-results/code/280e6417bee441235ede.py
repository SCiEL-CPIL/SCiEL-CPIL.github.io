import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _soft_max(v, tau=0.1):
    """Soft maximum of vector v."""
    v_max = np.max(v)
    # Prevent overflow
    if v_max > 700:
        return v_max
    return v_max + tau * np.log(np.sum(np.exp((v - v_max) / tau)))

def _autoconv_component(h, k_idx):
    """Compute (h*h)_k for a specific index k (0 to 2N-2)."""
    lo = max(0, k_idx - N + 1)
    hi = min(N - 1, k_idx)
    if lo > hi:
        return 0.0
    i_vals = np.arange(lo, hi + 1)
    j_vals = k_idx - i_vals
    return np.sum(h[i_vals] * h[j_vals])

def _is_sidon_like(indices, N):
    """Check if indices form a Sidon set (all pairwise sums distinct)."""
    sums = set()
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # 1. Parametric search for good initializations
    h_init = np.ones(N) / N
    s = _score_raw(h_init)
    if s > best_score:
        best_score = s
        best_h = h_init.copy()
        
    # Gaussians
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N/2, 10):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_gauss /= np.sum(h_gauss)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # Triangular
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N, 10):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                h_tri /= np.sum(h_tri)
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()

    # 2. Structured Search for Sidon-like Supports
    # Sidon sets minimize autoconvolution collisions, keeping the denominator low.
    sidon_inits = []
    
    # Try to construct small Sidon sets and embed them
    # A simple Sidon set construction: powers of 2 or quadratic residues
    # For N=50, we can pick a subset of size k that is Sidon.
    
    # Method 1: Greedy Sidon set construction
    def construct_sidon_set(k, N):
        S = [0]
        while len(S) < k:
            found = False
            for candidate in range(1, N):
                if candidate in S:
                    continue
                # Check if adding candidate breaks Sidon property
                new_S = S + [candidate]
                if _is_sidon_like(new_S, N):
                    S = new_S
                    found = True
                    break
            if not found:
                break
        return S

    # Generate several Sidon sets of different sizes
    for k in [5, 6, 7, 8, 9, 10]:
        if time.time() > deadline - 50: break
        S = construct_sidon_set(k, N)
        if len(S) == k:
            h_s = np.zeros(N)
            h_s[S] = 1.0
            # Add some noise or weight variation to break symmetry?
            # Let's try uniform weights first, then optimization will adjust.
            h_s /= np.sum(h_s)
            sidon_inits.append(h_s)
            
            # Also try weighted Sidon sets (geometric progression on weights)
            for r in [0.8, 1.2, 0.5, 2.0]:
                h_w = np.zeros(N)
                for i, idx in enumerate(S):
                    h_w[idx] = r**i
                h_w /= np.sum(h_w)
                sidon_inits.append(h_w)

    # Method 2: Random Sidon-like sets (random permutation, check)
    for _ in range(50):
        if time.time() > deadline - 40: break
        k = np.random.randint(5, 12)
        S = np.random.choice(N, k, replace=False)
        if _is_sidon_like(S, N):
            h_s = np.zeros(N)
            h_s[S] = 1.0
            h_s /= np.sum(h_s)
            sidon_inits.append(h_s)

    # Evaluate and update best
    for h_init in sidon_inits:
        s = _score_raw(h_init)
        if s > best_score:
            best_score = s
            best_h = h_init.copy()

    # Random sparse initialization (original)
    for _ in range(50):
        if time.time() > deadline: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(5, 15)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = 1.0
        h_rand += np.random.rand(N) * 0.1
        h_rand /= np.sum(h_rand)
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()

    # Optimization using L-BFGS-B on log-scale variables with Soft-Max surrogate
    def neg_obj_soft(x):
        h = np.exp(x)
        h = h / np.sum(h)
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        sm = _soft_max(h_conv, tau=0.01)
        if sm == 0:
            return 0.0
        return -(l2_sq / sm)

    h_start = best_h.copy()
    h_start[h_start < 1e-8] = 1e-8
    x0 = np.log(h_start)
    
    try:
        res = minimize(neg_obj_soft, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12})
        if res.success:
            h_candidate = np.exp(res.x)
            h_candidate /= np.sum(h_candidate)
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate
    except Exception:
        pass

    # Try multiple restarts from Sidon inits and best
    inits = [best_h.copy()]
    inits.extend(sidon_inits[:10]) # Top 10 Sidon inits
    
    # Add some random sparse inits
    for _ in range(10):
        h_s = np.zeros(N)
        k = np.random.randint(5, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # Add smoothed random inits
    for _ in range(10):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r /= np.sum(h_r)
        inits.append(h_r)

    # Optimize from each init
    for h_init in inits:
        if time.time() > deadline - 20: break
        
        if np.sum(h_init) == 0: continue
        h_init = h_init / np.sum(h_init)
        h_init[h_init < 1e-8] = 1e-8
        x_init = np.log(h_init)
        
        try:
            res = minimize(neg_obj_soft, x_init, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
            if res.success:
                h_candidate = np.exp(res.x)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    # NEW: Iterative peak-index refinement (fixed-point on argmax)
    if time.time() < deadline - 25:
        h_ref = best_h / np.sum(best_h)
        
        for iteration in range(10):
            if time.time() > deadline - 5:
                break
            
            h_conv_ref = np.convolve(h_ref, h_ref, mode='full')
            peak_idx = np.argmax(h_conv_ref)
            
            def neg_obj_fixed_peak(x, k_idx):
                h = np.exp(x)
                h = h / np.sum(h)
                l2_sq = np.sum(h**2)
                val = _autoconv_component(h, k_idx)
                if val <= 0:
                    return 0.0
                return -(l2_sq / val)
            
            h_ref[h_ref < 1e-8] = 1e-8
            x_ref = np.log(h_ref)
            
            try:
                res = minimize(neg_obj_fixed_peak, x_ref, args=(peak_idx,),
                              method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-14})
                if res.success:
                    h_candidate = np.exp(res.x)
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
                    h_ref = h_candidate
            except Exception:
                break

    # Final polish with tighter tolerance on the exact (non-soft) objective
    if time.time() < deadline - 10:
        h_start = best_h / np.sum(best_h)
        h_start[h_start < 1e-8] = 1e-8
        x0 = np.log(h_start)
        
        def neg_obj_exact(x):
            h = np.exp(x)
            h = h / np.sum(h)
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            max_conv = np.max(h_conv)
            if max_conv == 0: return 0.0
            return -(l2_sq / max_conv)

        try:
            res = minimize(neg_obj_exact, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-14})
            if res.success:
                h_candidate = np.exp(res.x)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    return list(best_h)

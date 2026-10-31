import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50
DEADLINE_MARGIN = 15.0  # Return 15s before the 150s kill

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    The ratio is homogeneous of degree 0, so scaling h does not change the value.
    We assume h is non-negative.
    """
    h = np.asarray(h, dtype=np.float64)
    # Enforce non-negativity for safety, though optimizers should maintain it
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    
    # Normalize to sum=1 to improve numerical stability and simplify ratio
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    
    # Full convolution
    # Using fftconvolve for speed, though for N=50 direct is also fast.
    # fftconvolve might have tiny numerical errors, but usually fine.
    # Let's stick to np.convolve for exactness if possible, or check.
    # For N=50, np.convolve is O(N^2) = 2500 ops, extremely fast.
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    
    if conv_inf < 1e-15:
        return 0.0
        
    return h2_sq / conv_inf

def neg_ratio_soft(params, beta=100.0):
    """
    Smooth surrogate objective: Minimize peak of autoconvolution (soft-max)
    subject to fixed L2 norm (via normalization).
    
    We maximize ||h||_2^2 / (||h||_1 ||h*h||_inf).
    Let h = exp(params).
    Ratio R = (sum h^2) / (sum h * max(conv(h))).
    Since R is scale invariant, we can fix sum(h^2) = 1 (or similar) and minimize sum(h) * max(conv(h)).
    Alternatively, fix sum(h)=1 and minimize max(conv(h)) / sum(h^2).
    
    Let's stick to the direct ratio but smooth the max.
    Objective to minimize: -log( sum(h^2) / (sum(h) * soft_max(conv(h))) )
    = log(sum(h) * soft_max(conv(h))) - log(sum(h^2))
    
    soft_max(x) = log(sum(exp(beta * x))) / beta
    
    This is differentiable.
    """
    h = np.exp(params)
    if not np.all(np.isfinite(h)):
        return 1e10
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 1e10
        
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    
    conv = np.convolve(h_norm, h_norm, mode='full')
    
    # Soft max of conv
    # conv values are small, beta scales them
    # Use stable log-sum-exp
    c_max = np.max(conv)
    # If c_max is very small, log-sum-exp might be unstable or just c_max
    if c_max < 1e-15:
        return 1e10
        
    # log(sum(exp(beta * (conv - c_max)))) + beta * c_max
    # We want soft_max = (1/beta) * log(sum(exp(beta * conv)))
    # = c_max + (1/beta) * log(sum(exp(beta * (conv - c_max))))
    
    shifted = beta * (conv - c_max)
    # Clip for numerical stability, though max is 0 so exp <= 1
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    soft_max_val = c_max + (1.0 / beta) * log_sum_exp
    
    if soft_max_val < 1e-15:
        return 1e10
        
    # We want to minimize: - Ratio = - (h2_sq / soft_max_val)
    # Or maximize log(Ratio) = log(h2_sq) - log(soft_max_val)
    # Minimize -log(Ratio) = log(soft_max_val) - log(h2_sq)
    
    return np.log(soft_max_val) - np.log(h2_sq)

def construct():
    t0 = time.time()
    deadline = t0 + 150.0 - DEADLINE_MARGIN
    
    def elapsed():
        return time.time() - t0

    # Initial best: single spike
    best_h = np.zeros(N, dtype=np.float64)
    best_h[N // 2] = 1.0
    best_score = compute_ratio(best_h)
    
    candidates = []
    
    # Helper to add candidate
    def add_candidate(h):
        nonlocal best_h, best_score
        h = np.maximum(h, 0.0)
        if np.sum(h) < 1e-15:
            return
        sc = compute_ratio(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
        candidates.append((sc, h.copy()))
        
    # --- Phase 1: Generate Structured Candidates ---
    
    # 1. Single spike (already set)
    add_candidate(best_h)
    
    # 2. Two spikes
    for i in range(N):
        for j in range(i+1, N):
            h = np.zeros(N)
            h[i] = 0.5
            h[j] = 0.5
            add_candidate(h)
            if elapsed() > deadline - 5:
                return list(best_h)
                
    # 3. Three spikes
    # Sample a subset of triplets to save time
    # Focus on "spread out" triplets
    for i in range(0, N, 2):
        for j in range(i+2, N, 2):
            for k in range(j+2, N, 2):
                h = np.zeros(N)
                h[i] = 1.0/3
                h[j] = 1.0/3
                h[k] = 1.0/3
                add_candidate(h)
                if elapsed() > deadline - 5:
                    return list(best_h)
                    
    # 4. Sidon-like sets
    # A Sidon set ensures all pair sums are distinct, minimizing collisions in convolution.
    # This tends to lower ||h*h||_inf relative to ||h||_2^2.
    
    def find_sidon_set_greedy(n, start_idx=0, step=1):
        """
        Greedy construction of a Sidon set.
        """
        A = []
        used_sums = set()
        # We want to pick indices from 0 to N-1
        for x in range(start_idx, N, step):
            valid = True
            new_sums = []
            for a in A:
                s = a + x
                if s in used_sums:
                    valid = False
                    break
                new_sums.append(s)
            if 2 * x in used_sums:
                valid = False
            if valid:
                A.append(x)
                used_sums.update(new_sums)
                used_sums.add(2 * x)
        return A

    # Try different steps and starts to find diverse Sidon sets
    for step in [1, 2, 3]:
        for start in range(0, 5, step):
            A = find_sidon_set_greedy(N, start_idx=start, step=step)
            if len(A) >= 2:
                h = np.zeros(N)
                for idx in A:
                    h[idx] = 1.0
                add_candidate(h)
                if elapsed() > deadline - 5:
                    return list(best_h)

    # 5. Random Sparse Sets
    rng = np.random.default_rng(42)
    for _ in range(20):
        k = rng.integers(2, 10)
        idx = rng.choice(N, size=k, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0
        add_candidate(h)
        if elapsed() > deadline - 5:
            return list(best_h)

    # 6. Uniform
    add_candidate(np.ones(N))
    
    # 7. Gaussian
    x = np.arange(N)
    for sigma in [2, 5, 10, 20]:
        h = np.exp(-0.5 * ((x - N/2) / sigma)**2)
        add_candidate(h)

    # Sort candidates by score, keep top few for refinement
    candidates.sort(key=lambda x: -x[0])
    
    # --- Phase 2: L-BFGS-B Refinement with Soft-Max Surrogate ---
    
    # We refine the top candidates using the smooth surrogate.
    top_candidates = candidates[:5]
    
    for sc, h_init in top_candidates:
        if elapsed() > deadline - 2:
            break
            
        # Ensure h_init has no zeros for log transformation
        h_safe = h_init + 1e-10
        x0 = np.log(h_safe)
        
        try:
            # Use moderate beta for stability, then increase if needed? 
            # Start with beta=50 for smoother landscape
            res = minimize(neg_ratio_soft, x0, method='L-BFGS-B',
                          args=(50.0,),
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if elapsed() > deadline - 1:
                break
                
            h_opt = np.exp(res.x)
            h_opt = np.maximum(h_opt, 0.0)
            if np.sum(h_opt) > 1e-15:
                h_opt = h_opt / np.sum(h_opt)
            
            sc_opt = compute_ratio(h_opt)
            if sc_opt > best_score:
                best_score = sc_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # --- Phase 2b: Random Smooth Starts for Spectral Design ---
    # Try to find better basins by starting from random smooth profiles
    # and optimizing the soft-max objective.
    if elapsed() < deadline - 10:
        for _ in range(10):
            if elapsed() > deadline - 2:
                break
            # Generate random smooth profile
            # Low frequency noise
            freqs = np.fft.rfftfreq(N)
            # Mask low frequencies
            mask = freqs < 0.1
            noise = np.zeros(N)
            # Add random phase and amplitude to low freqs
            for i in range(len(freqs)):
                if mask[i]:
                    if i == 0:
                        noise[i] = rng.normal(0, 1)
                    else:
                        # Real part of FFT
                        pass 
            # Simpler: just use a random polynomial or spline
            # Let's just use random Gaussian bumps
            k_bumps = rng.integers(2, 5)
            centers = rng.uniform(0, N, k_bumps)
            widths = rng.uniform(1, 5, k_bumps)
            amps = rng.uniform(0.1, 1.0, k_bumps)
            
            x = np.arange(N)
            h_rand = np.zeros(N)
            for c, w, a in zip(centers, widths, amps):
                h_rand += a * np.exp(-0.5 * ((x - c) / w)**2)
            
            if np.sum(h_rand) < 1e-6:
                continue
                
            h_safe = h_rand + 1e-10
            x0 = np.log(h_safe)
            
            try:
                res = minimize(neg_ratio_soft, x0, method='L-BFGS-B',
                              args=(100.0,), # Higher beta for sharper peak suppression
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12})
                
                if elapsed() > deadline - 1:
                    break
                    
                h_opt = np.exp(res.x)
                h_opt = np.maximum(h_opt, 0.0)
                if np.sum(h_opt) > 1e-15:
                    h_opt = h_opt / np.sum(h_opt)
                
                sc_opt = compute_ratio(h_opt)
                if sc_opt > best_score:
                    best_score = sc_opt
                    best_h = h_opt.copy()
                    
            except Exception:
                pass

    # --- Phase 3: Final Polish ---
    # If we have time, do one more high-precision pass on the best solution found.
    # Use very high beta for soft-max to approximate true max closely.
    if elapsed() < deadline - 1:
        h_safe = best_h + 1e-12
        x0 = np.log(h_safe)
        try:
            # Use high beta
            res = minimize(neg_ratio_soft, x0, method='L-BFGS-B',
                          args=(1000.0,),
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15})
            h_opt = np.exp(res.x)
            h_opt = np.maximum(h_opt, 0.0)
            if np.sum(h_opt) > 1e-15:
                h_opt = h_opt / np.sum(h_opt)
            sc_opt = compute_ratio(h_opt)
            if sc_opt > best_score:
                best_score = sc_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Ensure non-negative
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

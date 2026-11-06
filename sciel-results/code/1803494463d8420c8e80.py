import numpy as np
import time

def _supnorm(coeffs, n_pts=4096, refine_rounds=5, refine_points=10):
    """
    Estimate max |P(z)| for |z|=1 using dense sampling + golden-section refinement.
    coeffs: list/array of length n
    """
    n = len(coeffs)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    
    # Dense sampling
    angles = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    z = np.exp(1j * angles)
    k = np.arange(n)
    vals = np.sum(coeffs[None, :] * np.power(z[:, None], k[None, :]), axis=1)
    magnitudes = np.abs(vals)
    
    # Find top candidates for refinement
    if n_pts > refine_points:
        top_idx = np.argpartition(magnitudes, -refine_points)[-refine_points:]
    else:
        top_idx = np.arange(n_pts)
    
    best_mag = float(np.max(magnitudes))
    
    # Refine around the top peaks
    window = 2.0 * np.pi / n_pts * 2
    
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    inv_golden = 1.0 - 1.0 / golden_ratio
    
    for idx in top_idx:
        center = angles[idx]
        a = max(0.0, center - window)
        b = min(2.0 * np.pi, center + window)
        
        if b <= a:
            continue
            
        c = b - inv_golden * (b - a)
        d = a + inv_golden * (b - a)
        
        def _eval(angle):
            return np.abs(np.sum(coeffs * np.exp(1j * k * angle)))
        
        fc = _eval(c)
        fd = _eval(d)
        
        for _ in range(refine_rounds):
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + inv_golden * (b - a)
                fd = _eval(d)
            else:
                b = d
                d = c
                fd = fc
                c = b - inv_golden * (b - a)
                fc = _eval(c)
        
        fa = _eval(a)
        fb = _eval(b)
        
        local_best = max(fa, fb, fc, fd)
        if local_best > best_mag:
            best_mag = local_best
    
    return best_mag

def _score(coeffs):
    """Score = supnorm / sqrt(n)"""
    n = len(coeffs)
    return _supnorm(coeffs) / np.sqrt(n)

def _fast_supnorm(coeffs, M, n_pts):
    """Fast sup-norm estimate using precomputed matrix"""
    n = len(coeffs)
    k = np.arange(n)
    vals = coeffs @ M.T
    return np.max(np.abs(vals))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    N = 24
    
    # Precompute sampling matrix for fast batch evaluation
    n_pts_search = 2048
    angles = np.linspace(0.0, 2.0 * np.pi, n_pts_search, endpoint=False)
    z = np.exp(1j * angles)
    k = np.arange(N)
    M = np.power(z[:, None], k[None, :])
    
    best_coeffs = np.ones(N, dtype=np.float64)
    best_fast_score = np.inf
    
    # Batch size for exhaustive search
    batch_size = 50000
    total_candidates = 1 << N
    
    powers = 1 << np.arange(N, dtype=np.uint32)
    
    for start in range(0, total_candidates, batch_size):
        if time.time() > deadline - 15:
            break
        
        end = min(start + batch_size, total_candidates)
        
        indices = np.arange(start, end, dtype=np.uint32)
        
        bits = (indices[:, None] & powers[None, :]) != 0
        C = np.where(bits, 1.0, -1.0)
        
        V = C @ M.T
        mags = np.max(np.abs(V), axis=1)
        
        min_idx_in_batch = np.argmin(mags)
        min_val = mags[min_idx_in_batch]
        
        if min_val < best_fast_score:
            best_fast_score = min_val
            best_coeffs = C[min_idx_in_batch].copy()
            
    # Local search: try flipping each coefficient
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N):
                old_val = best_coeffs[i]
                best_coeffs[i] = -old_val
                new_score = _fast_supnorm(best_coeffs, M, n_pts_search)
                if new_score < best_fast_score:
                    best_fast_score = new_score
                    improved = True
                else:
                    best_coeffs[i] = old_val
    
    # Final high-precision refinement
    best_score = _score(best_coeffs)
    
    return [int(c) for c in best_coeffs]

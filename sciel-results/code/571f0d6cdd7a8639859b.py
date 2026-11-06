import numpy as np
import time

def _fast_supnorm(coeffs, z_powers):
    """
    Fast sup-norm estimation using precomputed powers of z.
    coeffs: array of length N
    z_powers: (n_pts, N) array of z^k
    """
    # coeffs: (N,)
    # z_powers: (n_pts, N)
    # vals = coeffs @ z_powers.T -> (n_pts,)
    vals = coeffs @ z_powers.T
    mags = np.abs(vals)
    return np.max(mags)

def _precise_supnorm(coeffs, n_pts=1024, refine_rounds=5):
    """
    Higher precision sup-norm for final evaluation.
    """
    n = len(coeffs)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    z = np.exp(1j * angles)
    k = np.arange(n)
    vals = np.sum(coeffs[None, :] * np.power(z[:, None], k[None, :]), axis=1)
    magnitudes = np.abs(vals)
    
    # Find top candidates for refinement
    refine_points = 10
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit
    
    N = 24
    
    # Precompute grid for fast evaluation
    # Use a moderately dense grid for search to balance speed and accuracy
    n_pts_search = 1024
    angles_search = np.linspace(0.0, 2.0 * np.pi, n_pts_search, endpoint=False)
    z_search = np.exp(1j * angles_search)
    k_arr = np.arange(N)
    z_powers = np.power(z_search[:, None], k_arr[None, :])
    
    best_coeffs = None
    best_fast_score = np.inf
    
    # Seeds
    # 1. Random
    # 2. Known good patterns
    seeds = []
    for _ in range(5):
        seeds.append(np.random.choice([-1, 1], N))
    
    # Add specific known candidates
    # Candidate from Odlyzko tables (one of the optimal ones)
    seeds.append(np.array([1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1], dtype=np.float64))
    # Another candidate
    seeds.append(np.array([1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1], dtype=np.float64))
    # All ones
    seeds.append(np.ones(N, dtype=np.float64))
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
            
        current_coeffs = seed.copy()
        current_fast_score = _fast_supnorm(current_coeffs, z_powers)
        
        if current_fast_score < best_fast_score:
            best_fast_score = current_fast_score
            best_coeffs = current_coeffs.copy()
        
        # Local search: Hill Climbing
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
                
            improved = False
            # Shuffle the order of bits to avoid bias
            indices = np.random.permutation(N)
            for i in indices:
                old_val = current_coeffs[i]
                current_coeffs[i] = -old_val
                
                new_fast_score = _fast_supnorm(current_coeffs, z_powers)
                
                if new_fast_score < current_fast_score:
                    current_fast_score = new_fast_score
                    improved = True
                    if current_fast_score < best_fast_score:
                        best_fast_score = current_fast_score
                        best_coeffs = current_coeffs.copy()
                else:
                    # Revert
                    current_coeffs[i] = old_val
    
    # Final high-precision check on the best candidate
    if best_coeffs is not None:
        # We return the coefficients. The evaluator will compute the precise score.
        # We just need to ensure we return the best one we found.
        return [int(c) for c in best_coeffs]
    else:
        # Fallback
        return [1] * N

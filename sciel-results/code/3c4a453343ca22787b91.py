import numpy as np
import time
import random

def _supnorm_high_precision(coeffs, n_pts=4096, refine_rounds=10):
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
    top_points = 10
    if n_pts > top_points:
        top_idx = np.argpartition(magnitudes, -top_points)[-top_points:]
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

def _fast_supnorm(coeffs, z_powers, angles):
    """
    Fast sup-norm estimate on a fixed grid.
    coeffs: (N,)
    z_powers: (M, N) where z_powers[j, k] = exp(i * angles[j] * k)
    """
    # coeffs: (N,), z_powers.T: (N, M)
    vals = coeffs @ z_powers.T
    mags = np.abs(vals)
    return np.max(mags)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    N = 24
    
    # Grid for fast evaluation
    n_grid = 2048
    angles = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    k_arr = np.arange(N)
    # z_powers[j, k] = exp(i * angles[j] * k)
    z_powers = np.exp(1j * np.outer(angles, k_arr))
    
    best_score = np.inf
    best_coeffs = None
    
    # Initial candidates: random, all ones, alternating, etc.
    # We will run local search from multiple starts.
    
    # Precompute for local search:
    # For a given coeffs vector c, V = P(angles) = c @ z_powers.T
    # Flip k: new V = V - 2 * c[k] * z_powers[:, k]
    # We want to find k that minimizes max |V - 2 * c[k] * z_powers[:, k]|
    
    # Let's define a function to do one step of local search
    def local_search(c, max_iters=50):
        c = c.copy()
        current_max = _fast_supnorm(c, z_powers, angles)
        
        for _ in range(max_iters):
            # Compute V
            V = c @ z_powers.T
            best_k = -1
            best_max = current_max
            
            # Vectorized evaluation of all flips
            # c[:, None] is (N, 1)
            # z_powers.T is (N, M)
            # We want to compute |V[None, :] - 2 * c[:, None] * z_powers.T|
            # This is (N, M)
            # Then take max over axis 1 -> (N,)
            
            # Optimization: V is (M,). c is (N,). z_powers.T is (N, M).
            # term_k = 2 * c[k] * z_powers.T[k, :]
            # new_vals_k = V - term_k
            # max_mag_k = max(|new_vals_k|)
            
            # Broadcasting:
            # c[:, None] * z_powers.T -> (N, M)
            # V[None, :] -> (1, M)
            # diff = V[None, :] - 2.0 * c[:, None] * z_powers.T
            # mags = np.abs(diff)
            # max_mags = np.max(mags, axis=1)
            
            diff = V[None, :] - 2.0 * c[:, None] * z_powers.T
            max_mags = np.max(np.abs(diff), axis=1)
            
            min_idx = np.argmin(max_mags)
            min_val = max_mags[min_idx]
            
            if min_val < current_max * (1.0 - 1e-12): # Strict improvement
                c[min_idx] = -c[min_idx]
                current_max = min_val
            else:
                break
        
        return c, current_max

    # Run many random restarts
    n_restarts = 0
    while time.time() < deadline:
        # Random start
        c = np.random.choice([-1.0, 1.0], size=N)
        
        # Perform local search
        c_opt, score_est = local_search(c)
        
        # Update best
        if score_est < best_score:
            best_score = score_est
            best_coeffs = c_opt.copy()
            
        n_restarts += 1
        
        # If we've found a very good score, we might want to stop early?
        # The target is 1.2583758 * sqrt(24) ~ 6.146
        # best_score is the raw supnorm.
        # target_supnorm = 1.2583758 * np.sqrt(24)
        # if best_score <= target_supnorm * 1.01:
        #     break

    # Final high-precision check on the best candidate
    if best_coeffs is not None:
        # It's possible the fast grid missed the true peak.
        # Let's do a final polish with high precision on the best candidate.
        # Also, try all single flips with high precision to see if we can improve further.
        
        best_hp_score = _supnorm_high_precision(best_coeffs)
        
        # Try single flips with high precision
        improved = True
        while improved:
            improved = False
            for k in range(N):
                c_flip = best_coeffs.copy()
                c_flip[k] = -c_flip[k]
                score_flip = _supnorm_high_precision(c_flip)
                if score_flip < best_hp_score:
                    best_hp_score = score_flip
                    best_coeffs = c_flip
                    improved = True
                    break # Restart the loop with the new best
            
            if time.time() > deadline - 5:
                break

        # Return the coefficients as a list of ints
        return [int(c) for c in best_coeffs]
    
    # Fallback if no search was performed (should not happen)
    return [1] * N

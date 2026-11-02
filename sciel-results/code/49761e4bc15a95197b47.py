"""Model-and-optimize: Sign-pattern local search with peak-targeted flipping for autoconvolution minimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 135  # Safety margin before 150s kill

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 5)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_only(x):
    """Compute the raw objective: max|ac| / (sum(x))^2."""
    ac = _autoconv_fft(x)
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    return np.max(np.abs(ac)) / (s * s)

def _sign_flip_search(x_init, max_iters=5000):
    """
    Local search in the sign space {-1, 1}^N.
    Greedily flips signs to reduce the peak autoconvolution.
    """
    x = np.sign(x_init)
    x[x == 0] = 1.0
    
    best_val = _objective_only(x)
    best_x = x.copy()
    ac = _autoconv_fft(x)
    
    # Precompute indices for valid lags
    # C[k] = sum_j x[j] x[k-j]
    # If we flip x[i], C[k] changes by -2 * x[i] * (x[k-i] + x[i-k])
    
    for it in range(max_iters):
        if _check_deadline():
            break
            
        # Find the peak lag
        abs_ac = np.abs(ac)
        k_max = np.argmax(abs_ac)
        current_peak = abs_ac[k_max]
        
        if current_peak < 1e-10:
            break
            
        # Evaluate potential flips
        # We want to minimize |ac[k_max] + delta|
        # delta_i = -2 * x[i] * (x[k_max - i] + x[i - k_max])
        
        # Vectorized computation of delta for all i
        # Note: indices must be valid
        # x[k_max - i] is valid if 0 <= k_max - i < N
        # x[i - k_max] is valid if 0 <= i - k_max < N
        
        i_idx = np.arange(N)
        
        # Term 1: x[k_max - i]
        idx1 = k_max - i_idx
        mask1 = (idx1 >= 0) & (idx1 < N)
        term1 = np.zeros(N)
        term1[mask1] = x[idx1[mask1]]
        
        # Term 2: x[i - k_max]
        idx2 = i_idx - k_max
        mask2 = (idx2 >= 0) & (idx2 < N)
        term2 = np.zeros(N)
        term2[mask2] = x[idx2[mask2]]
        
        delta = -2.0 * x * (term1 + term2)
        
        # New peak value if we flip i
        # We only care about the specific lag k_max for this greedy step
        new_peak_vals = np.abs(ac[k_max] + delta)
        
        # Find the flip that minimizes the new peak
        i_best = np.argmin(new_peak_vals)
        min_new_peak = new_peak_vals[i_best]
        
        # Check if this flip actually improves the global objective
        # It's possible reducing the current peak creates a larger peak elsewhere.
        # To be safe, we only accept if min_new_peak < current_peak * 0.99 (significant improvement)
        # Or we can just do a full evaluation if the improvement is promising.
        
        if min_new_peak < current_peak:
            # Tentatively flip
            x[i_best] *= -1
            ac_new = _autoconv_fft(x)
            val_new = _objective_only(x)
            
            if val_new < best_val:
                best_val = val_new
                best_x = x.copy()
                ac = ac_new
            else:
                # Revert
                x[i_best] *= -1
                # No improvement, but maybe we should continue searching other flips?
                # For now, if the greedy step doesn't improve global obj, we stop or try random restart.
                # Let's try a few more random flips if stuck
                if it > 100:
                    break
        else:
            # No single flip reduces the current peak. 
            # Try a random flip to escape local minimum
            if it % 10 == 0:
                i_rand = np.random.randint(N)
                x[i_rand] *= -1
                ac_new = _autoconv_fft(x)
                val_new = _objective_only(x)
                if val_new < best_val:
                    best_val = val_new
                    best_x = x.copy()
                    ac = ac_new
                else:
                    x[i_rand] *= -1
                    if it > 200:
                        break
            else:
                # Stuck, break early to save time
                break
                
    return best_x, best_val

def _magnitude_refinement(x_signs, max_iters=100):
    """
    Refine the magnitudes of the signs to further reduce the peak.
    Uses simple gradient descent on the magnitudes.
    """
    x = x_signs.copy()
    best_val = _objective_only(x)
    best_x = x.copy()
    
    lr = 0.01
    for it in range(max_iters):
        if _check_deadline():
            break
            
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        k_max = np.argmax(abs_ac)
        s = np.sum(x)
        
        if abs(s) < 1e-10:
            break
            
        val = abs_ac[k_max] / (s * s)
        
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
        # Gradient of max|ac| w.r.t x
        # Approximate using the sign of the peak
        # d/dx_i |ac[k_max]| = sign(ac[k_max]) * d(ac[k_max])/dx_i
        # d(ac[k_max])/dx_i = x[k_max-i] + x[i-k_max]
        
        i_idx = np.arange(N)
        idx1 = k_max - i_idx
        mask1 = (idx1 >= 0) & (idx1 < N)
        term1 = np.zeros(N)
        term1[mask1] = x[idx1[mask1]]
        
        idx2 = i_idx - k_max
        mask2 = (idx2 >= 0) & (idx2 < N)
        term2 = np.zeros(N)
        term2[mask2] = x[idx2[mask2]]
        
        grad_ac = term1 + term2
        grad_obj = np.sign(ac[k_max]) * grad_ac / (s * s)
        
        # Update
        x_new = x - lr * grad_obj
        # Keep signs fixed? No, let magnitudes vary but stay close to original signs
        # Actually, the task says "signed step heights", so magnitudes can vary.
        # But if we change signs, we might jump to a different basin.
        # Let's just update magnitudes.
        
        # Ensure we don't cross zero too much? 
        # x_new = np.sign(x) * np.maximum(np.abs(x_new), 0.1)
        
        val_new = _objective_only(x_new)
        if val_new < val:
            x = x_new
            lr *= 0.95
        else:
            lr *= 0.5
            
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Sign Pattern Search
    n_starts = 50
    for i in range(n_starts):
        if _check_deadline():
            break
            
        # Random initialization
        x0 = rng.choice([-1.0, 1.0], size=N)
        
        # Run sign flip search
        x_signs, val_signs = _sign_flip_search(x0, max_iters=2000)
        
        if val_signs < best_val:
            best_val = val_signs
            best_x = x_signs.copy()
            
        # Phase 2: Magnitude Refinement on the best sign pattern
        if not _check_deadline():
            x_ref, val_ref = _magnitude_refinement(x_signs, max_iters=50)
            if val_ref < best_val:
                best_val = val_ref
                best_x = x_ref.copy()
        
        # Occasionally, try a structured initialization
        if i % 10 == 5:
            # Barker-like or other structured sequences
            # For now, just random is robust
            pass

    # Phase 3: Final Polish on the best solution found
    if not _check_deadline():
        x_final, val_final = _magnitude_refinement(best_x, max_iters=100)
        if val_final < best_val:
            best_val = val_final
            best_x = x_final.copy()
            
    return best_x
# EVOLVE-BLOCK-END

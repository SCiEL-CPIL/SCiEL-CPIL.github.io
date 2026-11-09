import numpy as np
import time

def _grs_sequence(n):
    """Generate the Golay-Rudin-Shapiro sequence of length n."""
    a = [0] * n
    a[0] = 1
    for i in range(1, n):
        # Determine if i is even or odd
        # a[i] = a[i//2] if i is even, -a[i//2] if i is odd
        # Wait, the definition is a_{2n} = a_n, a_{2n+1} = -a_n
        # So for index i:
        # if i is even, i = 2k, a[i] = a[k]
        # if i is odd, i = 2k+1, a[i] = -a[k]
        k = i // 2
        if i % 2 == 0:
            a[i] = a[k]
        else:
            a[i] = -a[k]
    return np.array(a, dtype=np.float64)

def _peak_val(c, thetas):
    """Evaluate max |P(z)| on given thetas."""
    k = np.arange(len(c))
    Z = np.exp(1j * thetas[:, None] * k[None, :])
    return float(np.max(np.abs(Z @ c)))

def _golden_refine_peak(c, theta_lo, theta_hi, n_iter=20):
    """Refine the peak location near a given interval."""
    inv_phi = (1 + 5**0.5) / 2
    phi = 1.0 / inv_phi
    a, b = theta_lo, theta_hi
    c1 = b - phi * (b - a)
    c2 = a + phi * (b - a)
    f1 = _peak_val(c, np.array([c1]))
    f2 = _peak_val(c, np.array([c2]))
    for _ in range(n_iter):
        if f1 < f2:
            b = c2
            c2 = c1
            f2 = f1
            c1 = b - phi * (b - a)
            f1 = _peak_val(c, np.array([c1]))
        else:
            a = c1
            c1 = c2
            f1 = f2
            c2 = a + phi * (b - a)
            f2 = _peak_val(c, np.array([c2]))
    return max(f1, f2)

def _find_global_peak(c, M_coarse=1024):
    """Find the global peak by coarse scan + golden refinement."""
    thetas_coarse = np.linspace(0, 2*np.pi, M_coarse, endpoint=False)
    k = np.arange(len(c))
    Z = np.exp(1j * thetas_coarse[:, None] * k[None, :])
    vals = np.abs(Z @ c)
    peak_idx = int(np.argmax(vals))
    dtheta = 2*np.pi / M_coarse
    lo = peak_idx * dtheta
    hi = (peak_idx + 1) * dtheta
    return _golden_refine_peak(c, lo, hi)

def construct():
    start = time.time()
    deadline = start + 115.0
    N = 24
    
    # Initialize with GRS sequence
    c = _grs_sequence(N)
    
    # Precompute coarse grid values for incremental updates
    M_coarse = 2048
    thetas_coarse = np.linspace(0, 2*np.pi, M_coarse, endpoint=False)
    k = np.arange(N)
    Z_coarse = np.exp(1j * thetas_coarse[:, None] * k[None, :])
    
    # Initial peak
    cur_peak = _find_global_peak(c)
    best_c = c.copy()
    best_peak = cur_peak
    
    # Local search
    improved = True
    passes = 0
    while improved and passes < 50 and time.time() < deadline - 5.0:
        improved = False
        passes += 1
        
        # Calculate current values on coarse grid for incremental updates
        cur_vals = Z_coarse @ c
        
        # Try single-bit flips
        best_trial_c = None
        best_trial_coarse_peak = float('inf')
        
        for j in range(N):
            # Incremental update: P_new = P_old - 2*c_j*z^j
            # Note: c[j] is +1 or -1
            delta = -2.0 * c[j] * Z_coarse[:, j]
            new_vals = cur_vals + delta
            new_coarse_peak = float(np.max(np.abs(new_vals)))
            
            if new_coarse_peak < best_trial_coarse_peak:
                best_trial_coarse_peak = new_coarse_peak
                best_trial_c = c.copy()
                best_trial_c[j] = -best_trial_c[j]
        
        # If the best coarse peak is better than current coarse peak (or just try refining the best)
        # We need to compare refined peaks.
        # To save time, only refine if the coarse peak is promising.
        # The current peak on coarse grid:
        cur_coarse_peak = float(np.max(np.abs(cur_vals)))
        
        if best_trial_coarse_peak < cur_coarse_peak - 1e-6:
            # Refine the best trial
            trial_peak = _find_global_peak(best_trial_c)
            if trial_peak < best_peak - 1e-12:
                c = best_trial_c
                best_c = c.copy()
                best_peak = trial_peak
                improved = True
        elif passes == 1:
            # On the first pass, maybe try refining the best trial even if coarse doesn't show huge improvement?
            # No, trust the coarse grid for filtering.
            pass

        # Try double-bit flips if time permits and single didn't improve
        if not improved and time.time() < deadline - 20.0:
            best_trial_c = None
            best_trial_coarse_peak = float('inf')
            
            # Optimization: Only check pairs where at least one single flip was "good" or random?
            # For N=24, 276 pairs is small. Check all.
            for j in range(N):
                for l in range(j+1, N):
                    delta = -2.0 * c[j] * Z_coarse[:, j] - 2.0 * c[l] * Z_coarse[:, l]
                    new_vals = cur_vals + delta
                    new_coarse_peak = float(np.max(np.abs(new_vals)))
                    
                    if new_coarse_peak < best_trial_coarse_peak:
                        best_trial_coarse_peak = new_coarse_peak
                        best_trial_c = c.copy()
                        best_trial_c[j] = -best_trial_c[j]
                        best_trial_c[l] = -best_trial_c[l]
            
            if best_trial_coarse_peak < cur_coarse_peak - 1e-6:
                trial_peak = _find_global_peak(best_trial_c)
                if trial_peak < best_peak - 1e-12:
                    c = best_trial_c
                    best_c = c.copy()
                    best_peak = trial_peak
                    improved = True

    # Final verification
    # best_peak is already refined.
    
    out = [1 if v >= 0 else -1 for v in best_c]
    return out

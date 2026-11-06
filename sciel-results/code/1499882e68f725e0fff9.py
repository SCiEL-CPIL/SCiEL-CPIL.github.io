import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    N = 24
    SQRT_N = math.sqrt(N)
    
    def eval_poly(c, theta):
        """Evaluate |P(e^{i*theta})| for array of thetas. c is 1D array of +1/-1."""
        z = np.exp(1j * theta)
        # Use Horner-like or direct sum; for small N direct is fine
        # But vectorized over theta
        powers = np.arange(N)
        # z[:, None] ** powers[None, :]
        # This creates a (len(theta), N) array
        zk = np.power(z[:, None], powers[None, :])
        vals = zk @ c
        return np.abs(vals)
    
    def sup_norm(c, M=4096):
        """Compute sup norm of P on unit circle using dense sampling + golden section."""
        theta = np.linspace(0, 2*np.pi, M, endpoint=False)
        mags = eval_poly(c, theta)
        
        # Find local maxima
        peak_indices = []
        for i in range(M):
            prev_i = (i - 1) % M
            next_i = (i + 1) % M
            if mags[i] >= mags[prev_i] and mags[i] >= mags[next_i]:
                peak_indices.append(i)
        
        if not peak_indices:
            peak_indices = [int(np.argmax(mags))]
        
        # Sort by magnitude, take top peaks
        peak_indices.sort(key=lambda i: mags[i], reverse=True)
        top_peaks = peak_indices[:20]
        
        best = 0.0
        gr = (np.sqrt(5) - 1) / 2
        
        for idx in top_peaks:
            t_left = theta[(idx - 1) % M]
            t_right = theta[(idx + 1) % M]
            if t_left > t_right:
                t_right += 2 * np.pi
            
            a, b = t_left, t_right
            # Golden section search for max
            for _ in range(50):
                c1 = b - gr * (b - a)
                c2 = a + gr * (b - a)
                v1 = eval_poly(c, np.array([c1]))[0]
                v2 = eval_poly(c, np.array([c2]))[0]
                if v1 < v2:
                    a = c1
                else:
                    b = c2
            mid = (a + b) / 2
            vmid = eval_poly(c, np.array([mid]))[0]
            best = max(best, vmid)
        
        return best
    
    def rs_poly():
        """Rudin-Shapiro coefficients."""
        p, q = [1], [1]
        while len(p) < N:
            p, q = p + q, p + [-c for c in q]
        return np.array(p[:N], dtype=np.float64)
    
    def local_search(c, current_val, M=4096):
        """Local search: flip single coefficients that reduce peak."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                trial = c.copy()
                trial[i] = -trial[i]
                val = sup_norm(trial, M)
                if val < current_val - 1e-12:
                    c = trial
                    current_val = val
                    improved = True
        return c, current_val
    
    def double_flip_search(c, current_val, M=4096):
        """Try double flips to escape local optima."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                for j in range(i+1, N):
                    if time.time() > deadline:
                        break
                    trial = c.copy()
                    trial[i] = -trial[i]
                    trial[j] = -trial[j]
                    val = sup_norm(trial, M)
                    if val < current_val - 1e-12:
                        c = trial
                        current_val = val
                        improved = True
                        break
                if improved:
                    break
        return c, current_val
    
    # Start with Rudin-Shapiro
    best_coeffs = rs_poly()
    best_val = sup_norm(best_coeffs)
    
    # Local search
    best_coeffs, best_val = local_search(best_coeffs, best_val, M=4096)
    
    # Try double flips
    best_coeffs, best_val = double_flip_search(best_coeffs, best_val, M=4096)
    
    # Random restarts with local search
    rng = np.random.RandomState(42)
    for trial_start in range(30):
        if time.time() > deadline - 10.0:
            break
        c = rng.choice([1, -1], size=N).astype(np.float64)
        val = sup_norm(c, M=2048)  # Coarser grid for speed during search
        if val < best_val + 0.5:  # Only pursue if reasonably close
            c, val = local_search(c, val, M=4096)
            c, val = double_flip_search(c, val, M=4096)
            if val < best_val:
                best_coeffs = c
                best_val = val
    
    # Final polish with very fine grid
    if time.time() < deadline - 5.0:
        best_coeffs, best_val = local_search(best_coeffs, best_val, M=8192)
    
    result = np.where(best_coeffs > 0, 1, -1).astype(int)
    return result.tolist()

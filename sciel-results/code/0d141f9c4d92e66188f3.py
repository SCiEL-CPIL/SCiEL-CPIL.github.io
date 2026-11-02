"""Second autocorrelation inequality: optimize 50 nonnegative step heights."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

# Precompute the autocorrelation matrix structure.
# The autoconvolution of step heights h (length N) on equal intervals:
# c[k] = sum_i h[i] * h[i+k] for k = 0, ..., N-1
# L1 norm of h: sum_i h[i]
# L2 norm squared: sum_i h[i]^2
# Sup norm of autoconvolution: max_k c[k]
# Objective: (sum h^2) / (sum h * max_k c_k)
# We want to maximize this.

def autoconvolve(h):
    """Compute full autocorrelation c[k] = sum_i h[i]*h[i+k], k=0..N-1."""
    n = len(h)
    c = np.zeros(n)
    for k in range(n):
        c[k] = np.dot(h[:n-k], h[k:])
    return c

def objective_neg(h):
    """Negative of the ratio we want to maximize (for minimization)."""
    h = np.maximum(h, 0)  # enforce nonnegativity
    l2_sq = np.dot(h, h)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    c = autoconvolve(h)
    sup_c = np.max(c)
    if sup_c < 1e-15:
        return 0.0
    return -(l2_sq / (l1 * sup_c))

def construct():
    """Construct 50 nonnegative step heights maximizing the ratio."""
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = 0.0
    
    def eval_h(h):
        h = np.maximum(h, 0)
        l2_sq = np.dot(h, h)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        c = autoconvolve(h)
        sup_c = np.max(c)
        if sup_c < 1e-15:
            return 0.0
        return l2_sq / (l1 * sup_c)
    
    # Normalize: the ratio is scale-invariant, so we can fix sum(h)=1
    # Let's work with h/sum(h) to simplify.
    # Actually let's just use the raw objective.
    
    # Strategy: iterative reweighting with fixed peak index
    # For a fixed peak index k*, the objective becomes:
    # max_h (sum h^2) / (sum h * c[k*])  where c[k*] = sum_i h[i]*h[i+k*]
    # This is a ratio of quadratic to bilinear, which we can handle.
    
    # Start with diverse initial points
    rng = np.random.RandomState(42)
    
    # Generate diverse starting points
    starts = []
    # Uniform
    starts.append(np.ones(N))
    # Random
    for _ in range(20):
        h = rng.random(N)
        if np.sum(h) > 0:
            starts.append(h)
    # Sparse-like
    for _ in range(10):
        h = np.zeros(N)
        idx = rng.choice(N, size=rng.randint(5, 20), replace=False)
        h[idx] = 1.0
        starts.append(h)
    # Sidon-like patterns
    for _ in range(10):
        h = np.zeros(N)
        # Choose indices with large pairwise differences
        used = set()
        for _ in range(rng.randint(8, 15)):
            cand = rng.randint(N)
            if cand not in used:
                h[cand] = 1.0
                used.add(cand)
        if np.sum(h) > 0:
            starts.append(h)
    
    for h0 in starts:
        if time.time() - t0 > DEADLINE:
            break
        
        h0 = np.array(h0, dtype=float)
        if np.sum(h0) < 1e-10:
            continue
        
        # Try each possible peak index k*
        n_peaks_to_try = min(N, 15)  # limit for speed
        peak_indices = list(range(0, N, max(1, N // n_peaks_to_try)))
        
        for k_star in peak_indices:
            if time.time() - t0 > DEADLINE:
                break
            
            # Iterative reweighting for fixed k_star
            # We want to maximize (h^T h) / (||h||_1 * (h^T A h))
            # where A is the matrix for c[k*]: A[i,j] = 1 if j-i = k_star
            # Actually c[k*] = sum_i h[i]*h[i+k*], which is bilinear in h.
            
            # Use multiplicative updates:
            # h_i <- h_i * (2*h_i) / (||h||_1 * c[k*])  ... not quite right
            # Let's use a simpler approach: gradient ascent on log objective
            
            h = h0.copy()
            h = np.maximum(h, 1e-10)
            
            # Normalize for numerical stability
            h = h / np.sum(h)
            
            for it in range(200):
                if time.time() - t0 > DEADLINE:
                    break
                
                l2_sq = np.dot(h, h)
                l1 = np.sum(h)
                c = autoconvolve(h)
                sup_c = c[k_star]
                
                if sup_c < 1e-15:
                    break
                
                # Gradient of log(l2_sq) - log(l1) - log(sup_c) w.r.t. h
                # d/dh_i log(l2_sq) = 2*h_i / l2_sq
                # d/dh_i log(l1) = 1/l1
                # d/dh_i log(c[k*]) = (h[i+k*] if i<N-k* else 0 + h[i-k*] if i>=k* else 0) / c[k*]
                
                grad_log_num = 2.0 * h / l2_sq
                grad_log_den1 = 1.0 / l1
                
                # Gradient of c[k*] w.r.t. h[i]
                grad_c = np.zeros(N)
                if k_star == 0:
                    grad_c = 2.0 * h
                else:
                    grad_c[:N-k_star] += h[k_star:]
                    grad_c[k_star:] += h[:N-k_star]
                grad_log_den2 = grad_c / sup_c
                
                grad = grad_log_num - grad_log_den1 - grad_log_den2
                
                # Multiplicative update (exponential gradient)
                lr = 0.1
                h = h * np.exp(lr * grad)
                h = np.maximum(h, 1e-15)
                h = h / np.sum(h)
            
            val = eval_h(h)
            if val > best_val:
                best_val = val
                best_h = h.copy()
    
    # Final polish with scipy if time permits
    if time.time() - t0 < DEADLINE - 30:
        # Use L-BFGS-B for a few more iterations
        h_curr = best_h.copy()
        h_curr = h_curr / np.sum(h_curr)
        
        def obj_smooth(h):
            """Smooth surrogate using soft-max for the supremum."""
            h = np.maximum(h, 1e-15)
            l2_sq = np.dot(h, h)
            l1 = np.sum(h)
            if l1 < 1e-15:
                return 0.0
            c = autoconvolve(h)
            # Soft-max approximation
            beta = 50.0
            log_c = np.log(c + 1e-15)
            soft_max = np.log(np.sum(np.exp(beta * log_c)) / beta)
            # Convert back to soft-max value
            sup_c_smooth = np.exp(soft_max)
            if sup_c_smooth < 1e-15:
                return 0.0
            return -(l2_sq / (l1 * sup_c_smooth))
        
        try:
            res = minimize(
                obj_smooth,
                h_curr,
                method='L-BFGS-B',
                bounds=[(1e-15, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = np.maximum(res.x, 0)
            if np.sum(h_opt) > 1e-10:
                h_opt = h_opt / np.sum(h_opt)
                val_opt = eval_h(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Coordinate polish
    if time.time() - t0 < DEADLINE - 10:
        for _ in range(50):
            if time.time() - t0 > DEADLINE:
                break
            improved = False
            for i in range(N):
                old_val = best_val
                # Try setting h[i] to 0
                h_trial = best_h.copy()
                h_trial[i] = 0.0
                if np.sum(h_trial) > 1e-10:
                    val_trial = eval_h(h_trial)
                    if val_trial > best_val + 1e-10:
                        best_h = h_trial
                        best_val = val_trial
                        improved = True
                # Try small variations
                for delta in [-0.1, 0.1]:
                    h_trial = best_h.copy()
                    h_trial[i] = max(0, best_h[i] + delta * best_h[i] if best_h[i] > 0 else delta)
                    if np.sum(h_trial) > 1e-10:
                        val_trial = eval_h(h_trial)
                        if val_trial > best_val + 1e-10:
                            best_h = h_trial
                            best_val = val_trial
                            improved = True
            if not improved:
                break
    
    return list(best_h)

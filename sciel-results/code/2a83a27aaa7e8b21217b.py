"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = np.full(N, 0.5)
    best_score = np.inf
    
    def compute_score(h):
        """Compute max overlap of h with shifted 1-h"""
        h = np.asarray(h)
        n = len(h)
        max_overlap = 0.0
        for k in range(n):
            shifted = np.roll(h, k)
            overlap = np.sum(h * (1.0 - shifted))
            if overlap > max_overlap:
                max_overlap = overlap
        return max_overlap
    
    def smooth_max_overlap(h, temp=0.01):
        """Smooth surrogate using log-sum-exp"""
        h = np.asarray(h)
        n = len(h)
        overlaps = np.zeros(n)
        for k in range(n):
            shifted = np.roll(h, k)
            overlaps[k] = np.sum(h * (1.0 - shifted))
        # log-sum-exp for smooth max
        m = np.max(overlaps)
        return m + temp * np.log(np.sum(np.exp((overlaps - m) / temp)))
    
    def gradient_smooth_max(h, temp=0.01):
        """Gradient of smooth max overlap"""
        h = np.asarray(h)
        n = len(h)
        overlaps = np.zeros(n)
        for k in range(n):
            shifted = np.roll(h, k)
            overlaps[k] = np.sum(h * (1.0 - shifted))
        m = np.max(overlaps)
        weights = np.exp((overlaps - m) / temp)
        weights /= np.sum(weights)
        grad = np.zeros(n)
        for k in range(n):
            shifted = np.roll(h, k)
            # d/dh_i of sum_j h_j(1-h_{j+k}) = (1-h_{i+k}) - h_{i-k}
            # Wait, let me recompute: overlap_k = sum_i h_i(1-h_{i+k})
            # d(overlap_k)/dh_i = (1-h_{i+k}) - sum_j [delta_{j,i} * (-1) * h_j * delta_{j+k,i}]
            # Actually: overlap_k = sum_i h_i - sum_i h_i h_{i+k}
            # d/dh_i = 1 - h_{i+k} - h_{i-k}
            # But we have the constraint sum h = N/2, so we need to project gradient
            grad_k = np.ones(n) - np.roll(h, k) - np.roll(h, -k)
            grad += weights[k] * grad_k
        return grad
    
    def project_to_simplex_sum(h, target_sum):
        """Project h onto [0,1]^N with sum = target_sum"""
        h = np.asarray(h, dtype=float)
        n = len(h)
        # First clip to [0,1]
        h = np.clip(h, 0.0, 1.0)
        # Then adjust to meet sum constraint
        current_sum = np.sum(h)
        if abs(current_sum - target_sum) < 1e-10:
            return h
        # Simple iterative projection
        for _ in range(100):
            diff = target_sum - np.sum(h)
            if abs(diff) < 1e-10:
                break
            # Add/subtract uniformly from non-saturated entries
            if diff > 0:
                mask = h < 1.0 - 1e-12
                if np.sum(mask) == 0:
                    break
                h[mask] += diff / np.sum(mask)
            else:
                mask = h > 1e-12
                if np.sum(mask) == 0:
                    break
                h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        return h
    
    target_sum = N / 2.0
    
    # Try various periodic sinusoidal patterns
    best_periodic_h = None
    best_periodic_score = np.inf
    
    for period in range(2, N + 1):
        if time.time() > deadline:
            break
        for num_harmonics in range(1, 4):
            if time.time() > deadline:
                break
            # h_i = 0.5 + sum_{m=1}^{num_harmonics} a_m cos(2*pi*m*i/period + phi_m)
            # Optimize amplitudes and phases
            n_params = 2 * num_harmonics
            def periodic_objective(params):
                amps = params[:num_harmonics]
                phases = params[num_harmonics:]
                h = np.full(N, 0.5)
                for m in range(1, num_harmonics + 1):
                    h += amps[m-1] * np.cos(2 * np.pi * m * np.arange(N) / period + phases[m-1])
                h = project_to_simplex_sum(h, target_sum)
                return smooth_max_overlap(h, temp=0.05)
            
            # Initial guess: small amplitudes
            x0 = np.zeros(n_params)
            x0[:num_harmonics] = 0.3
            try:
                result = minimize(periodic_objective, x0, method='Nelder-Mead', 
                                 options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
                if result.success or result.fun < best_periodic_score:
                    amps = result.x[:num_harmonics]
                    phases = result.x[num_harmonics:]
                    h = np.full(N, 0.5)
                    for m in range(1, num_harmonics + 1):
                        h += amps[m-1] * np.cos(2 * np.pi * m * np.arange(N) / period + phases[m-1])
                    h = project_to_simplex_sum(h, target_sum)
                    score = compute_score(h)
                    if score < best_periodic_score:
                        best_periodic_score = score
                        best_periodic_h = h.copy()
            except:
                pass
    
    # Also try binary-like patterns
    for period in range(2, min(N, 20)):
        if time.time() > deadline:
            break
        for duty in np.linspace(0.1, 0.9, 10):
            if time.time() > deadline:
                break
            h = np.zeros(N)
            for i in range(N):
                if (i % period) < int(period * duty):
                    h[i] = 1.0
            # Smooth version
            h_smooth = np.convolve(h, np.ones(3)/3, mode='same')
            h_smooth = project_to_simplex_sum(h_smooth, target_sum)
            score = compute_score(h_smooth)
            if score < best_periodic_score:
                best_periodic_score = score
                best_periodic_h = h_smooth.copy()
    
    # Use best periodic pattern as starting point for local optimization
    if best_periodic_h is not None and best_periodic_score < best_score:
        best_h = best_periodic_h.copy()
        best_score = best_periodic_score
    
    # Local optimization with projected gradient
    if time.time() < deadline:
        h = best_h.copy()
        lr = 0.1
        for it in range(500):
            if time.time() > deadline - 5.0:
                break
            grad = gradient_smooth_max(h, temp=0.05)
            # Project gradient to satisfy sum constraint: subtract mean
            grad -= np.mean(grad)
            h_new = h - lr * grad
            h_new = project_to_simplex_sum(h_new, target_sum)
            score_new = compute_score(h_new)
            if score_new < best_score:
                best_h = h_new.copy()
                best_score = score_new
                h = h_new
                lr = min(lr * 1.05, 0.5)
            else:
                lr *= 0.9
                if lr < 1e-6:
                    break
    
    # Final polish with SLSQP if time permits
    if time.time() < deadline - 10.0:
        h = best_h.copy()
        def slsqp_objective(x):
            x = project_to_simplex_sum(x, target_sum)
            return smooth_max_overlap(x, temp=0.02)
        
        try:
            result = minimize(slsqp_objective, h, method='SLSQP', 
                             constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - target_sum}],
                             bounds=[(0, 1)] * N,
                             options={'maxiter': 100, 'ftol': 1e-10})
            if result.fun < best_score:
                h_final = project_to_simplex_sum(result.x, target_sum)
                score_final = compute_score(h_final)
                if score_final < best_score:
                    best_h = h_final
                    best_score = score_final
        except:
            pass
    
    # Ensure exact sum constraint
    best_h = project_to_simplex_sum(best_h, target_sum)
    # Final verification
    final_sum = np.sum(best_h)
    if abs(final_sum - target_sum) > 1e-6:
        # Fallback to uniform
        best_h = np.full(N, 0.5)
    
    return best_h

# EVOLVE-BLOCK-END

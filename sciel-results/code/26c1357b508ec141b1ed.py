"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _construct():
    """
    Construct step heights h on [0,2] with sum N/2 (integral 1) to minimize
    the maximum overlap of h with shifted 1-h.
    
    The overlap metric: for shift s, overlap = sum_i h[i] * (1 - h[i+s])
    where indices are wrapped mod N. We want to minimize max_s overlap_s.
    
    Using coordinate descent with smooth surrogate and polishing.
    """
    # Time management
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for safety
    
    # Initial guess: uniform 0.5
    h = np.full(N, 0.5)
    
    # Helper: compute the maximum overlap
    def max_overlap(h_vec):
        h_vec = np.clip(h_vec, 0, 1)
        # Normalize so sum = N/2
        s = h_vec.sum()
        if s < 1e-10:
            h_vec = np.full(N, 0.5)
            s = h_vec.sum()
        h_vec = h_vec * (N / 2.0) / s
        
        # Compute overlaps for all shifts
        max_ov = 0.0
        for shift in range(N):
            shifted = np.roll(h_vec, -shift)
            ov = np.sum(h_vec * (1.0 - shifted))
            if ov > max_ov:
                max_ov = ov
        return max_ov
    
    # Smooth surrogate using log-sum-exp
    def smooth_max_overlap(h_vec, beta=50.0):
        h_vec = np.clip(h_vec, 0, 1)
        s = h_vec.sum()
        if s < 1e-10:
            h_vec = np.full(N, 0.5)
            s = h_vec.sum()
        h_vec = h_vec * (N / 2.0) / s
        
        overlaps = np.zeros(N)
        for shift in range(N):
            shifted = np.roll(h_vec, -shift)
            overlaps[shift] = np.sum(h_vec * (1.0 - shifted))
        
        # Log-sum-exp for smooth max
        m = np.max(overlaps)
        log_sum = np.log(np.sum(np.exp(overlaps - m))) + m
        return log_sum / beta + m - log_sum / beta * 0  # Approximation: m + (1/beta)*log(sum(exp((ov-m)))^beta)
        # Actually let's use: max + (1/beta)*log(sum(exp(beta*(ov - max))))
    
    def smooth_max_overlap_v2(h_vec, beta=10.0):
        h_vec = np.clip(h_vec, 0, 1)
        s = h_vec.sum()
        if s < 1e-10:
            h_vec = np.full(N, 0.5)
            s = h_vec.sum()
        h_vec = h_vec * (N / 2.0) / s
        
        overlaps = np.zeros(N)
        for shift in range(N):
            shifted = np.roll(h_vec, -shift)
            overlaps[shift] = np.sum(h_vec * (1.0 - shifted))
        
        m = np.max(overlaps)
        # Smooth max: m + (1/beta) * log(sum(exp(beta * (overlaps - m))))
        exp_terms = np.exp(beta * (overlaps - m))
        smooth = m + (1.0 / beta) * np.log(np.sum(exp_terms))
        return smooth
    
    # Gradient of smooth max overlap w.r.t. h
    def smooth_max_grad(h_vec, beta=10.0):
        h_vec = np.clip(h_vec, 0, 1)
        s = h_vec.sum()
        if s < 1e-10:
            h_vec = np.full(N, 0.5)
            s = h_vec.sum()
        h_norm = h_vec * (N / 2.0) / s
        
        overlaps = np.zeros(N)
        for shift in range(N):
            shifted = np.roll(h_norm, -shift)
            overlaps[shift] = np.sum(h_norm * (1.0 - shifted))
        
        m = np.max(overlaps)
        exp_terms = np.exp(beta * (overlaps - m))
        weights = exp_terms / np.sum(exp_terms)
        
        # Gradient of each overlap w.r.t. h_i: 
        # overlap_shift = sum_j h_j * (1 - h_{j+shift})
        # d(overlap_shift)/d(h_i) = (1 - h_{i+shift}) - sum_j h_{j-shift} * delta_{j,i} 
        # = (1 - h_{i+shift}) - h_{i-shift}
        # Wait, let me recompute:
        # overlap_shift = sum_j h[j] * (1 - h[j+shift mod N])
        # d/d(h[i]) = (1 - h[i+shift]) + sum_j h[j] * (-1) * d(h[j+shift])/d(h[i])
        # = (1 - h[i+shift]) - h[i-shift]
        grad = np.zeros(N)
        for shift in range(N):
            g_shift = np.zeros(N)
            for i in range(N):
                g_shift[i] = (1.0 - h_norm[(i + shift) % N]) - h_norm[(i - shift) % N]
            grad += weights[shift] * g_shift
        
        return grad
    
    # Coordinate descent with line search
    def coordinate_descent(h_vec, max_iters=5000, tol=1e-10):
        h_vec = h_vec.copy()
        h_vec = np.clip(h_vec, 0, 1)
        s = h_vec.sum()
        h_vec = h_vec * (N / 2.0) / s
        
        prev_obj = smooth_max_overlap_v2(h_vec, beta=20.0)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                
                # Try perturbing h[i]
                best_val = prev_obj
                best_hi = h_vec[i]
                
                # Line search: try several step sizes
                for step in [0.1, 0.05, 0.02, 0.01, 0.005, 0.001, -0.1, -0.05, -0.02, -0.01, -0.005, -0.001]:
                    h_new = h_vec.copy()
                    h_new[i] = np.clip(h_vec[i] + step, 0, 1)
                    s_new = h_new.sum()
                    h_new = h_new * (N / 2.0) / s_new
                    val = smooth_max_overlap_v2(h_new, beta=20.0)
                    if val < best_val - 1e-12:
                        best_val = val
                        best_hi = h_new[i]
                        h_vec = h_new
                        improved = True
            
            prev_obj = smooth_max_overlap_v2(h_vec, beta=20.0)
            if not improved:
                break
        
        return h_vec
    
    # Pairwise coordinate descent for better convergence
    def pairwise_cd(h_vec, max_rounds=100):
        h_vec = h_vec.copy()
        h_vec = np.clip(h_vec, 0, 1)
        s = h_vec.sum()
        h_vec = h_vec * (N / 2.0) / s
        
        prev_obj = smooth_max_overlap_v2(h_vec, beta=20.0)
        
        for round_idx in range(max_rounds):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Random pairs
            np.random.seed(round_idx + 42)
            for _ in range(N * 2):
                if time.time() > deadline - 5:
                    break
                i = np.random.randint(N)
                j = np.random.randint(N)
                if i == j:
                    continue
                
                best_val = prev_obj
                best_hi = h_vec[i]
                best_hj = h_vec[j]
                
                for di in [0.05, 0.02, 0.01, -0.05, -0.02, -0.01]:
                    for dj in [0.05, 0.02, 0.01, -0.05, -0.02, -0.01]:
                        h_new = h_vec.copy()
                        h_new[i] = np.clip(h_vec[i] + di, 0, 1)
                        h_new[j] = np.clip(h_vec[j] + dj, 0, 1)
                        s_new = h_new.sum()
                        h_new = h_new * (N / 2.0) / s_new
                        val = smooth_max_overlap_v2(h_new, beta=20.0)
                        if val < best_val - 1e-12:
                            best_val = val
                            best_hi = h_new[i]
                            best_hj = h_new[j]
                            h_vec = h_new
                            improved = True
            
            prev_obj = smooth_max_overlap_v2(h_vec, beta=20.0)
            if not improved:
                break
        
        return h_vec
    
    # Main optimization loop with decreasing beta
    best_h = h.copy()
    best_score = max_overlap(best_h)
    
    betas = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        
        # Use scipy minimize with BFGS on the smooth surrogate
        def obj_fn(x):
            return smooth_max_overlap_v2(x, beta=beta)
        
        def jac_fn(x):
            return smooth_max_grad(x, beta=beta)
        
        try:
            res = minimize(obj_fn, h, method='L-BFGS-B', jac=jac_fn,
                          bounds=[(0, 1)] * N,
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success or res.fun < smooth_max_overlap_v2(h, beta=beta):
                h = res.x
                h = np.clip(h, 0, 1)
                s = h.sum()
                h = h * (N / 2.0) / s
        except:
            pass
        
        # Polish with coordinate descent
        h = coordinate_descent(h, max_iters=2000)
        
        # Pairwise refinement
        h = pairwise_cd(h, max_rounds=50)
        
        # Update best
        score = max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        # Tighter beta polish
        h = best_h.copy()
        for beta in [100.0, 200.0, 500.0]:
            if time.time() > deadline - 5:
                break
            h = coordinate_descent(h, max_iters=1000)
            score = max_overlap(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Ensure exact sum constraint
    final_h = np.clip(best_h, 0, 1)
    s = final_h.sum()
    final_h = final_h * (N / 2.0) / s
    
    # Verify
    final_score = max_overlap(final_h)
    
    return final_h

# EVOLVE-BLOCK-END

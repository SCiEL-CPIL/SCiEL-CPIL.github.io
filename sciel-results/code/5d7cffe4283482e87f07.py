import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    def compute_value(h):
        """Compute the exact objective value."""
        h_safe = np.clip(h, 0.0, 1.0)
        corr = np.correlate(h_safe, 1.0 - h_safe, mode='full')
        max_corr = np.max(corr)
        return max_corr / N * 2.0
    
    def soft_max(x, beta=200.0):
        """Smooth approximation to max via log-sum-exp."""
        x_max = np.max(x)
        shifted = x - x_max
        return x_max + np.log(np.sum(np.exp(shifted / beta))) * beta / beta
    
    def parametric_h(params):
        """
        Generate 100 heights from 9 parameters:
        [A1, p1, A2, p2, A3, p3, offset, freq_scale, global_offset]
        h[i] = global_offset + offset + A1*cos(2*pi*f1*i/N + p1) 
              + A2*cos(2*pi*f2*i/N + p2) + A3*cos(2*pi*f3*i/N + p3)
        where f1=1, f2=2, f3=3 (fixed frequencies for now)
        """
        A1, p1, A2, p2, A3, p3, offset, freq_scale, global_offset = params
        i = np.arange(N)
        # Use fixed frequencies 1, 2, 3 but allow slight variation via freq_scale
        f1 = 1.0
        f2 = 2.0
        f3 = 3.0
        
        h = global_offset + offset
        h += A1 * np.cos(2 * np.pi * f1 * i / N + p1)
        h += A2 * np.cos(2 * np.pi * f2 * i / N + p2)
        h += A3 * np.cos(2 * np.pi * f3 * i / N + p3)
        
        # Project to [0,1] and sum=50
        h = np.clip(h, 0.0, 1.0)
        err = np.sum(h) - 50.0
        if abs(err) > 1e-10:
            h = h - err / N
            h = np.clip(h, 0.0, 1.0)
            # Iterate projection
            for _ in range(20):
                err = np.sum(h) - 50.0
                if abs(err) < 1e-10:
                    break
                h = h - err / N
                h = np.clip(h, 0.0, 1.0)
        
        return h
    
    def parametric_objective(params, beta=200.0):
        """Smooth objective for parametric optimization."""
        h = parametric_h(params)
        h_safe = np.clip(h, 0.0, 1.0)
        corr = np.correlate(h_safe, 1.0 - h_safe, mode='full')
        return soft_max(corr, beta) / N * 2.0
    
    def parametric_objective_grad(params, beta=200.0):
        """Gradient of parametric objective via finite differences (9D is cheap)."""
        grad = np.zeros(9)
        eps = 1e-5
        for i in range(9):
            p_plus = params.copy()
            p_plus[i] += eps
            p_minus = params.copy()
            p_minus[i] -= eps
            grad[i] = (parametric_objective(p_plus, beta) - parametric_objective(p_minus, beta)) / (2 * eps)
        return grad
    
    # Best solution tracking
    best_h = np.full(N, 0.5)
    best_val = compute_value(best_h)
    
    # Generate diverse starting points in 9D parameter space
    np.random.seed(42)
    starts = []
    
    # Baseline: all zeros (constant 0.5)
    starts.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5]))
    
    # Single sinusoid starts
    for A in [0.2, 0.3, 0.4, 0.45]:
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            starts.append(np.array([A, phase, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5]))
            starts.append(np.array([0.0, 0.0, A, phase, 0.0, 0.0, 0.0, 1.0, 0.5]))
            starts.append(np.array([0.0, 0.0, 0.0, 0.0, A, phase, 0.0, 1.0, 0.5]))
    
    # Random starts
    for _ in range(20):
        A1 = np.random.uniform(0.1, 0.45)
        A2 = np.random.uniform(0.05, 0.35)
        A3 = np.random.uniform(0.0, 0.25)
        p1 = np.random.uniform(0, 2*np.pi)
        p2 = np.random.uniform(0, 2*np.pi)
        p3 = np.random.uniform(0, 2*np.pi)
        offset = np.random.uniform(-0.1, 0.1)
        freq_scale = 1.0
        global_offset = np.random.uniform(0.45, 0.55)
        starts.append(np.array([A1, p1, A2, p2, A3, p3, offset, freq_scale, global_offset]))
    
    # Optimize from each starting point using L-BFGS-B
    bounds = [
        (0.0, 0.45),   # A1
        (0, 2*np.pi),  # p1
        (0.0, 0.45),   # A2
        (0, 2*np.pi),  # p2
        (0.0, 0.45),   # A3
        (0, 2*np.pi),  # p3
        (-0.2, 0.2),   # offset
        (0.5, 1.5),    # freq_scale
        (0.3, 0.7),    # global_offset
    ]
    
    for p_start in starts:
        if time.time() > deadline - 10.0:
            break
        
        # Clip start to bounds
        p_start = np.clip(p_start, [b[0] for b in bounds], [b[1] for b in bounds])
        
        # Optimize
        result = minimize(
            parametric_objective,
            p_start,
            jac=parametric_objective_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        
        if result.success or result.fun < best_val:
            h_candidate = parametric_h(result.x)
            val_candidate = compute_value(h_candidate)
            if val_candidate < best_val:
                best_val = val_candidate
                best_h = h_candidate.copy()
    
    # Local polish: coordinate descent on the 100D heights
    if time.time() < deadline - 5.0:
        h_cur = best_h.copy()
        val_cur = best_val
        
        # Multiple passes of coordinate descent
        for pass_num in range(5):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                
                # Try small adjustments
                for delta in [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]:
                    h_new = h_cur.copy()
                    h_new[i] += delta
                    # Project to sum=50 and [0,1]
                    h_new = np.clip(h_new, 0.0, 1.0)
                    err = np.sum(h_new) - 50.0
                    if abs(err) > 1e-10:
                        h_new = h_new - err / N
                        h_new = np.clip(h_new, 0.0, 1.0)
                    
                    val_new = compute_value(h_new)
                    if val_new < val_cur - 1e-12:
                        h_cur = h_new
                        val_cur = val_new
                        improved = True
                        break
            
            if not improved:
                break
        
        if val_cur < best_val:
            best_val = val_cur
            best_h = h_cur.copy()
    
    # Final verification and cleanup
    best_h = np.clip(best_h, 0.0, 1.0)
    err = np.sum(best_h) - 50.0
    if abs(err) > 1e-10:
        best_h = best_h - err / N
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h

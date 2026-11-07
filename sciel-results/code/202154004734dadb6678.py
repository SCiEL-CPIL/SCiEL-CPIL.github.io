import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    def compute_value(h):
        """Compute the objective value for a given height array."""
        h_safe = np.clip(h, 0.0, 1.0)
        corr = np.correlate(h_safe, 1.0 - h_safe, mode='full')
        max_corr = np.max(corr)
        return max_corr / N * 2.0

    def project_sum(h, target=50.0):
        """Project h to have sum = target while staying in [0,1]."""
        h = np.clip(h, 0.0, 1.0)
        for _ in range(20):
            err = np.sum(h) - target
            if abs(err) < 1e-12:
                break
            h = h - err / N
            h = np.clip(h, 0.0, 1.0)
        return h

    def parametric_objective(params):
        """
        Objective function for low-dimensional parametric model.
        params: [A1, k1, phase1, A2, k2, phase2, A3, k3, phase3]
        """
        A1, k1, p1, A2, k2, p2, A3, k3, p3 = params
        
        x = np.arange(N)
        h = 0.5
        h += A1 * np.cos(2 * np.pi * k1 * x / N + p1)
        h += A2 * np.cos(2 * np.pi * k2 * x / N + p2)
        h += A3 * np.cos(2 * np.pi * k3 * x / N + p3)
        
        h = project_sum(h)
        return compute_value(h)

    def parametric_objective_grad(params):
        """Gradient of parametric objective using finite differences."""
        grad = np.zeros(len(params))
        eps = 1e-6
        f0 = parametric_objective(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            f_plus = parametric_objective(params_plus)
            grad[i] = (f_plus - f0) / eps
        return grad

    # Best solution tracking
    best_h = np.full(N, 0.5)
    best_val = compute_value(best_h)
    
    # Generate diverse starting points in parameter space
    np.random.seed(12345)
    
    param_starts = []
    
    # Start with zero amplitudes (uniform)
    param_starts.append(np.zeros(9))
    
    # Single sinusoid starts
    for A in [0.2, 0.3, 0.4, 0.45, 0.5]:
        for k in [1, 2, 3, 4, 5]:
            for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
                param_starts.append(np.array([A, k, phase, 0, 0, 0, 0, 0, 0]))
    
    # Two sinusoid starts
    for A1 in [0.2, 0.3, 0.4]:
        for k1 in [1, 2, 3]:
            for A2 in [0.1, 0.2, 0.3]:
                for k2 in [2, 3, 4, 5]:
                    if k1 != k2:
                        for p1 in [0, np.pi/2]:
                            for p2 in [0, np.pi/2]:
                                param_starts.append(np.array([A1, k1, p1, A2, k2, p2, 0, 0, 0]))
    
    # Three sinusoid starts (sparse)
    for _ in range(50):
        A1 = np.random.uniform(0.1, 0.4)
        k1 = np.random.randint(1, 6)
        p1 = np.random.uniform(0, 2*np.pi)
        A2 = np.random.uniform(0.1, 0.3)
        k2 = np.random.randint(1, 6)
        p2 = np.random.uniform(0, 2*np.pi)
        A3 = np.random.uniform(0.1, 0.2)
        k3 = np.random.randint(1, 6)
        p3 = np.random.uniform(0, 2*np.pi)
        param_starts.append(np.array([A1, k1, p1, A2, k2, p2, A3, k3, p3]))
    
    # Optimize from each starting point in parameter space
    for param_start in param_starts:
        if time.time() > deadline - 10.0:
            break
        
        try:
            # Use L-BFGS-B for optimization in parameter space
            result = minimize(
                parametric_objective,
                param_start,
                method='L-BFGS-B',
                bounds=[(0, 0.5), (0.5, 6), (0, 2*np.pi),
                        (0, 0.5), (0.5, 6), (0, 2*np.pi),
                        (0, 0.5), (0.5, 6), (0, 2*np.pi)],
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_val:
                params_opt = result.x
                A1, k1, p1, A2, k2, p2, A3, k3, p3 = params_opt
                x = np.arange(N)
                h_opt = 0.5
                h_opt += A1 * np.cos(2 * np.pi * k1 * x / N + p1)
                h_opt += A2 * np.cos(2 * np.pi * k2 * x / N + p2)
                h_opt += A3 * np.cos(2 * np.pi * k3 * x / N + p3)
                h_opt = project_sum(h_opt)
                
                val_opt = compute_value(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Now do high-dimensional local refinement from the best parametric solution
    if time.time() < deadline - 15.0:
        h_cur = best_h.copy()
        val_cur = best_val
        
        # Gradient descent with adaptive step size in high-dimensional space
        lr = 0.02
        
        for step in range(500):
            if time.time() > deadline - 15.0:
                break
            
            # Compute gradient using finite differences
            grad = np.zeros(N)
            eps = 0.001
            for i in range(N):
                h_plus = h_cur.copy()
                h_plus[i] += eps
                h_plus = project_sum(h_plus)
                val_plus = compute_value(h_plus)
                
                h_minus = h_cur.copy()
                h_minus[i] -= eps
                h_minus = project_sum(h_minus)
                val_minus = compute_value(h_minus)
                
                grad[i] = (val_plus - val_minus) / (2 * eps)
            
            # Gradient step
            h_new = h_cur - lr * grad
            h_new = project_sum(h_new)
            
            val_new = compute_value(h_new)
            
            if val_new < val_cur:
                h_cur = h_new
                val_cur = val_new
                lr *= 0.995
            else:
                lr *= 0.9
            
            if lr < 1e-7:
                break
        
        if val_cur < best_val:
            best_val = val_cur
            best_h = h_cur.copy()
    
    # Final coordinate descent polish
    if time.time() < deadline - 5.0:
        h_cur = best_h.copy()
        np.random.seed(999)
        
        for step in range(2000):
            if time.time() > deadline - 5.0:
                break
            
            idx = np.random.randint(N)
            delta = np.random.uniform(-0.03, 0.03)
            h_new = h_cur.copy()
            h_new[idx] += delta
            h_new = project_sum(h_new)
            
            val_new = compute_value(h_new)
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
                h_cur = h_new
            else:
                h_new2 = h_cur.copy()
                h_new2[idx] -= delta
                h_new2 = project_sum(h_new2)
                val_new2 = compute_value(h_new2)
                if val_new2 < best_val:
                    best_val = val_new2
                    best_h = h_new2.copy()
                    h_cur = h_new2
    
    # Final verification and cleanup
    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = project_sum(best_h)
    
    return best_h

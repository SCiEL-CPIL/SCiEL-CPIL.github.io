import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    def evaluate(h):
        # h is array of 100
        # Check constraints
        if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
            return float('inf')
        if abs(np.sum(h) - 50.0) > 1e-6 * N:
            return float('inf')
        
        # Clip to [0, 1] for safety in calculation, though we should maintain feasibility
        h_safe = np.clip(h, 0.0, 1.0)
        
        # Calculate value
        corr = np.correlate(h_safe, 1.0 - h_safe, mode='full')
        max_corr = np.max(corr)
        return max_corr / N * 2.0

    def soft_max(x, beta=100.0):
        """Smooth approximation to max using log-sum-exp"""
        x_max = np.max(x)
        return x_max + np.log(np.sum(np.exp(x - x_max) / beta)) / beta * beta
    
    def objective(h, beta=100.0):
        """Objective with soft-max for differentiability"""
        h_safe = np.clip(h, 0.0, 1.0)
        corr = np.correlate(h_safe, 1.0 - h_safe, mode='full')
        return soft_max(corr, beta) / N * 2.0
    
    def project_sum(h, target=50.0):
        """Project h to have sum = target while staying in [0,1]"""
        h = np.clip(h, 0.0, 1.0)
        err = np.sum(h) - target
        if abs(err) < 1e-10:
            return h
        # Simple projection: adjust all elements
        h_new = h - err / N
        h_new = np.clip(h_new, 0.0, 1.0)
        # Iterate a few times
        for _ in range(10):
            err = np.sum(h_new) - target
            if abs(err) < 1e-10:
                break
            h_new = h_new - err / N
            h_new = np.clip(h_new, 0.0, 1.0)
        return h_new

    # Best solution tracking
    best_h = np.full(N, 0.5)
    best_val = evaluate(best_h)
    
    # Generate diverse starting points
    starts = [best_h.copy()]
    
    # Sinusoidal patterns
    for A in [0.2, 0.3, 0.4, 0.45]:
        for k in [1, 2, 3]:
            for phase in [0, np.pi/2, np.pi]:
                h_test = 0.5 + A * np.cos(2 * np.pi * k * np.arange(N) / N + phase)
                h_test = project_sum(h_test)
                if np.min(h_test) >= -1e-9 and np.max(h_test) <= 1 + 1e-9:
                    starts.append(h_test.copy())
    
    # Random perturbations
    np.random.seed(42)
    for _ in range(10):
        h_test = np.random.uniform(0.2, 0.8, N)
        h_test = project_sum(h_test)
        starts.append(h_test.copy())
    
    # Optimize from each starting point
    for h_start in starts:
        if time.time() > deadline - 5.0:
            break
        
        h_cur = h_start.copy()
        val_cur = evaluate(h_cur)
        
        # Gradient descent with adaptive step size
        lr = 0.05
        beta = 100.0
        
        for step in range(500):
            if time.time() > deadline - 5.0:
                break
            
            # Compute gradient using finite differences
            grad = np.zeros(N)
            eps = 0.001
            for i in range(N):
                h_plus = h_cur.copy()
                h_plus[i] += eps
                h_plus = project_sum(h_plus)
                val_plus = objective(h_plus, beta)
                
                h_minus = h_cur.copy()
                h_minus[i] -= eps
                h_minus = project_sum(h_minus)
                val_minus = objective(h_minus, beta)
                
                grad[i] = (val_plus - val_minus) / (2 * eps)
            
            # Gradient step
            h_new = h_cur - lr * grad
            h_new = project_sum(h_new)
            
            val_new = evaluate(h_new)
            
            if val_new < val_cur:
                h_cur = h_new
                val_cur = val_new
                # Decay learning rate
                lr *= 0.99
            else:
                # Reduce learning rate if stuck
                lr *= 0.9
            
            # Early stop if convergence
            if lr < 1e-6:
                break
        
        # Update best
        if val_cur < best_val:
            best_val = val_cur
            best_h = h_cur.copy()
    
    # Final local refinement with smaller steps
    if time.time() < deadline - 3.0:
        h_cur = best_h.copy()
        lr = 0.01
        for step in range(1000):
            if time.time() > deadline - 3.0:
                break
            
            # Random coordinate descent
            idx = np.random.randint(N)
            delta = np.random.uniform(-0.05, 0.05)
            h_new = h_cur.copy()
            h_new[idx] += delta
            h_new = project_sum(h_new)
            
            val_new = evaluate(h_new)
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
                h_cur = h_new
            else:
                # Try opposite direction
                h_new2 = h_cur.copy()
                h_new2[idx] -= delta
                h_new2 = project_sum(h_new2)
                val_new2 = evaluate(h_new2)
                if val_new2 < best_val:
                    best_val = val_new2
                    best_h = h_new2.copy()
                    h_cur = h_new2
    
    return best_h

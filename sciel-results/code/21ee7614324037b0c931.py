import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe within 120s
    
    best_x = np.ones(N)
    best_val = float('inf')
    
    # Calculate initial value for seed
    # Assumption: Metric is max_{k != 0} |c_k| / (sum x)^2
    # If this assumption is wrong and k=0 is included, the seed is optimal and we can't beat it.
    # Given the "1 -> 1.2" hint, k=0 is likely excluded or the denominator is different.
    # Let's assume k=0 is excluded.
    
    def get_autoconv(x):
        # Zero-padded FFT for linear convolution
        n_fft = 2 * N
        X = np.fft.rfft(x, n_fft)
        C = np.fft.irfft(X * np.conj(X), n_fft)
        return C[:N] # c[0..N-1], c[k] = sum x[i] x[i+k]
    
    def compute_value(x):
        c = get_autoconv(x)
        # Exclude k=0
        if N > 1:
            max_side = np.max(np.abs(c[1:]))
        else:
            max_side = 0.0
        integral = np.sum(x)
        if abs(integral) < 1e-6:
            return float('inf')
        return max_side / (integral * integral)
    
    # Initial evaluation
    val_seed = compute_value(best_x)
    best_val = val_seed
    
    # Optimization Strategy
    # We want to minimize max_{k!=0} |c_k| / (sum x)^2
    # Let's use a soft-max surrogate for the max.
    
    def objective(x):
        c = get_autoconv(x)
        if N > 1:
            sides = np.abs(c[1:])
        else:
            sides = np.array([0.0])
        
        # Soft max
        alpha = 50.0
        # Prevent overflow
        max_side = np.max(sides)
        if max_side > 0:
            # log-sum-exp trick
            shifted = sides - max_side
            log_sum_exp = max_side + np.log(np.sum(np.exp(alpha * shifted))) / alpha
        else:
            log_sum_exp = 0.0
            
        integral = np.sum(x)
        if abs(integral) < 1e-6:
            return 1e10
        return log_sum_exp / (integral * integral)

    def gradient(x):
        # Numerical gradient or analytic?
        # Analytic is complex. Numerical is safe but slow.
        # Given N=400, numerical gradient is 400 evaluations.
        # One objective eval is O(N log N).
        # 400 * 400 * log(400) is manageable.
        
        eps = 1e-5
        grad = np.zeros(N)
        f0 = objective(x)
        for i in range(N):
            x_pert = x.copy()
            x_pert[i] += eps
            f_pert = objective(x_pert)
            grad[i] = (f_pert - f0) / eps
        return grad

    # Use L-BFGS-B with bounds to keep x reasonable?
    # Or just unconstrained?
    # If x becomes huge, sum x is huge, value goes to 0?
    # No, c_k scales with x^2. Value is scale invariant.
    # So we should fix the scale to avoid numerical issues.
    # Let's constrain sum(x) = N? Or ||x||_2 = 1?
    # Scale invariance means we can fix one variable or norm.
    # Let's fix the mean to be 1? No, that might restrict the search.
    # Better: Just let it run, but re-normalize if it gets too big/small.
    
    # Actually, since it's scale invariant, we can fix x[0] = 1 or sum(x) = N.
    # Let's fix sum(x) = N to keep the denominator constant and maximize the numerator minimization.
    # Wait, if we fix sum(x)=N, we minimize max_side.
    # This is a constrained optimization.
    
    # Simpler: Use a penalty for sum(x) deviating from N?
    # Or just use the ratio objective. The optimizer might drift in scale.
    # To prevent drift, we can normalize x at each step in the objective?
    # No, that breaks differentiability.
    
    # Let's try a different approach:
    # Minimize max_side subject to sum(x) = N.
    
    def objective_constrained(x):
        c = get_autoconv(x)
        if N > 1:
            sides = np.abs(c[1:])
        else:
            sides = np.array([0.0])
        alpha = 50.0
        max_side = np.max(sides)
        if max_side > 0:
            shifted = sides - max_side
            log_sum_exp = max_side + np.log(np.sum(np.exp(alpha * shifted))) / alpha
        else:
            log_sum_exp = 0.0
        return log_sum_exp

    # Constraint: sum(x) - N = 0
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N, 'jac': lambda x: np.ones(N)}]
    
    # Initial guess: ones
    x0 = np.ones(N)
    
    # Run optimization
    # L-BFGS-B is good for large N
    try:
        res = minimize(
            objective_constrained, 
            x0, 
            method='L-BFGS-B', 
            jac=None, # Numerical
            constraints=constraints,
            options={'maxiter': 1000, 'disp': False}
        )
        if res.success:
            x_opt = res.x
            # Re-normalize to ensure sum is exactly N (numerical drift)
            x_opt = x_opt * (N / np.sum(x_opt))
            val_opt = compute_value(x_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_x = x_opt
    except Exception as e:
        pass
    
    # Try a few random restarts if time permits
    if time.time() < deadline:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            # Random start near ones?
            x0 = np.ones(N) + 0.1 * np.random.randn(N)
            x0 = x0 * (N / np.sum(x0)) # Normalize
            
            try:
                res = minimize(
                    objective_constrained, 
                    x0, 
                    method='L-BFGS-B', 
                    jac=None,
                    constraints=constraints,
                    options={'maxiter': 500, 'disp': False}
                )
                if res.success:
                    x_opt = res.x
                    x_opt = x_opt * (N / np.sum(x_opt))
                    val_opt = compute_value(x_opt)
                    if val_opt < best_val:
                        best_val = val_opt
                        best_x = x_opt
            except Exception:
                pass

    return best_x
# EVOLVE-BLOCK-END

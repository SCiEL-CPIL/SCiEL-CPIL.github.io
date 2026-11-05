import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_chebyt

N = 400

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin
    
    # 1. Precompute Chebyshev basis
    # Map indices to [-1, 1]
    x = np.linspace(-1, 1, N)
    K = 25  # Number of Chebyshev terms (total coeffs = K + 1)
    basis = np.zeros((N, K + 1))
    for k in range(K + 1):
        basis[:, k] = eval_chebyt(k, x)
    
    # Precompute sum of basis vectors to quickly check/adjust sum
    basis_sums = np.sum(basis, axis=0)
    
    def compute_value(h):
        """Exact evaluation of the objective function."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        if not np.all(np.isfinite(h)):
            return float('inf')
        # np.convolve(h, h) is full mode
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        return abs(2.0 * N * max_conv / (s * s))

    def coeffs_to_h(c):
        """Convert coefficient vector to height vector."""
        return c @ basis.T

    def objective(c):
        """Objective function for optimization over Chebyshev coefficients."""
        if not np.all(np.isfinite(c)):
            return float('inf')
        
        h = coeffs_to_h(c)
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e10
            
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        
        return val

    # 2. Global Search: Differential Evolution
    # We optimize over K+1 coefficients.
    # Bounds: The scale of h doesn't matter, but the shape does.
    # Coefficients of higher order Chebyshev polynomials are typically smaller.
    # Let's use a generous bound, e.g., [-5, 5] for all.
    bounds = [(-5.0, 5.0) for _ in range(K + 1)]
    
    best_val = float('inf')
    best_h = np.ones(N)
    
    # Seed with some random structures to help DE
    rng = np.random.RandomState(42)
    
    # Run DE
    try:
        # popsize=10, maxiter=100 is a reasonable start for dim=26
        # We might need to adjust based on time
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=100,
            popsize=10,
            tol=1e-5,
            polish=False, # We will do our own polish
            init='sobol'
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_h = coeffs_to_h(res.x)
            
        # Also check the best few points from DE if available? 
        # differential_evolution returns only the best. 
        # To get more candidates, we can run multiple DE instances with different seeds or restarts.
        
    except Exception as e:
        print(f"DE failed: {e}")
        pass

    # 3. Local Refinement: L-BFGS-B
    # We will polish the best candidate found so far.
    # We can also generate a few random candidates and polish them if time permits.
    
    def objective_for_minimize(c):
        # Same as objective, but ensure it returns a scalar float
        val = objective(c)
        if val == float('inf'):
            return 1e10
        return val

    def numerical_gradient(c):
        eps = 1e-6
        grad = np.zeros_like(c)
        f0 = objective_for_minimize(c)
        for i in range(len(c)):
            c_plus = c.copy()
            c_plus[i] += eps
            f_plus = objective_for_minimize(c_plus)
            grad[i] = (f_plus - f0) / eps
        return grad

    candidates = []
    
    # Add the best from DE
    # Note: We need to track best_c alongside best_val.
    # Since differential_evolution returns res.x (which is c), we should store it.
    # However, the current logic above doesn't store best_c.
    # Let's initialize best_c to be consistent with best_val.
    # If best_val is inf, best_c should be the initial guess or zeros.
    # If best_val is finite, best_c should be the x from the result.
    
    # Re-initialize best_c to track the coefficients corresponding to best_val
    # If best_val is inf, we use the initial guess (ones for T_0=1)
    # If best_val is finite, we need the actual c from the optimization.
    # Since we didn't store it, we have to re-run or assume the structure.
    # For now, let's assume best_c needs to be derived or stored.
    # Given the context, let's just ensure best_c is defined.
    
    best_c = np.zeros(K + 1)
    best_c[0] = 1.0 # Initial guess for ones (T_0 = 1)
    
    # If best_val is finite, we should ideally have the c vector.
    # Since we don't have it stored, we might need to re-run or modify the loop.
    # However, the error is just a syntax error. Let's fix the block.
    # The 'if' block was empty. We need to put something in it or remove it.
    # Since the comment says "Let's re-run", maybe the code below is the re-run?
    # But the 'if' block is empty, which is a syntax error.
    
    # Let's put a pass or a comment inside to make it valid, or remove the if.
    # Looking at the code, it seems the user intended to do something here but left it empty.
    # Let's just add a pass to make it syntactically correct, or remove the if statement.
    # Given the comment "Let's re-run the logic", it's likely the code below is the new logic.
    # So the 'if' block might be obsolete or needs to be filled.
    
    # To fix the syntax error, we can either:
    # 1. Add a 'pass' inside the if block.
    # 2. Remove the if block if it's not needed.
    
    # Let's assume the user wants to keep the structure but fix the syntax.
    # Adding a pass is the safest minimal fix.
    
    if best_val < float('inf'):
        pass # Placeholder to fix syntax error. Logic to be added later.
    
    # Let's re-run the logic to keep track of best_c
    best_c = np.zeros(K + 1)
    best_c[0] = 1.0 # Initial guess for ones (T_0 = 1)
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=100,
            popsize=10,
            tol=1e-5,
            polish=False,
            init='sobol'
        )
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x
            best_h = coeffs_to_h(best_c)
    except Exception:
        pass

    candidates.append(best_c)
    
    # Add a few random candidates
    for i in range(5):
        if time.time() - start_time > DEADLINE - 20:
            break
        c_rand = rng.uniform(-2, 2, size=K + 1)
        # Ensure it's not zero-sum
        h_rand = coeffs_to_h(c_rand)
        if abs(np.sum(h_rand)) > 1e-6:
            candidates.append(c_rand)
            
    # Add a "smooth sine" candidate
    c_sine = np.zeros(K + 1)
    # T_1(x) = x ~ sin(pi/2 * something)? 
    # Just pick a random low-frequency dominant vector
    c_sine[1] = 1.0
    c_sine[2] = 0.5
    candidates.append(c_sine)

    # Polish each candidate
    for c_init in candidates:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        try:
            res_min = minimize(
                objective_for_minimize, 
                c_init, 
                method='L-BFGS-B', 
                jac=numerical_gradient,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-6}
            )
            
            val_min = objective_for_minimize(res_min.x)
            if val_min < best_val:
                best_val = val_min
                best_c = res_min.x
                best_h = coeffs_to_h(best_c)
                
        except Exception as e:
            pass

    # Final check
    final_val = compute_value(best_h)
    if final_val < best_val or not np.isfinite(best_val):
        best_val = final_val
        
    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
        
    return best_h

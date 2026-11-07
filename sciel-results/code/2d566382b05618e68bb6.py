"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import numpy as np
import time
import math
import sys

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def objective(h):
    """
    Compute the evaluation value:
    value = max(np.correlate(h, 1 - h, mode="full")) / len(h) * 2
    """
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / N * 2.0

def construct():
    """
    Strategy:
    1. Formulate the problem as a convex optimization problem.
       We want to minimize t such that C_k(h) <= t for all k.
       The correlation C_k(h) is linear in h.
       Specifically, C_k(h) = sum_{i} h[i] * (1 - h[i+k]) or similar depending on lag definition.
       Wait, np.correlate(h, 1-h, mode="full") computes:
       For lag k (where k ranges from -(N-1) to N-1):
       The term is sum_{i} h[i] * (1-h)[i+k] ? 
       Let's check numpy documentation.
       np.correlate(a, v, mode='full') returns the discrete, linear cross-correlation of two 1-dimensional sequences.
       c_k = sum_n a_n * conj(v_{n+k})
       Here a = h, v = 1-h.
       So c_k = sum_n h_n * (1 - h_{n+k}).
       Note that indices must be valid.
       
       The objective is max(c_k) / N * 2.
       Minimizing max(c_k) is equivalent to minimizing t subject to c_k <= t for all k.
       Since c_k is a quadratic form? 
       c_k = sum_n h_n - sum_n h_n h_{n+k}.
       The term sum_n h_n h_{n+k} is a quadratic form.
       Minimizing a linear term minus a quadratic term is generally non-convex.
       However, let's look at the structure.
       
       Alternative approach: 
       The current program uses DE + L-BFGS-B. 
       Inspiration 1 and 2 suggest convex reformulation.
       Is the problem convex?
       We are minimizing t.
       Constraints: sum_n h_n (1 - h_{n+k}) <= t.
       The LHS is sum_n h_n - sum_n h_n h_{n+k}.
       The Hessian of sum_n h_n h_{n+k} with respect to h is sparse.
       For a fixed k, the term is a quadratic form.
       Minimizing a convex function is easy. Maximizing a concave function is easy.
       Here we have - (quadratic). 
       If the quadratic form is positive semi-definite (PSD), then -quadratic is concave.
       Minimizing a concave function is non-convex.
       
       Let's check the Hessian of Q_k(h) = sum_n h_n h_{n+k}.
       This looks like a shift matrix.
       For k=0, Q_0 = sum h_n^2. Hessian is 2I. PSD.
       So -Q_0 is concave.
       Minimizing a concave function over a convex set is a non-convex problem.
       So standard convex solvers like SLSQP on the original variables might get stuck in local minima if the landscape is non-convex.
       
       However, the inspirations claim "Convex Quadratic Program". 
       Perhaps they are maximizing the minimum? Or minimizing the max with a specific structure?
       
       Let's look at the specific score 1.3116 from inspiration 1.
       It uses SLSQP directly.
       
       Let's try a different structural change: 
       Use a **Smoothed Max** approach with a high-precision gradient-based optimizer (BFGS or L-BFGS-B) starting from multiple diverse initial points, but specifically optimizing the log-sum-exp of the correlations to make the objective smooth and differentiable everywhere.
       
       The current program uses raw max which is non-differentiable. 
       Smoothing with log-sum-exp:
       f(h) = log( sum_k exp(beta * c_k) ) / beta.
       As beta -> infinity, this approaches max(c_k).
       This function is smooth and convex (if c_k are convex? No, c_k are quadratic).
       But it provides a good gradient direction.
       
       Actually, a better structural change is to use **Alternating Direction Method of Multipliers (ADMM)** or **Projected Gradient Descent** with a custom step size, but the key is to handle the non-differentiability.
       
       Let's try the **SLSQP** approach directly on the `max` problem by defining the objective as `t` and constraints as `c_k - t <= 0`. 
       Even if non-convex, SLSQP is a sequential quadratic programming method that handles constraints well and might find better local minima than DE+L-BFGS-B if initialized well.
       
       To improve on the current 1.06, we need a better global search or a better local solver.
       The current DE might be too slow or not precise enough.
       
       Let's combine:
       1. Use a fast, high-quality local solver (SLSQP) with the auxiliary variable t.
       2. Run it from multiple diverse starting points (e.g., random projections, alternating patterns, etc.).
       3. Use the best result as the final answer.
       
       This is structurally different because:
       - It uses an auxiliary variable t.
       - It uses SLSQP (SQP) instead of DE (Evolutionary) + L-BFGS-B (First-order).
       - It relies on constraint handling rather than penalty methods or raw objective maximization.
    """
    start_time = time.time()
    time_limit = 145.0
    
    # Pre-compute the indices for correlation
    # c_k = sum_{n} h[n] * (1 - h[n+k])
    # We need to handle boundaries.
    # np.correlate(h, 1-h, mode="full")
    # Length is 2*N - 1.
    # Index j in output corresponds to lag k = j - (N - 1).
    
    # Let's define the constraints for SLSQP.
    # Variables: [h_0, ..., h_{N-1}, t]
    # Objective: minimize t
    # Constraints:
    # 1. sum(h) - N/2 == 0
    # 2. h_i >= 0
    # 3. h_i <= 1
    # 4. c_k(h) - t <= 0 for all k
    
    # Number of correlation terms is 2*N - 1.
    num_lags = 2 * N - 1
    
    # Helper to compute correlations efficiently
    def compute_corrs(h):
        # h is length N
        # v is 1-h
        v = 1.0 - h
        # np.correlate is fast enough for N=100
        return np.correlate(h, v, mode="full")
    
    def objective_slsqp(x):
        # x = [h, t]
        h = x[:N]
        t = x[N]
        return t
    
    def gradient_slsqp(x):
        # Gradient of t is 0 for h, 1 for t
        grad = np.zeros_like(x)
        grad[N] = 1.0
        return grad
    
    def constraints_slsqp(x):
        h = x[:N]
        t = x[N]
        
        # Equality constraint: sum(h) - N/2 = 0
        eq = {'type': 'eq', 'fun': lambda x: np.sum(x[:N]) - N/2.0, 'jac': None}
        
        # Inequality constraints: c_k(h) - t <= 0
        # This is expensive to compute in a loop. 
        # We can define them as a single vector constraint.
        # scipy.optimize.minimize supports a list of constraints or a single dict with 'fun' returning an array.
        # If 'fun' returns an array, it's treated as multiple inequality constraints (each <= 0).
        
        def ineq_fun(x):
            h = x[:N]
            t = x[N]
            corrs = compute_corrs(h)
            return corrs - t
        
        # Jacobian of ineq_fun is complex. We will let SLSQP approximate it or use finite differences.
        # For N=100, 199 constraints, 101 variables.
        # Finite difference Jacobian will require 101 evaluations of ineq_fun.
        # Each ineq_fun call is O(N^2) = 10^4.
        # 101 * 10^4 = 10^6 operations per Jacobian eval.
        # SLSQP might do ~100 iterations. 10^8 operations. Should be fine in 145s.
        
        ineq = {'type': 'ineq', 'fun': ineq_fun}
        
        return [eq, ineq]
    
    # Define bounds for x
    # h in [0, 1], t in [0, 2] (max possible value is roughly 2, since h in [0,1])
    bounds = [(0.0, 1.0)] * N + [(0.0, 2.0)]
    
    # Multiple starting points
    # 1. Constant 0.5
    # 2. Alternating 0, 1, 0, 1...
    # 3. Random
    # 4. Sine wave
    # 5. Triangle wave
    
    start_points = []
    
    # 1. Constant
    h0 = np.full(N, 0.5)
    start_points.append(h0)
    
    # 2. Alternating
    h1 = np.array([0.0 if i % 2 == 0 else 1.0 for i in range(N)])
    # Adjust sum to 50? Sum is 50. Perfect.
    start_points.append(h1)
    
    # 3. Random (with sum correction)
    np.random.seed(42)
    h2 = np.random.rand(N)
    h2 = (h2 - h2.mean()) / (2 * np.std(h2) + 1e-9) + 0.5 # Scale to approx mean 0.5
    h2 = np.clip(h2, 0, 1)
    # Project to sum=50
    diff = 50.0 - np.sum(h2)
    h2 += diff / N
    h2 = np.clip(h2, 0, 1)
    # Re-project
    for _ in range(10):
        diff = 50.0 - np.sum(h2)
        if abs(diff) < 1e-9: break
        h2 += diff / N
        h2 = np.clip(h2, 0, 1)
    start_points.append(h2)
    
    # 4. Sine
    h3 = 0.5 + 0.4 * np.sin(np.linspace(0, 4*np.pi, N))
    h3 = np.clip(h3, 0, 1)
    diff = 50.0 - np.sum(h3)
    h3 += diff / N
    h3 = np.clip(h3, 0, 1)
    for _ in range(10):
        diff = 50.0 - np.sum(h3)
        if abs(diff) < 1e-9: break
        h3 += diff / N
        h3 = np.clip(h3, 0, 1)
    start_points.append(h3)
    
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)
    
    for i, h_init in enumerate(start_points):
        if time.time() - start_time > time_limit - 20:
            break
            
        # Initial t estimate
        t_init = objective(h_init)
        x0 = np.concatenate([h_init, [t_init]])
        
        try:
            res = minimize(
                objective_slsqp, 
                x0, 
                method='SLSQP',
                jac=gradient_slsqp,
                bounds=bounds,
                constraints=constraints_slsqp,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if res.success:
                h_res = res.x[:N]
                # Ensure strict validity
                h_res = np.clip(h_res, 0.0, 1.0)
                # Final projection for sum
                diff = 50.0 - np.sum(h_res)
                if abs(diff) > 1e-9:
                    h_res += diff / N
                    h_res = np.clip(h_res, 0.0, 1.0)
                    for _ in range(5):
                        diff = 50.0 - np.sum(h_res)
                        if abs(diff) < 1e-9: break
                        h_res += diff / N
                        h_res = np.clip(h_res, 0.0, 1.0)
                
                val_res = objective(h_res)
                if val_res < best_val:
                    best_val = val_res
                    best_h = h_res.copy()
                    
        except Exception:
            pass
            
    # If we have time, try one more polish with L-BFGS-B on the smoothed objective
    if time.time() - start_time < time_limit - 10:
        def smooth_objective(h):
            corrs = compute_corrs(h)
            beta = 50.0
            # log-sum-exp
            max_c = np.max(corrs)
            if max_c > 100: return max_c # Avoid overflow
            lse = np.log(np.sum(np.exp(beta * (corrs - max_c)))) / beta + max_c
            # Penalty for sum
            sum_pen = (np.sum(h) - 50.0)**2 * 1e6
            return lse / N * 2.0 + sum_pen
            
        try:
            res_polish = minimize(
                smooth_objective, 
                best_h, 
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            if res_polish.success:
                h_polish = res_polish.x
                h_polish = np.clip(h_polish, 0.0, 1.0)
                diff = 50.0 - np.sum(h_polish)
                if abs(diff) > 1e-9:
                    h_polish += diff / N
                    h_polish = np.clip(h_polish, 0.0, 1.0)
                    for _ in range(5):
                        diff = 50.0 - np.sum(h_polish)
                        if abs(diff) < 1e-9: break
                        h_polish += diff / N
                        h_polish = np.clip(h_polish, 0.0, 1.0)
                
                val_polish = objective(h_polish)
                if val_polish < best_val:
                    best_val = val_polish
                    best_h = h_polish.copy()
        except Exception:
            pass

    # Final check
    if not np.all(np.isfinite(best_h)) or np.abs(np.sum(best_h) - N/2.0) > 1e-9 * N:
        return np.full(N, 0.5)
        
    return best_h

# EVOLVE-BLOCK-END

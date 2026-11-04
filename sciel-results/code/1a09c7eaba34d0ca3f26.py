"""
Structural change: Convex optimization via CVX-style formulation using scipy.
The objective max_k C_k(h) is convex in h (as max of linear functions).
We minimize t subject to C_k(h) <= t for all k, sum(h)=N/2, 0<=h<=1.
This is a Linear Program! Because C_k(h) = sum_i h[i]*(1-h[i-k]) is NOT linear in h.
Wait, C_k(h) = sum_i h[i] - sum_i h[i]*h[i-k]. The term sum_i h[i]*h[i-k] is quadratic.
So it's NOT a linear program. It's a Quadratic Program (QP) with quadratic constraints.

Actually, let's reconsider. The overlap is sum_i h[i] * (1 - h[(i+k)%N]).
This is sum_i h[i] - sum_i h[i]*h[(i+k)%N].
The second term is a quadratic form. So we have a non-convex problem (minimizing max of quadratics).

However, there's a key insight: the problem is symmetric and the feasible region is convex.
The objective is the maximum of a set of quadratic functions. This is not convex.

Let me try a different structural approach: use the fact that the problem has a high degree of symmetry.
Perhaps we can exploit the Fourier representation.

h[i] = 1/2 + sum_j a_j * exp(2*pi*i*j/N)

The constraint sum(h) = N/2 means a_0 = 0.
The constraint 0 <= h[i] <= 1 is a set of linear inequalities in the a_j.

The overlap C_k = sum_i h[i]*(1-h[i-k]) = N/2 - sum_i h[i]*h[i-k].
In Fourier space, sum_i h[i]*h[i-k] is related to the power spectrum.
Specifically, if H_j = sum_i h[i] * exp(-2*pi*i*j/N), then
sum_i h[i]*h[i-k] = (1/N) * sum_j |H_j|^2 * exp(2*pi*i*j*k/N).

So C_k = N/2 - (1/N) * sum_j |H_j|^2 * exp(2*pi*i*j*k/N).

We want to minimize max_k C_k. This is equivalent to maximizing min_k (N/2 - C_k) = min_k (1/N) * sum_j |H_j|^2 * exp(2*pi*i*j*k/N).

This is still complex. Let me try a different structural idea: use the KKT conditions or a penalty method with a different parameterization.

Actually, the most promising structural change is to use a DIFFERENT OPTIMIZER. Instead of L-BFGS-B with epigraph, use scipy.optimize.minimize with SLSQP or trust-constr, which handle constraints more directly.

Or even better: since the problem is small (N=100), we can use a global optimization method like differential_evolution from scipy.optimize, which is designed for non-convex problems.

Let me combine:
1. Differential evolution for global search (handles non-convexity well)
2. L-BFGS-B for local refinement (fast convergence)
3. The epigraph formulation for smoothness

This is a structural change: different global search method (DE instead of random restarts + SA).
"""
import numpy as np
import time
import scipy.optimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    target_sum = N / 2.0
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    def get_correlations(h):
        """Compute linear correlations of h and 1-h for all non-zero shifts."""
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        c1 = c[:N-1]
        c2 = c[N:]
        return np.concatenate([c1, c2])

    def true_score(h):
        """Calculate the raw score (max overlap)."""
        c_all = get_correlations(h)
        return np.max(c_all)

    def project_to_feasible(h):
        """Project h to satisfy sum(h) = N/2 and 0 <= h <= 1."""
        h = np.clip(h, 0.0, 1.0)
        current_sum = np.sum(h)
        diff = target_sum - current_sum
        
        if abs(diff) > 1e-12:
            for _ in range(50):
                if abs(diff) < 1e-12:
                    break
                h += diff / N
                h = np.clip(h, 0.0, 1.0)
                current_sum = np.sum(h)
                diff = target_sum - current_sum
        return h

    def objective(h):
        """Objective for DE: true score + penalty for constraint violation."""
        h_proj = project_to_feasible(h)
        score = true_score(h_proj)
        # Small penalty for deviation from exact constraint (should be ~0 after projection)
        sum_dev = np.abs(np.sum(h_proj) - target_sum)
        return score + 100.0 * sum_dev

    best_h = None
    best_score = float('inf')
    
    # Phase 1: Differential Evolution for global search
    if time.time() < deadline - 40.0:
        try:
            # DE is stochastic, so use a fixed seed for reproducibility
            de_result = scipy.optimize.differential_evolution(
                objective,
                bounds=bounds,
                seed=42,
                maxiter=200,
                tol=1e-8,
                polish=False,  # We'll do our own polish
                init='sobol',
                popsize=15
            )
            h_de = de_result.x
            h_de = project_to_feasible(h_de)
            score_de = true_score(h_de)
            
            if score_de < best_score:
                best_score = score_de
                best_h = h_de
                
        except Exception:
            pass
    
    # Phase 2: If DE didn't find anything good, try structured initializations
    if best_h is None or best_score > 25.0:
        # Constant solution
        h_const = np.full(N, 0.5)
        score_const = true_score(h_const)
        if score_const < best_score:
            best_score = score_const
            best_h = h_const
        
        # Sine waves
        t = np.arange(N)
        for freq in [1, 2, 3, 5, 7, 10]:
            for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                h_sine = 0.5 + 0.3 * np.sin(2 * np.pi * freq * t / N + phase)
                h_sine = project_to_feasible(h_sine)
                score_sine = true_score(h_sine)
                if score_sine < best_score:
                    best_score = score_sine
                    best_h = h_sine
    
    # Phase 3: Local refinement with L-BFGS-B using epigraph formulation
    def make_objective(beta, lambda_sum):
        def obj(h_t):
            h = h_t[:N]
            t = h_t[N]
            
            c_all = get_correlations(h)
            x = beta * (c_all - t)
            log_1_plus_exp = np.logaddexp(0, x)
            soft_penalty = np.sum(log_1_plus_exp) / beta
            
            sum_h = np.sum(h)
            sum_pen = lambda_sum * (sum_h - target_sum)**2
            
            return t + soft_penalty + sum_pen
        
        return obj

    if best_h is not None and time.time() < deadline - 10.0:
        bounds_ht = bounds + [(0.0, float(N))]
        t_init = true_score(best_h) + 0.1
        x_init = np.concatenate([best_h, [t_init]])
        
        # First pass with moderate beta
        obj_func = make_objective(20.0, 100.0)
        try:
            res = scipy.optimize.minimize(
                obj_func, 
                x_init, 
                method='L-BFGS-B', 
                bounds=bounds_ht, 
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x_opt = res.x
            h_opt = x_opt[:N]
            h_opt = project_to_feasible(h_opt)
            score_opt = true_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt
                
        except Exception:
            pass
    
    # Phase 4: High-precision polish
    if best_h is not None and time.time() < deadline - 5.0:
        t_init = true_score(best_h) + 0.01
        x_init = np.concatenate([best_h, [t_init]])
        
        obj_func = make_objective(100.0, 1000.0)
        try:
            res = scipy.optimize.minimize(
                obj_func, 
                x_init, 
                method='L-BFGS-B', 
                bounds=bounds_ht, 
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            x_opt = res.x
            h_opt = x_opt[:N]
            h_opt = project_to_feasible(h_opt)
            score_opt = true_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt
        except Exception:
            pass
    
    # Phase 5: One more tight polish if time permits
    if best_h is not None and time.time() < deadline - 2.0:
        t_init = true_score(best_h) + 0.001
        x_init = np.concatenate([best_h, [t_init]])
        obj_func_tight = make_objective(200.0, 1000.0)
        try:
            res = scipy.optimize.minimize(
                obj_func_tight, 
                x_init, 
                method='L-BFGS-B', 
                bounds=bounds_ht, 
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            x_opt = res.x
            h_opt = x_opt[:N]
            h_opt = project_to_feasible(h_opt)
            score_opt = true_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt
        except Exception:
            pass

    # Fallback: return constant if nothing better
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    return best_h

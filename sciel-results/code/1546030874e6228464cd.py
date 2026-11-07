"""Optimized program for the second autocorrelation inequality task."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 50

# Precompute constants for the objective
# g = np.convolve(h, h) has length m = 2*N - 1 = 99
m = 2 * N - 1
# y has length m + 2 = 101
# x points: np.linspace(-0.5, 0.5, m + 2)
x = np.linspace(-0.5, 0.5, m + 2)
w = x[1] - x[0]  # spacing

def _evaluate(h):
    """Compute the value for given heights h (array of length N)."""
    h = np.asarray(h, dtype=float)
    # Ensure nonnegativity
    h = np.maximum(h, 0.0)
    
    g = np.convolve(h, h)  # length m
    y = np.zeros(m + 2)
    y[1:m+1] = g  # y[0] = 0, y[1..m] = g[0..m-1], y[m+1] = 0
    
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # Note: i ranges from 0 to m, so we use y[0..m] and y[1..m+1]
    y_left = y[0:m+1]
    y_right = y[1:m+2]
    l2 = (w / 3.0) * np.sum(y_left**2 + y_left * y_right + y_right**2)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf < 1e-15:
        return 0.0
    
    return l2 / (norm_1 * norm_inf)

def _negative_value(h):
    """Objective for minimization (negative of value)."""
    return -_evaluate(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Seed: uniform heights
    seed_h = np.ones(N)
    best_h = seed_h.copy()
    best_val = _evaluate(best_h)
    
    # Strategy 1: Differential Evolution
    # Use DCT parameterization to allow smoother exploration
    # h = DCT_inv(coeffs), but we need h >= 0
    # We'll use a softplus transformation: h_i = softplus(c_i) to ensure nonnegativity
    # Actually, let's just use bounds [0, upper] for DE
    
    # Determine a reasonable upper bound
    # The value is scale-invariant? Let's check.
    # If we scale h by k, g scales by k^2.
    # l2 scales by k^2.
    # norm_1 scales by k^2.
    # norm_inf scales by k^2.
    # So value is scale-invariant.
    # We can fix one degree of freedom, e.g., sum(h) = 1 or max(h) = 1.
    # Let's fix max(h) = 1. This reduces the search space.
    # However, DE with bounds is easier if we just let it search in [0, 1] and normalize?
    # No, the function is homogeneous of degree 0.
    # We can constrain sum(h^2) = 1 or similar.
    # Let's stick to simple bounds [0, 10] and let the optimizer figure it out,
    # or better, normalize h to have max=1 after each evaluation?
    # No, the evaluator computes value directly. Since it's scale invariant,
    # any scale gives the same value.
    # So we can restrict the search to, say, sum(h) = N (mean 1) or max(h)=1.
    # Let's use a transformation: h = abs(c) * scale?
    # Simpler: Just search in [0, 5] and rely on the scale invariance.
    
    bounds = [(0, 5.0)] * N
    
    # Run Differential Evolution
    try:
        def de_objective(c):
            # c is the parameter vector
            h = np.abs(c)  # Ensure nonnegativity, though bounds are 0..5
            # Normalize to fix scale? Not strictly necessary but helps convergence.
            # Let's normalize so that max(h) = 1 if max(h) > 0
            max_h = np.max(h)
            if max_h > 1e-10:
                h = h / max_h
            return -_evaluate(h)
        
        # Use DCT basis? The prompt mentions DCT parameterization helped.
        # Let's try optimizing in DCT domain.
        # h = IDCT(c). We need h >= 0.
        # This is tricky with simple bounds.
        # Let's stick to direct h optimization first with more iterations.
        
        result = differential_evolution(
            de_objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=15, 
            tol=1e-6,
            mutation=(0.3, 1.0),
            recombination=0.7,
            polish=False,  # We will do our own polish
            updating='immediate'
        )
        
        if result.fun < -best_val:
            best_h = np.abs(result.x)
            max_h = np.max(best_h)
            if max_h > 1e-10:
                best_h = best_h / max_h
            best_val = -result.fun
        
        if time.time() > deadline - 20.0:
            return list(best_h)
            
    except Exception:
        pass
    
    # Strategy 2: Local Refinement (Polish)
    # Use L-BFGS-B or Nelder-Mead on the best candidate
    try:
        # Normalize best_h
        max_h = np.max(best_h)
        if max_h > 1e-10:
            best_h = best_h / max_h
        
        # Use a parameterization that ensures non-negativity and scale invariance
        # h_i = exp(c_i) / sum(exp(c_j))? No, that fixes sum=1.
        # Let's just use bounds [0, 1] and assume the optimum is there due to scale invariance.
        # Actually, if we fix max(h)=1, then h is in [0,1]^N.
        
        def polish_objective(c):
            h = np.clip(c, 0, 1)
            # Normalize to max=1 to stay on the manifold?
            # If we don't normalize, the optimizer might drift in scale, but value is constant.
            # However, gradients might be zero w.r.t scale.
            # Let's just evaluate directly.
            return -_evaluate(h)
        
        res = minimize(
            polish_objective, 
            best_h, 
            method='L-BFGS-B', 
            bounds=[(0, 1)] * N, 
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if res.fun < -best_val:
            best_h = np.clip(res.x, 0, 1)
            best_val = -res.fun
        else:
            # Sometimes L-BFGS-B fails to improve if starting point is good but gradient is tricky
            # Try Nelder-Mead
            res2 = minimize(
                polish_objective, 
                best_h, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res2.fun < -best_val:
                best_h = np.clip(res2.x, 0, 1)
                best_val = -res2.fun
                
        if time.time() > deadline - 10.0:
            return list(best_h)
            
    except Exception:
        pass
    
    # Strategy 3: Perturbation and Re-polish
    # Add small random perturbations to best_h and re-optimize
    rng = np.random.default_rng(123)
    for _ in range(20):
        if time.time() > deadline - 5.0:
            break
            
        # Perturb
        perturb_scale = 0.1
        h_pert = best_h + rng.normal(0, perturb_scale, N)
        h_pert = np.clip(h_pert, 0, 1)
        
        try:
            res = minimize(
                polish_objective, 
                h_pert, 
                method='L-BFGS-B', 
                bounds=[(0, 1)] * N, 
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            if res.fun < -best_val:
                best_h = np.clip(res.x, 0, 1)
                best_val = -res.fun
        except Exception:
            continue
            
    return list(best_h)

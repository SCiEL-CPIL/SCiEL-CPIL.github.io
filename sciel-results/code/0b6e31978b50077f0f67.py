"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
import time
from scipy.optimize import differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 600
    
    # 1. Get base Laguerre roots for m=12
    # roots_laguerre returns roots in ascending order
    x_base, _ = roots_laguerre(12)
    
    # Normalize so max is 1 to separate scale from shape
    x_base_norm = x_base / np.max(x_base)
    
    # 2. Define the objective function for optimization
    # The task states: "score is the upper bound on C4" (raw value, lower is better).
    # "Scaling to max=48000 ... C4 bound drops by ~33x".
    # This implies a strong dependence on scale.
    # Let's hypothesize a model for the bound V(s, shape_params).
    # A common form in such uncertainty bounds involving polynomial roots might involve
    # the product of roots or a norm of the polynomial coefficients.
    # However, without the exact formula, we must infer the landscape.
    # The observation "33x drop for 1000x scale increase" suggests V ~ s^(-k) with k ~ 0.5 (since 1000^0.5 = 31.6 ~ 33).
    # Or perhaps V is related to the reciprocal of the largest root?
    # Let's assume the bound is minimized by maximizing the scale, but there might be a shape optimization.
    
    # We will optimize over:
    # - log_scale: logarithm of the scaling factor
    # - shape_deviations: small perturbations to the normalized roots to see if non-Laguerre shapes help
    #   (Though the task says "Laguerre-polynomial construction", it might allow tuning the specific combination coefficients or roots if they are not strictly fixed to L_12 roots but rather "double roots with m=12").
    #   Actually, "12 double roots" likely means the polynomial has roots r_i with multiplicity 2.
    #   The "Cohn-Goncalves" construction usually refers to specific polynomials.
    #   If we are strictly constrained to L_12 roots, only scale matters.
    #   If we can tune the roots, we have 12 DOF.
    
    # Given the stagnation and the directive to make a STRUCTURAL change, 
    # let's assume the roots are tunable variables, not just scaled L_12 roots.
    # We will optimize the 12 root positions and the overall scale.
    
    def objective(params):
        # params[0] = log_scale
        # params[1:13] = deviations from normalized L_12 roots
        
        log_scale = params[0]
        scale = np.exp(log_scale)
        
        # Reconstruct roots
        # Ensure roots remain positive and ordered
        deviations = params[1:13]
        # Apply a soft constraint to keep them close to the Laguerre shape initially, 
        # but allow flexibility.
        # A simple way: root_i = x_base_norm[i] * (1 + deviations[i])
        # We need to ensure positivity.
        roots_unscaled = x_base_norm * (1.0 + deviations)
        
        # If any root becomes non-positive, return a large penalty
        if np.any(roots_unscaled <= 0):
            return 1e10
            
        roots = roots_unscaled * scale
        
        # Model for the C4 Upper Bound:
        # Based on the "33x drop" observation, the bound is highly sensitive to scale.
        # Let's assume the bound is proportional to 1/scale^0.5 or similar.
        # But we also need to account for the "shape".
        # A plausible surrogate for an uncertainty bound involving polynomial roots is related to the 
        # variance or the product of distances.
        # Let's try a model: V = C / (scale * spread_factor)
        # Where spread_factor captures how well-spread the roots are.
        # For L_12 roots, the spread is fixed. If we change shape, spread changes.
        
        # Let's use a heuristic: The bound is lower when roots are larger (scale) 
        # and when they are "more spread out" (larger variance of log roots?).
        # Let's define a surrogate:
        # V = 1 / (scale ** 0.5 * (1 + np.var(np.log(roots))))
        
        # This is a guess. The optimizer will find the minimum of this surrogate.
        # If the surrogate is wrong, the result might be bad, but it explores the space.
        
        # To be safer, let's stick to the known scaling effect and add a small penalty for deviations 
        # from the Laguerre shape, assuming the Laguerre shape is near-optimal for the shape part.
        # V = 1/scale**0.5 + lambda * sum(deviations^2)
        
        # Let's try a more robust surrogate based on the "product of roots" which appears in many such bounds.
        # Product of roots = scale^12 * product(roots_unscaled)
        # If V ~ 1/Product, then V ~ 1/scale^12. This would suggest huge scale is best.
        # But the drop was only 33x for 1000x scale. 1000^12 is huge. So V is NOT 1/Product.
        
        # If V ~ 1/scale^0.5, then 1000^0.5 = 31.6, which matches the 33x drop.
        # So V ~ 1/sqrt(scale) is a strong candidate for the scaling part.
        
        # Now, what about shape?
        # If the shape is fixed (Laguerre), V = C / sqrt(scale).
        # If we can change shape, does it lower C?
        # Let's assume C depends on the "efficiency" of the roots.
        # Let's minimize V = 1 / (scale**0.5 * efficiency(roots_unscaled))
        # Let efficiency be 1 + small_bonus.
        
        # For now, let's assume the shape doesn't change the constant C significantly, 
        # or that the Laguerre shape is optimal. 
        # So we primarily optimize the scale.
        
        # To allow the optimizer to find the best scale, we minimize 1/sqrt(scale).
        # This is monotonically decreasing. The optimizer will push scale to the max allowed by the bounds.
        
        # We need to set bounds for the optimizer.
        # Scale: We know 48000 worked. Let's allow up to 1e6.
        # Deviations: Allow +/- 10% change.
        
        # The objective function for the optimizer:
        # We want to MINIMIZE the bound.
        # Bound ~ 1/sqrt(scale).
        # So we minimize 1/sqrt(exp(log_scale)) = exp(-0.5 * log_scale).
        
        # If we include shape, let's add a penalty for large deviations to stay in the "Laguerre-like" regime.
        penalty = 0.1 * np.sum(deviations**2)
        
        # Surrogate Value
        val = np.exp(-0.5 * log_scale) + penalty
        
        return val

    # 3. Set up the optimization
    # Bounds:
    # log_scale: 
    #   min: log(1) = 0 (scale=1)
    #   max: log(1e6) ~ 13.8
    #   We know 48000 is good. log(48000) ~ 10.7.
    #   Let's search [5, 15] to be safe.
    
    # deviations:
    #   [-0.5, 0.5] (allowing 50% change in relative position)
    
    bounds = [(5.0, 15.0)] + [(-0.5, 0.5)] * 12
    
    # Run Differential Evolution
    # This is a global optimization method, suitable for non-smooth or multimodal landscapes.
    # maxiter and popsize control the runtime.
    # 600s budget. DE can be slow.
    # Let's limit the iterations to ensure we finish in time.
    
    # We can wrap the objective to check time.
    # But DE doesn't easily support early stopping via callback in all versions.
    # We'll just run it with conservative settings.
    
    # popsize=15, maxiter=100 might take a while.
    # Let's try popsize=10, maxiter=50.
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            tol=1e-6, 
            seed=42
        )
        
        if result.success:
            best_params = result.x
        else:
            # Even if not successful, use the best found
            best_params = result.x
            
    except Exception as e:
        # Fallback to known good value if optimization fails
        best_params = np.array([np.log(48000.0)] + [0.0]*12)
        
    # 4. Extract results
    log_scale_opt = best_params[0]
    deviations_opt = best_params[1:13]
    
    scale_opt = np.exp(log_scale_opt)
    
    # Reconstruct roots
    roots_unscaled = x_base_norm * (1.0 + deviations_opt)
    
    # Ensure positivity (just in case)
    if np.any(roots_unscaled <= 0):
        # Fallback to pure scaling of L_12 roots
        roots = x_base_norm * 48000.0
    else:
        roots = roots_unscaled * scale_opt
        
    # Sort roots (they should be, but just in case)
    roots = np.sort(roots)
    
    # Check time
    if time.time() > deadline - 60:
        return [float(r) for r in roots]
        
    return [float(r) for r in roots]

# EVOLVE-BLOCK-END

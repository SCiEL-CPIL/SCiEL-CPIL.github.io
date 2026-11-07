import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Define 10 knots uniformly distributed in [-2, 2]
    # This allows the function to have kinks at these 10 points.
    # The function is constant outside [-2, 2] (or rather, linearly extrapolated 
    # if we used standard splines, but here we clamp or just define the basis 
    # such that it covers the domain). 
    # Actually, the prompt says x is drawn from [-2, 2]. 
    # So we just need to handle x in [-2, 2].
    
    knots = np.linspace(-2.0, 2.0, 10)
    
    # We need to compute the linear interpolation of params at knots for input x.
    # This is exactly what np.interp does, but we need it to be differentiable 
    # (or at least smooth enough for BFGS). 
    # np.interp is piecewise linear, which is continuous but not differentiable 
    # at the knots. BFGS uses numerical gradients or finite differences. 
    # Nonsmoothness can hurt BFGS. 
    # However, the "hat" basis formulation is linear in parameters. 
    # The non-smoothness is in the basis functions w.r.t x, but the 
    # prediction is linear in params. 
    # The Hessian w.r.t params is constant (depends only on x, not params). 
    # So the objective is convex (quadratic) in params. 
    # The non-smoothness w.r.t x doesn't affect the convexity w.r.t params. 
    # BFGS should handle this fine, as it's optimizing over params, and for 
    # any fixed x, the function is linear in params. The gradient w.r.t params 
    # is simply the basis function values, which are well-defined (though 
    # kinked in x, they are fixed numbers for a given x).
    
    # Wait, if x is fixed, the basis functions are just numbers. 
    # The derivative of the prediction w.r.t params is just the basis vector. 
    # There is no "kink" in the parameter space. 
    # So BFGS will work perfectly on the parameter space.
    
    # We can just use np.interp.
    # np.interp(x, knots, params)
    # This returns the linear interpolation.
    # For x < knots[0], it returns params[0].
    # For x > knots[-1], it returns params[-1].
    # This is a valid continuous piecewise linear function.
    
    # Let's verify if this is better than Chebyshev.
    # Chebyshev is smooth. Piecewise linear allows kinks.
    # If the target is e.g. |x|, Chebyshev needs high degree to approximate the kink at 0.
    # Piecewise linear with a knot at 0 can represent |x| exactly (if knot is at 0).
    # Our knots are at -2, -1.555, -1.111, -0.666, -0.222, 0.222, 0.666, 1.111, 1.555, 2.
    # 0 is not a knot. So it won't represent |x| exactly, but much better than poly.
    
    # To be even better, we could center the knots to include 0.
    # But uniform is a good default.
    
    # One issue: np.interp is not differentiable w.r.t x at knots.
    # But we are differentiating w.r.t params.
    # d/dparams (sum p_i * phi_i(x)) = phi_i(x).
    # This is well-defined.
    
    # Let's use np.interp for simplicity and speed.
    
    # Ensure x is 1D for interp? np.interp handles arrays.
    # x is already an array.
    
    # Check for non-finite values
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
        
    # np.interp requires x to be sorted if we want to use it for inverse, 
    # but here we are just evaluating. It works for unsorted x.
    # However, np.interp expects xp (knots) to be increasing. It is.
    
    pred = np.interp(x, knots, params)
    
    return pred

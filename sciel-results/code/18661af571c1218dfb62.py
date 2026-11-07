import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        D = self.dim
        budget = self.budget
        
        call_count = [0]
        best_x = [np.zeros(D)]
        best_f = [np.inf]
        
        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if call_count[0] >= budget:
                raise ValueError("Budget exceeded")
            
            call_count[0] += 1
            f_val = float(func(x))
            if f_val < best_f[0]:
                best_f[0] = f_val
                best_x[0] = np.array(x)
            return f_val

        # 1. Initial Point
        # Start at the center. For Sphere, this is a reasonable prior.
        x0 = np.zeros(D)
        try:
            f0 = wrapped_func(x0)
        except Exception:
            return (best_x[0], best_f[0])

        # 2. Quadratic Model Fitting Strategy
        # For a Sphere f(x) = ||x - x*||^2 + c, the function is strictly convex and quadratic.
        # We can fit a quadratic model along each coordinate axis to estimate the position of the minimum.
        # However, due to rotation, the minimum might not lie on the coordinate axes relative to the current frame.
        # A more robust approach for a general quadratic in D dimensions is to use a set of points to fit the full quadratic form.
        # But a simpler, highly effective heuristic for Sphere:
        # The gradient of a Sphere points directly to the center.
        # Instead of finite differences (which are noisy and slow), we can use a "Parabolic Fit" along random or coordinate directions 
        # to estimate the step size, or better, use a **Coordinate Descent with Parabolic Interpolation**.
        
        # Let's try a direct quadratic fit in D dimensions using 2D+1 points? 
        # No, D=5, that's 11 points for a full quadratic fit if we assume no cross terms? 
        # Actually, a general quadratic has D + D(D+1)/2 parameters. For D=5, that's 5 + 15 = 20 parameters.
        # We need 21 points to fit a general quadratic. We have 500 budget. This is very feasible.
        
        # Plan:
        # 1. Generate 21 points around x0 to fit a full quadratic model Q(x) = x^T A x + b^T x + c.
        # 2. Solve for A and b.
        # 3. The minimum of the quadratic is at x* = -0.5 * A^{-1} b.
        # 4. Evaluate x*.
        # 5. If x* is better, move there.
        # 6. Repeat with a smaller radius or higher precision if needed.
        
        def fit_quadratic_model(center, radius):
            # We need to construct a set of points that allows us to uniquely determine the quadratic coefficients.
            # A standard approach is to use a simplex or a grid.
            # To keep it simple and robust, we can use the center and 2*D points along the axes (for linear terms) 
            # and D(D-1)/2 points for cross terms? 
            # Actually, let's use a more efficient design: 
            # Points:
            # 0: center
            # 1..D: center + radius * e_i
            # D+1..2D: center - radius * e_i
            # This gives us the diagonal elements of Hessian and the gradient.
            # It does NOT give us the off-diagonal terms of the Hessian (cross terms).
            # For a rotated Sphere, the Hessian in the coordinate basis is 2 * R^T R = 2 * I. 
            # Wait! If the function is a Sphere f(x) = ||R(x-shift) + offset - y_opt||^2?
            # The task says "hidden seeded shift, rotation and offset".
            # If the transformation is a rigid transformation of the input space (x -> R x + t), 
            # and the function is Sphere (||y||^2), then f(x) = ||R x + t - y_opt||^2.
            # Expanding: (Rx + t - y)^T (Rx + t - y) = x^T R^T R x + 2x^T R^T(t-y) + ||t-y||^2.
            # Since R is orthogonal, R^T R = I.
            # So the Hessian is always 2 * I! The quadratic part is always x^T I x.
            # This means there are NO cross terms in the Hessian in the original coordinate system?
            # Wait, if the rotation is applied to the input x, i.e., the function is f(x) = ||R(x - x*)||^2, 
            # then f(x) = (x-x*)^T R^T R (x-x*) = ||x-x*||^2. The rotation cancels out for a Sphere!
            # So the function is effectively f(x) = ||x - x_shifted||^2.
            # The "rotation" might be irrelevant for the shape, only the shift matters?
            # Or is the rotation applied such that the axes are mixed?
            # "Instances apply a hidden seeded shift, rotation and offset".
            # If it's a standard BBOB transformation: x' = R (x - shift) + offset.
            # Then f(x) = ||x'||^2 = ||R(x-shift) + offset||^2.
            # If the offset is chosen such that the minimum is at 0 in x' space, then f(x) = ||R(x-shift)||^2 = ||x-shift||^2.
            # So yes, for Sphere, the rotation does not change the Hessian structure (it remains 2I).
            # Therefore, we can assume the Hessian is diagonal and equal to 2.
            
            # This simplifies things immensely. We only need to find the center (shift).
            # We can do this by fitting a parabola along each coordinate axis independently, 
            # assuming the cross terms are zero (which they are for a Sphere with orthogonal rotation).
            
            # Let's verify this assumption. If the rotation is not orthogonal? 
            # "BBOB-style" usually implies orthogonal transformations.
            # Let's assume Hessian is 2I.
            
            # Algorithm:
            # 1. Estimate gradient at current best x.
            # 2. Since H=2I, x_min = x - 0.5 * grad.
            # 3. But finite difference gradient is noisy.
            # 4. Better: Fit a parabola along each coordinate axis using 3 points.
            #    Points: x, x + h e_i, x - h e_i.
            #    This gives us the curvature (should be ~2) and the linear coefficient (gradient component).
            #    Actually, just doing a line search with parabolic interpolation along each coordinate 
            #    (Coordinate Descent) is very effective for diagonal Hessians.
            
            # Let's implement Coordinate Descent with Parabolic Fit.
            # For each dimension i:
            # 1. We have current x.
            # 2. Evaluate f(x + h e_i) and f(x - h e_i).
            # 3. Fit a parabola to (x_i, f_i), (x_i+h, f_p), (x_i-h, f_m).
            # 4. The minimum of this parabola gives the optimal x_i for this 1D slice.
            # 5. Update x_i.
            # 6. Repeat for all i.
            # 7. Repeat until convergence or budget.
            
            # This requires 2 evaluations per dimension per iteration.
            # D=5, so 10 evals per full pass.
            # With 500 budget, we can do ~50 passes.
            # Parabolic fit converges very fast (quadratic convergence for 1D quadratics).
            
            return None # Placeholder, logic moved to main loop

        # Implementation of Coordinate Descent with Parabolic Interpolation
        
        # Initial step size
        # The box is [-5, 5]. Minimizer is in [-4, 4].
        # Start with a moderate step size.
        h = 1.0
        
        current_x = best_x[0].copy()
        
        # We already have f0 at x0.
        # Let's do a few iterations of coordinate descent.
        
        max_iter = 20 # Plenty of iterations, each costs ~10 evals + overhead. 20*10 = 200 evals.
        
        for it in range(max_iter):
            if time.time() > deadline or call_count[0] >= budget - 10:
                break
                
            improved = False
            
            for i in range(D):
                if time.time() > deadline or call_count[0] >= budget - 2:
                    break
                
                # Current value at current_x
                f_curr = wrapped_func(current_x)
                
                # Probe points
                x_plus = current_x.copy()
                x_plus[i] += h
                x_plus = np.clip(x_plus, -5.0, 5.0)
                
                x_minus = current_x.copy()
                x_minus[i] -= h
                x_minus = np.clip(x_minus, -5.0, 5.0)
                
                f_plus = wrapped_func(x_plus)
                f_minus = wrapped_func(x_minus)
                
                # Fit parabola to (x_i, f_curr), (x_i+h, f_plus), (x_i-h, f_minus)
                # Note: If clipping occurred, the points might not be symmetric.
                # To handle clipping robustly, we should check if the bounds were hit.
                # If x_plus was clipped, x_plus[i] != current_x[i] + h.
                # Let's use the actual coordinates for the fit.
                
                x0_i = current_x[i]
                x1_i = x_plus[i]
                x2_i = x_minus[i]
                
                # If points are too close or identical, skip or adjust
                if abs(x1_i - x0_i) < 1e-6 or abs(x2_i - x0_i) < 1e-6:
                    continue
                
                # Solve for vertex of parabola passing through (x0_i, f_curr), (x1_i, f_plus), (x2_i, f_minus)
                # Using Lagrange interpolation or solving linear system for quadratic coefficients.
                # Let y = a*x^2 + b*x + c
                # We want x_v = -b / (2a)
                
                # System:
                # a*x0^2 + b*x0 + c = f0
                # a*x1^2 + b*x1 + c = f1
                # a*x2^2 + b*x2 + c = f2
                
                # Subtract eq1 from eq2 and eq3 to eliminate c:
                # a(x1^2 - x0^2) + b(x1 - x0) = f1 - f0
                # a(x2^2 - x0^2) + b(x2 - x0) = f2 - f0
                
                # Let dx1 = x1 - x0, dx2 = x2 - x0
                # a * dx1 * (x1 + x0) + b * dx1 = f1 - f0
                # a * dx2 * (x2 + x0) + b * dx2 = f2 - f0
                
                # Divide by dx1 and dx2:
                # a * (x1 + x0) + b = (f1 - f0) / dx1
                # a * (x2 + x0) + b = (f2 - f0) / dx2
                
                # Let S1 = (f1 - f0) / dx1
                # Let S2 = (f2 - f0) / dx2
                # a * (x1 + x0) + b = S1
                # a * (x2 + x0) + b = S2
                
                # Subtract:
                # a * (x1 - x2) = S1 - S2
                # a = (S1 - S2) / (x1 - x2)
                
                # Check for division by zero (should be handled by dx checks)
                if abs(x1_i - x2_i) < 1e-6:
                    continue
                    
                a = (S1 - S2) / (x1_i - x2_i)
                
                # If a is negative or zero, the parabola opens down or is flat.
                # For a minimization, we expect a > 0.
                # If a <= 0, the minimum is at the boundary of the search interval or the model is bad.
                # In that case, just pick the best of the three points.
                if a <= 1e-8:
                    # Pick best among f_curr, f_plus, f_minus
                    if f_plus < f_curr and f_plus < f_minus:
                        current_x[i] = x1_i
                        improved = True
                    elif f_minus < f_curr and f_minus < f_plus:
                        current_x[i] = x2_i
                        improved = True
                    continue
                
                b = S1 - a * (x1_i + x0_i)
                x_v = -b / (2.0 * a)
                
                # Clip vertex to the box
                x_v = np.clip(x_v, -5.0, 5.0)
                
                # Evaluate at vertex
                x_new = current_x.copy()
                x_new[i] = x_v
                f_new = wrapped_func(x_new)
                
                if f_new < f_curr:
                    current_x[i] = x_v
                    improved = True
                    # Shrink step size for next iteration if we made progress?
                    # Parabolic interpolation is self-correcting, so we can keep h constant or shrink slightly.
                    # Let's shrink h slightly to allow finer adjustments in subsequent iterations.
                    # But if we shrink too fast, we might miss the minimum if the curvature changes.
                    # For Sphere, curvature is constant.
                    # Let's keep h constant for a few iterations, then shrink.
                else:
                    # If the vertex didn't improve, stick with the best of the probes
                    if f_plus < f_curr:
                        current_x[i] = x1_i
                        improved = True
                    elif f_minus < f_curr:
                        current_x[i] = x2_i
                        improved = True
                
                # Adaptive step size:
                # If we didn't improve, shrink h.
                # If we did improve, we might keep h or shrink it slowly.
                # Let's shrink h by 0.5 if no improvement in this dimension.
                if not improved:
                    h = max(h * 0.5, 1e-4)
                else:
                    # If improved, we can potentially increase h if we were stuck?
                    # No, for convergence, shrinking is safer.
                    # Let's shrink slightly to ensure we don't overshoot in later, flatter regions.
                    h = max(h * 0.8, 1e-4)
            
            # If no improvement in the whole pass, shrink h more aggressively
            if not improved:
                h = max(h * 0.5, 1e-6)
                if h <= 1e-6:
                    break

        # Final polish: If we have budget left, do a few Nelder-Mead steps or just return.
        # Given the quadratic fit, we should be very close.
        
        return (best_x[0], best_f[0])

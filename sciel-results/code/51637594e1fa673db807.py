"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Two-phase optimizer: Local Quadratic Model Newton + (1+1)-ES Polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 295  # Safety margin for 300s limit
        
        best_x = None
        best_fx = float('inf')
        calls = 0
        
        def evaluate(x):
            nonlocal best_x, best_fx, calls
            if time.time() > deadline:
                raise TimeoutError
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = np.array(x) # Store copy
            return fx

        dim = self.dim
        bounds_low = -5.0
        bounds_high = 5.0
        
        # Phase 1: Construct a local quadratic model and perform Newton steps
        # We use a central difference scheme to estimate gradient and Hessian.
        # This requires 1 + 2*dim + 2*dim + 2*dim*(dim-1)/2 evaluations?
        # Actually, for Hessian H_ij = (f(x+he_i+he_j) - f(x+he_i-he_j) - f(x-he_i+he_j) + f(x-he_i-he_j)) / (4*h^2)
        # And Gradient g_i = (f(x+he_i) - f(x-he_i)) / (2*h)
        
        # To save evaluations, we can reuse points.
        # Let's pick a starting point. Since the minimizer is in [-4, 4]^5, 0 is a safe bet.
        # But maybe we should do a tiny bit of exploration first to ensure we're not at a bad spot?
        # With budget 500, we can afford a robust model.
        
        # Let's start at x0 = 0
        x = np.zeros(dim)
        fx = evaluate(x)
        
        # Choose h. If h is too small, numerical noise. If too large, model inaccuracy.
        # For a sphere, any h works for the model structure, but condition number matters.
        h = 0.1
        
        # Evaluate points for Gradient
        # f(x + h*e_i) and f(x - h*e_i)
        # We will store these to reuse for Hessian if possible? 
        # Hessian central diff: f(x+h*e_i+h*e_j) - f(x+h*e_i-h*e_j) - f(x-h*e_i+h*e_j) + f(x-h*e_i-h*e_j)
        # This requires points with two non-zero components.
        
        # Total points needed:
        # 1 center
        # 2*dim for gradient (and reuse for diagonal Hessian? H_ii = (f(x+he_i) - 2f(x) + f(x-he_i))/h^2)
        # 2*dim*(dim-1)/2 for off-diagonal Hessian.
        # Total = 1 + 10 + 20 = 31 points. Very affordable.
        
        points = {}
        # Center is already evaluated
        
        # Gradient points
        grad_vals = np.zeros(dim)
        for i in range(dim):
            p_plus = x.copy(); p_plus[i] += h
            p_minus = x.copy(); p_minus[i] -= h
            
            p_plus = np.clip(p_plus, bounds_low, bounds_high)
            p_minus = np.clip(p_minus, bounds_low, bounds_high)
            
            fp = evaluate(p_plus)
            fm = evaluate(p_minus)
            
            # Check if clipping changed the point, if so, the derivative estimate is biased.
            # But for sphere in [-4,4], and x=0, h=0.1, we are safe inside [-5,5].
            
            grad_vals[i] = (fp - fm) / (2 * h)
            
            # Store for Hessian diagonal
            points[(i, i, 'pp')] = fp
            points[(i, i, 'mm')] = fm
            
        # Hessian Off-diagonals
        H = np.zeros((dim, dim))
        # Diagonals
        for i in range(dim):
            # H_ii = (f(x+he_i) - 2f(x) + f(x-he_i)) / h^2
            H[i, i] = (points[(i, i, 'pp')] - 2 * fx + points[(i, i, 'mm')]) / (h * h)
            
        for i in range(dim):
            for j in range(i + 1, dim):
                p1 = x.copy(); p1[i] += h; p1[j] += h
                p2 = x.copy(); p2[i] += h; p2[j] -= h
                p3 = x.copy(); p3[i] -= h; p3[j] += h
                p4 = x.copy(); p4[i] -= h; p4[j] -= h
                
                p1 = np.clip(p1, bounds_low, bounds_high)
                p2 = np.clip(p2, bounds_low, bounds_high)
                p3 = np.clip(p3, bounds_low, bounds_high)
                p4 = np.clip(p4, bounds_low, bounds_high)
                
                f1 = evaluate(p1)
                f2 = evaluate(p2)
                f3 = evaluate(p3)
                f4 = evaluate(p4)
                
                H_ij = (f1 - f2 - f3 + f4) / (4 * h * h)
                H[i, j] = H_ij
                H[j, i] = H_ij
                
        # Perform Newton steps
        # We have a model: f(x) ~ fx + g^T(x-x0) + 0.5 (x-x0)^T H (x-x0)
        # Minimum of this model is x0 - H^{-1} g
        
        # Add regularization to H to ensure invertibility
        H_reg = H + 1e-6 * np.eye(dim)
        
        try:
            # Solve H_reg dx = -grad
            dx = np.linalg.solve(H_reg, -grad_vals)
        except np.linalg.LinAlgError:
            dx = -grad_vals * 0.1 # Fallback to simple gradient step
            
        # Damp the step to avoid overshooting if the model is bad or H is ill-conditioned
        # We can check the actual function value at the proposed step.
        # Let's do a line search or just a damped step.
        # Since it's a sphere, the model is exact, so dx should take us to the optimum in 1 step (if H is perfect).
        # But numerical errors in H might make it diverge.
        
        # Let's try a few Newton steps with damping.
        current_x = x
        current_fx = fx
        
        for _ in range(5):
            if time.time() > deadline:
                break
            if calls >= self.budget:
                break
                
            # Recompute gradient and Hessian at current_x?
            # That would cost 30 evals per step. 5 steps = 150 evals.
            # We have 500 - 31 (initial) - 1 (center) = 468 left.
            # So we can afford ~15 recomputations.
            
            # Let's just do one robust Newton step from the initial model,
            # then switch to (1+1)-ES which is very efficient near the optimum.
            
            # Propose new x
            x_new = current_x + dx
            x_new = np.clip(x_new, bounds_low, bounds_high)
            
            # Check if x_new is actually better
            fx_new = evaluate(x_new)
            
            if fx_new < current_fx:
                current_x = x_new
                current_fx = fx_new
                
                # If we got very close (e.g. improvement is tiny), we can break early?
                # Or just let the ES polish it.
                
                # For the next Newton step, we would need to recompute H and g.
                # Let's do one more recompute if we improved significantly.
                if calls < self.budget - 40:
                     # Recompute model at current_x
                     x = current_x
                     fx = current_fx
                     
                     grad_vals = np.zeros(dim)
                     for i in range(dim):
                         p_plus = x.copy(); p_plus[i] += h
                         p_minus = x.copy(); p_minus[i] -= h
                         p_plus = np.clip(p_plus, bounds_low, bounds_high)
                         p_minus = np.clip(p_minus, bounds_low, bounds_high)
                         fp = evaluate(p_plus)
                         fm = evaluate(p_minus)
                         grad_vals[i] = (fp - fm) / (2 * h)
                         points[(i, i, 'pp')] = fp
                         points[(i, i, 'mm')] = fm
                     
                     H = np.zeros((dim, dim))
                     for i in range(dim):
                         H[i, i] = (points[(i, i, 'pp')] - 2 * fx + points[(i, i, 'mm')]) / (h * h)
                         
                     for i in range(dim):
                         for j in range(i + 1, dim):
                             p1 = x.copy(); p1[i] += h; p1[j] += h
                             p2 = x.copy(); p2[i] += h; p2[j] -= h
                             p3 = x.copy(); p3[i] -= h; p3[j] += h
                             p4 = x.copy(); p4[i] -= h; p4[j] -= h
                             
                             p1 = np.clip(p1, bounds_low, bounds_high)
                             p2 = np.clip(p2, bounds_low, bounds_high)
                             p3 = np.clip(p3, bounds_low, bounds_high)
                             p4 = np.clip(p4, bounds_low, bounds_high)
                             
                             f1 = evaluate(p1)
                             f2 = evaluate(p2)
                             f3 = evaluate(p3)
                             f4 = evaluate(p4)
                             
                             H_ij = (f1 - f2 - f3 + f4) / (4 * h * h)
                             H[i, j] = H_ij
                             H[j, i] = H_ij
                             
                     H_reg = H + 1e-6 * np.eye(dim)
                     try:
                         dx = np.linalg.solve(H_reg, -grad_vals)
                     except np.linalg.LinAlgError:
                         dx = -grad_vals * 0.1
                else:
                    break
            else:
                # If the step was bad, shrink it?
                # Or just break and rely on ES.
                # Since we are on a sphere, if the step was bad, it's likely due to H being wrong (e.g. indefinite).
                # But Sphere H is positive definite.
                # Maybe we just break and let ES take over from current_x.
                break
                
        # Phase 2: (1+1)-ES Polish
        # Start from the best point found so far
        x = best_x
        fx = best_fx
        
        # Initial sigma
        # If we are close to optimum, small sigma. If not, larger.
        # We can estimate distance from origin or just use a fixed reasonable value.
        # Given the Newton step, we should be close.
        sigma = 0.1
        
        # Adaptive sigma:
        # If success, increase sigma. If fail, decrease.
        
        remaining_budget = self.budget - calls
        
        for _ in range(remaining_budget):
            if time.time() > deadline:
                break
            if calls >= self.budget:
                break
                
            # Generate mutant
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, bounds_low, bounds_high)
            
            fy = evaluate(y)
            
            if fy < fx:
                x = y
                fx = fy
                # Increase sigma
                sigma *= 1.05
            else:
                # Decrease sigma
                sigma *= 0.95
                
            # Prevent sigma from becoming too small or too large
            if sigma < 1e-6:
                sigma = 1e-6
            if sigma > 5.0:
                sigma = 5.0

        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            
        return best_x, best_fx
# EVOLVE-BLOCK-END

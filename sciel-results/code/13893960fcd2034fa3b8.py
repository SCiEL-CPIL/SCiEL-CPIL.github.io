import numpy as np
import time

class Optimizer:
    """
    Optimizer using a Newton-like step based on finite difference gradients.
    Assumes the function is locally quadratic with Hessian approx 2*I (Sphere).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        # Start at the center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1
        
        # Initial step size for gradient estimation
        h = 1.0
        
        start_time = time.time()
        
        # We will iterate until budget or time limit
        # Each iteration costs 2*dim + 1 evaluations (2*dim for grad, 1 for new point)
        # Plus the initial 1.
        
        while calls < budget:
            # Safety margin for time
            if time.time() - start_time > 290:
                break
            
            # Check if we have enough budget for a full gradient step
            # 2*dim for gradient, 1 for the step
            if calls + 2 * dim + 1 > budget:
                # Not enough for a full step, try a partial step or break
                # If we have enough for 1 step and some grad? 
                # Just break to be safe, as partial grad is useless for Newton step.
                break

            # Estimate gradient using central differences
            grad = np.zeros(dim)
            for i in range(dim):
                xp = x.copy()
                xm = x.copy()
                xp[i] += h
                xm[i] -= h
                
                # Clip to bounds
                xp = np.clip(xp, -5, 5)
                xm = np.clip(xm, -5, 5)
                
                # If clipping prevented movement, use forward/backward diff?
                # For now, just evaluate. If xp==x, the diff is 0, which is wrong.
                # Check if movement was possible
                if np.allclose(xp, x):
                     # Use forward diff
                     xp = x.copy()
                     xp[i] += h
                     xp = np.clip(xp, -5, 5)
                     if np.allclose(xp, x):
                         grad[i] = 0.0
                         continue
                     f_p = func(xp)
                     calls += 1
                     grad[i] = (f_p - fx) / (xp[i] - x[i])
                elif np.allclose(xm, x):
                     # Use backward diff
                     xm = x.copy()
                     xm[i] -= h
                     xm = np.clip(xm, -5, 5)
                     if np.allclose(xm, x):
                         grad[i] = 0.0
                         continue
                     f_m = func(xm)
                     calls += 1
                     grad[i] = (fx - f_m) / (x[i] - xm[i])
                else:
                    f_p = func(xp)
                    f_m = func(xm)
                    calls += 2
                    grad[i] = (f_p - f_m) / (2 * h)

            # Newton step assuming Hessian = 2*I
            # x_new = x - 0.5 * grad
            x_new = x - 0.5 * grad
            x_new = np.clip(x_new, -5, 5)
            
            # If x_new is very close to x, we are likely at the minimum
            if np.linalg.norm(x_new - x) < 1e-8:
                break

            f_new = func(x_new)
            calls += 1
            
            if f_new < fx:
                x = x_new
                fx = f_new
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                # Decrease h to improve gradient accuracy as we get closer
                # But not too fast to avoid numerical noise
                h *= 0.5
                if h < 1e-5:
                    h = 1e-5
            else:
                # No improvement. 
                # This can happen if h is too large and the quadratic assumption fails,
                # or if we are at a boundary and the step goes outside.
                # Reduce h to get a more accurate local gradient.
                h *= 0.5
                if h < 1e-6:
                    # If we are stuck with very small h and no improvement, 
                    # try a random perturbation to escape potential local minima (though Sphere has none)
                    # or just break.
                    # Let's try a random jump if we are far from best?
                    # Actually, if Sphere, we should have converged.
                    break
        
        # Final high-precision polish pass: use smaller h for more accurate gradient
        if calls < budget and time.time() - start_time < 295:
            h_polish = 1e-4
            x_polish = x.copy()
            fx_polish = fx
            for _ in range(5):  # Max 5 polish iterations
                if calls + 2 * dim + 1 > budget:
                    break
                if time.time() - start_time > 295:
                    break
                
                grad_polish = np.zeros(dim)
                for i in range(dim):
                    xp = x_polish.copy()
                    xm = x_polish.copy()
                    xp[i] += h_polish
                    xm[i] -= h_polish
                    xp = np.clip(xp, -5, 5)
                    xm = np.clip(xm, -5, 5)
                    
                    if np.allclose(xp, x_polish) and np.allclose(xm, x_polish):
                        grad_polish[i] = 0.0
                        continue
                    elif np.allclose(xp, x_polish):
                        xm = x_polish.copy()
                        xm[i] -= h_polish
                        xm = np.clip(xm, -5, 5)
                        if np.allclose(xm, x_polish):
                            grad_polish[i] = 0.0
                            continue
                        f_m = func(xm)
                        calls += 1
                        grad_polish[i] = (fx_polish - f_m) / (x_polish[i] - xm[i])
                    elif np.allclose(xm, x_polish):
                        xp = x_polish.copy()
                        xp[i] += h_polish
                        xp = np.clip(xp, -5, 5)
                        if np.allclose(xp, x_polish):
                            grad_polish[i] = 0.0
                            continue
                        f_p = func(xp)
                        calls += 1
                        grad_polish[i] = (f_p - fx_polish) / (xp[i] - x_polish[i])
                    else:
                        f_p = func(xp)
                        f_m = func(xm)
                        calls += 2
                        grad_polish[i] = (f_p - f_m) / (2 * h_polish)
                
                x_new_polish = x_polish - 0.5 * grad_polish
                x_new_polish = np.clip(x_new_polish, -5, 5)
                
                if np.linalg.norm(x_new_polish - x_polish) < 1e-10:
                    break
                
                f_new_polish = func(x_new_polish)
                calls += 1
                
                if f_new_polish < fx_polish:
                    x_polish = x_new_polish
                    fx_polish = f_new_polish
                    if fx_polish < best_fx:
                        best_fx = fx_polish
                        best_x = x_polish.copy()
                else:
                    break
        
        return best_x, best_fx

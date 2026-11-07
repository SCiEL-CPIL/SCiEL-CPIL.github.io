import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        calls = 0
        deadline = time.time() + 290

        def evaluate(x):
            nonlocal calls
            if time.time() > deadline:
                raise TimeoutError()
            calls += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        # Phase 1: Probe the function structure
        # We evaluate the origin and the 5 unit axis vectors.
        # For a linear function f(x) = w^T x + b:
        # f(0) = b
        # f(e_i) = w_i + b
        # Thus w_i = f(e_i) - f(0)
        
        points = []
        values = []
        
        # Point 0: Origin
        x0 = np.zeros(dim)
        f0 = evaluate(x0)
        points.append(x0)
        values.append(f0)
        
        best_x = x0.copy()
        best_f = f0

        # Points 1-5: Unit axis vectors
        for i in range(dim):
            xi = np.zeros(dim)
            xi[i] = 1.0
            fi = evaluate(xi)
            points.append(xi)
            values.append(fi)
            
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()

        # Phase 2: Fit a linear model using least squares
        # We have 6 points. We want to find w and b such that X @ [w; b] ≈ y
        # X is 6x6 (identity + column of ones? No, let's construct properly)
        # Actually, we can just compute the gradient directly since the design matrix is orthogonal/simple.
        # w_i = f(e_i) - f(0)
        
        w = np.zeros(dim)
        for i in range(dim):
            w[i] = values[i+1] - values[0]
            
        b = values[0]

        # Verify linearity: Check if the model predicts the values well
        # For a perfect linear function, the residual should be 0.
        # If there is significant residual, the function might not be purely linear,
        # or numerical issues exist. However, the task states it is BBOB f5 (Linear slope).
        # The "hidden seeded shift, rotation and offset" preserves linearity.
        # f(x) = w^T (R x + s) + o = (R^T w)^T x + (w^T s + o)
        # So it is still linear in x.
        
        # Phase 3: Determine the optimal corner
        # The minimum of a linear function w^T x + b on [-5, 5]^dim is at the vertex
        # x*_i = -5 if w_i > 0
        # x*_i = +5 if w_i < 0
        # If w_i == 0, the coordinate doesn't matter. We pick 0 for stability.
        
        x_opt = np.zeros(dim)
        for i in range(dim):
            if w[i] > 1e-12:
                x_opt[i] = -5.0
            elif w[i] < -1e-12:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0

        # Evaluate the predicted optimum
        f_opt_pred = evaluate(x_opt)
        
        if f_opt_pred < best_f:
            best_f = f_opt_pred
            best_x = x_opt.copy()
            
        # Phase 4: Sanity Check / Fallback
        # If the residual was large, or if the predicted optimum wasn't actually the best,
        # we might have made an error. But for a linear function, this vertex IS the global min.
        # Let's check the residual to see if our linear assumption held.
        residuals = []
        for i in range(6):
            pred = np.dot(w, points[i]) + b
            residuals.append(values[i] - pred)
        max_residual = max(abs(r) for r in residuals)
        
        # If the function was NOT linear (which shouldn't happen for f5), 
        # the vertex might not be the min. 
        # However, given the budget is 500 and we've only used 7, we can afford a small local search
        # around the vertex if we suspect non-linearity, but for f5 it is exact.
        
        # To be extremely safe against any "hidden" non-linearity or numerical noise 
        # that might flip a sign incorrectly if w[i] is very close to 0, 
        # we can check the neighbors of the vertex if we have budget.
        # But for f5, w[i] is determined by the rotation/shift. 
        # If w[i] is effectively 0, the function is flat in that direction.
        
        # Let's just return the best found. The vertex is the theoretical global min for linear f.
        return best_x, best_f

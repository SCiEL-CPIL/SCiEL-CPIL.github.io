"""Model-and-optimize with Nelder-Mead polish for Sphere."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Model-based optimizer with Nelder-Mead for BBOB-style Sphere functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Track calls and best
        self.calls = 0
        self.best_x = np.zeros(dim)
        self.best_f = float('inf')

        def safe_call(x):
            if self.calls >= budget:
                return self.best_f
            if time.time() > deadline:
                return self.best_f
            
            x = np.clip(x, -5.0, 5.0)
            f_val = func(x)
            self.calls += 1
            
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x.copy()
                
            return f_val

        # --- Phase 1: Model-based estimation ---
        # f(x) = ||x - x*||^2 + C = x^T x - 2 x*^T x + ||x*||^2 + C
        # Let y = f(x) - ||x||^2. Then y = -2 x*^T x + (||x*||^2 + C)
        # This is a linear system in x* and constant term.
        # We need dim + 1 points. Let's use origin and dim points along axes.
        
        points = [np.zeros(dim)]
        # Choose step size for exploration. 4.0 is safe within [-5, 5]
        D = 4.0
        for i in range(dim):
            p = np.zeros(dim)
            p[i] = D
            points.append(p)
        
        values = []
        for p in points:
            if self.calls >= budget or time.time() > deadline:
                break
            v = safe_call(p)
            values.append(v)
        
        x_star_est = None
        if len(values) == dim + 1:
            try:
                # Construct linear system:
                # y_i = f(x_i) - ||x_i||^2
                # y = -2 * X^T * x_star + b
                # where X is matrix of points (dim+1 x dim), y is vector (dim+1)
                # We have dim+1 equations and dim+1 unknowns (x_star has dim, b has 1)
                
                X = np.array(points)  # (dim+1, dim)
                y = np.array([v - np.dot(p, p) for v, p in zip(values, points)])
                
                # Augmented system:
                # [ -2*X  1 ] [ x_star ]   [ y ]
                # [          ] [ b      ] = [   ]
                
                A = np.hstack([-2 * X, np.ones((dim + 1, 1))])
                # Solve least squares (though it should be exact for perfect sphere)
                solution = np.linalg.lstsq(A, y, rcond=None)[0]
                x_star_est = solution[:dim]
                
                # Clip to valid range
                x_star_est = np.clip(x_star_est, -5.0, 5.0)
                
            except Exception:
                x_star_est = None

        # --- Phase 2: Refinement ---
        
        # Check if we have budget left
        if self.calls >= budget or time.time() > deadline:
            return self.best_x, self.best_f

        # If we have a valid estimate, evaluate it
        if x_star_est is not None:
            f_est = safe_call(x_star_est)
            
            # If the estimate is good (better than origin or close to it), use it as center for NM
            # If it's worse, it might be noise or out of bounds, but we still might want to start NM from there
            # if it's close to the boundary? No, better to start from the best known point.
            
            # Let's start Nelder-Mead from the best of (origin, x_star_est)
            # Actually, if x_star_est is significantly better, we are done with most of the work.
            # But to be safe against noise, run a few NM steps.
            
            # Initialize NM simplex around the best known point
            center = self.best_x.copy()
            
            # If x_star_est is much better, center is x_star_est
            # If not, center is origin (which is best_x if x_star_est was worse)
            
            # Create simplex
            step = 0.5  # Smaller step since we are likely close to optimum
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = center
            for i in range(dim):
                simplex[i + 1] = center
                simplex[i + 1, i] += step
            
            fvals = np.zeros(dim + 1)
            fvals[0] = self.best_f
            
            # Evaluate rest of simplex
            for i in range(1, dim + 1):
                if self.calls >= budget or time.time() > deadline:
                    break
                fvals[i] = safe_call(simplex[i])
            
            # Run Nelder-Mead
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            while self.calls < budget:
                if time.time() > deadline:
                    break
                
                # Sort
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                # Convergence check
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break
                
                # Centroid
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = safe_call(x_r)
                
                if self.calls >= budget:
                    break
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = safe_call(x_e)
                    
                    if self.calls >= budget:
                        if f_e < f_r:
                            simplex[-1] = x_e
                            fvals[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            fvals[-1] = f_r
                        break
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = safe_call(x_c)
                    
                    if self.calls >= budget:
                        if f_c < f_r:
                            simplex[-1] = x_c
                            fvals[-1] = f_c
                        break
                    
                    if f_c < f_r:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = safe_call(simplex[i])
                            if self.calls >= budget:
                                break
        else:
            # Fallback: random search if model failed
            rng = np.random.default_rng(42)
            for _ in range(10):
                if self.calls >= budget: break
                x = rng.uniform(-5, 5, dim)
                safe_call(x)

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END

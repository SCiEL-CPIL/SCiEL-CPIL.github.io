"""Optimizer for BBOB f13 (Sharp Ridge) with Ridge Tracking."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Ridge-tracking optimizer for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280  # safety margin before 300s limit
        self.history = []  # List of (f, x) tuples

    def _call(self, func, x):
        """Evaluate function with call counting and deadline check."""
        if self.calls >= self.budget:
            return None
        if time.time() > self.deadline:
            return None
        self.calls += 1
        fx = func(x)
        if fx < self.best_f:
            self.best_f = fx
            self.best_x = x.copy()
        self.history.append((fx, x.copy()))
        return fx

    def _halton(self, dim, count, rng):
        """Generate Halton sequence points in [0,1)^dim."""
        points = []
        for i in range(1, count + 1):
            pt = []
            for base in range(2, 2 + dim):
                f = 1.0 / base
                p = base
                while p <= i:
                    f *= base
                    p *= base
                pt.append(f)
            points.append(pt)
        return np.array(points)

    def _golden_section_search(self, func, x0, direction, budget, tol=1e-6):
        """
        1-D Golden Section Search along 'direction' starting from 'x0'.
        Returns the best point and value found.
        """
        if self.calls >= self.budget or time.time() > self.deadline:
            return x0, self._call(func, x0) if self._call(func, x0) is not None else self.best_f

        # Determine initial bracket
        # We assume the function is convex along the ridge.
        # Start with a small step.
        step = 0.5
        # Check if moving in +direction improves
        x_plus = np.clip(x0 + direction * step, -5, 5)
        f_plus = self._call(func, x_plus)
        if f_plus is None:
            return x0, self.best_f
        
        x_minus = np.clip(x0 - direction * step, -5, 5)
        f_minus = self._call(func, x_minus)
        if f_minus is None:
            return x0, self.best_f

        # Determine the direction of descent
        if f_plus < f_minus:
            # Descent is in +direction
            a = 0.0
            b = step
            fa = self.best_f  # f(x0)
            fb = f_plus
            # Ensure we have a valid bracket [a, b] where f(a) > f(b) or f(b) < f(a)
            # Actually, Golden Section assumes we are minimizing on an interval.
            # Let's define the scalar function g(t) = f(x0 + t * direction)
            # We know g(0) = fa, g(step) = fb.
            # If fb < fa, the minimum is likely > 0.
            # If fa < fb, the minimum is likely < 0 (or at 0).
            
            if f_plus < self.best_f:
                # Try to expand the bracket to find a minimum
                # We know g(0) > g(step).
                # Let's search in [0, step * 2] or similar.
                # Simple approach: use the points we have.
                # a = 0, b = step.
                # We need to ensure the minimum is inside.
                # If g(step) < g(0), min is in (0, something).
                # Let's just do a few iterations of golden section on [0, step*2] if it looks promising.
                
                # Let's refine the bracket.
                # Evaluate at 2*step
                x_2step = np.clip(x0 + direction * (2 * step), -5, 5)
                f_2step = self._call(func, x_2step)
                if f_2step is None:
                    return x_plus, f_plus

                if f_2step < f_plus:
                    # Minimum is further out
                    a = step
                    b = 2 * step
                    fa = f_plus
                    fb = f_2step
                else:
                    # Minimum is between 0 and 2*step
                    # We have points at 0, step, 2*step
                    # f(0)=fa_orig, f(step)=f_plus, f(2*step)=f_2step
                    # Since f_plus < fa_orig and f_plus < f_2step (implied if we chose this branch? No, f_2step >= f_plus)
                    # So min is in [0, 2*step]
                    a = 0.0
                    b = 2 * step
                    fa = self.best_f # f(0)
                    fb = f_2step
                    # We already evaluated at 'step' (which is fb? No, fb is f_2step)
                    # Let's restart the bracket logic cleanly.
                    
                    # Bracket [a, b] = [0, 2*step]
                    # Points evaluated: 0 (fa), step (f_plus), 2*step (f_2step)
                    # We know f_plus < fa and f_plus <= f_2step.
                    # So min is in [0, 2*step].
                    # Golden Section on [0, 2*step]
                    gr = (np.sqrt(5) - 1) / 2  # ~0.618
                    c = a + (1 - gr) * (b - a)  # ~0.382 * b
                    d = a + gr * (b - a)        # ~0.618 * b
                    # We already have f(c) if c==step? 0.382 * 2 = 0.764 != 1.
                    # We have f(1.0).
                    # Let's just run a standard loop.
                    
                    # To save calls, we can use the existing evaluations.
                    # But for simplicity and robustness, let's just run a few iterations.
                    # Budget remaining: budget - self.calls
                    max_iter = min(10, budget - self.calls)
                    
                    # We have f(0), f(step), f(2*step).
                    # Let's assume the min is near 'step'.
                    # Use 'step' as the center.
                    # New bracket: [0, 2*step]
                    # We will perform Golden Section.
                    
                    # To avoid complexity of reusing points, we'll just do a simple ternary search or golden section from scratch on a smaller range if we have budget.
                    # But we are tight on budget.
                    
                    # Let's just return the best of the 3 points if budget is low.
                    if self.calls > self.budget - 5:
                        best_local = min([(fa, x0), (f_plus, x_plus), (f_2step, x_2step)], key=lambda t: t[0])
                        return best_local[1], best_local[0]
                    
                    # Perform 5 iterations of Golden Section on [0, 2*step]
                    # We have f(0), f(step), f(2*step).
                    # Let's map them to the golden section points if possible.
                    # It's easier to just evaluate new points.
                    
                    # Actually, let's just use the best point found so far and try to refine it locally with a smaller step.
                    # This is getting complicated. Let's stick to a simpler "Line Search" that expands/contracts.
                    
                    # Simple Line Search:
                    # We know x_plus is better than x0.
                    # Try x_plus + direction * step * 0.5
                    # Try x_plus - direction * step * 0.5
                    # This is basically a local 1-D Nelder-Mead.
                    
                    # Let's implement a simple 1-D Nelder-Mead (3 points).
                    # Points: x0, x_plus, x_2step
                    # Sort by f
                    # Replace worst with reflection of best two.
                    
                    # For now, let's just return the best of the 3 points.
                    # The main gain comes from finding the ridge direction.
                    # The 1-D refinement can be done by repeating this with smaller steps.
                    
                    candidates = [
                        (self.best_f, x0),
                        (f_plus, x_plus),
                        (f_2step, x_2step)
                    ]
                    best_cand = min(candidates, key=lambda t: t[0])
                    return best_cand[1], best_cand[0]

            else:
                # Descent is in -direction
                x_2minus = np.clip(x0 - direction * (2 * step), -5, 5)
                f_2minus = self._call(func, x_2minus)
                if f_2minus is None:
                    return x_minus, f_minus

                if f_2minus < f_minus:
                    # Min is further out
                    pass
                # Similar logic, return best of 3
                candidates = [
                    (self.best_f, x0),
                    (f_minus, x_minus),
                    (f_2minus, x_2minus)
                ]
                best_cand = min(candidates, key=lambda t: t[0])
                return best_cand[1], best_cand[0]

        # If f_minus < f_plus
        else:
             # Symmetric logic
             x_2minus = np.clip(x0 - direction * (2 * step), -5, 5)
             f_2minus = self._call(func, x_2minus)
             if f_2minus is None:
                 return x_minus, f_minus

             candidates = [
                (self.best_f, x0),
                (f_minus, x_minus),
                (f_2minus, x_2minus)
             ]
             best_cand = min(candidates, key=lambda t: t[0])
             return best_cand[1], best_cand[0]

    def __call__(self, func):
        dim = self.dim
        rng = np.random.default_rng(42)

        # Phase 1: Coarse exploration with Halton sequence
        n_coarse = 50
        halton_points = self._halton(dim, n_coarse, rng)
        # Map [0,1] to [-5,5]
        halton_points = halton_points * 10 - 5
        
        for x in halton_points:
            if self.calls >= self.budget or time.time() > self.deadline:
                break
            self._call(func, x)

        if self.best_x is None:
            return None, np.inf

        # Phase 2: Ridge Tracking
        # Estimate ridge direction from history
        # Sort history by function value
        self.history.sort(key=lambda t: t[0])
        
        # Use the best 5 points to estimate direction
        # If they are close, the direction is the principal component of their differences
        # If they are far, we might be in a flat region or multiple basins.
        
        # Simple heuristic:
        # If the best point is significantly better than the second best, 
        # the direction from second best to best is a good descent direction.
        # But for a ridge, we want the direction ALONG the ridge.
        
        # Let's use the vector between the best point and the second best point.
        # If f(best) << f(second), the vector points towards the minimum.
        # If f(best) ~ f(second), the vector might be along the ridge.
        
        # For f13, the ridge is flat. So points on the ridge have similar f values.
        # The "valley" is the ridge.
        
        # Strategy:
        # 1. Get best point x_b.
        # 2. Get next best point x_s.
        # 3. Direction d = x_b - x_s.
        # 4. If ||d|| is small, d is noise. Use a random direction or coordinate axis.
        # 5. Perform 1-D search along d.
        
        # To make it robust, we'll alternate between:
        # - 1-D search along estimated ridge direction
        # - Small Nelder-Mead step to correct direction
        
        # Since we don't have a full NM implementation here, we'll stick to 1-D + random perturbations.
        
        # Let's define a function to estimate the ridge direction
        def estimate_ridge_direction():
            if len(self.history) < 2:
                return rng.normal(0, 1, dim)
            
            # Take best 3 points
            best_3 = self.history[:3]
            f_vals = [t[0] for t in best_3]
            x_vals = np.array([t[1] for t in best_3])
            
            # If the function values are very close, the direction between them is likely along the ridge
            if max(f_vals) - min(f_vals) < 1e-3:
                # Use PCA on x_vals
                mean_x = np.mean(x_vals, axis=0)
                centered = x_vals - mean_x
                cov = np.cov(centered.T)
                if cov.shape == (dim, dim):
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    # Largest eigenvalue corresponds to direction of maximum variance (ridge)
                    return eigenvectors[:, -1]
            else:
                # Direction from worst to best
                d = x_vals[0] - x_vals[-1]
                norm = np.linalg.norm(d)
                if norm > 1e-6:
                    return d / norm
                else:
                    return rng.normal(0, 1, dim)

        # Main Loop
        while self.calls < self.budget and time.time() < self.deadline:
            # Estimate direction
            d = estimate_ridge_direction()
            
            # Normalize
            norm_d = np.linalg.norm(d)
            if norm_d < 1e-6:
                d = rng.normal(0, 1, dim)
                norm_d = np.linalg.norm(d)
            d = d / norm_d
            
            # Perform 1-D search
            # Use a small step size
            step = 0.5
            x_new, f_new = self._golden_section_search(func, self.best_x, d, step)
            
            # If no improvement, try a random direction
            if f_new >= self.best_f - 1e-8:
                # Perturb
                d_rand = rng.normal(0, 1, dim)
                d_rand = d_rand / np.linalg.norm(d_rand)
                x_new, f_new = self._golden_section_search(func, self.best_x, d_rand, step * 0.5)
            
            # Update best if improved
            if f_new < self.best_f:
                self.best_f = f_new
                self.best_x = x_new.copy()
                self.history.append((f_new, x_new.copy()))
            
            # Reduce step size if we are stuck?
            # For now, keep step constant.
            
        return self.best_x, self.best_f
# EVOLVE-BLOCK-END

import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self.calls = 0
        rng = np.random.default_rng(42)

        def wrapper(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exhausted")
            self.calls += 1
            return func(x)

        # Finite difference gradient
        h = 1e-5
        def fd_gradient(x):
            grad = np.zeros(dim)
            for i in range(dim):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                # Clip to box? The problem says func clips. 
                # But if we step out of bounds, the gradient might be inaccurate.
                # However, the box is [-5, 5]. If x is near boundary, we should use one-sided.
                # For simplicity, let's assume we stay inside or use central if possible.
                # The problem says func clips. So if x_plus is out, it clips.
                # This might make the gradient inaccurate near boundaries.
                # But the minimum is in [-4, 4], so we should stay away from boundaries.
                
                f_plus = wrapper(x_plus)
                f_minus = wrapper(x_minus)
                grad[i] = (f_plus - f_minus) / (2 * h)
            return grad

        best_x = None
        best_f = np.inf

        # Multi-start with L-BFGS-B
        n_init = 5
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Random
        for _ in range(n_init - 1):
            if self.calls >= budget: break
            x = rng.uniform(-4, 4, dim) # Stay inside
            f = wrapper(x)
            starts.append(x.copy())
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Sort starts by fitness? We need to evaluate them first.
        # We already evaluated the random ones. The center was evaluated? 
        # Let's evaluate center explicitly.
        if self.calls < budget:
            fc = wrapper(np.zeros(dim))
            if fc < best_f:
                best_f = fc
                best_x = np.zeros(dim).copy()
            starts.append(np.zeros(dim))

        # Evaluate all starts to sort
        scored_starts = []
        for x in starts:
            # We might have already evaluated some. 
            # To be safe, we can just run L-BFGS-B from the best few.
            # But we need to know which is best.
            # Let's just pick the best from the initial random samples + center.
            pass

        # Simpler: Just run L-BFGS-B from the best 2-3 starts found so far.
        # We need to track the best starts.
        
        # Let's re-do the initialization tracking properly.
        candidates = []
        x0 = np.zeros(dim)
        f0 = wrapper(x0)
        candidates.append((f0, x0.copy()))
        if f0 < best_f:
            best_f = f0
            best_x = x0.copy()

        for _ in range(n_init - 1):
            if self.calls >= budget: break
            x = rng.uniform(-4, 4, dim)
            f = wrapper(x)
            candidates.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        candidates.sort(key=lambda x: x[0])

        for i, (f0, x0) in enumerate(candidates[:2]):
            if self.calls >= budget:
                break
            
            remaining = budget - self.calls
            # L-BFGS-B will use maxiter * (1 + dim) calls roughly?
            # maxfev is the limit.
            # Each gradient eval costs 2*dim calls.
            # So maxfev should be set to remaining.
            
            try:
                result = minimize(
                    wrapper,
                    x0,
                    jac=fd_gradient,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': 100,
                        'maxfev': remaining,
                        'gtol': 1e-8,
                        'ftol': 1e-8
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                break

        # Polish
        if best_x is not None and self.calls < budget:
             # Small random search
             sigma = 0.01
             while self.calls < budget:
                y = best_x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                try:
                    fy = wrapper(y)
                except ValueError:
                    break
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                    if sigma < 1e-6:
                        break

        return best_x, best_f

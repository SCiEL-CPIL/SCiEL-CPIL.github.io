import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 275.0
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def run_nm(x0, max_iter, xatol=1e-8, fatol=1e-10):
            nonlocal calls
            if calls >= budget or time.time() >= deadline:
                return
            try:
                def objective(x):
                    return evaluate(x)
                minimize(
                    objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': xatol,
                        'fatol': fatol,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        def run_lbfgsb(x0, max_iter):
            nonlocal calls
            if calls >= budget or time.time() >= deadline:
                return
            try:
                def obj(x):
                    return evaluate(x)
                minimize(
                    obj,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * n,
                    options={
                        'maxiter': max_iter,
                        'maxls': 50
                    }
                )
            except Exception:
                pass

        def run_powell(x0, max_iter):
            nonlocal calls
            if calls >= budget or time.time() >= deadline:
                return
            try:
                def obj(x):
                    return evaluate(x)
                minimize(
                    obj,
                    x0,
                    method='Powell',
                    options={
                        'maxiter': max_iter,
                        'xtol': 1e-8,
                        'ftol': 1e-10
                    }
                )
            except Exception:
                pass

        def langevin_step(x, f, step_size, momentum_decay=0.9):
            """
            Simple Langevin dynamics step with momentum.
            Uses finite differences to estimate gradient.
            """
            nonlocal calls
            if calls >= budget or time.time() >= deadline:
                return x
            
            # Estimate gradient via central differences
            eps = 1e-5
            grad = np.zeros(n)
            for i in range(n):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                x_plus = np.clip(x_plus, -5.0, 5.0)
                x_minus = np.clip(x_minus, -5.0, 5.0)
                fp = evaluate(x_plus)
                fm = evaluate(x_minus)
                grad[i] = (fp - fm) / (2 * eps)
            
            # Langevin step with momentum
            noise = step_size * rng.standard_normal(n)
            x_new = x - step_size * grad + noise
            x_new = np.clip(x_new, -5.0, 5.0)
            
            # Accept if better (Metropolis-like)
            fx = evaluate(x)
            fx_new = evaluate(x_new)
            if fx_new < fx:
                return x_new
            else:
                # Accept with probability exp(-(fx_new - fx) / temperature)
                temp = max(0.1, (fx - best_f) * 0.1)
                if rng.random() < np.exp(-(fx_new - fx) / temp):
                    return x_new
                return x

        # Phase 1: Diverse Exploration with multiple starting points
        # Use a mix of strategies: random, grid-like, and Rosenbrock-valley inspired
        num_exploration = 10
        calls_per_exploration = max(15, (budget - 150) // num_exploration)
        
        starts = []
        # Origin
        starts.append(np.zeros(n))
        # Random points
        for _ in range(5):
            starts.append(rng.uniform(-4.5, 4.5, size=n))
        # Rosenbrock-valley inspired: points along the valley x_i = x_{i-1}^2
        for level in [0.5, 1.0, 2.0]:
            x_valley = np.zeros(n)
            x_valley[0] = level
            for i in range(1, n):
                x_valley[i] = x_valley[i-1] ** 2
            x_valley = np.clip(x_valley, -4.5, 4.5)
            starts.append(x_valley)
        # Additional diverse points
        starts.append(rng.uniform(-3, 3, size=n))
        starts.append(rng.uniform(-4, 4, size=n))
        
        for x0 in starts:
            if calls >= budget or time.time() >= deadline:
                break
            
            # Try L-BFGS-B first for fast convergence
            run_lbfgsb(x0, max(10, calls_per_exploration // 3))
            
            # Follow up with NM for local polish
            if calls < budget and time.time() < deadline:
                run_nm(x0, calls_per_exploration // 3, xatol=1e-8, fatol=1e-10)
            
            # Try Powell for different direction-based search
            if calls < budget and time.time() < deadline:
                run_powell(x0, calls_per_exploration // 3)

        # Phase 2: Langevin dynamics to escape local minima
        if best_x is not None and calls < budget and time.time() < deadline:
            x_current = best_x.copy()
            step_size = 0.3
            for _ in range(20):
                if calls >= budget or time.time() >= deadline:
                    break
                x_current = langevin_step(x_current, best_f, step_size)
                # Adapt step size
                if best_f < float('inf'):
                    step_size = max(0.05, step_size * 0.95)

        # Phase 3: Polish with tighter tolerances
        if best_x is not None and calls < budget and time.time() < deadline:
            num_polish = 5
            calls_per_polish = max(10, (budget - calls) // num_polish)
            for _ in range(num_polish):
                if calls >= budget or time.time() >= deadline:
                    break
                # Small perturbation around best
                x0_polish = best_x + rng.normal(0, 0.03, size=n)
                x0_polish = np.clip(x0_polish, -5.0, 5.0)
                
                # Use NM with very tight tolerances
                run_nm(x0_polish, calls_per_polish, xatol=1e-10, fatol=1e-12)
                
                # Also try L-BFGS-B for comparison
                if calls < budget and time.time() < deadline:
                    run_lbfgsb(x0_polish, calls_per_polish // 2)

        # Phase 4: Final high-precision polish
        if best_x is not None and calls < budget and time.time() < deadline:
            remaining = budget - calls
            if remaining > 20:
                run_nm(best_x, remaining, xatol=1e-12, fatol=1e-14)
                if calls < budget and time.time() < deadline:
                    run_lbfgsb(best_x, remaining // 2)

        return best_x, best_f

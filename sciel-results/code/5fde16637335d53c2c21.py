"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Quadratic-aware optimizer for Sphere: Finite-difference Newton step + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        t_start = time.time()
        deadline = 290.0  # safety margin below the 300s wall-clock limit

        # Count function evaluations against the budget.
        calls = [0]

        def f(x):
            calls[0] += 1
            return func(x)

        best_x = np.zeros(dim)
        best_fx = np.inf

        def update_best(s, fv):
            nonlocal best_fx, best_x
            if fv < best_fx:
                best_fx, best_x = fv, s.copy()

        # ---- Phase 1: Quadratic Model Estimation ----
        # Estimate gradient and curvature at the origin to get a Newton step.
        x0 = np.zeros(dim)
        f0 = f(x0)
        update_best(x0, f0)
        
        if calls[0] >= budget:
            return best_x, best_fx

        h = 0.1
        g = np.zeros(dim)
        curv_sum = 0.0
        
        for i in range(dim):
            if calls[0] >= budget: break
            ei = np.zeros(dim)
            ei[i] = h
            x_plus = x0 + ei
            x_minus = x0 - ei
            
            fp = f(x_plus)
            if calls[0] >= budget: break
            fm = f(x_minus)
            
            g[i] = (fp - fm) / (2 * h)
            curv_sum += (fp + fm - 2 * f0) / (h * h)
            
            update_best(x_plus, fp)
            update_best(x_minus, fm)

        if curv_sum > 1e-10:
            lambda_est = curv_sum / dim
            z_est = x0 - g / lambda_est
            z_est = np.clip(z_est, -5.0, 5.0)
            
            if calls[0] < budget:
                fz = f(z_est)
                update_best(z_est, fz)
        
        # ---- Phase 2: Multi-start CMA-ES ----
        # Use a deterministic multi-start strategy with progressively smaller step sizes.
        # This leverages the global convexity of the Sphere function.
        # We allocate the remaining budget to CMA-ES runs.
        
        # Remaining budget
        remaining = budget - calls[0]
        if remaining <= 0:
            return best_x, best_fx

        # Strategy:
        # 1. Start with a large step size (4.0) to explore the box.
        # 2. If stagnation occurs or a significant improvement is found, restart with smaller step.
        # 3. Use a simple CMA-ES implementation.
        
        def cma_es(x_start, sigma, budget_left):
            nonlocal calls, best_x, best_fx
            if budget_left <= 0:
                return
            
            # Initialize population
            # Population size lambda = 4 + floor(3 * log(dim))
            pop_size = 4 + int(3 * np.log(dim))
            
            # Means
            m = x_start.copy()
            sigma = sigma
            
            # Covariance matrix
            C = np.eye(dim)
            # Evolution paths
            ps = np.zeros(dim)
            pc = np.zeros((dim, dim))
            
            # Learning rates
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            cs = (2 + mu_eff / (dim + 2)) / (dim + 2 + 2 * mu_eff / dim)
            c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * ((mu_eff + 2) / (dim + 2) - 1))
            damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1) - 1) - 1) + cs
            dampc = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1) - 1) - 1) + cc
            
            # Cholesky factor of C
            try:
                B = np.linalg.cholesky(C)
            except:
                B = np.eye(dim)
            
            # Initial population
            ys = []
            xs = []
            fvals = []
            
            # Generate initial population
            for i in range(pop_size):
                if calls[0] >= budget:
                    return
                z = rng.standard_normal(dim)
                x = np.clip(m + sigma * (B @ z), -5.0, 5.0)
                ys.append(z)
                xs.append(x)
                fvals.append(f(x))
                update_best(x, fvals[-1])
            
            # Main loop
            while calls[0] < budget and (time.time() - t_start) < deadline:
                # Sort by function value
                order = np.argsort(fvals)
                ys = [ys[i] for i in order]
                xs = [xs[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Update mean
                m_old = m.copy()
                m = np.sum([weights[i] * xs[i] for i in range(mu)], axis=0)
                
                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (B.T @ (m - m_old) / (sigma * np.linalg.norm(B.T @ (m - m_old)) + 1e-30))
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 + 1)) < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - cc) ** (1 / (budget_left - calls[0] + 1))) if budget_left > calls[0] else False
                pc = (1 - cc) * pc + hsig * cc * (m - m_old) / sigma
                C = (1 + c1 * hsig * (2 - cc) + cmu) * C - c1 * np.outer(ps, ps) - cmu * np.sum([weights[i] * np.outer(ys[i], ys[i]) for i in range(mu)], axis=0)
                
                # Symmetrize C
                C = (C + C.T) / 2
                
                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
                
                # Update Cholesky
                try:
                    B = np.linalg.cholesky(C)
                except:
                    B = np.eye(dim)
                    C = np.eye(dim)
                    sigma *= 0.5 # shrink if covariance is invalid
                
                # Generate new population
                new_ys = []
                new_xs = []
                new_fvals = []
                
                # Reuse worst individuals
                for i in range(mu, pop_size):
                    new_ys.append(ys[i])
                    new_xs.append(xs[i])
                    new_fvals.append(fvals[i])
                    
                for i in range(mu):
                    if calls[0] >= budget:
                        return
                    z = rng.standard_normal(dim)
                    x = np.clip(m + sigma * (B @ z), -5.0, 5.0)
                    new_ys.append(z)
                    new_xs.append(x)
                    new_fvals.append(f(x))
                    update_best(x, new_fvals[-1])
                    
                ys, xs, fvals = new_ys, new_xs, new_fvals
                
                # Check for stagnation
                if np.allclose(fvals[:mu], fvals[0], rtol=1e-10, atol=1e-10):
                    break
            
        # Multi-start sequence
        # 1. Try from the Newton estimate or origin if Newton failed
        start_points = []
        if curv_sum > 1e-10:
            start_points.append(best_x.copy())
        start_points.append(np.zeros(dim))
        
        # 2. Try from corners/edges if needed? No, stick to deterministic.
        # Let's just use the best point found so far and the origin.
        # And maybe a point further out?
        
        # Allocate budget to CMA-ES runs
        # Run 1: Large step, from best_x
        # Run 2: Medium step, from best_x (if improved)
        # Run 3: Small step, from best_x (polish)
        
        # We will run CMA-ES with a fixed number of iterations or until budget is exhausted.
        # Since CMA-ES is expensive, we might need to be careful with the budget.
        # pop_size is ~ 14 for dim=5.
        # Each iteration costs pop_size - mu = 7 evaluations.
        
        # Let's just run one CMA-ES instance for the remaining budget, starting from best_x.
        # But the directive says "multi-start strategy that sequentially executes CMA-ES with progressively smaller initial step sizes".
        
        # So we split the budget.
        # Let's say 50% for large, 30% for medium, 20% for small.
        
        # However, if the first one finds the optimum, we don't need the others.
        # So we run them sequentially, updating the start point and reducing the step size.
        
        # Step sizes: 4.0, 1.0, 0.2
        step_sizes = [4.0, 1.0, 0.2]
        
        # We need to be careful not to exceed the budget.
        # We'll track the calls used by CMA-ES.
        
        # Actually, the `calls` counter is global. So we can just run them in a loop.
        # But we need to pass the current best as the start point.
        
        for sigma_init in step_sizes:
            if calls[0] >= budget:
                break
            
            # Start from the current best point
            x_start = best_x.copy()
            
            # Run CMA-ES
            # We don't know how many calls CMA-ES will use, so we just let it run until budget is exhausted or stagnation.
            # But we want to try the next sigma if the first one stagnates quickly.
            
            # Let's limit the number of iterations for the first run to see if it makes progress.
            # Or just run it and see.
            
            # To implement "restarts upon stagnation", we can check if the best value improves significantly.
            # If not, we break and try the next sigma.
            
            # Let's modify cma_es to return early if no improvement for a certain number of iterations.
            
            # For simplicity, let's just run the full CMA-ES for the remaining budget.
            # The multi-start aspect is handled by the fact that we start from the best point found by the previous run.
            # But the directive says "progressively smaller initial step sizes".
            # So we should explicitly change the sigma.
            
            # Let's run a short CMA-ES for each sigma.
            # Budget allocation: 
            # Run 1: 200 calls
            # Run 2: 150 calls
            # Run 3: 150 calls
            # Total: 500 (approx, leaving some for the initial probes)
            
            # But we already used ~11 calls for probing.
            # So we have ~489 left.
            
            # Let's allocate:
            # Run 1: 200
            # Run 2: 150
            # Run 3: 139
            
            # This is getting complicated. Let's just run one CMA-ES with a dynamic sigma?
            # No, the directive is specific.
            
            # Let's try this:
            # Run CMA-ES with sigma=4.0. If it stagnates (no improvement in 50 calls), break and try sigma=1.0.
            
            # We need to instrument cma_es to check for stagnation.
            
            pass # Placeholder, will implement below
        
        # Let's implement a simpler version of multi-start CMA-ES.
        # We will run CMA-ES for a fixed number of evaluations, then restart with a smaller sigma if needed.
        
        # Evaluation budget for each phase:
        # Phase 1: 200 evals
        # Phase 2: 150 evals
        # Phase 3: 150 evals
        
        # But we must respect the total budget.
        
        # Let's just run the CMA-ES for the remaining budget, but change the sigma if we detect stagnation.
        
        # Actually, the simplest way to satisfy the directive is to run three separate CMA-ES instances.
        # We will allocate a fixed budget to each.
        
        # Let's say:
        # Run 1: 200 evals
        # Run 2: 150 evals
        # Run 3: 150 evals
        
        # But we must ensure we don't exceed the total budget.
        
        # Let's calculate the remaining budget.
        # remaining = budget - calls[0]
        
        # We will allocate the remaining budget to the three runs proportionally or fixed.
        
        # Let's try:
        # Run 1: min(200, remaining)
        # Run 2: min(150, remaining - used1)
        # Run 3: min(150, remaining - used1 - used2)
        
        # This requires tracking the calls used by each run.
        
        # Let's modify cma_es to return the number of calls used.
        
        # Actually, we can just use the global `calls` counter.
        
        # Let's implement the multi-start loop.
        
        # Note: The `cma_es` function defined above uses the global `calls` and `best_x`, `best_fx`.
        # So we can just call it.
        
        # But we need to control the budget per run.
        # We can pass a `max_calls` parameter to `cma_es`.
        
        # Let's redefine `cma_es` to accept `max_calls`.
        
        # Wait, the `cma_es` function is already defined. We can't redefine it easily inside the loop.
        # We can just check `calls[0]` against a limit inside the loop.
        
        # Let's do that.
        
        # Run 1
        if calls[0] < budget:
            # Set a limit for this run
            limit1 = min(200, budget - calls[0])
            start_calls = calls[0]
            cma_es(best_x.copy(), 4.0, limit1)
            used1 = calls[0] - start_calls
            
        # Run 2
        if calls[0] < budget:
            limit2 = min(150, budget - calls[0])
            start_calls = calls[0]
            cma_es(best_x.copy(), 1.0, limit2)
            used2 = calls[0] - start_calls
            
        # Run 3
        if calls[0] < budget:
            limit3 = min(150, budget - calls[0])
            start_calls = calls[0]
            cma_es(best_x.copy(), 0.2, limit3)
            used3 = calls[0] - start_calls
            
        return best_x, best_fx
# EVOLVE-BLOCK-END

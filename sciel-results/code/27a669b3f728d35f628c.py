import numpy as np
import time

class Optimizer:
    """
    Multi-start CMA-ES optimizer with Sobol initialization.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        # Sobol sequence generation (simple implementation for low d)
        # Using a standard Sobol direction numbers table for d=5 is complex to hardcode.
        # We can use a Halton sequence which is simpler to implement and has similar low-discrepancy properties.
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(count, dim):
            points = []
            for i in range(1, count + 1):
                point = []
                # First few primes
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                for d in range(dim):
                    point.append(halton(i, primes[d]))
                points.append(point)
            return np.array(points)

        # Map [0, 1] to [-5, 5]
        def map_to_box(points):
            return -5.0 + 10.0 * points

        # Phase 1: Quasi-random exploration
        n_init = 20
        init_points = map_to_box(halton_sequence(n_init, dim))
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Evaluate initial points
        candidates = []
        for x in init_points:
            if time.time() > deadline:
                break
            fx = func(x)
            calls_made += 1
            candidates.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort candidates by function value
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Multi-start CMA-ES
        # Select top 3 starting points
        n_starts = 3
        start_points = [candidates[i][1] for i in range(min(n_starts, len(candidates)))]
        
        # Remaining budget
        remaining_budget = budget - calls_made
        
        # Split budget among starts
        # Give more budget to the best start? Or equal?
        # Equal is safer for multimodal.
        budget_per_start = remaining_budget // n_starts if n_starts > 0 else 0
        
        # CMA-ES implementation
        def run_cma_es(x0, sigma0, budget_limit):
            nonlocal best_x, best_f, calls_made
            
            d = dim
            # Lambda (population size)
            # Standard: 4 + floor(3 * log(d))
            # For d=5, log(5)~1.6, 3*1.6=4.8, floor=4, lambda=8.
            # Let's use lambda=6 for faster generations with small budget.
            lam = 6 
            
            # Weights
            mu = lam // 2
            w = np.zeros(mu)
            for i in range(mu):
                w[i] = np.log(mu + 0.5) - np.log(i + 1)
            w /= w.sum()
            mueff = 1.0 / np.sum(w**2)
            
            # Strategy parameters
            cs = (mueff + 2) / (d + mueff + 5)
            cm = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) / (d + 3)
            ds = 1 + 2 * max(0, np.sqrt((mueff + 1) / (d + 1)) - 1) + cs
            
            # State variables
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(d)
            Cinv = np.eye(d)
            B = np.eye(d)
            D = np.ones(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            
            # Initial evaluation
            # We already have x0 evaluated? No, x0 is from init points, which were evaluated.
            # But CMA-ES usually starts a new evaluation loop.
            # We can reuse the value of x0 to save one call if we track it.
            # For simplicity, we assume x0 is the current best in this sub-run.
            # We will generate lambda offspring.
            
            f_mean = func(x0) if calls_made < budget else best_f # We already evaluated x0 in Phase 1
            # Actually, we should NOT re-evaluate x0.
            # We need to track the best value found in this CMA-ES run.
            # Let's assume the input x0 has a known value.
            # To be safe and simple, we will just run the loop.
            # We have budget_limit calls left for this CMA-ES instance.
            
            # We need to be careful not to exceed the global budget.
            # The caller passes budget_limit.
            
            for gen in range(budget_limit // lam + 1):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break
                if calls_made + lam > budget:
                    # Evaluate only as many as fit
                    lam_current = budget - calls_made
                    if lam_current <= 0:
                        break
                else:
                    lam_current = lam

                # Sample offspring
                ys = []
                for i in range(lam_current):
                    if time.time() > deadline:
                        break
                    # Cholesky decomposition of C
                    # For d=5, np.linalg.cholesky is fast
                    try:
                        L = np.linalg.cholesky(C)
                    except:
                        # If C is not positive definite, reset
                        C = np.eye(d)
                        L = np.eye(d)
                    
                    z = np.random.standard_normal(d)
                    y = L @ z
                    xi = x_mean + sigma * y
                    xi = np.clip(xi, -5, 5)
                    fi = func(xi)
                    calls_made += 1
                    ys.append((fi, xi))
                    
                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()

                if not ys:
                    break

                # Sort by fitness
                ys.sort(key=lambda t: t[0])
                
                # Update mean
                new_mean = np.zeros(d)
                for i in range(min(mu, len(ys))):
                    new_mean += w[i] * ys[i][1]
                x_mean = new_mean
                
                # Update step size and covariance
                # Calculate ps
                # Need B, D, Cinv
                # Update B, D, Cinv every 1/d * mueff generations? Or every gen for simplicity?
                # Every gen is fine for d=5.
                
                # Recalculate eigendecomposition
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    D = np.sqrt(eigvals)
                    B = eigvecs
                    Cinv = B @ np.diag(1.0 / D) @ B.T
                except:
                    D = np.ones(d)
                    B = np.eye(d)
                    Cinv = np.eye(d)
                
                # ps = (1-cs)*ps + cs * sqrt(mueff) * Cinv^(1/2) * (x_mean_old - x_mean_new)/sigma
                # Note: x_mean_old is the mean BEFORE update.
                # We need to store old_mean.
                # Let's restructure:
                # 1. Generate ys
                # 2. Sort
                # 3. Calculate new_mean
                # 4. Update ps, pc, C, sigma
                # 5. Set x_mean = new_mean
                
                # We lost old_mean in the loop above. Let's fix the loop structure.
                # This manual CMA-ES is getting complex and error-prone.
                # Alternative: Use (1+1)-ES with adaptive sigma (1/5th rule) which is simpler and robust for d=5.
                # The current program already uses (1+1)-ES.
                # The main weakness of (1+1)-ES is getting stuck in local optima.
                # The main weakness of the current program is random restarts.
                # If we replace random restarts with Halton restarts, we might get a boost.
                # Let's try a simpler structural change: **Halton-initialized (1+1)-ES with more frequent, smaller restarts.**
                # Or **Population-based (1+lambda)-ES with lambda=4, adaptive sigma.**
                
                # Let's go with **Halton-initialized (1+1)-ES** but with a twist:
                # Use the Halton sequence to define the *restart points*.
                # Instead of random restarts, we cycle through Halton points.
                # This ensures that if the first run fails, the next restart is in a different, systematic part of the space.
                
                break # Abort CMA-ES attempt, switch to improved (1+1)-ES with Halton restarts.
            
            return best_x, best_f

        # Improved Strategy: Halton-initialized (1+1)-ES with systematic restarts
        # We already evaluated 20 Halton points.
        # We will run (1+1)-ES from the best point.
        # If it stagnates, we restart from the next best Halton point, then the next, etc.
        # This is much more structured than random restarts.
        
        # Sort init_points by their function values (we have them in candidates)
        # candidates is already sorted.
        
        # We will iterate through the candidates as potential restart seeds.
        # For each seed, we run (1+1)-ES until budget is exhausted or stagnation occurs.
        
        # (1+1)-ES Parameters
        sigma = 1.0
        no_improve = 0
        max_no_improve = 15 # Restart after 15 failed steps
        
        # We will process the candidates in order of their initial fitness.
        # This means we first exploit the best region. If it fails, we try the second best, etc.
        # This is a "greedy" multi-start.
        
        # To avoid spending all budget on the first (possibly local) optimum, 
        # we can limit the budget per restart.
        # But for AOCC, early progress is key. The best initial point is likely the best region.
        # So we should spend most of the budget there.
        
        # Let's do this:
        # 1. Start with the best Halton point.
        # 2. Run (1+1)-ES.
        # 3. If stagnation (no improve for 20 steps), restart from the next best Halton point.
        # 4. Continue until budget is out.
        
        # We need to track which Halton points we've tried.
        # candidates list contains (f, x).
        
        # Reset sigma for each restart
        for i, (f_init, x_init) in enumerate(candidates):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
            
            # Start ES from this point
            x = x_init.copy()
            fx = f_init
            sigma = 1.0 # Reset step size
            no_improve = 0
            
            # Run ES
            while calls_made < budget:
                if time.time() > deadline:
                    break
                
                # Generate offspring
                y = x + sigma * np.random.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    no_improve = 0
                    # Increase sigma on success
                    sigma = min(sigma * 1.1, 5.0)
                else:
                    no_improve += 1
                    # Decrease sigma on failure
                    sigma *= 0.9
                    if no_improve > max_no_improve:
                        break # Stagnation, break to try next restart point
            
            # If we broke due to stagnation, try next candidate
            # If we broke due to budget, the outer loop will break.
            # If we broke due to time, the outer loop will break.
            
            # Optional: If we used very little budget for this restart, move to next.
            # If we used a lot, maybe we are done?
            # The stagnation break handles this.

        return best_x, best_f

# EVOLVE-BLOCK-END

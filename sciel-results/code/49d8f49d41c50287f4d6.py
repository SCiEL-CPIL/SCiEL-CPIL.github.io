import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        eval_count = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def time_check():
            return (time.time() - start_time) > deadline

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def wrapped_func(x):
            nonlocal eval_count
            if time_check():
                return 1e100
            eval_count += 1
            val = func(x)
            update_best(x, val)
            return val

        rng = np.random.default_rng(42)

        # CMA-ES helper: runs a short CMA-ES from a given starting point
        def run_cmaes(start_mean, start_sigma, max_evals, seed_offset):
            nonlocal eval_count, best_x, best_f
            
            if eval_count >= budget or time_check():
                return
            
            # CMA-ES Parameters
            lam = 8
            mu = lam // 2  # 4
            
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Step size control
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            
            # Covariance matrix adaptation
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            # State variables
            x_mean = start_mean.copy()
            sigma = start_sigma
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            
            # Path variables
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Precompute constants
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim**2))
            c_hs = 1.1 + 2.0 / (dim + 1.7)
            
            generation = 0
            local_best_f = best_f
            stagnation_count = 0
            
            local_rng = np.random.default_rng(42 + seed_offset)
            
            while eval_count < budget and not time_check():
                remaining = budget - eval_count
                if remaining < lam:
                    break
                
                # 1. Sample new points
                ys = []
                for k in range(lam):
                    z = local_rng.standard_normal(dim)
                    y = B @ (D * z)
                    ys.append(y)
                
                xs = x_mean + sigma * np.array(ys)
                
                # 2. Evaluate
                fitnesses = []
                eval_xs = []
                for i in range(lam):
                    if eval_count >= budget or time_check():
                        break
                    x = xs[i]
                    f = wrapped_func(x)
                    fitnesses.append(f)
                    eval_xs.append(x)
                
                if len(fitnesses) < lam:
                    break
                
                fitnesses = np.array(fitnesses)
                eval_xs = np.array(eval_xs)
                
                # 3. Sort by fitness
                order = np.argsort(fitnesses)
                xs_sorted = eval_xs[order]
                ys_array = np.array(ys)
                ys_sorted = ys_array[order][:mu]
                
                # 4. Update mean
                new_mean = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
                
                # 5. Update sigma (step size)
                y_mean = (new_mean - x_mean) / sigma
                
                norm_y_mean = np.linalg.norm(y_mean)
                sqrt_trace_C = np.sqrt(np.trace(C))
                ratio = norm_y_mean / (chi_n * sqrt_trace_C) if sqrt_trace_C > 0 else 0
                
                p_sigma = (1 - c_sigma) * p_sigma + c_hs * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * ratio * y_mean
                
                # h_sigma calculation with proper generation count
                norm_p_sigma = np.linalg.norm(p_sigma)
                gen = generation + 1
                h_sigma = (norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * gen)) < (1.4 + 2 / (dim + 1)) * chi_n)
                
                sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
                generation = gen
                
                # 6. Update covariance matrix C
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
                
                # Rank-one update
                rank_one = np.outer(p_c, p_c)
                if not h_sigma:
                    rank_one += c_c * (2 - c_c) * C
                
                # Rank-mu update
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    rank_mu += weights[i] * np.outer(ys_sorted[i], ys_sorted[i])
                
                C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                
                # Update x_mean
                x_mean = new_mean
                
                # 7. Decompose C for next iteration
                try:
                    D2, B = np.linalg.eigh(C)
                    D2 = np.clip(D2, 1e-10, None)
                    D = np.sqrt(D2)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    B = np.eye(dim)
                    D = np.ones(dim)
                
                # Stagnation detection
                if best_f < local_best_f - 1e-10:
                    local_best_f = best_f
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                
                if sigma < 1e-5 or stagnation_count > 15:
                    break

        # Phase 1: Integer grid sampling to find promising basins
        # Evaluate a diverse set of integer and near-integer points
        init_points = []
        
        # Origin
        init_points.append(np.zeros(dim))
        
        # Random integer points in [-4, 4]^dim
        for _ in range(15):
            pt = rng.integers(-4, 5, size=dim).astype(float)
            init_points.append(pt)
            
        # Near-integer points with small noise
        for _ in range(15):
            pt = rng.integers(-4, 5, size=dim).astype(float)
            noise = rng.uniform(-0.2, 0.2, size=dim)
            init_points.append(pt + noise)
            
        # A few general random points
        for _ in range(5):
            init_points.append(rng.uniform(-5, 5, dim))

        # Evaluate initial points
        for p in init_points:
            if eval_count >= budget or time_check():
                break
            wrapped_func(p)

        # Phase 2: Multi-restart CMA-ES from top initial points
        # Sort initial points by fitness and run CMA-ES from the best few
        if eval_count < budget and not time_check():
            # Collect evaluated points and their fitnesses
            # We'll use the best few as restart points
            # Since we track best_f, we know the global best
            # Run CMA-ES from the current best with a modest budget
            cmaes_budget = min(200, budget - eval_count)
            if cmaes_budget >= lam * 2:
                run_cmaes(best_x.copy(), 0.5, cmaes_budget, 0)

        # Phase 3: If budget remains, try a second restart from a different integer point
        if eval_count < budget and not time_check():
            # Find a different promising starting point
            # Try a few more integer points and pick the best
            second_best_x = None
            second_best_f = np.inf
            for _ in range(10):
                if eval_count >= budget or time_check():
                    break
                pt = rng.integers(-4, 5, size=dim).astype(float)
                f = wrapped_func(pt)
                if f < second_best_f and f < best_f:
                    second_best_f = f
                    second_best_x = pt.copy()
            
            if second_best_x is not None and (budget - eval_count) >= lam * 2:
                run_cmaes(second_best_x, 0.5, min(150, budget - eval_count), 100)

        # Phase 4: Local refinement with coordinate descent
        if eval_count < budget and not time_check():
            local_x = best_x.copy()
            local_f = best_f
            step = 0.1
            max_local_evals = min(40, budget - eval_count)
            
            for _ in range(max_local_evals):
                if eval_count >= budget or time_check():
                    break
                
                improved = False
                for d in range(dim):
                    for direction in [1, -1]:
                        trial_x = local_x.copy()
                        trial_x[d] += direction * step
                        trial_x = np.clip(trial_x, -5, 5)
                        
                        if eval_count >= budget or time_check():
                            break
                        
                        trial_f = wrapped_func(trial_x)
                        if trial_f < local_f:
                            local_x = trial_x
                            local_f = trial_f
                            improved = True
                
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break

        return best_x, best_f

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

        # 1. Initialization: Exploit Rastrigin structure (integer grid)
        # The global minimum of Rastrigin is at integer coordinates.
        # Sampling near integers increases the chance of finding the global basin.
        rng = np.random.default_rng(42)
        
        # Generate a set of candidate points
        init_points = []
        
        # Add origin
        init_points.append(np.zeros(dim))
        
        # Add random integer points in [-4, 4]^dim
        # This targets the likely location of the global minimum
        for _ in range(20):
            pt = rng.integers(-4, 5, size=dim).astype(float)
            init_points.append(pt)
            
        # Add points near integers with small noise
        for _ in range(20):
            pt = rng.integers(-4, 5, size=dim).astype(float)
            noise = rng.uniform(-0.1, 0.1, size=dim)
            init_points.append(pt + noise)
            
        # Add some general random points for coverage
        for _ in range(5):
            init_points.append(rng.uniform(-5, 5, dim))

        # Evaluate initial points
        for p in init_points:
            if eval_count >= budget or time_check():
                break
            wrapped_func(p)

        # 2. CMA-ES Phase
        # Use the best initial point as the starting mean for CMA-ES.
        # CMA-ES is robust to rotations and shifts, which are present in the task.
        
        # CMA-ES Parameters
        lam = 10
        mu = lam // 2  # 5
        
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
        x_mean = best_x.copy()
        sigma = 1.5  # Initial step size
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        
        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute constants
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim**2))
        c_hs = 1.1 + 2.0 / (dim + 1.7)
        
        # Track generations for proper h_sigma calculation
        generation = 0
        
        # Main CMA-ES Loop
        while eval_count < budget and not time_check():
            # 1. Sample new points
            ys = []
            for k in range(lam):
                z = rng.standard_normal(dim)
                y = B @ (D * z)
                ys.append(y)
            
            xs = x_mean + sigma * np.array(ys)
            
            # 2. Evaluate
            remaining = budget - eval_count
            num_evals = min(lam, remaining)
            if num_evals <= 0:
                break
            
            fitnesses = []
            eval_xs = []
            for i in range(num_evals):
                x = xs[i]
                f = wrapped_func(x)
                fitnesses.append(f)
                eval_xs.append(x)
                
                if eval_count >= budget:
                    break
            
            if eval_count >= budget:
                break
                
            # If we ran out of budget mid-evaluation, we can't do a full generation update
            if num_evals < lam:
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
            
            # Stagnation detection: if sigma is very small or no improvement for several generations
            if sigma < 1e-4:
                break
            
            # Check time
            if time_check():
                break

        # 3. Local refinement phase: use remaining budget for coordinate descent
        if eval_count < budget and not time_check():
            # Try local refinement from best point
            local_x = best_x.copy()
            local_f = best_f
            step = 0.1
            max_local_evals = min(30, budget - eval_count)
            
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

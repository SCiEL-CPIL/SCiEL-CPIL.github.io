"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer using a hybrid approach:
    1. Global scan using Grid + Halton sequence to identify promising basins.
    2. Local refinement using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        budget = self.budget
        dim = self.dim
        
        calls_used = 0

        def f(x):
            nonlocal calls_used
            if calls_used >= budget:
                return 1e10
            calls_used += 1
            val = func(x)
            return val

        # --- Phase 1: Global Scan ---
        # We use a mix of a coarse grid and a Halton sequence to ensure broad coverage.
        # dim=5 is small enough that a grid is feasible for a coarse scan.
        
        candidates = []
        
        # 1a. Coarse Grid Scan
        # 3 points per dimension -> 3^5 = 243 points. This might be too many for a "scan" if budget is tight, 
        # but 500 budget is decent. Let's stick to a smaller grid or fewer points if needed.
        # Let's use 2 points per dim? 2^5=32. Too sparse.
        # Let's use a specific set of points: corners + midpoints + halton.
        
        # Generate Halton points
        def halton_sequence(n, dim, seed=1):
            # Simple Halton generator
            def halton(index, base):
                f = 1.0
                r = 0.0
                while index > 0:
                    f /= base
                    r += f * (index % base)
                    index //= base
                return r
            
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2
                for i in range(n):
                    points[i, d] = halton(i + 1, base)
            return points

        # We will evaluate 30 Halton points and 30 Grid-like points (center, corners, etc.)
        n_halton = 30
        n_grid = 30
        
        points_to_eval = []
        
        # Grid-like points: Corners and Midpoints
        # For dim=5, corners are 2^5=32.
        for mask in range(2**dim):
            pt = np.zeros(dim)
            for d in range(dim):
                if mask & (1 << d):
                    pt[d] = 5.0
                else:
                    pt[d] = -5.0
            points_to_eval.append(pt)
        
        # Add center
        points_to_eval.append(np.zeros(dim))
        
        # Add Halton points mapped to [-5, 5]
        halton_pts = halton_sequence(n_halton, dim)
        halton_pts = halton_pts * 10.0 - 5.0
        for pt in halton_pts:
            points_to_eval.append(pt)

        # Shuffle to randomize order slightly (though order matters less if we take all)
        # But we might run out of budget. Let's prioritize.
        # Actually, just evaluate as many as we can.
        
        # To be safe with budget, let's limit global phase to 100 calls max.
        max_global_calls = min(100, budget // 4)
        
        evaluated = 0
        for pt in points_to_eval:
            if evaluated >= max_global_calls:
                break
            if time.time() > deadline:
                break
            pt_clipped = np.clip(pt, -5, 5)
            fp = f(pt_clipped)
            candidates.append((fp, pt_clipped.copy()))
            evaluated += 1

        if not candidates:
            # Fallback
            best_x = np.zeros(dim)
            best_f = f(best_x)
            candidates = [(best_f, best_x)]

        # Sort candidates
        candidates.sort(key=lambda x: x[0])
        
        # Keep top K candidates for restarts
        K = 5
        restart_candidates = [c[1] for c in candidates[:K]]
        restart_fits = [c[0] for c in candidates[:K]]

        global_best_x = restart_candidates[0].copy()
        global_best_f = restart_fits[0]

        # --- Phase 2: CMA-ES Local Refinement ---
        # We implement a lightweight CMA-ES.
        # Population size lambda = 4 + floor(3 * log(dim)) -> 4 + 3*2.3 = 10.9 -> 10
        lambda_pop = int(4 + 3 * np.log(dim))
        # Sigma initialization: distance from best to 2nd best, or fixed 1.0
        # We use 1.0 initially, then adapt.
        
        # CMA-ES Parameters
        sigma = 1.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Update rates
        mu = lambda_pop // 2  # Number of parents
        # Weights
        weights = np.arange(1, lambda_pop + 1)
        weights = weights ** 1.2  # Heuristic weighting
        weights = weights / weights.sum()
        mu_eff = 1.0 / np.sum(weights[:mu]**2)
        
        c_sigma = 1.0 / (dim + 3)
        c_c = (4 + mu_eff/dim) / (dim + 4 + 2*mu_eff/dim)
        c_1 = 2 / ((dim+1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff-2+1/mu_eff) / ((dim+2)**2 + mu_eff)))
        d_sigma = 1 + 2*max(0, np.sqrt((mu_eff-1)/(dim+1))-1) + c_sigma

        current_idx = 0
        x_mean = restart_candidates[current_idx].copy()
        # Initialize sigma based on local landscape if possible, else 1.0
        if len(restart_candidates) > 1:
            sigma = np.linalg.norm(restart_candidates[0] - restart_candidates[1]) / 2.0
            sigma = np.clip(sigma, 0.5, 5.0)
        else:
            sigma = 2.0
            
        best_f_cma = global_best_f
        x_best_cma = x_mean.copy()
        
        stagnation_count = 0
        max_stagnation = 20 # Generations without improvement
        
        # Main Loop
        while calls_used < budget and time.time() < deadline:
            # Generate population
            pop = []
            for i in range(lambda_pop):
                # Sample from N(x_mean, sigma^2 * C)
                z = np.random.default_rng(i + int(time.time()*1000)).standard_normal(dim)
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Regularize
                    L = np.linalg.cholesky(C + 1e-8 * np.eye(dim))
                
                # Standard CMA-ES sampling
                y = x_mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = f(y)
                pop.append((fy, y))
                
                # Early break if budget exhausted
                if calls_used >= budget:
                    break
                    
            if not pop:
                break

            # Sort population
            pop.sort(key=lambda x: x[0])
            
            # Check for improvement
            if pop[0][0] < best_f_cma:
                best_f_cma = pop[0][0]
                x_best_cma = pop[0][1].copy()
                stagnation_count = 0
                
                if best_f_cma < global_best_f:
                    global_best_f = best_f_cma
                    global_best_x = x_best_cma.copy()
            else:
                stagnation_count += 1
            
            # If stagnated, restart
            if stagnation_count > max_stagnation:
                current_idx += 1
                if current_idx < len(restart_candidates):
                    x_mean = restart_candidates[current_idx].copy()
                    # Reset CMA-ES parameters
                    C = np.eye(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    sigma = 1.5
                    best_f_cma = restart_fits[current_idx]
                    x_best_cma = x_mean.copy()
                    stagnation_count = 0
                else:
                    # Random restart
                    rng = np.random.default_rng(int(time.time()*1000))
                    x_mean = rng.uniform(-5, 5, dim)
                    fx_mean = f(x_mean)
                    if fx_mean < global_best_f:
                        global_best_f = fx_mean
                        global_best_x = x_mean.copy()
                    
                    C = np.eye(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    sigma = 2.0
                    best_f_cma = fx_mean
                    x_best_cma = x_mean.copy()
                    stagnation_count = 0
                    # Increment a virtual index or just continue
                    # To prevent infinite random restarts, we could limit them, 
                    # but budget will run out.
                    current_idx = len(restart_candidates) # Stay in random mode

            # Update Mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * pop[i][1]
            x_mean = x_new

            # Update Covariance Matrix
            # Calculate step
            delta_x = x_mean - x_mean_old # Wait, we need x_mean from previous iteration
            # I need to store x_mean_old. Let's refactor slightly.
            
            # Refactor: Store x_mean_old before updating
            # (This logic is getting complex inline. Let's implement a cleaner CMA-ES step below)
            # Actually, the above loop updates x_mean *before* using it to update C. 
            # Standard CMA-ES:
            # 1. Sample pop from N(mu, sigma, C)
            # 2. Evaluate
            # 3. Sort
            # 4. New mu = sum w_i y_i
            # 5. Update p_sigma, p_c, C, sigma
            
            # My current code:
            # x_mean is updated to x_new.
            # But I haven't updated C, sigma, etc. correctly in this specific snippet block above.
            # Let's implement a proper CMA-ES step function to be safe and correct.

            # RE-IMPLEMENTING THE STEP CORRECTLY WITHIN THE LOOP
            # I will discard the partial logic above and use a cleaner internal function for the step.
            # However, to keep it simple and fast, I will stick to the (1+1) logic if CMA-ES implementation is risky?
            # No, CMA-ES is better. Let's write it carefully.

            # Since I can't easily "undo" the x_mean update without storing it, 
            # let's assume the previous structure was just a sketch. 
            # I will rewrite the loop body completely in the final code.
            
            # For the purpose of this "thinking" block, I will just ensure the final code 
            # has a correct CMA-ES implementation.

            # If we detect an error or timeout in the complex CMA-ES, fallback to (1+1).
            # Given the complexity, maybe a simpler "Adaptive Step Size (1+1)" with 
            # direction memory (like Particle Swarm or simple gradient estimation) is safer?
            # Or just a robust (1+1) with larger restarts?
            
            # The prompt says "Make a STRUCTURAL change". CMA-ES is a strong structural change.
            # I will implement a simplified CMA-ES (just updating Sigma and Mean, maybe skipping Covariance adaptation initially? No, that's not CMA-ES).
            # Let's use a simplified "CMA-like" where we adapt sigma based on success rate, 
            # and use a population to estimate a "good" direction?
            
            # Actually, for dim=5, a simple (mu+lambda)-ES with adaptive sigma is often sufficient and much less error-prone than full CMA-ES in a constrained environment.
            # Let's use a (4+4)-ES with adaptive sigma.
            # 4 parents, 4 offspring.
            # Sigma adaptation: if >50% offspring are better, increase sigma, else decrease.
            
            pass # Placeholder for the actual code generation below

        return global_best_x, global_best_f

# --- ACTUAL CODE TO BE OUTPUT ---
# I will generate the code with a robust (mu+lambda)-ES approach, which is simpler than CMA-ES but better than (1+1) due to population diversity, and easier to implement correctly under time constraints.

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        budget = self.budget
        dim = self.dim
        calls_used = 0

        def f(x):
            nonlocal calls_used
            if calls_used >= budget:
                return 1e10
            calls_used += 1
            return func(x)

        # Phase 1: Global Scan
        def halton_sequence(n, dim):
            def halton(index, base):
                f = 1.0
                r = 0.0
                while index > 0:
                    f /= base
                    r += f * (index % base)
                    index //= base
                return r
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2
                for i in range(n):
                    points[i, d] = halton(i + 1, base)
            return points

        points_to_eval = []
        # Corners
        for mask in range(2**dim):
            pt = np.zeros(dim)
            for d in range(dim):
                if mask & (1 << d):
                    pt[d] = 5.0
                else:
                    pt[d] = -5.0
            points_to_eval.append(pt)
        # Center
        points_to_eval.append(np.zeros(dim))
        # Halton
        halton_pts = halton_sequence(50, dim)
        halton_pts = halton_pts * 10.0 - 5.0
        points_to_eval.extend(halton_pts)

        candidates = []
        max_global_calls = min(100, budget // 4)
        for pt in points_to_eval:
            if len(candidates) >= max_global_calls:
                break
            if time.time() > deadline:
                break
            pt_clipped = np.clip(pt, -5, 5)
            fp = f(pt_clipped)
            candidates.append((fp, pt_clipped.copy()))

        if not candidates:
            best_x = np.zeros(dim)
            best_f = f(best_x)
            candidates = [(best_f, best_x)]

        candidates.sort(key=lambda x: x[0])
        K = 5
        restart_candidates = [c[1] for c in candidates[:K]]
        restart_fits = [c[0] for c in candidates[:K]]

        global_best_x = restart_candidates[0].copy()
        global_best_f = restart_fits[0]

        # Phase 2: (mu+lambda)-ES with Adaptive Sigma
        mu = 4
        lam = 4
        rng = np.random.default_rng(0)
        
        current_idx = 0
        x_mean = restart_candidates[current_idx].copy()
        sigma = 1.0
        if len(restart_candidates) > 1:
            sigma = np.linalg.norm(restart_candidates[0] - restart_candidates[1]) / 2.0
            sigma = np.clip(sigma, 0.5, 5.0)
        else:
            sigma = 2.0

        best_f_es = global_best_f
        x_best_es = x_mean.copy()
        stagnation = 0
        max_stagnation = 15

        while calls_used < budget and time.time() < deadline:
            pop = []
            for i in range(lam):
                if calls_used >= budget or time.time() > deadline:
                    break
                y = np.clip(x_mean + sigma * rng.standard_normal(dim), -5, 5)
                fy = f(y)
                pop.append((fy, y))
            
            if not pop:
                break

            pop.sort(key=lambda x: x[0])
            
            # Adapt Sigma
            # If the best offspring is better than the previous best, increase sigma?
            # Simple rule: if best offspring < best parent (mean), increase sigma slightly, else decrease.
            # Or: if any offspring is better than x_best_es, increase, else decrease.
            if pop[0][0] < best_f_es:
                sigma *= 1.1
                best_f_es = pop[0][0]
                x_best_es = pop[0][1].copy()
                stagnation = 0
                if best_f_es < global_best_f:
                    global_best_f = best_f_es
                    global_best_x = x_best_es.copy()
            else:
                sigma *= 0.9
                stagnation += 1

            # Update Mean: weighted average of top mu offspring
            weights = np.ones(mu) / mu
            x_new = np.sum([weights[i] * pop[i][1] for i in range(min(mu, len(pop)))], axis=0)
            x_mean = x_new

            if stagnation > max_stagnation:
                current_idx += 1
                if current_idx < len(restart_candidates):
                    x_mean = restart_candidates[current_idx].copy()
                    sigma = 1.5
                    best_f_es = restart_fits[current_idx]
                    x_best_es = x_mean.copy()
                    stagnation = 0
                else:
                    # Random restart
                    x_mean = rng.uniform(-5, 5, dim)
                    fx = f(x_mean)
                    if fx < global_best_f:
                        global_best_f = fx
                        global_best_x = x_mean.copy()
                    sigma = 2.0
                    best_f_es = fx
                    x_best_es = x_mean.copy()
                    stagnation = 0
                    current_idx = len(restart_candidates)

        return global_best_x, global_best_f

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer with adaptive restart strategy for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin: 300s limit, leave 10s
        
        dim = self.dim
        total_budget = self.budget
        
        # Global best tracking
        global_best_x = np.zeros(dim)
        global_best_f = np.inf
        calls_made = 0
        
        # CMA-ES Parameters
        # Population size lambda = 4 + floor(3 * ln(n))
        lam = 4 + int(3 * np.log(dim))
        # Number of selected individuals mu = floor(lam / 2)
        mu = lam // 2
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        # Variance-effective population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size sigma
        initial_sigma = 2.0 
        
        # Restart strategy
        # We split the budget into phases. 
        # Phase 1: Explore broadly. If stagnation or convergence is detected, restart with a new center.
        # To maximize AOCC (early progress), we should not waste too much time on one restart if it fails.
        # However, CMA-ES is very efficient. 
        # Let's use a single robust run first, but with a mechanism to detect if we are stuck 
        # and then perform a local refinement or a smaller restart if budget allows.
        # Given dim=5 and budget=500, a single CMA-ES run is usually sufficient if initialized well.
        # But f7 has steps. CMA-ES handles steps okay, but might get stuck on plateaus.
        # A common trick for f7 is to use a larger initial sigma and let it converge.
        
        # Let's implement a simple restart loop.
        # We will allocate chunks of budget to CMA-ES runs.
        # If the first run consumes most of the budget, that's fine.
        # If it stagnates early, we can try another.
        
        # Actually, for 500 calls in 5D, a single CMA-ES run is typically very strong.
        # Let's stick to one main run but ensure robustness.
        # We will use a fixed seed for reproducibility in local tests, but the task implies 
        # hidden shifts. Using a fixed seed is fine as the shift is hidden but deterministic per instance.
        # Wait, the evaluator creates Optimizer and calls it. The seed inside should probably be fixed 
        # or random? The task says "hidden seeded shift". The optimizer's internal randomness 
        # can be fixed for determinism or random for robustness. 
        # Let's use a fixed seed for the optimizer's RNG to ensure consistent behavior 
        # across the 3 instances, relying on the CMA-ES adaptation to handle the shifts.
        
        rng = np.random.default_rng(42)
        
        # Initial mean: Center of the box [0, 0, ..., 0] is a reasonable start.
        # Or random? Center is safer for global coverage initially.
        mean = np.zeros(dim)
        
        # Covariance matrix C = I
        C = np.eye(dim)
        # Eigen decomposition for updates
        # C = B * D * B.T
        B = np.eye(dim)
        D = np.ones(dim)
        
        sigma = initial_sigma
        
        # Path lengths for adaptation
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Adaptation parameters
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        cmu = min(1 - cs, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        
        # Track best
        best_x = mean.copy()
        best_f = func(best_x)
        calls_made += 1
        
        # Population storage
        pop = np.empty((lam, dim))
        fit = np.empty(lam)
        
        # Main Loop
        # We run until budget is exhausted or time limit.
        # CMA-ES generation cost is lam evaluations.
        
        while calls_made < total_budget:
            if time.time() > deadline:
                break
            
            # Check if we have enough budget for a full generation
            if total_budget - calls_made < lam:
                # If not enough for a full gen, we can still do a few random perturbations 
                # or just stop. Stopping is cleaner for CMA-ES integrity.
                # But we can squeeze out remaining calls as simple random search around best.
                while calls_made < total_budget:
                    if time.time() > deadline:
                        break
                    # Simple random perturbation
                    step = sigma * rng.standard_normal(dim)
                    candidate = best_x + step
                    candidate = np.clip(candidate, -5, 5)
                    f_cand = func(candidate)
                    calls_made += 1
                    if f_cand < best_f:
                        best_f = f_cand
                        best_x = candidate.copy()
                break

            # Generate Population
            for i in range(lam):
                # Sample from N(mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                x = mean + sigma * (B @ (D * z))
                x = np.clip(x, -5, 5)
                pop[i] = x
                fit[i] = func(x)
                calls_made += 1
                
                if fit[i] < best_f:
                    best_f = fit[i]
                    best_x = x.copy()
                    if best_f < global_best_f:
                        global_best_f = best_f
                        global_best_x = best_x.copy()
                
                if time.time() > deadline:
                    # If we run out of time mid-generation, we can't update CMA-ES state properly.
                    # We break out and return best found so far.
                    return global_best_x, global_best_f

            # Sort by fitness
            idx = np.argsort(fit)
            pop_sorted = pop[idx]
            fit_sorted = fit[idx]
            
            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * pop_sorted[i]
            
            # Update path lengths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ (D @ (B.T @ y_w))) / sigma
            c_sigma = 1 + 2 * max((mueff - 1) / (dim + mueff - 3) - 1, 0) * cs
            pc = (1 - cc) * pc + c_sigma * (y_w / sigma - pc)
            
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            
            # Update covariance matrix
            delta_sigma = (1 + c_sigma) * cc * (np.linalg.norm(pc)**2 / dim - 1)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2)) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
            
            C = (1 + cmu * delta_sigma - cs * (1 - cs) - cmu) * C
            for i in range(mu):
                y_i = (pop_sorted[i] - mean) / sigma
                C += cmu * weights[i] * np.outer(y_i, y_i)
            
            # Update mean
            mean = mean + y_w
            
            # Update eigen decomposition (approximate update or full recompute)
            # For dim=5, full recompute is cheap
            D, B = np.linalg.eigh(C)
            # Ensure D is positive
            D = np.maximum(D, 1e-12)
            # Sort D and B accordingly? Eigh returns ascending order. 
            # CMA-ES usually wants descending? No, the formula works with any consistent basis.
            # But we need to ensure B columns correspond to D eigenvalues.
            # np.linalg.eigh returns eigenvalues in ascending order.
            # We should probably sort them in descending order for stability? 
            # Actually, the update formula C = B D B.T holds regardless.
            # However, standard CMA-ES implementations often sort to keep the largest eigenvalue first.
            idx_sort = np.argsort(D)[::-1]
            D = D[idx_sort]
            B = B[:, idx_sort]
            
            # Convergence check: if sigma is very small, we are converged.
            if sigma < 1e-8:
                break

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END

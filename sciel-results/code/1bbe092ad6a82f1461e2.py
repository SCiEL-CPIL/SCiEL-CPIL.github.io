import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline - 3.0

        calls_made = 0
        best_x = None
        best_fx = float('inf')

        def record(x, fx):
            nonlocal calls_made, best_x, best_fx
            calls_made += 1
            fx = float(fx)
            if fx < best_fx:
                best_fx = fx
                best_x = np.asarray(x, dtype=np.float64).copy()

        # Phase 1: Global Scan
        # Use a stratified random sampling for better coverage
        n_scan = 80
        scan_points = []
        # Divide the domain into a grid, pick one random point per cell
        # 4x4x4x4x4 = 1024 cells. We only have 80 points.
        # Better: Just use random sampling with a minimum distance constraint (simple rejection or just random).
        # For dim=5, 80 random points is sparse.
        # Let's use a grid of 3x3x3x3x3 = 243 points, and pick the best 80? No, evaluate 80.
        # Let's do a simple random scan but ensure spread.
        
        # Actually, for f21, the peaks are localized. Random might miss them.
        # The previous code used a 4^5 grid with jitter. That's 1024 points. It picked 60.
        # Let's stick to the grid approach but pick more or better ones.
        
        n_grid = 4
        grid_points = []
        for i in range(n_grid ** dim):
            coords = []
            tmp = i
            for d in range(dim):
                coords.append(tmp % n_grid)
                tmp //= n_grid
            x = np.array([(-5 + 10 * (c + 0.5) / n_grid) for c in coords])
            # Add jitter
            x += rng.uniform(-0.5, 0.5) * (10.0 / n_grid)
            x = np.clip(x, -5, 5)
            grid_points.append(x)
            
        # Shuffle and pick n_scan
        idx = rng.permutation(len(grid_points))[:n_scan]
        
        for i in idx:
            if not time_ok() or calls_made >= budget:
                break
            x = grid_points[i]
            fx = func(x)
            record(x, fx)
            
        if calls_made == 0:
             # Fallback
             x0 = rng.uniform(-4, 4, dim)
             fx0 = func(x0)
             record(x0, fx0)
             return best_x, best_fx

        # Phase 2: CMA-ES with Restarts
        # We will maintain a list of "promising" start points.
        # The best point from the scan is the first.
        # If CMA-ES converges, we restart.
        
        # Store scan results to pick restart points
        # Note: record() updates best_x. We need to store all scan points and their values.
        # Since record() doesn't return the value, we need to capture it.
        
        # Let's re-do the scan to store values.
        # Actually, we can just restart from the current best_x with a larger sigma.
        # Or, we can keep track of the top K scan points.
        
        scan_results = []
        # We lost the individual scan values in the loop above because we used record().
        # Let's modify the scan phase to store values.
        
        # Re-evaluating the scan phase logic:
        # The previous loop updated best_x. It did not store all points.
        # For restarts, it's better to restart from the *current* best point with a perturbed mean or larger sigma.
        # However, if the current best is in a shallow local minimum, and there's a deeper one in a different basin,
        # restarting from the current best won't find the other basin.
        # So we should store the top K scan points.
        
        # Let's rewrite the scan to store top K.
        # But I already wrote the loop. I'll just assume the best_x is a good start.
        # To get multiple basins, I can add a small random perturbation to the start point for each restart.
        
        def run_cma_es(center, sigma_init, max_evals):
            nonlocal calls_made, best_x, best_fx, time_ok, deadline, rng, dim, budget
            
            if calls_made >= budget or not time_ok():
                return
            
            # CMA-ES Parameters
            pop_size = max(4, int(4 + 3 * np.log(dim)))
            # For dim=5, pop_size = 4 + 3*1.6 = 8.8 -> 8.
            pop_size = 8
            
            # State
            mean = np.array(center, dtype=np.float64)
            sigma = sigma_init
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            
            # Adaptation
            c_sigma = (1 + 2 / max(dim - 2, 1) + pop_size / (2 * dim)) / (dim + 4 + pop_size / dim)
            c_c = (4 - dim / 3 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
            c_mu = min(1 - c_1, 2 * ((pop_size / 2 - 1 + 1. / pop_size) - c_1) / ((dim + 2) ** 2 + pop_size))
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1. / np.sum(weights ** 2)
            chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1 / (21 * dim ** 2))
            
            # Memory
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            h_sigma = 0.0
            count_evals = 0
            
            # Loop
            while calls_made < budget and time_ok() and count_evals < max_evals:
                # Sample
                population = []
                for i in range(pop_size):
                    z = rng.standard_normal(dim)
                    y = B @ (D * z)
                    x = mean + sigma * y
                    x = np.clip(x, -5, 5)
                    population.append(x)
                
                # Evaluate
                fvals = []
                for x in population:
                    if not time_ok() or calls_made >= budget:
                        break
                    fx = func(x)
                    record(x, fx)
                    fvals.append(fx)
                
                if len(fvals) < pop_size:
                    break
                
                # Sort
                order = np.argsort(fvals)
                y_sorted = [population[i] for i in order]
                
                # Update Mean
                y_mean = np.zeros(dim)
                for i in range(mu):
                    y_mean += weights[i] * (y_sorted[i] - mean) / sigma
                mean_new = mean + sigma * y_mean
                mean_new = np.clip(mean_new, -5, 5)
                
                # Update Sigma (CSA)
                # Norm of y_mean
                y_mean_norm = np.linalg.norm(y_mean)
                # Correction for recent bias
                bias_correction = np.sqrt(1 - np.exp(-2 * count_evals / (mu_eff * dim)))
                
                # Update p_sigma
                # The standard formula:
                # p_sigma = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)*mu_eff) * (1/sigma) * (mean - mean_old)
                # (mean - mean_old)/sigma is y_mean
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
                
                # Hysteresis
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (count_evals + 1) / dim)) < 
                           (1.4 + 2 / (dim + 1)) * chi_n)
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = max(1e-10, min(sigma, 10.0))
                
                # Update Covariance
                y_c = y_mean
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
                delta_h = (1 - h_sigma) * np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (2 * (count_evals + 1) / dim)) + h_sigma
                
                C = (1 + c_1 * delta_h * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
                for i in range(mu):
                    C += c_mu * weights[i] * np.outer(y_sorted[i] - mean, y_sorted[i] - mean)
                
                # Symmetrize and Eigen-decompose
                C = (C + C.T) / 2
                try:
                    eigenvalues, eigenvectors = np.linalg.eigh(C)
                    eigenvalues = np.maximum(eigenvalues, 1e-10)
                    C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                    B = eigenvectors
                    D = np.sqrt(eigenvalues)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    B = np.eye(dim)
                    D = np.ones(dim)
                    sigma = max(sigma, 0.1) # Reset if matrix broke
                
                mean = mean_new
                count_evals += pop_size
                
                # Convergence Check
                if np.linalg.norm(sigma * D) < 1e-6 * max(1.0, np.linalg.norm(mean)):
                    break

        # Restart Strategy
        # We have `budget - calls_made` left.
        # We will run CMA-ES multiple times.
        # Start from best_x.
        # If it finishes, restart from best_x with a larger sigma?
        # Or restart from a random point near best_x?
        
        remaining_budget = budget - calls_made
        if remaining_budget > 0:
            # Try 3 restarts
            restarts = 3
            for r in range(restarts):
                if calls_made >= budget or not time_ok():
                    break
                
                # Choose start point
                if r == 0:
                    start = best_x.copy()
                    sigma_init = 1.0
                else:
                    # Perturb the best point
                    # Or pick a random point in the box
                    # Let's try a random point first, then best point if that fails?
                    # No, random might miss.
                    # Let's try: Start from best_x, but with a larger sigma.
                    # If the function is multimodal, a larger sigma might jump to a better basin.
                    start = best_x.copy()
                    sigma_init = 2.0 * (1 + r) # Increasing sigma
                    # Add a random offset to break symmetry
                    start = start + rng.uniform(-0.5, 0.5, dim)
                    start = np.clip(start, -5, 5)

                # Allocate budget for this run
                # Divide remaining budget among restarts
                evals_per_run = remaining_budget // (restarts - r + 1)
                # Ensure at least 20 evals
                evals_per_run = max(20, evals_per_run)
                
                run_cma_es(start, sigma_init, evals_per_run)
                
                remaining_budget = budget - calls_made

        # Phase 3: Final Polish
        remaining = budget - calls_made
        if remaining > 0 and best_x is not None and time_ok():
            x = best_x.copy()
            fx = best_fx
            sigma_ref = 0.1
            
            while remaining > 0 and time_ok():
                y = np.clip(x + sigma_ref * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                record(y, fy)
                remaining -= 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma_ref *= 1.05
                else:
                    sigma_ref *= 0.9
                    if sigma_ref < 0.005:
                        sigma_ref = 0.05

        return best_x, best_fx

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
        # We will run CMA-ES starting from the best scan point, then restart with different strategies.
        
        def run_cma_es(center, sigma_init, max_evals):
            nonlocal calls_made, best_x, best_fx, time_ok, deadline, rng, dim, budget
            
            if calls_made >= budget or not time_ok():
                return
            
            # CMA-ES Parameters
            pop_size = max(4, int(4 + 3 * np.log(dim)))
            # For dim=5, pop_size = 4 + 3*1.609 = 4 + 4.827 = 8.827 -> 8
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
                    # Rank-mu update: use deviation from the OLD mean
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
                
                # Convergence Check: stop if the distribution is too narrow
                # or if we've made very little progress
                sigma_max = np.max(sigma * D)
                sigma_min = np.min(sigma * D)
                if sigma_max < 1e-8 * max(1.0, np.linalg.norm(mean)):
                    break
                # Also check if the best value hasn't improved in last 3 generations
                if count_evals > 3 * pop_size and np.all(np.array(fvals[order[:3]]) >= best_fx - 1e-10):
                    # Check if we've been stuck for a while
                    if count_evals > 10 * pop_size:
                        break

        # Restart Strategy
        # We have `budget - calls_made` left.
        # We will run CMA-ES multiple times with different strategies.
        
        remaining_budget = budget - calls_made
        if remaining_budget > 0:
            # Try up to 3 restarts with different initialization strategies
            restarts = 3
            for r in range(restarts):
                if calls_made >= budget or not time_ok():
                    break
                
                # Choose start point and sigma based on restart number
                if r == 0:
                    # First restart: start from best_x with moderate sigma
                    start = best_x.copy()
                    sigma_init = 0.5
                elif r == 1:
                    # Second restart: random point in the box to explore new basins
                    start = rng.uniform(-4, 4, dim)
                    sigma_init = 1.5
                else:
                    # Third restart: start from best_x with larger sigma to escape local minima
                    start = best_x.copy() + rng.uniform(-1.0, 1.0, dim)
                    start = np.clip(start, -5, 5)
                    sigma_init = 2.0

                # Allocate budget for this run
                # Divide remaining budget among remaining restarts
                evals_per_run = remaining_budget // (restarts - r + 1)
                # Ensure at least 20 evals
                evals_per_run = max(20, evals_per_run)
                
                run_cma_es(start, sigma_init, evals_per_run)
                
                remaining_budget = budget - calls_made

        # Phase 3: Final Polish using a finer CMA-ES run
        remaining = budget - calls_made
        if remaining > 0 and best_x is not None and time_ok():
            # Run a final CMA-ES with small sigma and tight convergence
            run_cma_es(best_x.copy(), 0.01, remaining)

        return best_x, best_fx

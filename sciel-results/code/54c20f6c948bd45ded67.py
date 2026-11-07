"""
Coarse-to-fine grid search followed by CMA-ES refinement for BBOB f3 (Rastrigin).
The grid phase systematically covers the box to identify the most promising region,
then CMA-ES refines within that region using the remaining budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 330s kill

        n = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0
        rng = np.random.default_rng(42)

        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= budget

        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Coarse-to-fine grid search
        # Start with a coarse grid, then refine around the best cell
        grid_calls = 0
        max_grid_calls = min(100, budget // 5)  # Use at most 100 or 20% of budget for grid
        
        # Coarse grid: 3 points per dimension = 3^5 = 243 points, too many
        # Use 2 points per dimension = 2^5 = 32 points for coarse
        # Then refine: 3 points per dimension in the best region
        
        # Coarse grid evaluation
        coarse_points = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        for m in range(2):
                            coarse_points.append([
                                -5.0 + 5.0 * (i + 0.5) / 2,
                                -5.0 + 5.0 * (j + 0.5) / 2,
                                -5.0 + 5.0 * (k + 0.5) / 2,
                                -5.0 + 5.0 * (l + 0.5) / 2,
                                -5.0 + 5.0 * (m + 0.5) / 2
                            ])
        
        # Evaluate coarse grid
        coarse_results = []
        for x in coarse_points:
            if check_deadline() or grid_calls >= max_grid_calls:
                break
            x = np.clip(x, -5, 5)
            f = func(x)
            update_best(x, f)
            grid_calls += 1
            coarse_results.append((f, x))
        
        if best_x is None:
            # Fallback: evaluate origin
            x = np.zeros(n)
            f = func(x)
            update_best(x, f)
            return best_x, best_f
        
        # Sort coarse results to find the best region
        coarse_results.sort(key=lambda t: t[0])
        
        # Phase 1b: Fine grid around the best coarse cell
        # The best coarse cell center is coarse_results[0][1]
        # Refine in a smaller region around it
        best_coarse_x = coarse_results[0][1]
        fine_radius = 2.5  # Half the coarse cell size
        
        # Create fine grid: 3 points per dimension in the refined region
        fine_points = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            fine_points.append([
                                best_coarse_x[0] - fine_radius + 2 * fine_radius * (i + 0.5) / 3,
                                best_coarse_x[1] - fine_radius + 2 * fine_radius * (j + 0.5) / 3,
                                best_coarse_x[2] - fine_radius + 2 * fine_radius * (k + 0.5) / 3,
                                best_coarse_x[3] - fine_radius + 2 * fine_radius * (l + 0.5) / 3,
                                best_coarse_x[4] - fine_radius + 2 * fine_radius * (m + 0.5) / 3
                            ])
        
        # Evaluate fine grid (but only if we have budget left)
        if not check_deadline() and grid_calls < max_grid_calls:
            for x in fine_points:
                if check_deadline() or grid_calls >= max_grid_calls:
                    break
                x = np.clip(x, -5, 5)
                f = func(x)
                update_best(x, f)
                grid_calls += 1
        
        # Phase 2: CMA-ES refinement starting from the best point found
        remaining = budget - calls_made
        if remaining > 10 and not check_deadline():
            self._run_cmaes(func, best_x.copy(), best_f, remaining)
        
        return best_x, best_f

    def _run_cmaes(self, func, x_start, f_start, max_calls):
        """Run CMA-ES refinement from the given starting point."""
        start_time = time.time()
        deadline = start_time + 280  # safety margin
        
        n = self.dim
        calls_made = 0
        
        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= max_calls
        
        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Initialize CMA-ES parameters
        lambda_pop = 4 + int(3 * np.log(n))
        lambda_pop = min(lambda_pop, max(2, max_calls // 2))
        if lambda_pop < 2:
            return
        
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)

        sigma = 0.5  # Start with moderate sigma for local refinement
        mean = x_start.copy()

        C = np.eye(n)
        C_sqrt = np.eye(n)

        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (weights[1] - 1) / ((n + 2)**2 + mu_eff))

        ps = np.zeros(n)
        pc = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

        no_improve = 0
        last_best = f_start

        while not check_deadline():
            # Check if we can afford a full generation
            if calls_made + lambda_pop > max_calls:
                remaining = max_calls - calls_made
                if remaining < mu:
                    break
                lambda_pop = remaining
                mu = max(1, lambda_pop // 2)
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= np.sum(weights)
                mu_eff = 1.0 / np.sum(weights**2)

            offspring = []
            for i in range(lambda_pop):
                z = np.random.standard_normal(n)
                y = mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                offspring.append((fy, y))
                if check_deadline():
                    break

            if len(offspring) < mu:
                break

            offspring.sort(key=lambda t: t[0])

            # Check improvement
            if offspring[0][0] < last_best - 1e-10:
                last_best = offspring[0][0]
                no_improve = 0
            else:
                no_improve += 1

            # Recombination
            y_w = np.zeros(n)
            for i in range(mu):
                y_w += weights[i] * offspring[i][1]

            old_mean = mean.copy()
            mean = y_w

            # Update paths
            ys = (mean - old_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * ys
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / chi_n
                    < (1.4 + 2 / (n + 1)) * cc)

            if hsig:
                pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * ys
            else:
                pc = (1 - cs) * pc

            # Update covariance
            rank_one = np.outer(ys, ys)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                zi = (offspring[i][1] - old_mean) / sigma
                rank_mu += weights[i] * np.outer(zi, zi)

            C = ((1 - c1 - cmu) * C
                 + c1 * (np.outer(pc, pc) + (1 - hsig) * C)
                 + cmu * rank_mu)

            C = (C + C.T) / 2.0

            try:
                C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(n))
            except np.linalg.LinAlgError:
                C = np.eye(n)
                C_sqrt = np.eye(n)

            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))

            if sigma < 1e-14:
                break
            if no_improve > 30 and sigma < 0.05:
                break

# EVOLVE-BLOCK-END

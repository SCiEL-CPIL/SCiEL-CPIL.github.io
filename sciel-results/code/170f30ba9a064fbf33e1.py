"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES for BBOB f21 (Gallagher 101 peaks)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Stratified initialization to find diverse basins
        # Use a grid of points to cover the search space
        n_init = min(50, budget // 10)
        
        # Create stratified points
        init_points = []
        # Add center
        init_points.append(np.zeros(dim))
        
        # Add random points
        for _ in range(n_init - 1):
            init_points.append(rng.uniform(-4, 4, dim))
        
        init_points = np.array(init_points)
        
        for x in init_points:
            if calls_made >= budget:
                break
            if time.time() > deadline - 5:
                break
            evaluate(x)

        # Phase 2: Multi-restart CMA-ES from the best few initial points
        # Sort initial points by fitness
        init_fits = []
        for x in init_points:
            if calls_made >= budget:
                break
            if time.time() > deadline - 5:
                break
            # We already evaluated these, so just record
            # Actually we need to track which ones we evaluated
            pass
        
        # Let's re-evaluate to get fitnesses (but we already have best_f)
        # Better approach: keep track of all evaluated points
        evaluated_points = []
        
        # Re-do initialization with tracking
        best_x = None
        best_f = np.inf
        calls_made = 0
        evaluated_points = []

        def evaluate_tracked(x):
            nonlocal calls_made, best_x, best_f, evaluated_points
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            evaluated_points.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Stratified initialization
        n_init = min(50, budget // 10)
        init_points = [np.zeros(dim)]
        for _ in range(n_init - 1):
            init_points.append(rng.uniform(-4, 4, dim))

        for x in init_points:
            if calls_made >= budget:
                break
            if time.time() > deadline - 5:
                break
            evaluate_tracked(x)

        # Sort by fitness and pick top k for CMA-ES restarts
        evaluated_points.sort(key=lambda p: p[0])
        
        # Pick up to 5 best distinct starting points
        start_points = []
        for f, x in evaluated_points:
            if len(start_points) >= 5:
                break
            # Check if this point is sufficiently different from existing start points
            is_distinct = True
            for sx in start_points:
                if np.linalg.norm(x - sx) < 0.5:
                    is_distinct = False
                    break
            if is_distinct:
                start_points.append(x)

        if not start_points:
            start_points = [evaluated_points[0][1]]

        # Phase 2: CMA-ES from each start point
        n_restarts = len(start_points)
        budget_per_restart = max(10, (budget - calls_made) // n_restarts)

        for idx, x0 in enumerate(start_points):
            if calls_made >= budget:
                break
            if time.time() > deadline - 10:
                break

            # CMA-ES parameters
            lambda_ = max(4, int(4 + 3 * np.log(dim)))
            sigma = 1.0  # initial step size
            x_mean = x0.copy()
            f_best_local = best_f  # start from global best knowledge
            
            # CMA-ES state
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Step sizes for adaptation
            c_sigma = (lambda_ + 2) / (dim + lambda_ + 5)
            mu_sigma = (lambda_ + 2) / (dim + lambda_ + 5)
            c_c = 4 / (dim + 4)
            mu_c = (lambda_ + 2) / (dim + 4)
            c_1 = 2 / ((dim + 1.3) ** 2 + lambda_)
            c_mu = min(1 - c_1, 2 * ((lambda_ / 2 - 1 + 2 / dim)) / ((dim + 2) ** 2 + lambda_))
            
            # Inverse square root of C for sampling
            def invsqrtm(C):
                # Use eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            C_inv_sqrt = invsqrtm(C)

            for gen in range(budget_per_restart):
                if calls_made >= budget:
                    break
                if time.time() > deadline - 5:
                    break

                # Generate offspring
                offspring = []
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * C_inv_sqrt @ z
                    y = np.clip(y, -5, 5)
                    offspring.append(y)
                
                # Evaluate offspring
                offs_f = []
                for y in offspring:
                    if calls_made >= budget:
                        break
                    if time.time() > deadline - 5:
                        break
                    offs_f.append(evaluate_tracked(y))

                if len(offs_f) < lambda_:
                    break

                # Sort offspring by fitness
                order = np.argsort(offs_f)
                mu = max(1, lambda_ // 2)
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mu_eff = 1.0 / np.sum(weights ** 2)

                # Update mean
                x_new_mean = np.zeros(dim)
                for i in range(mu):
                    x_new_mean += weights[i] * offspring[order[i]]
                
                # Update evolution paths
                old_x_mean = x_mean.copy()
                x_mean = x_new_mean

                # Success rate for sigma adaptation
                # Count how many offspring improved over the current best
                improved = sum(1 for i in range(mu) if offs_f[order[i]] < f_best_local)
                
                # Adapt sigma using 1/5 rule approximation
                if improved > lambda_ * 0.2:
                    sigma *= 1.2
                elif improved < lambda_ * 0.1:
                    sigma *= 0.8
                sigma = np.clip(sigma, 0.01, 5.0)

                # Update covariance matrix (simplified)
                # Recompute C_inv_sqrt periodically
                if gen % 5 == 0:
                    # Update C based on successful steps
                    v = (x_mean - old_x_mean) / sigma if sigma > 1e-10 else np.zeros(dim)
                    C = (1 - c_1) * C + c_1 * (np.outer(v, v) + (1 - chi_n * c_sigma * (1 - c_sigma)) * np.eye(dim))
                    # Ensure positive semi-definite
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    C_inv_sqrt = invsqrtm(C)

                # Update local best
                local_best_f = min(offs_f)
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = offspring[order[0]].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END

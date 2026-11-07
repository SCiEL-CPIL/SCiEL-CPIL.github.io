"""Rewritten optimizer: population-based CMA-ES-like with restarts and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based evolutionary optimizer with restarts and adaptive sigma."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # 300s limit, keep 20s margin

        # Global best tracking
        global_best_x = None
        global_best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal call_count
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            call_count += 1
            return x, f

        # CMA-ES-like parameters
        lambda_ = max(4, dim + 3)  # population size
        mu = lambda_ // 2          # number of parents

        def run_cmaes(x0, f0, sigma, remaining_budget):
            nonlocal global_best_x, global_best_f, call_count
            nonlocal deadline

            # CMA-ES state
            mean = x0.copy()
            best_local_x = x0.copy()
            best_local_f = f0

            # Covariance matrix C and its inverse square root
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            step_size = 1.0

            # Adaptation parameters
            c_sigma = (mu + 2) / (dim + mu + 5)
            d_sigma = 1 + 2 * max(0, (mu - 1) / (dim + mu - 3)) + c_sigma
            c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu)
            c_mu = min(1 - c_1, 2 * (mu - 2 + 1. / mu) / ((dim + 2) ** 2 + mu))

            pc = np.zeros(dim)
            ps = np.zeros(dim)

            # Initial population
            rng = np.random.default_rng(12345)

            def sample_population():
                """Generate lambda samples from N(mean, sigma^2 * C)."""
                # Use Cholesky-like decomposition via eigenvalue decomposition for simplicity
                # For efficiency, we use a simple approach: multiply by sqrt(C)
                # Actually, let's use the standard CMA-ES update with eigendecomposition
                # But for speed, we'll use a simplified version
                
                # Generate standard normals
                Z = rng.standard_normal((lambda_, dim))
                # Transform using C_inv_sqrt transpose (which is sqrt(C))
                # In CMA-ES, we have C = B * D * B^T, and we maintain B and D
                # For simplicity, let's use a different approach: 
                # We'll maintain the covariance matrix and use np.linalg.cholesky when needed
                # But that's expensive. Instead, let's use a simpler adaptation.
                
                # Simplified: use diagonal covariance approximation
                # Actually, let's just do a proper but simple CMA-ES
                # For dim=5, we can afford eigendecomposition each iteration
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    sqrt_eig = np.sqrt(eigvals)
                    # Transform: Z @ (eigvecs * sqrt_eig)
                    samples = Z @ (eigvecs * sqrt_eig).T + mean
                    return samples
                except np.linalg.LinAlgError:
                    return mean[None, :] + sigma * Z

            # Pre-compute weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            iterations = 0
            max_iter = remaining_budget // lambda_

            for it in range(max_iter):
                if call_count >= budget:
                    break
                if time.time() > deadline - 5:
                    break

                # Sample population
                pop = sample_population()
                # Clip to box
                pop = np.clip(pop, -5.0, 5.0)

                # Evaluate
                fvals = []
                for x in pop:
                    if call_count >= budget:
                        break
                    if time.time() > deadline - 5:
                        break
                    _, f = evaluate(x)
                    fvals.append(f)

                if len(fvals) < mu:
                    break

                fvals = np.array(fvals)
                # Sort by fitness
                idx = np.argsort(fvals)
                pop_sorted = pop[idx]
                fvals_sorted = fvals[idx]

                # Update best
                if fvals_sorted[0] < best_local_f:
                    best_local_f = fvals_sorted[0]
                    best_local_x = pop_sorted[0].copy()

                if best_local_f < global_best_f:
                    global_best_f = best_local_f
                    global_best_x = best_local_x.copy()

                # Update mean
                mean_old = mean.copy()
                mean = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)

                # Update path accumulators
                # ps = (1-c_sigma)*ps + sqrt(c_sigma*(2-c_sigma)*mu_eff) * C_inv_sqrt @ (mean - mean_old)/sigma
                try:
                    C_inv_sqrt @ (mean - mean_old) / sigma
                    ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (mean - mean_old) / sigma)
                except Exception:
                    ps = (1 - c_sigma) * ps

                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1)))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
                
                # Update pc
                if hsig:
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - mean_old) / sigma
                else:
                    pc = (1 - c_c) * pc

                # Update covariance
                delta_hsig = (1.0 if hsig else 0.0) - (1 - c_c)
                C = (1 + c_1 * delta_hsig) * C + c_1 * np.outer(pc, pc)
                for i in range(mu):
                    y_i = (pop_sorted[i] - mean_old) / sigma
                    C += c_mu * weights[i] * np.outer(y_i, y_i)

                # Ensure symmetry and positive semi-definite
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_inv_sqrt = np.eye(dim)

                # Adapt sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                iterations += 1

            return best_local_x, best_local_f

        # Phase 1: Initial exploration with several restarts
        n_restarts = 3
        budget_per_restart = budget // (n_restarts + 1)

        for r in range(n_restarts):
            if call_count >= budget:
                break
            if time.time() > deadline - 10:
                break

            # Random starting point
            rng = np.random.default_rng(42 + r)
            x0 = rng.uniform(-4.5, 4.5, dim)
            f0 = func(x0)
            call_count += 1

            if f0 < global_best_f:
                global_best_f = f0
                global_best_x = x0.copy()

            # Run CMA-ES from this point
            remaining = budget - call_count
            if remaining <= 0:
                break
            _, _ = run_cmaes(x0, f0, 1.0, remaining)

        # Phase 2: Final exploitation around global best
        if global_best_x is not None and call_count < budget:
            if time.time() > deadline - 10:
                return global_best_x, global_best_f

            # Local refinement with smaller sigma
            remaining = budget - call_count
            if remaining > 10:
                _, _ = run_cmaes(global_best_x, global_best_f, 0.1, remaining)

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END

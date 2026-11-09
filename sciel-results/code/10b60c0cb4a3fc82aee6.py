"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free evolutionary strategy with large population and anisotropic initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget

        # Initialize with a diverse set of points
        # Use a mix of random points and points near the center
        n_init = min(10, budget)
        X = []
        F = []

        # Start with a point near center
        x0 = np.zeros(dim)
        f0 = func(x0)
        X.append(x0)
        F.append(f0)

        # Add random points
        for _ in range(n_init - 1):
            x = rng.uniform(-4, 4, dim)
            fx = func(x)
            X.append(x)
            F.append(fx)

        X = np.array(X)
        F = np.array(F)

        best_idx = np.argmin(F)
        best_x = X[best_idx].copy()
        best_f = F[best_idx]

        calls_made = len(X)

        # Population size
        lam = min(50, budget - calls_made)
        if lam < 2:
            lam = max(2, budget - calls_made)

        # Initial sigma - start moderately large
        sigma = 2.0

        # Anisotropic covariance: start with identity, adapt based on progress
        # For ellipsoidal functions, we want to explore along axes
        C = np.eye(dim)
        # Precompute Cholesky
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(dim)

        # Track best values for stagnation detection
        best_f_history = [best_f]
        stagnant_steps = 0
        stagnation_threshold = 20  # steps without improvement

        for gen in range(budget - calls_made):
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = []
            offspring_f = []

            for i in range(lam):
                # Sample from multivariate normal with current covariance
                z = rng.multivariate_normal(np.zeros(dim), C, check_valid='ignore')
                y = best_x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                offspring.append(y)
                offspring_f.append(fy)

            calls_made += lam

            offspring = np.array(offspring)
            offspring_f = np.array(offspring_f)

            # Update best
            gen_best_idx = np.argmin(offspring_f)
            gen_best_f = offspring_f[gen_best_idx]

            if gen_best_f < best_f:
                best_f = gen_best_f
                best_x = offspring[gen_best_idx].copy()
                stagnant_steps = 0
                # Slightly increase sigma on improvement
                sigma *= 1.1
            else:
                stagnant_steps += 1
                # Decrease sigma when no improvement
                sigma *= 0.95

            # Update best_f_history
            best_f_history.append(best_f)

            # Adapt covariance based on recent successful steps
            # Simple adaptation: use the direction of improvement if available
            if stagnant_steps == 0:
                # Keep covariance mostly as is, slight regularization
                C = 0.9 * C + 0.1 * np.eye(dim)
            else:
                # If stagnant, randomize covariance to escape local optima
                if stagnant_steps % stagnation_threshold == 0:
                    # Reset to a more isotropic exploration
                    C = np.eye(dim) * (1 + rng.random())
                    sigma = max(sigma * 1.5, 0.5)

            # Ensure C is positive semi-definite
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)

            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 0.01, 5.0)

            # Check if we've used all budget
            if calls_made >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END

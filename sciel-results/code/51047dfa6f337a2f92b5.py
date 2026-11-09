"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """Hybrid Halton + CMA-ES + Nelder-Mead optimizer for BBOB f4 (Büche-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton(self, index, base):
        """Generate Halton sequence value."""
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def _halton_point(self, index, dim):
        """Generate a Halton sequence point in [0, 1]^dim."""
        point = np.zeros(dim)
        for i in range(dim):
            # Use primes for different dimensions
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            point[i] = self._halton(index + 1, primes[i % len(primes)])
        return point

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin
        rng = np.random.default_rng(42)
        dim = self.dim

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= self.budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Halton-sequence initialization for better global coverage
        # Use 40 Halton points in [-4, 4]^5
        num_initial = 40
        initial_points = []
        for i in range(num_initial):
            if calls >= self.budget or time.time() > deadline:
                break
            # Map Halton point from [0,1] to [-4, 4]
            h = self._halton_point(i, dim)
            x = h * 8.0 - 4.0
            fx = evaluate(x)
            if fx is None:
                break
            initial_points.append((fx, x.copy()))

        # Add a few random points for diversity
        for i in range(10):
            if calls >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-4.0, 4.0, size=dim)
            fx = evaluate(x)
            if fx is None:
                break
            initial_points.append((fx, x.copy()))

        # Sort by fitness and take top K basins
        initial_points.sort(key=lambda item: item[0])
        top_k = min(6, len(initial_points))

        # Phase 2: CMA-ES from top basins
        pop_size = 10
        sigma_min = 0.005
        sigma_max = 5.0
        max_local_steps = 35

        for idx in range(top_k):
            if calls >= self.budget or time.time() > deadline:
                break

            f_start, x_start = initial_points[idx]
            x_mean = x_start.copy()
            sigma = 0.5

            # CMA-ES state
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Adaptation parameters
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (weights[0] - 0.5) / ((dim + 2) ** 2 + mu_eff))

            stall_count = 0
            best_in_cma = f_start
            for gen in range(max_local_steps):
                if calls >= self.budget or time.time() > deadline:
                    break

                # Generate offspring
                offspring = []
                for i in range(pop_size):
                    if calls >= self.budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5.0, 5.0)
                    fy = evaluate(y)
                    if fy is None:
                        break
                    offspring.append((fy, y))

                if len(offspring) < pop_size:
                    break

                # Sort offspring by fitness
                offspring.sort(key=lambda item: item[0])

                # Select best mu individuals
                selected = offspring[:mu]

                # Track best in this CMA-ES run
                if selected[0][0] < best_in_cma:
                    best_in_cma = selected[0][0]

                # Check for stall
                if best_in_cma >= f_start * 0.999 and gen > 5:
                    stall_count += 1
                    if stall_count >= 8:
                        break
                else:
                    stall_count = 0

                # Update mean
                new_mean = sum(weights[i] * selected[i][1] for i in range(mu))

                # Update covariance matrix
                y_vec = (new_mean - x_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) /
                          np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1 + 3 / dim) * np.sqrt(1 - (1 - c_sigma) ** 2)

                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec

                delta_h = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (gen + 1)) /
                          np.sqrt(1 - (1 - c_c) ** (2 * (gen + 1)))) < (1 + 2 / dim) * np.sqrt(1 - (1 - c_c) ** 2)

                C = ((1 + c_1 + c_mu - c_1) * C +
                    c_1 * (np.outer(p_c, p_c) - C) +
                    c_mu * sum(weights[i] * np.outer(selected[i][1] - x_mean, selected[i][1] - x_mean) for i in range(mu)) / sigma ** 2)

                # Ensure positive semi-definiteness
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, sigma_min, sigma_max)

                # Update x_mean
                x_mean = new_mean

                # Update C_sqrt via Cholesky decomposition (approximate)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))

        # Phase 3: Nelder-Mead local polish for fast convergence
        if best_x is not None and calls < self.budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f

            # Build initial simplex
            simplex = [x.copy()]
            for i in range(dim):
                point = x.copy()
                # Adaptive step size based on current best
                step = max(0.1, 0.05 * (1.0 + abs(fx)))
                point[i] += step
                point = np.clip(point, -5.0, 5.0)
                simplex.append(point)

            # Evaluate initial simplex
            vals = []
            for i, sp in enumerate(simplex):
                if i == 0:
                    vals.append(fx)
                else:
                    if calls >= self.budget or time.time() > deadline:
                        break
                    vals.append(func(sp))
                    calls += 1
            else:
                # Nelder-Mead iterations
                for _ in range(80):
                    if calls >= self.budget or time.time() > deadline:
                        break

                    # Sort simplex by fitness
                    order = np.argsort(vals)
                    simplex = [simplex[i] for i in order]
                    vals = [vals[i] for i in order]

                    # Find worst point
                    worst_idx = dim
                    centroid = np.mean([simplex[j] for j in range(dim)], axis=0)

                    # Reflect
                    reflection = centroid + 1.0 * (centroid - simplex[worst_idx])
                    reflection = np.clip(reflection, -5.0, 5.0)
                    if calls < self.budget and time.time() < deadline:
                        f_reflect = func(reflection)
                        calls += 1
                        if f_reflect < vals[0]:
                            # Expansion
                            expansion = centroid + 2.0 * (centroid - simplex[worst_idx])
                            expansion = np.clip(expansion, -5.0, 5.0)
                            if calls < self.budget and time.time() < deadline:
                                f_expansion = func(expansion)
                                calls += 1
                                if f_expansion < f_reflect:
                                    simplex[worst_idx] = expansion
                                    vals[worst_idx] = f_expansion
                                    if f_expansion < best_f:
                                        best_f = f_expansion
                                        best_x = expansion.copy()
                                else:
                                    simplex[worst_idx] = reflection
                                    vals[worst_idx] = f_reflect
                                    if f_reflect < best_f:
                                        best_f = f_reflect
                                        best_x = reflection.copy()
                            else:
                                simplex[worst_idx] = reflection
                                vals[worst_idx] = f_reflect
                                if f_reflect < best_f:
                                    best_f = f_reflect
                                    best_x = reflection.copy()
                        elif f_reflect < vals[worst_idx]:
                            simplex[worst_idx] = reflection
                            vals[worst_idx] = f_reflect
                            if f_reflect < best_f:
                                best_f = f_reflect
                                best_x = reflection.copy()
                        else:
                            # Contraction
                            contraction = centroid + 0.5 * (simplex[worst_idx] - centroid)
                            contraction = np.clip(contraction, -5.0, 5.0)
                            if calls < self.budget and time.time() < deadline:
                                f_contraction = func(contraction)
                                calls += 1
                                if f_contraction < vals[worst_idx]:
                                    simplex[worst_idx] = contraction
                                    vals[worst_idx] = f_contraction
                                    if f_contraction < best_f:
                                        best_f = f_contraction
                                        best_x = contraction.copy()
                                else:
                                    # Shrink
                                    for i in range(1, dim + 1):
                                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                        if calls < self.budget and time.time() < deadline:
                                            vals[i] = func(simplex[i])
                                            calls += 1
                                            if vals[i] < best_f:
                                                best_f = vals[i]
                                                best_x = simplex[i].copy()

        # Phase 4: Final (1+1)-ES polish
        if best_x is not None and calls < self.budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            for step in range(30):
                if calls >= self.budget or time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.1, 0.5)
                else:
                    sigma = max(sigma * 0.9, 0.01)

        return best_x, best_f
# EVOLVE-BLOCK-END

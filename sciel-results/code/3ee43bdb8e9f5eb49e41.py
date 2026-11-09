"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-fidelity CMA-ES with Halton initialization for f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s kill
        budget = self.budget
        dim = self.dim

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def elapsed():
            return time.time() - start_time

        def time_ok():
            return elapsed() < deadline - 5

        def remaining():
            return budget - calls_made

        def halton(index, base):
            """Generate Halton sequence value for given index and base."""
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_point(index, dim):
            """Generate a point in [0,1]^dim using Halton sequence."""
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            point = []
            for d in range(dim):
                point.append(halton(index + 1, primes[d]))
            return np.array(point)

        # Phase 1: Halton sequence for systematic space-filling search
        n_halton = min(60, budget // 8)
        for i in range(n_halton):
            if calls_made >= budget or not time_ok():
                break
            # Map from [0,1]^dim to [-5,5]^dim
            x = 10.0 * halton_point(i, dim) - 5.0
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        # Phase 2: Multi-fidelity CMA-ES
        # Start with larger sigma, then adapt based on progress
        x = best_x.copy()
        sigma = 1.0  # larger initial step for exploration
        lambda_ = 8  # population size
        mu = 4
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights ** 2)

        c1 = 1 / (dim + 3)
        cmu = 2 / ((dim + 2) ** 2 + 5)
        cc = (4 + dim / 4) / (dim + 4 + 2 / dim)
        cs = (dim + 1.7) / (dim + 9)
        damps = 1 + 2 * max(0, np.sqrt((dim - 1) / (dim + 3)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        step_size = sigma

        # Precompute Cholesky factor for sampling
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(dim)

        # Track progress for adaptive sigma
        prev_best = best_f
        no_improve_count = 0
        max_no_improve = 15

        for gen in range(3000):
            if calls_made >= budget or not time_ok():
                break

            # Generate lambda_ offspring
            offspring = []
            for i in range(lambda_):
                if calls_made >= budget or not time_ok():
                    break
                z = np.random.standard_normal(dim)
                y = L @ z
                xi = x + step_size * y
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls_made += 1
                offspring.append((fi, xi))

            if len(offspring) == 0:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            new_best_f = best_f
            for fi, xi in offspring:
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
                    new_best_f = fi

            # Check for improvement
            if new_best_f < prev_best - 1e-8:
                prev_best = new_best_f
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Take top mu
            mu_actual = min(mu, len(offspring))
            selected = [offspring[i][1] for i in range(mu_actual)]
            selected_f = [offspring[i][0] for i in range(mu_actual)]

            # Compute mean of selected
            new_x = np.zeros(dim)
            for i in range(mu_actual):
                new_x += weights[i] * selected[i]

            # Update step size (CSA)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi_n * (C @ (new_x - x))
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1 + 2 / (dim + 1)) * chi_n
            step_size *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            step_size = np.clip(step_size, 1e-10, 50)

            # Adaptive sigma: if no improvement for a while, increase sigma to escape local minima
            if no_improve_count >= max_no_improve:
                step_size = min(step_size * 2.0, 10.0)
                no_improve_count = 0
                # Also reset some state to encourage exploration
                pc = np.zeros(dim)
                ps = np.zeros(dim)

            # Update covariance
            v = (new_x - x) / step_size
            rank_one = c1 * (v @ v.T + (1 - hsig) * cc * (2 - cc) * pc @ pc.T)
            rank_mu = cmu * sum(weights[i] * np.outer(selected[i] - x, selected[i] - x) / step_size ** 2 for i in range(mu_actual))
            C = (1 - c1 - cmu) * C + rank_one + rank_mu

            # Update pc
            pc = (1 - cc) * pc + cc * hsig * v

            x = new_x

            # Ensure C is symmetric positive definite and update Cholesky
            C = (C + C.T) / 2
            try:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))

            # If converged, do a local restart
            if step_size < 1e-5:
                # Try several perturbations with different scales
                found_improvement = False
                for scale in [0.5, 1.0, 2.0, 3.0]:
                    if calls_made >= budget or not time_ok():
                        break
                    x_try = x + np.random.normal(0, scale, dim)
                    x_try = np.clip(x_try, -5, 5)
                    f_try = func(x_try)
                    calls_made += 1
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()
                        x = x_try.copy()
                        found_improvement = True
                        break
                if not found_improvement:
                    # Reset and continue with larger step
                    step_size = max(step_size * 5.0, 0.5)
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END

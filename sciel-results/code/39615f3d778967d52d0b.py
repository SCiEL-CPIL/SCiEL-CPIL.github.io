"""Optimizer for BBOB f3 (Rastrigin) with lattice-based global search + CMA-ES refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lattice-based global search followed by CMA-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        calls_made = 0
        dim = self.dim
        budget = self.budget

        def check_deadline():
            return time.time() < deadline

        def evaluate(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            calls_made += 1
            return func(x)

        # Phase 1: Lattice-based global search
        # Rastrigin has periodic structure with period 2 in each dimension.
        # Evaluate points at integer and half-integer offsets to find the global basin.
        # The shift is in [-4, 4]^5, so the minimizer is at some shifted location.
        # We'll use a coarse grid to find the best region, then refine.

        best_x = None
        best_f = np.inf

        # Generate candidate points: integers and half-integers in [-4.5, 4.5]
        # This covers the range where the shifted minimizer can be.
        # Actually, let's think more carefully. The minimizer is in [-4, 4]^5.
        # Rastrigin: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # The global minimum of the unshifted Rastrigin is at x=0 with f=0.
        # With shift s, rotation R, and offset, the minimizer is at some point in [-4,4]^5.
        # The periodic structure means that local minima occur near integer points (shifted).

        # Strategy: sample a grid of points at integer and half-integer coordinates
        # within [-4, 4]^5. This gives 9^5 = 59049 points, which is too many.
        # Instead, use a smarter approach: sample integer points first (9^5 is still too many).
        # Let's use a smaller set: sample points at multiples of 1.0 in [-4, 4] for each dim.
        # That's 9 values per dim, 9^5 = 59049. Still too many for 500 budget.

        # Better strategy: Use a hierarchical approach.
        # 1. Sample a coarse grid (e.g., 3 points per dim: -3, 0, 3) = 3^5 = 243 points
        # 2. Refine around the best point with a finer grid
        # 3. Then use CMA-ES for final refinement

        # Actually, with only 500 calls, we need to be very efficient.
        # Let's use a Sobol-like quasi-random sequence for the global phase,
        # then switch to CMA-ES.

        # Phase 1: Quasi-random global search with Sobol sequence
        # Use the first ~200 calls for global exploration
        n_global = min(200, budget // 2)

        # Generate Sobol-like points using a simple LHS or Halton sequence
        # For simplicity, use a stratified sampling approach
        def halton_sequence(n, dim):
            """Generate Halton sequence points in [0,1]^dim."""
            points = np.zeros((n, dim))
            for i in range(n):
                for d in range(dim):
                    base = d + 2  # bases: 2, 3, 4, 5, 6
                    f = 1.0
                    i_val = i + 1
                    result = 0.0
                    while i_val > 0:
                        f /= base
                        result += f * (i_val % base)
                        i_val //= base
                    points[i, d] = result
            return points

        # Generate Halton points and map to [-4.5, 4.5]^5
        halton_pts = halton_sequence(n_global, dim)
        # Map from [0,1] to [-4.5, 4.5]
        halton_pts = halton_pts * 9.0 - 4.5

        for i in range(n_global):
            if not check_deadline():
                break
            if calls_made >= budget:
                break
            x = halton_pts[i]
            fx = evaluate(x)
            if fx is not None:
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 2: Local refinement around the best point with a fine grid
        # Use the remaining budget for CMA-ES
        if best_x is None:
            # Fallback: random start
            best_x = np.zeros(dim)
            fx = evaluate(best_x)
            if fx is not None:
                best_f = fx

        # Phase 3: CMA-ES for local refinement
        if calls_made < budget and check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.5  # start with moderate step size
            c1 = 1 / (dim + 4)
            cmu = 2 / ((dim + 2) ** 2 + 5)
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            cs = (1 + 2 / (dim + 1)) / (dim + 1 + 2)
            ds = 1 + 2 * max(0, np.sqrt((dim + 1 / dim - 2) * (cs - 1 / (dim + 1))) + cs)
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Population size
            lambda_ = max(4, int(4 + 3 * np.log(dim)))
            mu = lambda_ // 2

            while calls_made < budget and check_deadline():
                # Generate offspring
                offspring = []
                for j in range(lambda_):
                    if not check_deadline() or calls_made >= budget:
                        break
                    z = np.random.default_rng(hash((j, calls_made)) % (2**32)).standard_normal(dim)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = evaluate(y)
                    if fy is not None:
                        offspring.append((fy, y))

                if len(offspring) < mu:
                    break

                # Sort by fitness
                offspring.sort(key=lambda t: t[0])
                best_offspring_f, best_offspring_x = offspring[0]

                # Update mean
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w /= w.sum()
                x_new = sum(w[i] * offspring[i][1] for i in range(mu))

                # Update covariance matrix
                d = x_new - x
                C += c1 * (np.outer(d, d) + (1 - (1 - mu_eff) * c1 * c1 / (c1 * cmu)) * C) + \
                     cmu * sum(w[i] * np.outer(offspring[i][1] - x, offspring[i][1] - x) for i in range(mu)) - C
                mu_eff = 1 / sum(w ** 2)

                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (B @ (D ** -1) @ (d / sigma))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made - n_global)))) < \
                       (1.4 + 2 / (dim + 1)) * chiN
                pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * d / sigma

                # Update B and D
                C += 1e-12 * np.eye(dim)  # regularization
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                B = eigvecs

                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                x = x_new
                if best_offspring_f < fx:
                    fx = best_offspring_f
                    best_x = best_offspring_x.copy()

        # Ensure we've made at least some calls
        return best_x, best_f
# EVOLVE-BLOCK-END

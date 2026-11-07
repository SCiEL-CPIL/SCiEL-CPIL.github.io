"""Multi-start CMA-ES style optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            try:
                val = func(np.asarray(x, dtype=float))
                if not np.isfinite(val):
                    return np.inf
                return float(val)
            except Exception:
                return np.inf

        # Phase 1: CMA-ES style population-based search
        # Use a small population for efficiency
        pop_size = max(2 * dim + 2, 8)
        # But we only have budget=500 total calls, so pop_size must be small
        # Let's use pop_size = 4 for dim=5, giving ~125 generations
        pop_size = min(4, max(2, dim))  # For dim=5, pop_size=4

        # Initial population: spread out
        rng = np.random.default_rng(42)
        
        # Start with a reasonable initial point
        x_mean = np.zeros(dim)
        sigma = 1.5
        lambda_ = pop_size

        # CMA-ES parameters
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1) - 1)) + c_sigma
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

        # State
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Initial population
        pops = []
        for i in range(lambda_):
            z = rng.standard_normal(dim)
            x_i = x_mean + sigma * (B @ (D * z))
            x_i = np.clip(x_i, -5, 5)
            pops.append(x_i)
        fvals = [safe_call(x) for x in pops]

        best_x = pops[np.argmin(fvals)]
        best_f = min(fvals)

        calls_made = lambda_
        gen = 0

        while calls_made < budget:
            if time.time() > deadline - 5:
                break

            # Sort by fitness
            order = np.argsort(fvals)
            pops = [pops[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pops[0].copy()

            # Compute weighted mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += weights[i] * pops[i]

            # Update step size control
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / (D @ B).T @ (x_new_mean - x_mean)
            norm_ps = np.linalg.norm(ps)

            # Update covariance matrix
            diff = (x_new_mean - x_mean) / sigma
            cmat_update = np.outer(diff, diff)
            for i in range(mu):
                z_i = np.linalg.solve(D @ B, (pops[i] - x_mean) / sigma)
                cmat_update += weights[i] * np.outer(z_i, z_i)
            
            h_sigma = (norm_ps / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi_N
            cmat = (1 - (c_1 + c_mu)) * cmat + c_1 * (np.outer(pc, pc) + (1 - h_sigma) * c_sigma * (2 - c_sigma) * np.eye(dim)) + c_mu * cmat_update

            # Update evolution path
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff

            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(cmat)
            eigvals = np.maximum(eigvals, 1e-12)
            D = np.sqrt(eigvals)
            B = eigvecs
            cmat = (B * D) @ B.T

            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_N - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            x_mean = x_new_mean

            # Generate new population
            new_pops = []
            new_fvals = []
            for i in range(lambda_):
                if calls_made >= budget:
                    break
                if time.time() > deadline - 3:
                    break
                z = rng.standard_normal(dim)
                x_i = x_mean + sigma * (B @ (D * z))
                x_i = np.clip(x_i, -5, 5)
                f_i = safe_call(x_i)
                new_pops.append(x_i)
                new_fvals.append(f_i)
                calls_made += 1
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()

            if len(new_pops) == 0:
                break

            pops = new_pops
            fvals = new_fvals
            gen += 1

        # Phase 2: Local refinement with Nelder-Mead on the best point
        remaining_calls = budget - calls_made
        if remaining_calls > 10 and time.time() < deadline - 10:
            best_x, best_f = self._nelder_mead(func, best_x, best_f, remaining_calls, deadline, safe_call)

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, max_calls, deadline, safe_call):
        """Nelder-Mead refinement."""
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = f0
        calls = 1

        # Initialize simplex
        simplex = [x.copy()]
        fvals = [fx]
        for i in range(dim):
            xi = x.copy()
            xi[i] += 0.1 if xi[i] < 5 else -0.1
            xi = np.clip(xi, -5, 5)
            fi = safe_call(xi)
            simplex.append(xi)
            fvals.append(fi)
            calls += 1
            if fi < fx:
                fx = fi
                x = xi.copy()

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5

        while calls < max_calls:
            if time.time() > deadline - 3:
                break

            # Sort
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            if fvals[0] < fx:
                fx = fvals[0]
                x = simplex[0].copy()

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]

            # Reflection
            xr = centroid + alpha * (centroid - worst)
            xr = np.clip(xr, -5, 5)
            fr = safe_call(xr)
            calls += 1
            if fr < fx:
                fx = fr
                x = xr.copy()

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - worst)
                xe = np.clip(xe, -5, 5)
                fe = safe_call(xe)
                calls += 1
                if fe < fx:
                    fx = fe
                    x = xe.copy()
                simplex[-1] = xe if fe < fr else xr
                fvals[-1] = fe if fe < fr else fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + 0.5 * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - 0.5 * (centroid - worst)
                xc = np.clip(xc, -5, 5)
                fc = safe_call(xc)
                calls += 1
                if fc < fx:
                    fx = fc
                    x = xc.copy()
                if fc < (fr if fr < fvals[-1] else fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = safe_call(simplex[i])
                        calls += 1
                        if fvals[i] < fx:
                            fx = fvals[i]
                            x = simplex[i].copy()

        return x, fx
# EVOLVE-BLOCK-END

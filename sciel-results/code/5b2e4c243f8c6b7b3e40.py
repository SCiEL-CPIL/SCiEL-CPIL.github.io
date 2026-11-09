"""Two-stage Lattice-Pruned CMA-ES for BBOB f3 (Rastrigin)."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        calls_made = 0
        budget = self.budget
        dim = self.dim

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = float(func(x))
            calls_made += 1
            return val

        # Track best
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # ============================================================
        # STAGE 1: Coarse lattice search to find the global basin
        # Rastrigin has period 2, so we sample on a grid with step ~1.0
        # over [-5, 5] to identify the lowest-cost region.
        # We use a structured but not exhaustive grid to save budget.
        # ============================================================

        # Generate a coarse grid: 11 points per dimension would be 161051,
        # too many. Instead, use a stratified sampling approach.
        # Key insight: Rastrigin minima are at even integers (after shift/rotation).
        # We'll sample a set of points that covers the space well.

        rng = np.random.default_rng(42)

        # First, sample the center and corners plus structured points
        # Use a 5^5 = 3125 point grid with step 2.5 (too coarse)
        # Better: use a hybrid - 3^5 = 243 points on a step-5 grid,
        # plus random samples to fill gaps.

        # Actually, let's be smarter. The box is [-5,5]^5.
        # Rastrigin: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # After shift s and rotation R: f(x) = 10*5 + sum((Rx+s)_i^2 - 10*cos(2*pi*(Rx+s)_i))
        # Minima where each (Rx+s)_i is an even integer.

        # Strategy: sample ~100 points spread across the space to find
        # the basin with lowest function value, then refine locally.

        # Use Latin Hypercube Sampling for good coverage with few points
        n_lhs = 80
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            # Divide into n_lhs equal intervals in [-5, 5]
            edges = np.linspace(-5, 5, n_lhs + 1)
            for i in range(n_lhs):
                lhs_points[perm[i], d] = rng.uniform(edges[i], edges[i+1])

        # Also add the origin and some structured points
        structured = np.array([
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 4.0,
            np.ones(dim) * -4.0,
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
            np.ones(dim) * 3.0,
            np.ones(dim) * -3.0,
        ])

        # Evaluate all initial points
        all_initial = np.vstack([lhs_points, structured])
        for x in all_initial:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            f = safe_call(x)
            if f is not None:
                update_best(x, f)

        # ============================================================
        # STAGE 2: Local CMA-ES refinement around the best point
        # ============================================================

        if best_x is None:
            # Fallback: just return what we have
            return best_x, best_f

        # CMA-ES implementation
        # Start from best_x with a moderate sigma
        x_mean = best_x.copy()
        f_mean = best_f

        # Initial sigma: based on the grid spacing, try ~0.5
        sigma = 0.5
        pop_size = 4 + int(3 * np.log(dim))  # = 4 + 3*ln(5) ≈ 8
        pop_size = max(8, min(pop_size, 15))

        # CMA-ES parameters
        c_sigma = (pop_size + 2) / ((dim + 2) ** 2 + pop_size)
        d_sigma = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        c_mu = min(1 - c_1, 2 * ((pop_size / 2 + 1) / (dim + 2) - 1) / ((dim + 2) ** 2 + pop_size + 1))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        C_eigvals = np.ones(dim)
        C_eigvecs = np.eye(dim)
        # Precompute inverse square root of C
        C_eigvals_sqrt_inv = 1.0 / np.sqrt(C_eigvals)

        # Step size update
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def update_CMA(x_mean, sigma, p_sigma, p_c, C, C_eigvals, C_eigvecs,
                       c_sigma, c_c, c_1, c_mu, d_sigma, pop_size, dim,
                       func_eval, calls_made, budget, deadline):
            """One generation of CMA-ES. Returns updated state and calls_made."""
            # Generate population
            # Eigendecomposition is already maintained
            B = C_eigvecs
            D_inv_sqrt = 1.0 / np.sqrt(C_eigvals)

            ws = []
            x_w = []
            f_w = []

            for i in range(pop_size):
                z = rng.standard_normal(dim)
                x_i = x_mean + sigma * (B @ (D_inv_sqrt * z))
                x_i = np.clip(x_i, -5.0, 5.0)
                f_i = func_eval(x_i)
                if f_i is None:
                    # Out of budget or time
                    return x_mean, sigma, p_sigma, p_c, C, C_eigvals, C_eigvecs, calls_made, True
                ws.append(x_i)
                x_w.append(x_i)
                f_w.append(f_i)
                calls_made += 1

            if len(f_w) < pop_size:
                # Pad with best
                min_idx = np.argmin(f_w)
                while len(f_w) < pop_size:
                    f_w.append(f_w[min_idx])
                    x_w.append(x_w[min_idx])

            # Sort by fitness
            order = np.argsort(f_w)
            x_sorted = np.array([x_w[i] for i in order])
            f_sorted = [f_w[i] for i in order]

            # Weights
            w = np.zeros(pop_size)
            m_eff = 2.0 / (np.sum(np.exp(-2 * np.arange(pop_size))) + 1e-10)
            for i in range(pop_size):
                w[i] = np.log(m_eff * pop_size) - np.log(i + 1)
            w = np.maximum(w, 0)
            w /= np.sum(w)
            mu_eff = 1.0 / np.sum(w ** 2)

            # Update mean
            x_new_mean = np.sum(w[:, None] * x_sorted, axis=0)
            x_new_mean = np.clip(x_new_mean, -5.0, 5.0)

            # Update step size (CSA)
            y_sigma = (x_new_mean - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update C
            y_c = x_new_mean - x_mean
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c / sigma

            # Rank-one and rank-mu update
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(y_c / sigma, y_c / sigma) +
                        (c_c * (2 - c_c)) * (np.outer(p_c, p_c) - C if np.linalg.norm(p_c) / sigma > 1.7 + 2 / (dim + 1) else 0))
            C += c_mu * np.sum(w[:int(mu_eff) + 1][:, None] * np.outer(
                (x_sorted[:int(mu_eff) + 1] - x_new_mean) / sigma,
                (x_sorted[:int(mu_eff) + 1] - x_new_mean) / sigma), axis=0)

            # Ensure symmetry
            C = (C + C.T) / 2

            # Eigendecomposition (every 10 generations to save time)
            try:
                C_eigvals, C_eigvecs = np.linalg.eigh(C)
                C_eigvals = np.maximum(C_eigvals, 1e-20)
            except:
                pass

            return x_new_mean, sigma, p_sigma, p_c, C, C_eigvals, C_eigvecs, calls_made, False

        # Run CMA-ES for remaining budget
        max_generations = (budget - calls_made) // pop_size
        max_generations = max(1, min(max_generations, 200))

        for gen in range(max_generations):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

            x_mean, sigma, p_sigma, p_c, C, C_eigvals, C_eigvecs, calls_made, done = \
                update_CMA(x_mean, sigma, p_sigma, p_c, C, C_eigvals, C_eigvecs,
                           c_sigma, c_c, c_1, c_mu, d_sigma, pop_size, dim,
                           safe_call, calls_made, budget, deadline)
            if done:
                break

            # Update best
            if x_mean is not None:
                f_mean = safe_call(x_mean)
                if f_mean is not None:
                    update_best(x_mean, f_mean)

        # ============================================================
        # STAGE 3: Final local refinement with Nelder-Mead or Powell
        # ============================================================

        if best_x is not None and calls_made < budget and time.time() < deadline:
            try:
                from scipy.optimize import minimize

                # Use Nelder-Mead for final polish
                res = minimize(
                    lambda x: safe_call(x),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True},
                    tol=1e-10
                )
                if res.success or res.fun < best_f:
                    if safe_call(res.x) is not None:
                        update_best(res.x, res.fun)
            except:
                pass

        if best_x is not None and calls_made < budget and time.time() < deadline:
            try:
                from scipy.optimize import minimize

                # Powell method for additional refinement
                res = minimize(
                    lambda x: safe_call(x),
                    best_x,
                    method='Powell',
                    options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-12},
                    tol=1e-12
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except:
                pass

        return best_x, best_f

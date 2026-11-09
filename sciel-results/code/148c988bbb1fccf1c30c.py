import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        n = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: CMA-ES Global Search ---
        lambda_ = max(4 + int(3 * np.log(n)), 8)
        mu = lambda_ // 2

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)

        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))

        mean = np.zeros(n)
        sigma = 2.5
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Reserve ~40% of budget for multi-start Nelder-Mead polish
        cmaes_budget_limit = int(total_budget * 0.6)

        def get_C_sqrt(C_mat):
            try:
                L = np.linalg.cholesky(C_mat + 1e-12 * np.eye(n))
                return L
            except:
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.clip(eigvals, 1e-12, None)
                return eigvecs @ np.diag(np.sqrt(eigvals))

        # Track all evaluated points for potential Nelder-Mead starts
        evaluated_points = []

        while calls_made < cmaes_budget_limit:
            if time.time() > deadline:
                break

            z = np.random.standard_normal((lambda_, n))
            C_sqrt = get_C_sqrt(C)
            y = np.einsum('ij,jk->ik', z, C_sqrt.T)
            x_pop = mean + sigma * y

            f_pop = []
            for i in range(lambda_):
                if calls_made >= cmaes_budget_limit or time.time() > deadline:
                    break
                fx = evaluate(x_pop[i])
                f_pop.append(fx)
                evaluated_points.append(x_pop[i].copy())

            if len(f_pop) < mu:
                break

            f_pop = np.array(f_pop)
            idx = np.argsort(f_pop)

            x_w = np.zeros(n)
            for i in range(mu):
                x_w += weights[i] * x_pop[idx[i]]

            C_sqrt = get_C_sqrt(C)
            try:
                C_inv_sqrt = np.linalg.inv(C_sqrt)
            except:
                C_inv_sqrt = np.eye(n)

            y_w = (x_w - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mueff) * (C_inv_sqrt @ y_w)

            norm_p_sigma = np.linalg.norm(p_sigma)
            threshold = (1.4 + 2 / (n + 1)) * np.sqrt((mueff + 1) / (mueff + 3))
            h_sigma = 1.0 if norm_p_sigma < threshold else 0.0

            p_c = (1 - c_c) * p_c + h_sigma * c_c * (x_w - mean) / sigma

            C_update = np.zeros((n, n))
            for i in range(mu):
                y_i = (x_pop[idx[i]] - mean) / sigma
                C_update += weights[i] * np.outer(y_i, y_i)

            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (1 - c_c) * C) + c_mu * C_update
            C = 0.5 * (C + C.T)

            mean = x_w
            sigma *= np.exp((c_sigma / d_sigma) * (norm_p_sigma / np.sqrt(mueff) - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)

            if sigma < 1e-9:
                break

        # --- Phase 2: Multi-Start Nelder-Mead Local Polish ---
        # Collect diverse starting points: best from CMA-ES + some random points
        start_points = []

        # Add the best point found
        if best_x is not None:
            start_points.append(best_x.copy())

        # Add some of the best evaluated points from CMA-ES
        if evaluated_points:
            # Sort evaluated points by their function values (approximate)
            # We'll just take a few diverse ones
            for pt in evaluated_points[:10]:
                if pt is not None and len(start_points) < 5:
                    # Check if it's sufficiently different from existing starts
                    is_diverse = True
                    for sp in start_points:
                        if np.linalg.norm(pt - sp) < 0.5:
                            is_diverse = False
                            break
                    if is_diverse:
                        start_points.append(pt.copy())

        # Add a few random points in the box for diversity
        while len(start_points) < 8 and calls_made < total_budget:
            rand_pt = np.random.uniform(-4, 4, n)
            is_diverse = True
            for sp in start_points:
                if np.linalg.norm(rand_pt - sp) < 1.0:
                    is_diverse = False
                    break
            if is_diverse:
                start_points.append(rand_pt)

        # Nelder-Mead implementation
        def nelder_mead(start, budget_left, max_iter=100):
            nonlocal calls_made, best_x, best_f
            if calls_made >= total_budget or time.time() > deadline:
                return

            # Initialize simplex
            simplex = [start.copy()]
            f_simplex = [evaluate(start)]
            calls_made_local = 1

            # Generate initial simplex
            for i in range(n):
                x_new = start.copy()
                x_new[i] += 0.1 if start[i] < 4.9 else -0.1
                x_new = np.clip(x_new, -5, 5)
                if calls_made >= total_budget or time.time() > deadline:
                    return
                fx = evaluate(x_new)
                simplex.append(x_new.copy())
                f_simplex.append(fx)

            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma_nm = 0.5  # shrink

            for iteration in range(max_iter):
                if calls_made >= total_budget or time.time() > deadline:
                    return

                # Sort simplex by function value
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                if calls_made >= total_budget or time.time() > deadline:
                    return
                f_r = evaluate(x_r)

                if f_r < f_simplex[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    if calls_made >= total_budget or time.time() > deadline:
                        return
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_simplex[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    if calls_made >= total_budget or time.time() > deadline:
                        return
                    f_c = evaluate(x_c)

                    if f_c < f_simplex[-1]:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            if calls_made >= total_budget or time.time() > deadline:
                                return
                            f_simplex[i] = evaluate(simplex[i])

                # Check convergence
                f_range = max(f_simplex) - min(f_simplex)
                if f_range < 1e-10:
                    break

        # Run Nelder-Mead from each start point with remaining budget
        nm_budget_per_start = max(10, (total_budget - calls_made) // max(1, len(start_points)))

        for start_pt in start_points:
            if calls_made >= total_budget or time.time() > deadline:
                break
            nelder_mead(start_pt, nm_budget_per_start, max_iter=80)

        # Final fallback
        if best_x is None:
            x = np.zeros(n)
            best_f = func(x)
            best_x = x

        return best_x, best_f

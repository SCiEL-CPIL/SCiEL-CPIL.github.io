"""Combined optimizer: CMA-ES global search + paraboloid basin-jumping + local ES polish."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with quadratic basin-jumping and local ES for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        n = self.dim
        total_budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def fit_quadratic(center, points, values):
            """Fit a full quadratic model and return the predicted minimizer."""
            m = len(points)
            if m < n + 1:
                return None

            X = np.ones((m, 1 + n + n * (n + 1) // 2))
            for k, p in enumerate(points):
                d = p - center
                col = 1
                for i in range(n):
                    X[k, col] = d[i]
                    col += 1
                for i in range(n):
                    for j in range(i, n):
                        X[k, col] = d[i] * d[j]
                        col += 1
            y = np.array(values)

            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                return None

            g = coeffs[1:n + 1].copy()
            H = np.zeros((n, n))
            idx = n + 1
            for i in range(n):
                for j in range(i, n):
                    hval = coeffs[idx]
                    idx += 1
                    if i == j:
                        H[i, j] = 2.0 * hval
                    else:
                        H[i, j] = hval
                        H[j, i] = hval

            try:
                eigvals = np.linalg.eigvalsh(H)
                if np.any(eigvals <= 0):
                    return None
            except:
                return None

            try:
                dx = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(H, -g, rcond=None)[0]
                except:
                    return None

            model_center = coeffs[0]
            model_new = coeffs[0] + g @ dx + 0.5 * dx @ H @ dx
            if model_new >= model_center:
                return None

            new_x = center + dx
            new_x = np.clip(new_x, -5, 5)
            return new_x

        def do_es_around(x, fx, budget_limit, sigma_init=0.5):
            """Local (1+λ)-ES around a point."""
            nonlocal calls_made, best_x, best_f
            sigma = sigma_init
            lam = 4
            no_improve = 0
            max_no_improve = 15

            while calls_made < budget_limit and time.time() < deadline:
                if no_improve >= max_no_improve:
                    break

                offspring = []
                offspring_f = []
                for i in range(lam):
                    if calls_made >= budget_limit or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    y = np.clip(x + sigma * z, -5, 5)
                    fy = evaluate(y)
                    offspring.append(y)
                    offspring_f.append(fy)

                if not offspring:
                    break

                best_idx = np.argmin(offspring_f)
                best_y = offspring[best_idx]
                best_yf = offspring_f[best_idx]

                if best_yf < fx:
                    x = best_y
                    fx = best_yf
                    no_improve = 0
                    sigma *= 1.15
                else:
                    no_improve += 1
                    sigma *= 0.85

                sigma = np.clip(sigma, 1e-6, 3.0)

            return x, fx

        def paraboloid_jump(center, fx, h=0.8):
            """Use 2n+1 evaluations to fit a paraboloid and jump to predicted minimum."""
            nonlocal calls_made, best_x, best_f
            if calls_made >= total_budget or time.time() > deadline:
                return center, fx

            points = [center]
            values = [fx]

            for i in range(n):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                x_plus = np.clip(center + h * np.eye(n)[i], -5, 5)
                x_minus = np.clip(center - h * np.eye(n)[i], -5, 5)
                fp = evaluate(x_plus)
                fm = evaluate(x_minus)
                points.extend([x_plus, x_minus])
                values.extend([fp, fm])

            jump_x = fit_quadratic(center, points, values)
            if jump_x is not None:
                jump_f = evaluate(jump_x)
                if jump_f < fx:
                    fx = jump_f
                    center = jump_x.copy()
                    # Follow up with local ES
                    es_budget = min(total_budget, calls_made + 60)
                    center, fx = do_es_around(center, fx, es_budget, sigma_init=0.3)

            return center, fx

        # ============================================================
        # Phase 1: CMA-ES for global/intermediate exploration
        # ============================================================
        lambda_ = max(4 + int(3 * np.log(n)), 10)
        mu = lambda_ // 2

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

        mean = np.zeros(n)
        sigma = 2.5  # Large initial step to cover [-5,5]^5
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        cmaes_budget_limit = int(total_budget * 0.55)
        gen_count = 0

        while calls_made < cmaes_budget_limit and time.time() < deadline:
            if time.time() > deadline:
                break

            # Sample population
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
            except:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                L = eigvecs @ np.diag(np.sqrt(eigvals))

            z = rng.standard_normal((lambda_, n))
            y = z @ L
            x_pop = mean + sigma * y

            # Evaluate
            f_pop = np.empty(lambda_)
            for i in range(lambda_):
                if calls_made >= cmaes_budget_limit or time.time() > deadline:
                    break
                f_pop[i] = evaluate(x_pop[i])

            if calls_made >= cmaes_budget_limit or time.time() > deadline:
                break

            # Sort by fitness
            idx = np.argsort(f_pop)

            # Update mean
            x_w = np.zeros(n)
            for i in range(mu):
                x_w += weights[i] * x_pop[idx[i]]

            # Compute C_inv_sqrt for p_sigma update
            try:
                C_inv_sqrt = np.linalg.inv(L)
            except:
                C_inv_sqrt = np.eye(n)

            # Update p_sigma
            y_w = (x_w - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mueff) * (C_inv_sqrt @ y_w)

            # Update p_c
            h_sigma = 1.0
            if np.linalg.norm(p_sigma) > (1.4 + 2 / (n + 1)) * np.sqrt((mueff + 1) / (mueff + 3)):
                h_sigma = 0.0
            p_c = (1 - c_c) * p_c + h_sigma * c_c * (x_w - mean) / sigma

            # Update C
            C_update = np.zeros((n, n))
            for i in range(mu):
                y_i = (x_pop[idx[i]] - mean) / sigma
                C_update += weights[i] * np.outer(y_i, y_i)

            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (1 - c_c) * C) + c_mu * C_update

            mean = x_w

            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(mueff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            gen_count += 1

            # Periodic paraboloid jumps during CMA-ES to escape local basins
            if gen_count % 15 == 0 and calls_made < cmaes_budget_limit and time.time() < deadline:
                if best_x is not None:
                    best_x, best_f = paraboloid_jump(best_x.copy(), best_f, h=0.8)

            # Convergence check
            if sigma < 1e-8:
                break

        # ============================================================
        # Phase 2: Paraboloid basin-jumping on the best point
        # ============================================================
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            # Multiple paraboloid jumps with decreasing step sizes
            for h_val in [1.5, 0.8, 0.4]:
                if calls_made >= total_budget or time.time() > deadline:
                    break
                best_x, best_f = paraboloid_jump(best_x.copy(), best_f, h=h_val)

        # ============================================================
        # Phase 3: Final local ES polishing
        # ============================================================
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            remaining = total_budget - calls_made
            if remaining > 10:
                best_x, best_f = do_es_around(best_x, best_f, total_budget, sigma_init=0.15)

        # Fallback
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f

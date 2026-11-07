"""Black-box continuous optimizer for the rotated Rosenbrock (f9) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES + gradient descent hybrid for rotated Rosenbrock (f9)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 330s limit

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        def estimate_gradient(x, h=None):
            """Estimate gradient using central finite differences with adaptive h."""
            nonlocal calls
            if h is None:
                # Adaptive h based on current function value magnitude
                h = max(1e-5, min(1e-3, 0.01 * max(1.0, abs(best_f))))
            grad = np.zeros(n)
            for i in range(n):
                if calls >= budget or time.time() > deadline:
                    return None
                x_plus = x.copy()
                x_plus[i] += h
                x_minus = x.copy()
                x_minus[i] -= h
                f_plus = func(x_plus)
                calls += 1
                if calls >= budget or time.time() > deadline:
                    return None
                f_minus = func(x_minus)
                calls += 1
                grad[i] = (f_plus - f_minus) / (2 * h)
            return grad

        def gradient_descent(x0, f0, max_iter=5000, h_start=None):
            """Variable-step gradient descent with backtracking line search."""
            nonlocal calls, best_x, best_f

            x = x0.copy()
            f = f0
            step_size = 0.1
            max_step = 1.0
            min_step = 1e-8
            c1 = 1e-4
            rho = 0.5

            for it in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break

                h = h_start
                if h is None:
                    h = max(1e-5, min(1e-3, 0.01 * max(1.0, abs(f))))
                grad = estimate_gradient(x, h=h)
                if grad is None:
                    break

                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-10:
                    break

                direction = -grad / grad_norm

                alpha = step_size
                f_new = None
                for _ in range(20):
                    if time.time() > deadline or calls >= budget:
                        break
                    x_trial = x + alpha * direction
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls += 1

                    if f_trial < f - c1 * alpha * grad_norm:
                        f_new = f_trial
                        break
                    alpha *= rho

                if f_new is None:
                    step_size *= 0.5
                    if step_size < min_step:
                        break
                    continue

                x = x_trial
                f = f_new

                if f < best_f:
                    best_f = f
                    best_x = x.copy()

                if f < f0 - 0.01 * abs(f0):
                    step_size = min(step_size * 1.5, max_step)
                else:
                    step_size *= 0.8

                f0 = f

            return x, f

        def cma_es(x0, f0, max_gen=100, sigma_init=None):
            """CMA-ES with small population for exploration."""
            nonlocal calls, best_x, best_f

            lam = 8
            mu = 4
            w = np.log(np.arange(mu, 0, -1)) - np.log(np.arange(lam, mu, -1))
            w /= np.sum(w)
            mueff = 1.0 / np.sum(w ** 2)

            sigma = sigma_init if sigma_init is not None else 1.0
            ps = np.zeros(n)
            pc = np.zeros(n)
            C = np.eye(n)
            Csqrt = np.eye(n)
            Csqrt_inv = np.eye(n)

            cs = (mueff + 2) / (n + mueff + 5)
            ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            x_mean = x0.copy()
            f_best_local = f0
            x_best_local = x0.copy()

            # Initial population
            pop_x = [x0.copy()]
            pop_f = [f0]

            for i in range(1, lam):
                if calls >= budget or time.time() > deadline:
                    return x_best_local, f_best_local
                z = rng.standard_normal(n)
                xi = x_mean + sigma * (Csqrt @ z)
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                if fi < f_best_local:
                    f_best_local = fi
                    x_best_local = xi.copy()
                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()
                pop_x.append(xi)
                pop_f.append(fi)

            if calls >= budget:
                return x_best_local, f_best_local

            for gen in range(max_gen):
                if time.time() > deadline or calls >= budget:
                    break

                if sigma < 1e-8:
                    break

                order = np.argsort(pop_f)
                pop_x = [pop_x[i] for i in order]
                pop_f = [pop_f[i] for i in order]

                # Update mean
                m = np.zeros(n)
                for i in range(mu):
                    m += w[i] * pop_x[i]

                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / (sigma * chiN) * (Csqrt_inv @ (m - x_mean))

                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) > (1.4 + 2 / (n + 1)) * chiN:
                    sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1) / 2)
                else:
                    if np.linalg.norm(ps) / chiN < (1.4 + 2 / (n + 1)):
                        pc = (1 - cc) * pc + cc * (Csqrt_inv @ (m - x_mean) / sigma)
                    else:
                        pc = (1 - cc) * pc + (1 + cc) * (Csqrt_inv @ (m - x_mean) / sigma)

                    hsig = min(1, (np.linalg.norm(ps) / chiN) ** 2 / (1 - (1 - cs) ** (2 * (gen + 1))))
                    rank_one = np.outer(pc, pc)
                    rank_one[pc < 0] *= (1 + (2 / (n + 1) - 1) * hsig)
                    C = (1 + c1 * (hsig - 1)) * C + c1 * rank_one
                    rank_mu = np.zeros((n, n))
                    for i in range(mu):
                        z = Csqrt_inv @ (pop_x[i] - m) / sigma
                        rank_mu += w[i] * np.outer(z, z)
                    C += cmu * rank_mu

                    C = (C + C.T) / 2
                    C += 1e-12 * np.eye(n)

                    try:
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.maximum(eigvals, 1e-15)
                        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                        Csqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                        Csqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
                    except np.linalg.LinAlgError:
                        C = np.eye(n)
                        Csqrt = np.eye(n)
                        Csqrt_inv = np.eye(n)

                x_mean = m

                # Generate new population
                new_pop_x = []
                new_pop_f = []
                for i in range(lam):
                    if calls >= budget or time.time() > deadline:
                        return x_best_local, f_best_local
                    z = rng.standard_normal(n)
                    xi = x_mean + sigma * (Csqrt @ z)
                    xi = np.clip(xi, -5, 5)
                    fi = func(xi)
                    calls += 1
                    if fi < f_best_local:
                        f_best_local = fi
                        x_best_local = xi.copy()
                        if fi < best_f:
                            best_f = fi
                            best_x = xi.copy()
                    new_pop_x.append(xi)
                    new_pop_f.append(fi)

                pop_x = new_pop_x
                pop_f = new_pop_f

                # Convergence check
                if pop_f[-1] - pop_f[0] < 1e-10:
                    break

            return x_best_local, f_best_local

        def nelder_mead(x0, f0, max_iter=500):
            """Nelder-Mead simplex method for final polish."""
            nonlocal calls, best_x, best_f

            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [f0]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.1
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                simplex.append(xi)
                fvals.append(fi)

            for it in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break

                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                if fvals[-1] - fvals[0] < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls += 1

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget or time.time() > deadline:
                                return simplex[0], fvals[0]

                # Track best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()

            min_idx = np.argmin(fvals)
            return simplex[min_idx], fvals[min_idx]

        # Phase 1: Evaluate origin and a few random points
        f_origin = func(np.zeros(n))
        calls += 1
        best_x = np.zeros(n).copy()
        best_f = f_origin

        for i in range(5):
            if calls >= budget or time.time() > deadline:
                break
            x_try = rng.uniform(-4, 4, n)
            f_try = func(x_try)
            calls += 1
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: CMA-ES for exploration (use ~40% of remaining budget)
        remaining = budget - calls
        cma_budget = int(0.4 * remaining)
        max_gen = max(20, cma_budget // 8)
        x_cma, f_cma = cma_es(best_x.copy(), best_f, max_gen=max_gen)

        if f_cma < best_f:
            best_f = f_cma
            best_x = x_cma.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: Gradient descent for fast local convergence
        x_grad, f_grad = gradient_descent(best_x.copy(), best_f, max_iter=2000)

        if f_grad < best_f:
            best_f = f_grad
            best_x = x_grad.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3b: High-precision polish with tighter gradient tolerance
        if budget - calls > 20:
            x_polish, f_polish = gradient_descent(best_x.copy(), best_f, max_iter=500, h_start=1e-6)
            if f_polish < best_f:
                best_f = f_polish
                best_x = x_polish.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 4: Nelder-Mead for final polish
        x_nm, f_nm = nelder_mead(best_x.copy(), best_f, max_iter=200)

        if f_nm < best_f:
            best_f = f_nm
            best_x = x_nm.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 4b: Final high-precision gradient polish after Nelder-Mead
        if budget - calls > 10:
            x_final, f_final = gradient_descent(best_x.copy(), best_f, max_iter=200, h_start=1e-7)
            if f_final < best_f:
                best_f = f_final
                best_x = x_final.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 5: If budget remains, do another CMA-ES restart from best with smaller sigma
        if budget - calls > 50:
            x_restart = best_x + 0.3 * rng.standard_normal(n)
            x_restart = np.clip(x_restart, -5, 5)
            f_restart = func(x_restart)
            calls += 1
            if f_restart < best_f:
                best_f = f_restart
                best_x = x_restart.copy()

            x_cma2, f_cma2 = cma_es(x_restart, f_restart, max_gen=30, sigma_init=0.5)
            if f_cma2 < best_f:
                best_f = f_cma2
                best_x = x_cma2.copy()

            if calls < budget and time.time() < deadline:
                x_grad2, f_grad2 = gradient_descent(best_x.copy(), best_f, max_iter=500)
                if f_grad2 < best_f:
                    best_f = f_grad2
                    best_x = x_grad2.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END

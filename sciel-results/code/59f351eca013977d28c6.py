"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: random init, CMA-ES-like global search, Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Random initialization with a few samples to find a good starting point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Quick random sampling to find a better starting point
        for _ in range(min(20, budget - 1)):
            if time.time() > deadline:
                break
            y = rng.uniform(-5, 5, dim)
            fy = func(y)
            calls += 1
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()
                x = y.copy()
                fx = fy

        # Phase 2: CMA-ES-like evolution strategy for global search
        sigma = 1.0
        c1 = (2 + 2 / (dim + 1)) / ((dim + 2) ** 2 + 2)
        cmu = c1 * 2 / (dim * (dim + 1))
        cs = (dim + 1.7) / (dim + 9)
        c_sigma = 1 + 2 * max(0, np.sqrt((dim - 1) / (dim + 1)) - 1) + cs
        d_sigma = 1 + 2 * max(0, np.sqrt((dim - 1) / (dim + 1)) - 1) + cs
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        lambda_pop = max(4, int(4 + 3 * np.log(dim)))
        weights = np.log(lambda_pop + 0.5) - np.log(np.arange(1, lambda_pop + 1))
        weights /= weights.sum()
        mu = int(np.sum(weights > 0))
        nu = 1.0 / (c1 * dim * 2)

        pop_size = lambda_pop
        remaining = budget - calls
        global_budget = int(0.7 * remaining)
        
        for gen in range(global_budget // pop_size):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Generate population
            D, B = np.linalg.eigh(C)
            B_inv = B @ np.diag(1.0 / np.sqrt(np.maximum(D, 1e-20)))

            for i in range(pop_size):
                if time.time() > deadline:
                    break
                if calls >= budget:
                    break
                
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ (np.sqrt(np.maximum(D, 0)) @ z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                
                # Store for update
                if not hasattr(self, '_pop'):
                    self._pop = []
                self._pop.append((y, fy, z))

            if not hasattr(self, '_pop') or len(self._pop) < pop_size:
                break

            # Sort by fitness
            self._pop.sort(key=lambda t: t[1])
            selected = self._pop[:mu]

            # Update mean
            new_x = sum(weights[i] * selected[i][0] for i in range(mu))
            new_x = np.clip(new_x, -5, 5)

            # Update covariance
            diff = [selected[i][0] - x for i in range(mu)]
            delta_x = sum(weights[i] * diff[i] for i in range(mu))
            pc = (1 - c1) * pc + c1 * (delta_x / sigma)
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + \
                cmu * sum(weights[i] * np.outer(diff[i], diff[i]) for i in range(mu))

            # Update sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (delta_x / sigma)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            x = new_x
            if hasattr(self, '_pop'):
                del self._pop

        # Phase 3: Nelder-Mead polish for local refinement
        polish_budget = budget - calls
        if polish_budget > 0:
            x_curr = best_x.copy()
            f_curr = best_fx

            # Initialize simplex
            simplex = [x_curr.copy()]
            for i in range(dim):
                xi = x_curr.copy()
                step = 0.1 if abs(xi[i]) > 1e-8 else 0.1
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)

            f_simplex = [f_curr]
            for i in range(1, dim + 1):
                fi = func(simplex[i])
                calls += 1
                f_simplex.append(fi)
                if fi < best_fx:
                    best_fx = fi
                    best_x = simplex[i].copy()

            # Nelder-Mead iterations
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5

            while calls < budget and polish_budget > 0:
                if time.time() > deadline:
                    break
                polish_budget -= 1

                # Sort simplex
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]

                # Centroid excluding worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                fr = func(xr)
                calls += 1
                if fr < best_fx:
                    best_fx = fr
                    best_x = xr.copy()

                if f_simplex[0] <= fr < f_simplex[-2]:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                elif fr < f_simplex[0]:
                    # Expansion
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < best_fx:
                        best_fx = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        f_simplex[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_simplex[-1] = fr
                else:
                    # Contraction
                    if fr < f_simplex[-1]:
                        # Outside contraction
                        xc = np.clip(centroid + rho * (xr - centroid), -5, 5)
                    else:
                        # Inside contraction
                        xc = np.clip(centroid - rho * (centroid - simplex[-1]), -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < best_fx:
                        best_fx = fc
                        best_x = xc.copy()

                    if fc < min(fr, f_simplex[-1]):
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = np.clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]), -5, 5)
                            fi = func(simplex[i])
                            calls += 1
                            f_simplex[i] = fi
                            if fi < best_fx:
                                best_fx = fi
                                best_x = simplex[i].copy()

            # Final fine-tuning with smaller steps
            if calls < budget:
                for step_size in [0.01, 0.001, 0.0001]:
                    if calls >= budget or time.time() > deadline:
                        break
                    for _ in range(10):
                        if calls >= budget or time.time() > deadline:
                            break
                        for i in range(dim):
                            if calls >= budget or time.time() > deadline:
                                break
                            for direction in [1, -1]:
                                if calls >= budget or time.time() > deadline:
                                    break
                                y = best_x.copy()
                                y[i] += direction * step_size
                                y = np.clip(y, -5, 5)
                                fy = func(y)
                                calls += 1
                                if fy < best_fx:
                                    best_fx = fy
                                    best_x = y.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END

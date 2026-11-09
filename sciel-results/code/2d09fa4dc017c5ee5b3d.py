"""Adaptive Hyperparameter Annealing (AHA) with population-based search for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Adaptive population-based optimizer for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 295.0
        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0
        n = self.dim

        def check_time():
            return (time.time() - start_time) < time_limit

        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            calls_made += 1
            fx = func(x)
            if np.isnan(fx):
                fx = 1e6
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Initial exploration with stratified sampling
        n_explore = 50
        if calls_made + n_explore <= self.budget:
            n_lhs = 35
            n_rand = n_explore - n_lhs
            
            lhs_points = np.zeros((n_lhs, n))
            for i in range(n):
                perm = rng.permutation(n_lhs)
                u = (perm + 0.5) / n_lhs
                lhs_points[:, i] = -5 + 10 * u
            
            rand_points = rng.uniform(-5, 5, size=(n_rand, n))
            
            all_points = np.vstack([lhs_points, rand_points])
            
            for x in all_points:
                if not check_time() or calls_made >= self.budget:
                    break
                safe_func(x)

        # Phase 2: Adaptive population-based search
        pop_size = 10
        if calls_made < self.budget and best_f < np.inf:
            pop = []
            pop_f = []
            step_sizes = []
            
            base = best_x.copy()
            for i in range(pop_size):
                if i == 0:
                    x = base.copy()
                else:
                    radius = 2.0 / (i + 1)
                    x = base + rng.normal(0, radius, size=n)
                    x = np.clip(x, -5, 5)
                fx = safe_func(x)
                if not check_time() or calls_made >= self.budget:
                    break
                pop.append(x.copy())
                pop_f.append(fx)
                step_sizes.append(0.5)

            if len(pop) < 2:
                return best_x, best_f

            max_iter = 250
            for iteration in range(max_iter):
                if not check_time() or calls_made >= self.budget:
                    break

                order = np.argsort(pop_f)
                pop = [pop[i] for i in order]
                pop_f = [pop_f[i] for i in order]
                step_sizes = [step_sizes[i] for i in order]

                if pop_f[0] < best_f:
                    best_f = pop_f[0]
                    best_x = pop[0].copy()

                if len(pop) > 1:
                    spread = np.max(pop_f) - np.min(pop_f)
                    if spread < 1e-10:
                        break

                new_pop = [pop[0].copy()]
                new_pop_f = [pop_f[0]]
                new_step_sizes = [step_sizes[0]]

                anneal_factor = max(0.1, 1.0 - iteration / 200.0)

                for i in range(1, pop_size):
                    if not check_time() or calls_made >= self.budget:
                        break

                    tournament_size = 3
                    candidates = rng.choice(len(pop), size=min(tournament_size, len(pop)), replace=False)
                    parent_idx = candidates[np.argmin([pop_f[c] for c in candidates])]
                    parent = pop[parent_idx]
                    parent_step = step_sizes[parent_idx]

                    current_step = parent_step * anneal_factor

                    direction = rng.normal(0, 1, size=n)
                    direction /= (np.linalg.norm(direction) + 1e-10)
                    child = parent + current_step * direction
                    child = np.clip(child, -5, 5)

                    fx = safe_func(child)
                    if not check_time() or calls_made >= self.budget:
                        break

                    if fx < pop_f[parent_idx]:
                        new_step = min(2.0, current_step * 1.2)
                    else:
                        new_step = max(0.05, current_step * 0.7)

                    new_pop.append(child.copy())
                    new_pop_f.append(fx)
                    new_step_sizes.append(new_step)

                if len(new_pop) < 2:
                    break

                pop = new_pop
                pop_f = new_pop_f
                step_sizes = new_step_sizes

        # Phase 3: Final polish with Nelder-Mead
        if check_time() and calls_made < self.budget and best_f < np.inf:
            self._nelder_mead(func, best_x.copy(), best_f, max_steps=40, initial_delta=0.1,
                              check_time=check_time, safe_func=safe_func)

        return best_x, best_f

    def _nelder_mead(self, func, x_start, f_start, max_steps=50, initial_delta=0.5,
                     check_time=None, safe_func=None):
        dim = self.dim
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_shrink = 0.5

        if safe_func is None:
            def safe_func(x):
                fx = func(x)
                if np.isnan(fx):
                    fx = 1e6
                return fx

        if check_time is None:
            def check_time():
                return True

        simplex = [x_start.copy()]
        for i in range(dim):
            xi = x_start.copy()
            xi[i] += initial_delta
            xi = np.clip(xi, -5, 5)
            if np.allclose(xi, x_start):
                xi[i] = x_start[i] + initial_delta * 0.1
                xi = np.clip(xi, -5, 5)
            simplex.append(xi)

        simplex_f = [f_start]
        for i in range(1, dim + 1):
            if not check_time():
                return
            fx = safe_func(simplex[i])
            simplex_f.append(fx)

        for step in range(max_steps):
            if not check_time():
                break

            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)

            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fxr = safe_func(xr)
            if not check_time():
                break

            if fxr < simplex_f[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fxe = safe_func(xe)
                if not check_time():
                    break
                if fxe < fxr:
                    simplex[-1] = xe
                    simplex_f[-1] = fxe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
            elif fxr < simplex_f[-2]:
                simplex[-1] = xr
                simplex_f[-1] = fxr
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fxc = safe_func(xc)
                if not check_time():
                    break
                if fxc < simplex_f[-1]:
                    simplex[-1] = xc
                    simplex_f[-1] = fxc
                else:
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if not check_time():
                            return
                        simplex_f[i] = safe_func(simplex[i])

            if np.max(simplex_f) - np.min(simplex_f) < 1e-14:
                break

# EVOLVE-BLOCK-END

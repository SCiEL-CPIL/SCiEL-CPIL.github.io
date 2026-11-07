"""Rewritten optimizer using Particle Swarm Optimization (PSO) with velocity clamping and local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """PSO-based derivative-free optimizer with inertia decay and local NM polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s limit
        self._call_count = 0
        best_x = None
        best_f = np.inf

        def counted_func(x):
            nonlocal best_x, best_f
            if self._call_count >= self.budget:
                return np.inf
            self._call_count += 1
            val = func(np.clip(x, -5, 5))
            if val < best_f:
                best_f = val
                best_x = np.clip(x, -5, 5).copy()
            return val

        def nelder_mead_local(start_x, max_evals):
            """Quick local NM polish around start_x."""
            nonlocal best_x, best_f, deadline, self
            if self._call_count >= self.budget or time.time() > deadline:
                return
            x0 = np.clip(start_x, -5, 5)
            f0 = counted_func(x0)

            simplex = [x0.copy()]
            for i in range(self.dim):
                xi = x0.copy()
                xi[i] = x0[i] + 0.3 if x0[i] < 4.7 else x0[i] - 0.3
                simplex.append(xi)

            simplex_f = [f0]
            for i in range(1, self.dim + 1):
                simplex_f.append(counted_func(simplex[i]))

            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            iters = 0
            while iters < max_evals and self._call_count < self.budget and time.time() < deadline:
                iters += 1
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = counted_func(xr)

                if fr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = counted_func(xe)
                    simplex[-1] = xe if fe < fr else xr
                    simplex_f[-1] = fe if fe < fr else fr
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    fc = counted_func(xc)
                    if fc < fr:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex_f[i] = counted_func(simplex[i])

        # PSO parameters
        n_particles = 20
        w_max = 0.9
        w_min = 0.3
        c1 = 1.5  # cognitive
        c2 = 1.5  # social

        rng = np.random.default_rng(42)

        # Initialize particles
        positions = rng.uniform(-5, 5, (n_particles, self.dim))
        velocities = rng.uniform(-1, 1, (n_particles, self.dim))
        personal_bests = np.copy(positions)
        personal_best_f = np.full(n_particles, np.inf)

        # Evaluate initial positions
        for i in range(n_particles):
            if self._call_count >= self.budget or time.time() > deadline:
                break
            f = counted_func(positions[i])
            personal_best_f[i] = f
            personal_bests[i] = positions[i].copy()

        global_best = personal_bests[np.argmin(personal_best_f)].copy()
        global_best_f = np.min(personal_best_f)

        # PSO loop
        total_pso_budget = int(self.budget * 0.75)  # 75% for PSO, 25% for local polish
        pso_calls_used = 0
        max_pso_iters = total_pso_budget // n_particles

        for it in range(max_pso_iters):
            if self._call_count >= self.budget or time.time() > deadline:
                break
            pso_calls_used += 1

            # Linearly decay inertia weight
            progress = min(1.0, (it + 1) / max_pso_iters)
            w = w_max - (w_max - w_min) * progress

            for i in range(n_particles):
                if self._call_count >= self.budget or time.time() > deadline:
                    break

                r1 = rng.random(self.dim)
                r2 = rng.random(self.dim)

                velocities[i] = (w * velocities[i]
                                 + c1 * r1 * (personal_bests[i] - positions[i])
                                 + c2 * r2 * (global_best - positions[i]))

                # Clamp velocity to prevent divergence
                v_max = 2.0
                velocities[i] = np.clip(velocities[i], -v_max, v_max)

                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], -5, 5)

                f = counted_func(positions[i])
                if f < personal_best_f[i]:
                    personal_best_f[i] = f
                    personal_bests[i] = positions[i].copy()
                    if f < global_best_f:
                        global_best_f = f
                        global_best = positions[i].copy()

        # Local polish: run NM from top few particles
        polish_budget = self.budget - self._call_count
        if polish_budget > 10 and best_x is not None:
            # Sort particles by personal best fitness
            order = np.argsort(personal_best_f)
            top_k = min(3, n_particles)
            for k in range(top_k):
                if self._call_count >= self.budget or time.time() > deadline:
                    break
                alloc = max(5, polish_budget // (top_k - k))
                nelder_mead_local(personal_bests[order[k]], alloc)

        return best_x, best_f
# EVOLVE-BLOCK-END

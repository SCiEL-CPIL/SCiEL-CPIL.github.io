"""Rewritten optimizer using PSO with Nelder-Mead local polishing for Lunacek bi-Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """PSO-based derivative-free optimizer with Nelder-Mead local polish."""

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
            x_clipped = np.clip(x, -5, 5)
            f = float(func(x_clipped))
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        def nelder_mead(start_x, max_calls=50):
            """Nelder-Mead local optimizer."""
            nonlocal deadline
            dim = self.dim
            # Initial simplex
            simplex = [start_x.copy()]
            for i in range(dim):
                point = start_x.copy()
                point[i] += 0.5 if point[i] < 4.5 else -0.5
                simplex.append(point)

            fs = [counted_func(s) for s in simplex]
            if self._call_count >= self.budget:
                return

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_calls):
                if time.time() > deadline or self._call_count >= self.budget:
                    break

                # Sort simplex
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = counted_func(xr)
                if self._call_count >= self.budget:
                    break

                if fr < fs[0]:
                    # Try expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = counted_func(xe)
                    if self._call_count >= self.budget:
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        fs[-1] = fe
                    else:
                        simplex[-1] = xr
                        fs[-1] = fr
                elif fr < fs[-2]:
                    simplex[-1] = xr
                    fs[-1] = fr
                else:
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = counted_func(xc)
                    if self._call_count >= self.budget:
                        break
                    if fc < fs[-1]:
                        simplex[-1] = xc
                        fs[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fs[i] = counted_func(simplex[i])
                            if self._call_count >= self.budget:
                                return

                # Check convergence
                if len(set(np.round(fs, 10))) == 1:
                    break

        def pso_optimize(n_particles, max_iter, w=0.7, c1=1.49445, c2=1.49445, v_max_frac=0.5):
            """Particle Swarm Optimization."""
            nonlocal deadline
            dim = self.dim
            rng = np.random.default_rng(42)

            # Initialize particles
            positions = rng.uniform(-4, 4, (n_particles, dim))
            velocities = rng.uniform(-1, 1, (n_particles, dim))
            pbests = positions.copy()
            pbest_f = np.array([counted_func(p) for p in positions])
            if self._call_count >= self.budget:
                return

            gbest_idx = np.argmin(pbest_f)
            gbest = pbests[gbest_idx].copy()
            gbest_f = pbest_f[gbest_idx]

            v_max = v_max_frac * 10.0  # box width is 10

            for it in range(max_iter):
                if time.time() > deadline or self._call_count >= self.budget:
                    break

                for i in range(n_particles):
                    if self._call_count >= self.budget:
                        return

                    r1 = rng.random(dim)
                    r2 = rng.random(dim)

                    velocities[i] = w * velocities[i] + c1 * r1 * (pbests[i] - positions[i]) + c2 * r2 * (gbest - positions[i])
                    velocities[i] = np.clip(velocities[i], -v_max, v_max)

                    positions[i] = positions[i] + velocities[i]
                    positions[i] = np.clip(positions[i], -5, 5)

                    f = counted_func(positions[i])
                    if self._call_count >= self.budget:
                        return

                    if f < pbest_f[i]:
                        pbest_f[i] = f
                        pbests[i] = positions[i].copy()
                        if f < gbest_f:
                            gbest_f = f
                            gbest = positions[i].copy()
                            gbest_idx = i

                # Adaptive inertia weight
                w = 0.9 - 0.4 * it / max(1, max_iter)

            return gbest, gbest_f

        # Phase 1: PSO global search
        n_particles = 20
        pso_iters = 10  # 20 particles * 10 iters = 200 calls for PSO
        gbest, gbest_f = pso_optimize(n_particles, pso_iters)

        # Phase 2: Local polish with Nelder-Mead from best PSO result
        if self._call_count < self.budget - 10:
            nelder_mead(gbest, max_calls=30)

        # Phase 3: If budget remains, do another short PSO from current best
        remaining = self.budget - self._call_count
        if remaining > 30:
            # Second PSO with smaller population, focused near current best
            n_particles2 = 10
            pso_iters2 = min(5, remaining // 10)
            if pso_iters2 > 0:
                # Initialize around current best
                rng2 = np.random.default_rng(43)
                positions2 = rng2.normal(gbest, 1.0, (n_particles2, self.dim))
                positions2 = np.clip(positions2, -5, 5)
                velocities2 = rng2.uniform(-0.5, 0.5, (n_particles2, self.dim))
                pbests2 = positions2.copy()
                pbest_f2 = np.array([counted_func(p) for p in positions2])
                if self._call_count < self.budget:
                    gbest_idx2 = np.argmin(pbest_f2)
                    gbest2 = pbests2[gbest_idx2].copy()
                    gbest_f2 = pbest_f2[gbest_idx2]

                    w2 = 0.7
                    c1, c2 = 1.49445, 1.49445
                    v_max2 = 0.3 * 10.0

                    for it in range(pso_iters2):
                        if time.time() > deadline or self._call_count >= self.budget:
                            break
                        for i in range(n_particles2):
                            if self._call_count >= self.budget:
                                break
                            r1 = rng2.random(self.dim)
                            r2 = rng2.random(self.dim)
                            velocities2[i] = w2 * velocities2[i] + c1 * r1 * (pbests2[i] - positions2[i]) + c2 * r2 * (gbest2 - positions2[i])
                            velocities2[i] = np.clip(velocities2[i], -v_max2, v_max2)
                            positions2[i] = positions2[i] + velocities2[i]
                            positions2[i] = np.clip(positions2[i], -5, 5)
                            f = counted_func(positions2[i])
                            if self._call_count >= self.budget:
                                break
                            if f < pbest_f2[i]:
                                pbest_f2[i] = f
                                pbests2[i] = positions2[i].copy()
                                if f < gbest_f2:
                                    gbest_f2 = f
                                    gbest2 = positions2[i].copy()
                        w2 = 0.9 - 0.4 * it / max(1, pso_iters2)

                    # Final polish
                    if self._call_count < self.budget - 5:
                        nelder_mead(gbest2, max_calls=15)

        return best_x, best_f
# EVOLVE-BLOCK-END

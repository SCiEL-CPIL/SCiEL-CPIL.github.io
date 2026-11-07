"""Multi-phase optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase: Latin hypercube sampling + Nelder-Mead + coordinate descent."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, calls_state, max_iter=100, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0):
        """Nelder-Mead simplex method with adaptive step control."""
        dim = self.dim
        calls_made = [calls_state[0]]

        # Initialize simplex
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            step = 0.25 if xi[i] == 0 else 0.05 * abs(xi[i])
            xi[i] += step
            simplex.append(xi)

        # Evaluate initial simplex
        fvals = []
        for i in range(dim + 1):
            if calls_made[0] >= self.budget:
                break
            f = func(simplex[i])
            fvals.append(f)
            calls_made[0] += 1

        if len(fvals) < dim + 1:
            return simplex[0], fvals[0] if fvals else np.inf

        best_idx = np.argmin(fvals)
        best_x = simplex[best_idx].copy()
        best_f = fvals[best_idx]

        for iteration in range(max_iter):
            if calls_made[0] >= self.budget:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Check convergence
            if abs(fvals[-1] - fvals[0]) < 1e-12:
                break

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls_made[0] += 1

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls_made[0] += 1
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
                calls_made[0] += 1
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls_made[0] += 1
                    if calls_made[0] >= self.budget:
                        break

            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = simplex[min_idx].copy()

        return best_x, best_f

    def _coordinate_descent(self, func, x0, calls_state, max_iter=50, step_init=0.5):
        """Cyclic coordinate descent with adaptive step size."""
        dim = self.dim
        calls_made = [calls_state[0]]
        x = x0.copy()
        f = func(x)
        calls_made[0] += 1
        step = step_init

        for iteration in range(max_iter):
            if calls_made[0] >= self.budget:
                break
            improved = False
            for i in range(dim):
                if calls_made[0] >= self.budget:
                    break
                # Try positive and negative steps
                for direction in [1, -1]:
                    x_new = x.copy()
                    x_new[i] += direction * step
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    calls_made[0] += 1
                    if f_new < f:
                        x = x_new
                        f = f_new
                        improved = True
                        break
                if calls_made[0] >= self.budget:
                    break
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
            else:
                step = min(step * 1.2, 2.0)

        return x, f

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # 10s safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_made = [0]

        best_x = None
        best_f = np.inf

        def evaluate(x):
            """Evaluate and track best."""
            nonlocal best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = func(x)
            calls_made[0] += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Latin Hypercube Sampling for global exploration
        n_lhs = min(30, budget // 5)
        rng = np.random.default_rng(42)

        # Create LHS points
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, d] = -5 + 10 * (perm + 0.5) / n_lhs

        # Evaluate LHS points
        for i in range(n_lhs):
            if calls_made[0] >= budget or time.time() > deadline:
                break
            evaluate(lhs_points[i])

        if calls_made[0] >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Nelder-Mead from top candidates
        # Sort evaluated points by function value
        evaluated_points = []
        # Re-evaluate LHS points to get their values (or track them)
        # Actually, let's track them properly
        # Let me redo this with proper tracking

        # Reset and redo with proper tracking
        calls_made[0] = 0
        best_x = None
        best_f = np.inf
        evaluated = []  # list of (f, x)

        def evaluate2(x):
            """Evaluate and track."""
            nonlocal best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = func(x)
            calls_made[0] += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            evaluated.append((f, x.copy()))
            return f

        # Phase 1: Latin Hypercube Sampling
        n_lhs = min(25, budget // 6)
        rng = np.random.default_rng(42)
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, d] = -5 + 10 * (perm + 0.5) / n_lhs

        for i in range(n_lhs):
            if calls_made[0] >= budget or time.time() > deadline:
                break
            evaluate2(lhs_points[i])

        if calls_made[0] >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Nelder-Mead from top 3 candidates
        evaluated.sort(key=lambda t: t[0])
        top_candidates = [x for _, x in evaluated[:3]]

        nm_budget_per = (budget - calls_made[0]) // 4
        for cand in top_candidates:
            if calls_made[0] >= budget or time.time() > deadline:
                break
            x_opt, f_opt = self._nelder_mead(func, cand, calls_made, max_iter=nm_budget_per)
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        if calls_made[0] >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: Coordinate descent polish on best point
        cd_budget = budget - calls_made[0]
        if cd_budget > 10:
            x_cd, f_cd = self._coordinate_descent(func, best_x, calls_made, max_iter=cd_budget)
            if f_cd < best_f:
                best_f = f_cd
                best_x = x_cd.copy()

        # Phase 4: If budget remains, try another Nelder-Mead from a different start
        if calls_made[0] < budget - 20 and time.time() < deadline:
            # Try a random point
            x_rand = rng.uniform(-4, 4, dim)
            x_opt2, f_opt2 = self._nelder_mead(func, x_rand, calls_made, max_iter=30)
            if f_opt2 < best_f:
                best_f = f_opt2
                best_x = x_opt2.copy()

        # Final coordinate descent polish
        if calls_made[0] < budget - 5 and time.time() < deadline:
            x_final, f_final = self._coordinate_descent(func, best_x, calls_made, max_iter=10, step_init=0.1)
            if f_final < best_f:
                best_f = f_final
                best_x = x_final.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            if calls_made[0] < budget:
                best_f = func(best_x)
                calls_made[0] += 1

        return best_x, best_f
# EVOLVE-BLOCK-END

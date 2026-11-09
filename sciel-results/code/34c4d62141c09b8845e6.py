# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Tchebycheff decomposition-based multi-objective optimizer for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # --- Generate Tchebycheff weight vectors on the simplex ---
        # For 3 objectives, use a grid of weight vectors
        n_w = 20  # points per axis in the simplex grid
        weights = []
        for i in range(n_w + 1):
            for j in range(n_w + 1 - i):
                k = n_w - i - j
                w = np.array([i, j, k], dtype=float) / n_w
                weights.append(w)
        weights = np.array(weights)
        n_sub = len(weights)

        # --- Initial population: one solution per weight vector ---
        # Use a biased initialization toward the Pareto front region
        # For DTLZ4, the Pareto front is at x_1..x_{n-2} in [0,1], x_{n-1}..x_n at 0
        # But we don't exploit that; use random init
        X = np.zeros((n_sub, dim))
        for i in range(n_sub):
            X[i] = lo + span * rng.random(dim)

        F = np.array([problem(x) for x in X])
        used = n_sub

        # Reference point (nadir) for Tchebycheff
        z = np.max(F, axis=0)

        # Track best solutions per subproblem
        best_X = X.copy()
        best_F = F.copy()
        best_tcheb = np.zeros(n_sub)

        # Compute initial Tchebycheff values
        for i in range(n_sub):
            best_tcheb[i] = np.max(weights[i] * np.abs(best_F[i] - z))

        # --- Local search helper: Nelder-Mead on Tchebycheff scalarization ---
        def tcheb_objective(x, w, z_ref):
            f = problem(x)
            return np.max(w * np.abs(f - z_ref))

        def nelder_mead(x0, w, z_ref, max_iter=50):
            """Simple Nelder-Mead minimizer for Tchebycheff scalarization."""
            f0 = tcheb_objective(x0, w, z_ref)
            # Create initial simplex
            simplex = [x0.copy()]
            for d in range(dim):
                x_new = x0.copy()
                step = 0.05 * span[d] if span[d] > 0 else 0.05
                x_new[d] += step
                x_new = np.clip(x_new, lo, hi)
                simplex.append(x_new)
            fvals = [tcheb_objective(s, w, z_ref) for s in simplex]

            for _ in range(max_iter):
                # Sort by fitness
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Reflection
                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + 1.0 * (centroid - simplex[-1])
                xr = np.clip(xr, lo, hi)
                fr = tcheb_objective(xr, w, z_ref)

                if fr < fvals[-2]:
                    if fr < fvals[0]:
                        # Expansion
                        xe = centroid + 2.0 * (centroid - simplex[-1])
                        xe = np.clip(xe, lo, hi)
                        fe = tcheb_objective(xe, w, z_ref)
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                        xc = np.clip(xc, lo, hi)
                        fc = tcheb_objective(xc, w, z_ref)
                        if fc < fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                fvals[i] = tcheb_objective(simplex[i], w, z_ref)
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (centroid - simplex[-1])
                        xc = np.clip(xc, lo, hi)
                        fc = tcheb_objective(xc, w, z_ref)
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                fvals[i] = tcheb_objective(simplex[i], w, z_ref)

                # Check convergence
                if max(fvals) - min(fvals) < 1e-10:
                    break

            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]

        # --- Main optimization loop ---
        # Use remaining budget for local search on promising subproblems
        # and global exploration

        # Sort subproblems by current Tchebycheff value (best first)
        while used < budget:
            if time.time() > deadline - 30:
                break

            # Update nadir reference point
            z = np.max(best_F, axis=0)

            # Recompute Tchebycheff values with updated z
            for i in range(n_sub):
                best_tcheb[i] = np.max(weights[i] * np.abs(best_F[i] - z))

            # Select subproblems to improve: those with highest Tchebycheff values
            # and some random ones for exploration
            # Focus on worst subproblems to improve coverage
            worst_indices = np.argsort(best_tcheb)[-max(1, n_sub // 5):]
            # Also add some random subproblems
            random_indices = rng.integers(0, n_sub, size=min(3, n_sub))
            to_improve = list(set(list(worst_indices) + list(random_indices)))

            # Shuffle to vary the order
            rng.shuffle(to_improve)

            for idx in to_improve:
                if used >= budget or time.time() > deadline - 10:
                    break

                w = weights[idx]
                x0 = best_X[idx].copy()

                # Try Nelder-Mead local search
                x_new, f_new = nelder_mead(x0, w, z, max_iter=30)
                used += 1

                # Evaluate the new solution's actual objectives
                f_actual = problem(x_new)
                used += 1  # count the extra evaluation

                # Check if it improves the Tchebycheff value
                tcheb_new = np.max(w * np.abs(f_actual - z))
                if tcheb_new < best_tcheb[idx]:
                    best_X[idx] = x_new
                    best_F[idx] = f_actual
                    best_tcheb[idx] = tcheb_new

                # Also try a random perturbation for diversity
                if used < budget and time.time() < deadline - 5:
                    x_rand = x0 + 0.1 * span * rng.standard_normal(dim)
                    x_rand = np.clip(x_rand, lo, hi)
                    f_rand = problem(x_rand)
                    used += 1
                    tcheb_rand = np.max(w * np.abs(f_rand - z))
                    if tcheb_rand < best_tcheb[idx]:
                        best_X[idx] = x_rand
                        best_F[idx] = f_rand
                        best_tcheb[idx] = tcheb_rand

        # --- Final non-dominated archive ---
        # Collect all best solutions and filter non-dominated
        all_F = best_F
        n = len(all_F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(all_F[j] <= all_F[i]) and np.any(all_F[j] < all_F[i]):
                    dominated[i] = True
                    break

        non_dom_mask = ~dominated
        archive_F = all_F[non_dom_mask]
        archive_X = best_X[non_dom_mask]

        return archive_X
# EVOLVE-BLOCK-END

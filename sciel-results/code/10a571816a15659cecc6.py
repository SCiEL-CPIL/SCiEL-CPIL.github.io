"""Multi-objective WFG7 optimizer: Tchebycheff decomposition + adaptive local search + NM polish."""
import time
import numpy as np


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.RandomState(0)

        used = 0

        def remaining():
            return self.budget - used

        def time_left():
            return deadline - time.time()

        # --- Tchebycheff weight vectors (3^3 = 27, but we use a subset of 25) ---
        # Generate a grid of weight vectors for 3 objectives
        # Use 3 levels per objective: 0, 0.5, 1 -> 27 combos, drop a few to get 25
        levels = np.array([0.0, 0.5, 1.0])
        weights = []
        for a in levels:
            for b in levels:
                for c in levels:
                    w = np.array([a, b, c])
                    if np.sum(w) > 1e-12:
                        weights.append(w / np.sum(w))
        # We have 26 non-zero vectors; use all 26 (or 25)
        weights = np.array(weights[:25])

        # --- Initial population: one point per weight vector ---
        # For each weight, start at a random point
        n_w = len(weights)
        X = np.zeros((n_w, self.dim))
        F = np.zeros((n_w, self.n_obj))
        steps = np.full(n_w, 0.1)  # adaptive step size per sub-problem

        for i in range(n_w):
            X[i] = lo + (hi - lo) * rng.rand(self.dim)
            F[i] = np.asarray(problem(X[i]), dtype=float)
            used += 1

        # --- Tchebycheff scalarization ---
        # First find the ideal point (min of each objective)
        ideal = F.min(axis=0)

        def tcheb(x, f, w):
            # Tchebycheff: max_i w_i * |f_i - z_i| / w_i  (with z_i = ideal)
            # Actually standard: max_i w_i * |f_i - z_i|
            # But to handle different scales, use: max_i (f_i - z_i) * w_i  (since we minimize)
            # Use the standard form: max_i w_i * (f_i - z_i)
            diff = f - ideal
            # Clip negatives to 0 (since ideal is the minimum)
            diff = np.maximum(diff, 0.0)
            return np.max(diff * w)

        # --- Main search loop ---
        # For each weight vector, do adaptive local search
        # Track best solution per weight
        best_F = F.copy()
        best_X = X.copy()

        # Precompute: for each weight, the current best tcheb value
        tcheb_vals = np.array([tcheb(best_X[i], best_F[i], weights[i]) for i in range(n_w)])

        # Main loop: iterate through weights, do a step for each
        max_iter = 50
        for it in range(max_iter):
            if used >= self.budget:
                break
            if time_left() < 5.0:
                break

            for i in range(n_w):
                if used >= self.budget:
                    break
                if time_left() < 3.0:
                    break

                w = weights[i]
                x = best_X[i]
                f = best_F[i]
                cur_tcheb = tcheb_vals[i]

                # Generate a perturbation
                step = steps[i]
                # Direction: random, but biased toward reducing the worst objective
                # Find which objective contributes most to the Tchebycheff value
                diff = np.maximum(f - ideal, 0.0)
                contrib = diff * w
                worst_idx = np.argmax(contrib)

                # Perturb: random direction, but nudge the worst objective down
                delta = rng.randn(self.dim) * step
                # Add a small bias: move in a random direction
                x_new = x + delta
                x_new = np.clip(x_new, lo, hi)

                f_new = np.asarray(problem(x_new), dtype=float)
                used += 1

                new_tcheb = tcheb(x_new, f_new, w)

                if new_tcheb < cur_tcheb - 1e-10:
                    # Improvement: accept, and maybe grow step
                    best_X[i] = x_new
                    best_F[i] = f_new
                    tcheb_vals[i] = new_tcheb
                    steps[i] = min(0.5, steps[i] * 1.1)
                else:
                    # No improvement: shrink step
                    steps[i] = max(0.001, steps[i] * 0.8)

        # --- Polish phase: Nelder-Mead on best non-dominated points ---
        # Find non-dominated points
        def is_dominated(f, F):
            """Check if f is dominated by any point in F."""
            for fj in F:
                if np.all(fj <= f) and np.any(fj < f):
                    return True
            return False

        # Get non-dominated set
        non_dom_idx = []
        for i in range(n_w):
            if not is_dominated(best_F[i], best_F):
                non_dom_idx.append(i)

        # Polish the top few non-dominated points with NM
        # Sort by hypervolume contribution (approximate)
        ref = np.array([3.0, 5.0, 7.0])
        # Approximate contribution: product of (ref - f) for points better than ref
        contributions = []
        for i in non_dom_idx:
            if np.all(best_F[i] < ref):
                contrib = np.prod(ref - best_F[i])
            else:
                contrib = 0.0
            contributions.append((contrib, i))
        contributions.sort(reverse=True)

        # Polish top 5 with NM
        n_polish = min(5, len(contributions))
        for _, idx in contributions[:n_polish]:
            if used >= self.budget:
                break
            if time_left() < 10.0:
                break

            w = weights[idx]
            x0 = best_X[idx].copy()

            # NM on Tchebycheff value
            def tcheb_obj(x):
                nonlocal used
                if used >= self.budget:
                    return 1e10
                if time.time() > deadline:
                    return 1e10
                x = np.clip(x, lo, hi)
                f = np.asarray(problem(x), dtype=float)
                used += 1
                return tcheb(x, f, w)

            # Simple NM implementation
            dim = self.dim
            simplex = [x0.copy()]
            for d in range(dim):
                pt = x0.copy()
                pt[d] += 0.05 * (hi[d] - lo[d])
                simplex.append(np.clip(pt, lo, hi))

            vals = [tcheb_obj(pt) for pt in simplex]

            nm_iter = 0
            max_nm_iter = 30
            while nm_iter < max_nm_iter and used < self.budget and time_left() > 5.0:
                nm_iter += 1
                # Sort by objective value
                order = np.argsort(vals)
                simplex = [simplex[i] for i in order]
                vals = [vals[i] for i in order]

                worst = simplex[-1]
                val_worst = vals[-1]
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - worst)
                x_ref = np.clip(x_ref, lo, hi)
                val_ref = tcheb_obj(x_ref)

                if val_ref < vals[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - worst)
                    x_exp = np.clip(x_exp, lo, hi)
                    val_exp = tcheb_obj(x_exp)
                    if val_exp < val_ref:
                        simplex[-1] = x_exp
                        vals[-1] = val_exp
                    else:
                        simplex[-1] = x_ref
                        vals[-1] = val_ref
                elif val_ref < vals[-2]:
                    simplex[-1] = x_ref
                    vals[-1] = val_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (worst - centroid)
                    x_con = np.clip(x_con, lo, hi)
                    val_con = tcheb_obj(x_con)
                    if val_con < val_worst:
                        simplex[-1] = x_con
                        vals[-1] = val_con
                    else:
                        # Shrink
                        for k in range(1, len(simplex)):
                            simplex[k] = simplex[0] + 0.5 * (simplex[k] - simplex[0])
                            vals[k] = tcheb_obj(simplex[k])

                # Update best for this weight if improved
                best_val = min(vals)
                best_pt = simplex[vals.index(best_val)]
                f_best = np.asarray(problem(best_pt), dtype=float)
                used += 1
                if tcheb(best_pt, f_best, w) < tcheb_vals[idx]:
                    best_X[idx] = best_pt
                    best_F[idx] = f_best
                    tcheb_vals[idx] = tcheb(best_pt, f_best, w)

        # --- Final: return all best points ---
        return best_X

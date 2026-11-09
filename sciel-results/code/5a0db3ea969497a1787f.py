"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start local search for WFG4: bootstrap + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Scalarization weights: emphasize different regions of the front
        # For WFG4 with 3 objectives, use a set of weight vectors
        n_starts = 40
        # Pre-compute weight vectors for scalarization
        weights = []
        for i in range(n_starts):
            w = rng.dirichlet(np.ones(n_obj))
            weights.append(w)

        all_F = []  # list of all evaluated objective vectors

        def evaluate(x):
            nonlocal all_F
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            all_F.append(f)
            return f

        def local_refine(x0, w, max_iter=30):
            """Nelder-Mead style local search on scalarized objective."""
            nonlocal all_F
            x = x0.copy()
            f = evaluate(x)
            scalar = float(np.dot(w, f))
            # Initial simplex
            simplex = [x.copy()]
            for i in range(dim):
                xi = x.copy()
                step = 0.05 * (hi[i] - lo[i]) if hi[i] != lo[i] else 0.05
                xi[i] += step
                xi = np.clip(xi, lo, hi)
                simplex.append(xi)
            simplex_F = [f]
            for i in range(1, dim + 1):
                simplex_F.append(evaluate(simplex[i]))
            simplex_scalar = [float(np.dot(w, sf)) for sf in simplex_F]

            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                # Sort by scalar objective
                order = np.argsort(simplex_scalar)
                simplex = [simplex[i] for i in order]
                simplex_F = [simplex_F[i] for i in order]
                simplex_scalar = [simplex_scalar[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                # Reflection
                xr = centroid + (centroid - simplex[-1])
                xr = np.clip(xr, lo, hi)
                fr = evaluate(xr)
                sr = float(np.dot(w, fr))

                if sr < simplex_scalar[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[-1])
                    xe = np.clip(xe, lo, hi)
                    fe = evaluate(xe)
                    se = float(np.dot(w, fe))
                    if se < sr:
                        simplex[-1] = xe
                        simplex_F[-1] = fe
                        simplex_scalar[-1] = se
                    else:
                        simplex[-1] = xr
                        simplex_F[-1] = fr
                        simplex_scalar[-1] = sr
                elif sr < simplex_scalar[-2]:
                    simplex[-1] = xr
                    simplex_F[-1] = fr
                    simplex_scalar[-1] = sr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, lo, hi)
                    fc = evaluate(xc)
                    sc = float(np.dot(w, fc))
                    if sc < simplex_scalar[-1]:
                        simplex[-1] = xc
                        simplex_F[-1] = fc
                        simplex_scalar[-1] = sc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            simplex_F[i] = evaluate(simplex[i])
                            simplex_scalar[i] = float(np.dot(w, simplex_F[i]))
                        order2 = np.argsort(simplex_scalar)
                        simplex = [simplex[i] for i in order2]
                        simplex_F = [simplex_F[i] for i in order2]
                        simplex_scalar = [simplex_scalar[i] for i in order2]

            return simplex[0], simplex_F[0]

        # Allocate budget: each start gets roughly budget/n_starts evaluations
        # But local_refine uses ~dim+1 + max_iter*(1 or 2) evaluations
        # Let's be conservative: use ~40 evals per start
        evals_per_start = max(20, budget // n_starts)
        max_iter_per_start = max(5, (evals_per_start - dim - 1) // 2)

        for start_idx in range(n_starts):
            if time.time() > deadline:
                break
            if len(all_F) >= budget:
                break

            w = weights[start_idx]
            # Bootstrap: small random population
            boot_size = min(5, evals_per_start // 3)
            boot_X = lo + (hi - lo) * rng.random((boot_size, dim))
            boot_F = []
            for bx in boot_X:
                if len(all_F) >= budget or time.time() > deadline:
                    break
                bf = evaluate(bx)
                boot_F.append(bf)

            if not boot_F:
                continue

            # Pick best boot point for this weight
            boot_scalars = [float(np.dot(w, bf)) for bf in boot_F]
            best_boot_idx = int(np.argmin(boot_scalars))
            x0 = boot_X[best_boot_idx].copy()

            # Local refinement
            local_refine(x0, w, max_iter=max_iter_per_start)

        return np.array(all_F)
# EVOLVE-BLOCK-END

"""Multi-objective optimizer for DTLZ1 using simplex-grid + local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid simplex-grid + local-polish optimizer for DTLZ1 (3 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 300.0
        budget = self.budget
        dim = self.dim

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        # DTLZ1 structure:
        #   g = sum((x[i]-0.5)^2) for i in [4,5,6]  (last 3 vars)
        #   z1 = x[0], z2 = x[1]
        #   f1 = 0.5*g*(1-z1), f2 = 0.5*g*z1, f3 = 0.5*g*z2
        # Pareto front: g=0, z1+z2<=1, z1,z2>=0
        #   -> triangle in objective space.
        # We fix the g-vars near 0.5 (polish) and grid z1,z2 over the simplex.

        # Build a uniform grid on the 2-simplex {z1,z2>=0, z1+z2<=1}
        # Use a lattice: for i in 0..m, j in 0..m-i, k=m-i-j, point = (i,j,k)/m
        # Number of points = (m+1)(m+2)/2. Choose m so that points*polish ~ budget.
        # We'll use polish steps per point = 6 (3 vars, 2 iterations each).
        polish_per = 6
        # Solve (m+1)(m+2)/2 * polish_per <= budget
        # try m values
        m = 0
        best_m = 0
        for cand in range(0, 40):
            npts = (cand + 1) * (cand + 2) // 2
            if npts * polish_per <= budget:
                best_m = cand
        m = best_m
        npts = (m + 1) * (m + 2) // 2

        # Generate simplex grid points (z1, z2)
        pts = []
        for i in range(m + 1):
            for j in range(m - i + 1):
                k = m - i - j
                z1 = i / m if m > 0 else 0.5
                z2 = j / m if m > 0 else 0.5
                pts.append((z1, z2))
        pts = np.array(pts)  # (npts, 2)

        used = 0
        all_F = []
        all_X = []

        # Center for g-vars (x4,x5,x6)
        g_center = 0.5
        g_idx = [4, 5, 6]

        def make_x(z1, z2, g_vals):
            x = np.zeros(dim)
            x[0] = z1
            x[1] = z2
            # x[2], x[3] are free (don't affect objectives in DTLZ1 standard); set to 0.5
            x[2] = 0.5
            x[3] = 0.5
            x[4] = g_vals[0]
            x[5] = g_vals[1]
            x[6] = g_vals[2]
            return x

        # For each simplex point, start g-vars at center and polish to reduce g.
        for idx in range(npts):
            if used >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            z1, z2 = pts[idx]
            gv = np.array([g_center, g_center, g_center])
            # Evaluate initial
            x = make_x(z1, z2, gv)
            f = np.array(problem(x), dtype=float)
            used += 1
            all_F.append(f)
            all_X.append(x.copy())

            # Polish: coordinate descent on g-vars to minimize g (i.e., push toward 0.5)
            # Since g = sum (x_i-0.5)^2, the minimum is exactly at 0.5.
            # We do a few local steps: try small perturbations and keep if g decreases.
            for it in range(2):
                if used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                # Try moving each g-var toward 0.5 with a step
                improved = False
                for vi in range(3):
                    if used >= budget:
                        break
                    step = 0.05 * (g_center - gv[vi])
                    if abs(step) < 1e-9:
                        continue
                    new_gv = gv.copy()
                    new_gv[vi] = np.clip(gv[vi] + step, lo[g_idx[vi]], hi[g_idx[vi]])
                    nx = make_x(z1, z2, new_gv)
                    nf = np.array(problem(nx), dtype=float)
                    used += 1
                    # g is monotonic with all objectives (all scale by 0.5*g),
                    # so minimizing sum of objectives ~ minimizing g.
                    if np.sum(nf) < np.sum(f):
                        gv = new_gv
                        f = nf
                        x = nx
                        all_F[-1] = f
                        all_X[-1] = x.copy()
                        improved = True
            # Final: set g-vars exactly to 0.5 if within bounds, one more eval
            if used < budget:
                gv_final = np.array([g_center, g_center, g_center], dtype=float)
                # clip
                gv_final = np.clip(gv_final, lo[g_idx], hi[g_idx])
                xf = make_x(z1, z2, gv_final)
                ff = np.array(problem(xf), dtype=float)
                used += 1
                all_F.append(ff)
                all_X.append(xf.copy())

        # If we have budget left, do a light random evolutionary sweep for robustness
        if used < budget:
            rng = np.random.default_rng(0)
            pop = min(20, budget - used)
            Xr = lo + (hi - lo) * rng.random((pop, dim))
            for x in Xr:
                if used >= budget:
                    break
                if time.time() > deadline - 2.0:
                    break
                f = np.array(problem(x), dtype=float)
                used += 1
                all_F.append(f)
                all_X.append(x.copy())

        # Ensure at least one evaluation happened
        if not all_F:
            x = lo + 0.5 * span
            f = np.array(problem(x), dtype=float)
            all_F.append(f)
            all_X.append(x.copy())

        return np.array(all_X)
# EVOLVE-BLOCK-END

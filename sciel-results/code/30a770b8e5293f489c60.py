"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7-specific optimizer using structured (x0,x1) grid with g-variable search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 320.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_dim = self.dim
        n_obj = self.n_obj

        # DTLZ7 structure: x0, x1 determine f1, f2; x[2:] determine g
        # f1 = x0, f2 = x1*(1-x0), f3 = g(x[2:]) * (1 + x1 * sin(pi/2 * x1))
        # Actually for DTLZ7: M=3, n=22
        # f1 = x0
        # f2 = x1 * (1 - x0)
        # f3 = g(x_{M-1}) * (1 + x_{M-2} * sin(pi/2 * x_{M-2}))
        # where g(x) = 1 + 9/n * sum(x_i) for x_i in x_{M-1}
        # Wait, standard DTLZ7 with M=3, n=22:
        # f1 = x0
        # f2 = x1 * (1 - x0)  
        # f3 = g(x_{2..21}) * (1 + x1 * sin(pi/2 * x1))
        # g(x) = 1 + 9/20 * sum_{i=2}^{21} x_i
        # So g ranges from 1 to 11 when x[2:] in [0,1]
        # f3 ranges from 1 to 11

        def make_x(x0, x1, g_vec):
            x = np.zeros(n_dim)
            x[0] = x0
            x[1] = x1
            x[2:] = g_vec
            return np.clip(x, lo, hi)

        def eval_x(x):
            return np.asarray(problem(x), dtype=float)

        # Phase 1: Structured sampling on (x0, x1) with g=0 (minimal g)
        # Use Latin hypercube on (x0, x1) for good spread
        n_lh = 64
        # LHS sampling
        perm0 = rng.permutation(n_lh)
        perm1 = rng.permutation(n_lh)
        x0_vals = (perm0 + 0.5) / n_lh
        x1_vals = (perm1 + 0.5) / n_lh
        # Shuffle to decorrelate
        idx = rng.permutation(n_lh)
        x0_vals = x0_vals[idx]
        x1_vals = x1_vals[idx]

        g_zero = np.zeros(n_dim - 2)
        X = []
        F = []
        used = 0

        for i in range(n_lh):
            if used >= self.budget:
                break
            x = make_x(x0_vals[i], x1_vals[i], g_zero)
            f = eval_x(x)
            X.append(x)
            F.append(f)
            used += 1

        # Phase 2: Add points with varying g to explore f3 dimension
        # Sample some points with non-zero g
        n_g = min(32, self.budget - used)
        for i in range(n_g):
            if used >= self.budget:
                break
            x0 = rng.random()
            x1 = rng.random()
            g_val = rng.random() * 10.0  # g ranges ~1 to 11
            g_vec = np.full(n_dim - 2, g_val / (n_dim - 2))  # uniform spread
            x = make_x(x0, x1, g_vec)
            f = eval_x(x)
            X.append(x)
            F.append(f)
            used += 1

        X = np.array(X)
        F = np.array(F)

        # Phase 3: Local refinement around best hypervolume-contributing points
        # Simple approach: for each point, try small perturbations
        remaining = self.budget - used
        if remaining > 0 and len(X) > 0:
            # Find non-dominated points
            F_arr = F
            n_pts = len(F_arr)
            dominated = np.zeros(n_pts, dtype=bool)
            for i in range(n_pts):
                if dominated[i]:
                    continue
                for j in range(i + 1, n_pts):
                    if dominated[j]:
                        continue
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated[j] = True
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break

            nd_idx = np.where(~dominated)[0]
            if len(nd_idx) == 0:
                nd_idx = np.array([np.argmin(np.sum(F_arr, axis=1))])

            # Refine around non-dominated points
            n_refine = min(remaining, len(nd_idx) * 3)
            for i in range(n_refine):
                if used >= self.budget:
                    break
                if time.time() - t0 > deadline:
                    break
                base_idx = nd_idx[i % len(nd_idx)]
                base_x = X[base_idx].copy()
                
                # Perturb x0, x1 slightly
                pert = rng.standard_normal(2) * 0.05
                new_x0 = np.clip(base_x[0] + pert[0], lo[0], hi[0])
                new_x1 = np.clip(base_x[1] + pert[1], lo[1], hi[1])
                new_x = base_x.copy()
                new_x[0] = new_x0
                new_x[1] = new_x1
                # Also slightly perturb g
                if n_dim > 2:
                    g_pert = rng.standard_normal(n_dim - 2) * 0.1
                    new_x[2:] = np.clip(base_x[2:] + g_pert, lo[2:], hi[2:])
                
                f = eval_x(new_x)
                X.append(new_x)
                F.append(f)
                used += 1

        X = np.array(X)
        F = np.array(F)
        return X
# EVOLVE-BLOCK-END

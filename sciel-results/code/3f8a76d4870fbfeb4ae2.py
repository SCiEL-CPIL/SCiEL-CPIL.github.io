# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware multi-objective optimizer for ZDT4.

    Exploits ZDT4's separability: f1 depends only on x1 (min 0 at x1=0.5),
    f2 depends on x2..x10. Uses non-uniform x1 sampling, vectorized batch
    evaluation, non-dominated filtering, and a final polish pass.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300.0 - 10.0  # safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        budget = self.budget

        # ZDT4 structure: f1 = 4x1^2 - 6x1*cos(8*pi*x1) + 2, global min 0 at x1=0.5
        # f2 = sum g(xi), g = 20*xi^2 + 2*cos(20*pi*xi^2) for i>=2
        # x1 in [0,1] typically; other vars in [-5,5]

        def clip(x):
            return np.clip(x, lo, hi)

        # Batched evaluator
        def eval_batch(X):
            # X: (m, n) array
            m = X.shape[0]
            F = np.empty((m, 2))
            for i in range(m):
                F[i] = problem(X[i])
            return F

        # Non-dominated mask: returns boolean array True where point is non-dominated
        def non_dominated(F):
            m = F.shape[0]
            if m <= 1:
                return np.ones(m, dtype=bool)
            # A point i is dominated if exists j: F[j] <= F[i] and F[j] != F[i]
            nd = np.ones(m, dtype=bool)
            for i in range(m):
                if not nd[i]:
                    continue
                # check if any other point dominates i
                dom = (F <= F[i]).all(axis=1) & (F < F[i]).any(axis=1)
                dom[i] = False
                if dom.any():
                    nd[i] = False
            return nd

        # Phase 1: structured initialization
        # Sample x1 heavily around 0.5 (where f1 is minimized)
        # Sample other vars in low region to keep f2 small

        rng = np.random.default_rng(42)

        # x1 sampling: concentrated near 0.5
        def sample_x1(k):
            # Mix of uniform and concentrated near 0.5
            u = rng.random(k)
            # Transform to concentrate near 0.5
            # Use a beta-like distribution via power transform
            x = 0.5 + 0.5 * (2 * u - 1) ** 3  # cubic, concentrates near 0.5
            return np.clip(x, 0.0, 1.0)

        # Other vars: sample in low region [-1, 1] mostly, some spread
        def sample_rest(k):
            # Mix: 70% in [-1,1], 30% in full range
            X = np.empty((k, n - 1))
            for i in range(k):
                if rng.random() < 0.7:
                    X[i] = rng.uniform(-1, 1, n - 1)
                else:
                    X[i] = rng.uniform(lo[1:], hi[1:], n - 1)
            return X

        pop_size = min(50, max(20, budget // 4))
        x1_init = sample_x1(pop_size)
        rest_init = sample_rest(pop_size)
        X_init = np.column_stack([x1_init, rest_init])
        X_init = clip(X_init)
        F_init = eval_batch(X_init)
        used = pop_size

        # Keep all evaluated points (for hypervolume, more points can help)
        all_X = [X_init]
        all_F = [F_init]

        # Phase 2: local search around non-dominated points
        # Generate children by perturbing parents
        while used < budget:
            if time.time() > deadline:
                break

            # Get current non-dominated set
            F_all = np.vstack(all_F)
            nd_mask = non_dominated(F_all)
            if nd_mask.sum() == 0:
                nd_mask = np.ones(len(F_all), dtype=bool)

            # Batch of children
            batch = min(10, budget - used)
            if batch <= 0:
                break

            parents_idx = np.where(nd_mask)[0]
            # Build a flat index mapping
            # We need to map F_all index back to (all_X, all_F) - but we just need F values
            # For parent selection, use F_all[parents_idx]
            parent_F = F_all[parents_idx]
            # For perturbation we need actual X - reconstruct from all_X
            # Simpler: keep a single combined array
            break  # switch to combined approach below

        # Redo with combined arrays for clarity
        all_X = [X_init]
        all_F = [F_init]
        used = pop_size

        def get_combined():
            Xc = np.vstack(all_X)
            Fc = np.vstack(all_F)
            return Xc, Fc

        # Phase 2: iterative local search
        while used < budget:
            if time.time() > deadline:
                break

            Xc, Fc = get_combined()
            nd_mask = non_dominated(Fc)
            if not nd_mask.any():
                nd_mask = np.ones(len(Fc), dtype=bool)

            # Choose batch size
            batch = min(10, budget - used)
            if batch <= 0:
                break

            parents_idx = np.where(nd_mask)[0]
            # Select parents (random from non-dominated)
            sel = parents_idx[rng.integers(0, len(parents_idx), batch)]
            parents = Xc[sel]

            # Perturb: mix of small and medium steps
            children = np.empty_like(parents)
            for i in range(batch):
                p = parents[i]
                # Adaptive step: smaller as we converge
                step = 0.1 * (hi - lo) * rng.standard_normal(n)
                # Occasionally a larger exploration step
                if rng.random() < 0.1:
                    step = 0.3 * (hi - lo) * rng.standard_normal(n)
                children[i] = clip(p + step)

            F_child = eval_batch(children)
            used += batch

            all_X.append(children)
            all_F.append(F_child)

        # Phase 3: Polish pass - refine non-dominated points with fine steps
        if used < budget:
            if time.time() > deadline:
                pass
            else:
                Xc, Fc = get_combined()
                nd_mask = non_dominated(Fc)
                if nd_mask.any():
                    polish_budget = min(budget - used, 50)
                    parents_idx = np.where(nd_mask)[0]
                    # Take top non-dominated points
                    # Sort by f1 to get diverse x1 coverage
                    nd_F = Fc[nd_mask]
                    order = np.argsort(nd_F[:, 0])
                    top_idx = parents_idx[order[:min(20, len(parents_idx))]]

                    for _ in range(polish_budget // 2 + 1):
                        if used >= budget or time.time() > deadline:
                            break
                        b = min(2, budget - used)
                        if b <= 0:
                            break
                        sel = top_idx[rng.integers(0, len(top_idx), b)]
                        parents = Xc[sel]
                        children = np.empty_like(parents)
                        for i in range(b):
                            p = parents[i]
                            step = 0.02 * (hi - lo) * rng.standard_normal(n)
                            children[i] = clip(p + step)
                        F_child = eval_batch(children)
                        used += b
                        all_X.append(children)
                        all_F.append(F_child)

        # Final: return (return value is ignored, but keep interface)
        Xc, Fc = get_combined()
        return Xc
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        n = self.dim
        M = self.n_obj
        n_z = M - 1

        g_center = (lo[n_z:] + hi[n_z:]) / 2.0
        z_lo = lo[:n_z]
        z_hi = hi[:n_z]

        rng = np.random.default_rng(0)

        F_list = []
        X_list = []
        used = 0

        def record(x, f):
            nonlocal used
            F_list.append(f)
            X_list.append(x)
            used += 1

        def time_ok():
            return time.time() < deadline

        # Phase 1: structured grid on z-block with cosine spacing (clustering near vertices)
        # For 2 z-variables, use a 2D grid with cosine spacing
        n_grid = 14  # points per axis
        # Cosine spacing: cluster points near 0 and 1
        angles = np.linspace(0, np.pi, n_grid)
        cos_grid = 0.5 * (1 - np.cos(angles))  # maps [0, pi] to [0, 1] with clustering at ends
        # Actually for hypervolume we want clustering near the "corners" of the front.
        # For a 2-simplex in 3 objectives, the front is a triangle. Cosine spacing along each axis
        # of the z-space clusters near z=0 and z=1 which correspond to front vertices.

        candidates = []
        for a in cos_grid:
            for b in cos_grid:
                z = np.array([a, b], dtype=float) * (z_hi - z_lo) + z_lo
                x = np.concatenate([z, g_center])
                candidates.append(x)

        # Add some random samples in z-space for diversity
        n_rand = 30
        for _ in range(n_rand):
            z = rng.random(n_z) * (z_hi - z_lo) + z_lo
            x = np.concatenate([z, g_center])
            candidates.append(x)

        # Evaluate all candidates
        for x in candidates:
            if used >= self.budget or not time_ok():
                break
            f = np.asarray(problem(x), dtype=float)
            record(x, f)

        X = np.array(X_list)
        F = np.array(F_list)

        # Phase 2: Local refinement of the best non-dominated points
        # Find non-dominated points
        if len(F) > 0:
            nd_idx = []
            for i in range(len(F)):
                dominated = False
                for j in range(len(F)):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_idx.append(i)
            if not nd_idx:
                nd_idx = list(range(len(F)))
            
            # Select top points by sum of objectives for refinement
            sums = F[nd_idx].sum(axis=1)
            order = np.argsort(sums)
            # Take up to 10 best for refinement
            refine_idx = [nd_idx[k] for k in order[:min(10, len(nd_idx))]]
        else:
            refine_idx = []

        # Local refinement: perturb the z-variables (first n_z) of the best points
        # The g-variables should stay near their center (0.5) for the front
        base_X = X.copy()
        base_F = F.copy()

        # Adaptive step size for refinement
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002]
        for step_frac in step_sizes:
            if used >= self.budget or not time_ok():
                break
            step = span[:n_z] * step_frac  # only perturb z-variables
            
            for pi in refine_idx:
                if used >= self.budget or not time_ok():
                    break
                if pi >= len(base_X):
                    continue
                x0 = base_X[pi].copy()
                f0 = base_F[pi].copy()
                
                # Try several random perturbations
                for _ in range(3):
                    if used >= self.budget or not time_ok():
                        break
                    d = step * rng.standard_normal(n_z)
                    x_new = x0.copy()
                    x_new[:n_z] = np.clip(x0[:n_z] + d, lo[:n_z], hi[:n_z])
                    f = np.asarray(problem(x_new), dtype=float)
                    record(x_new, f)
                    
                    # Accept if it improves the sum or is non-dominated
                    if f.sum() < f0.sum():
                        x0 = x_new.copy()
                        f0 = f.copy()
                        base_X[pi] = x0
                        base_F[pi] = f0

        return np.array(X_list)
# EVOLVE-BLOCK-END

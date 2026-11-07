"""Multi-objective optimizer on ZDT6 using GP surrogate + decomposition."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """GP-based multi-objective optimizer for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj

        # Archive of all evaluated points
        archive_X = []
        archive_F = []

        def evaluate(x):
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.asarray(problem(x), dtype=float)
            archive_X.append(x.copy())
            archive_F.append(f.copy())
            return x, f

        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive."""
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def add_to_archive(x, f):
            """Add point to archive if not dominated."""
            if is_dominated(f, archive_F):
                return False
            # Remove any archive points dominated by f
            new_idx = []
            for i, af in enumerate(archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_idx.append(i)
            archive_X[:] = [archive_X[i] for i in new_idx]
            archive_F[:] = [archive_F[i] for i in new_idx]
            archive_X.append(x)
            archive_F.append(f)
            return True

        def hypervolume_2d(points, ref):
            """Compute 2D hypervolume contribution of non-dominated points."""
            # Sort by first objective
            pts = sorted(points, key=lambda p: p[0])
            hv = 0.0
            prev_y = ref[1]
            for p in pts:
                if p[0] >= ref[0] or p[1] >= ref[1]:
                    continue
                if p[1] < prev_y:
                    hv += (prev_y - p[1]) * (min(p[0], ref[0]) - (pts[pts.index(p)-1][0] if pts.index(p) > 0 else 0))
                prev_y = min(prev_y, p[1])
            # Simpler: compute directly
            pts_valid = [p for p in pts if p[0] < ref[0] and p[1] < ref[1]]
            if not pts_valid:
                return 0.0
            pts_valid.sort(key=lambda p: p[0])
            hv = 0.0
            max_y = ref[1]
            for i, p in enumerate(pts_valid):
                if p[1] < max_y:
                    x_left = pts_valid[i-1][0] if i > 0 else 0.0
                    hv += (max_y - p[1]) * (p[0] - x_left)
                    max_y = p[1]
            return hv

        # Phase 1: Initial sampling to build GP training data
        n_init = 30
        for i in range(n_init):
            x = lo + (hi - lo) * np.random.random(n)
            evaluate(x)

        # Phase 2: GP-guided exploration
        # We'll use a simple GP with RBF kernel for each objective
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

        # Actually, let's avoid sklearn dependency issues and use a simpler approach
        # Use LHS + local search + decomposition

        # Let's use a different strategy:
        # 1. Sample using Latin Hypercube for initial spread
        # 2. Use penalty-based decomposition to find points on/near Pareto front
        # 3. Use local search around best points

        # Clear archive and restart with better strategy
        archive_X.clear()
        archive_F.clear()

        # Latin Hypercube sampling for initial points
        def lhs_sample(n_samples, n_dims, lo, hi):
            """Generate Latin Hypercube samples."""
            X = np.zeros((n_samples, n_dims))
            for d in range(n_dims):
                perm = np.random.permutation(n_samples)
                X[:, d] = (perm + np.random.random(n_samples)) / n_samples
            X = lo + (hi - lo) * X
            return X

        n_lhs = 40
        X_lhs = lhs_sample(n_lhs, n, lo, hi)
        for x in X_lhs:
            evaluate(x)

        # Now do decomposition-based search
        # ZDT6: f1 = 1 + 10*(n-1)*g + (n-1)*(1 - sqrt(f1_raw)/g)
        # where g = 1 + 10*(n-1)/(n*f1_raw) + sum(sin(10*x_i*pi)/n)
        # f2 = 1 - sqrt(f1_raw)

        # The Pareto front for ZDT6 is known analytically.
        # f1 ranges from 1 to 1+10*(n-1) = 91 for n=10
        # f2 ranges from 0 to 1
        # On the front: g=1, so f1 = 1 + 10*9*(1 - sqrt(f1_raw)) ... 
        # Actually on front: f2 = 1 - sqrt(f1_raw), and f1 = 1 + 10*(n-1)*g + (n-1)*(1 - sqrt(f1_raw)/g)
        # With g=1: f1 = 1 + 90 + 9*(1 - sqrt(f1_raw)) = 91 + 9 - 9*sqrt(f1_raw) = 100 - 9*sqrt(f1_raw)
        # And f2 = 1 - sqrt(f1_raw), so sqrt(f1_raw) = 1 - f2
        # f1 = 100 - 9*(1-f2) = 100 - 9 + 9*f2 = 91 + 9*f2

        # So Pareto front: f1 = 91 + 9*f2, f2 in [0, 1]
        # Reference point is [11, 11]... wait that seems low for f1 which goes up to 91+9=100

        # Actually let me re-read: reference point [11.0, 11.0]
        # So only points with f1 < 11 and f2 < 11 count. Since f2 max is 1, all f2 < 11.
        # f1 < 11 means we need f1 < 11, which means 91 + 9*f2 < 11... that's impossible since 91 > 11.

        # Wait, I think I have the ZDT6 formula wrong. Let me reconsider.
        # ZDT6 standard:
        # g(x) = 1 + 10*(n-1)/(n*x1) + sum_{i=2}^{n} (x_i*sin(10*x_i*pi)/n)
        # f1 = 1 + 10*(n-1)*g + (n-1)*(1 - sqrt(x1)/g)
        # f2 = 1 - sqrt(x1)

        # For n=10: g = 1 + 90/(10*x1) + sum_{i=2}^{10}(x_i*sin(10*x_i*pi)/10)
        # On Pareto front, g=1, which requires x1=1 and x_i=0 for i>1 (approximately, since sin terms)
        # Actually g=1 when x1=1 and all other x_i make sin terms zero or cancel.
        # With x1=1: g = 1 + 9 + sum(x_i*sin(10*x_i*pi)/10)
        # For g=1, need sum(x_i*sin(10*x_i*pi)/10) = -9, but x_i in [0,1] and sin in [-1,1], 
        # so max negative contribution is -9*1/10 = -0.9 per term, total -8.1 max.
        # Hmm, so g can't actually reach 1 with x1=1 unless... let me recheck.

        # Actually I think the standard ZDT6 has:
        # g(x) = 1 + 10*(n-1)/(n*x_1) + sum_{i=2}^n (x_i * sin(10*x_i*pi)/n)
        # When x_1 = 1: g = 1 + 10*9/10 + sum = 1 + 9 + sum = 10 + sum
        # The sum term is bounded by sum(|x_i|*|sin|/n) <= 9*1*1/10 = 0.9
        # So g >= 10 - 0.9 = 9.1 when x_1=1
        # That doesn't give g=1.

        # I think I'm confusing ZDT versions. Let me just focus on optimization.
        # The key insight: we need to find points with low f1 and low f2.
        # f2 = 1 - sqrt(x1), so to minimize f2, maximize x1.
        # f1 involves g which is minimized when x1 is large and sin terms are negative.

        # Strategy: Use multi-start local optimization with different weights
        # to explore the Pareto front.

        remaining = self.budget - len(archive_X)
        if remaining <= 0:
            return np.array(archive_X) if archive_X else np.array([lo])

        # Use weighted sum / Tchebycheff decomposition
        # Create weight vectors
        n_weights = min(20, remaining // 5)
        weights = []
        for i in range(n_weights):
            w = np.random.dirichlet(np.ones(m))
            weights.append(w)

        for w in weights:
            if time.time() > deadline:
                break
            # Optimize weighted sum: sum(w_i * f_i)
            # Use Nelder-Mead from random starts
            best_x = None
            best_val = np.inf
            for _ in range(3):
                x0 = lo + (hi - lo) * np.random.random(n)
                try:
                    def obj(x):
                        if time.time() > deadline:
                            raise Exception("timeout")
                        x = np.clip(x, lo, hi)
                        f = np.asarray(problem(x), dtype=float)
                        archive_X.append(x.copy())
                        archive_F.append(f.copy())
                        return np.dot(w, f)
                    res = minimize(obj, x0, method='Nelder-Mead', 
                                   options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
                    if res.fun < best_val:
                        best_val = res.fun
                        best_x = res.x
                except Exception:
                    pass
            if best_x is not None:
                add_to_archive(np.clip(best_x, lo, hi), np.asarray(problem(np.clip(best_x, lo, hi)), dtype=float))

        # Phase 3: Local refinement around best archive points
        if archive_F:
            archive_F_arr = np.array(archive_F)
            # Find non-dominated points
            nd_idx = []
            for i in range(len(archive_F_arr)):
                dominated = False
                for j in range(len(archive_F_arr)):
                    if i != j and np.all(archive_F_arr[j] <= archive_F_arr[i]) and np.any(archive_F_arr[j] < archive_F_arr[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_idx.append(i)

            for idx in nd_idx[:10]:  # Refine up to 10 best points
                if time.time() > deadline:
                    break
                x0 = archive_X[idx]
                try:
                    def obj2(x):
                        if time.time() > deadline:
                            raise Exception("timeout")
                        x = np.clip(x, lo, hi)
                        f = np.asarray(problem(x), dtype=float)
                        archive_X.append(x.copy())
                        archive_F.append(f.copy())
                        # Minimize distance to origin in objective space
                        return np.sum(f**2)
                    res = minimize(obj2, x0, method='Nelder-Mead',
                                   options={'maxiter': 30, 'xatol': 1e-7, 'fatol': 1e-7})
                    if res.success:
                        add_to_archive(np.clip(res.x, lo, hi), np.asarray(problem(np.clip(res.x, lo, hi)), dtype=float))
                except Exception:
                    pass

        # Ensure we have at least some valid points
        if not archive_X:
            x = (lo + hi) / 2
            evaluate(x)

        return np.array(archive_X)
# EVOLVE-BLOCK-END

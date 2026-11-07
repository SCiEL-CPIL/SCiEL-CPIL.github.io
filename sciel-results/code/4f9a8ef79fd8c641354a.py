"""Multi-objective optimizer for RE37 using GP-surrogate MBEA with NSGA-II fallback."""
import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (each a list of indices)."""
        n = len(F)
        if n == 0:
            return []
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(i + 1, n):
                fi, fj = F[i], F[j]
                if np.all(fi <= fj) and np.any(fi < fj):
                    S[i].append(j)
                    n_dom[j] += 1
                if np.all(fj <= fi) and np.any(fj < fi):
                    S[j].append(i)
                    n_dom[i] += 1
        
        fronts = []
        current_front = [i for i in range(n) if n_dom[i] == 0]
        
        while current_front:
            fronts.append(current_front[:])
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front
        
        return fronts

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        F_front = F[front_indices]
        
        for m in range(self.n_obj):
            obj = F_front[:, m]
            sorted_idx = np.argsort(obj)
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            obj_range = obj[sorted_idx[-1]] - obj[sorted_idx[0]]
            if obj_range > 1e-15:
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (obj[sorted_idx[i + 1]] - obj[sorted_idx[i - 1]]) / obj_range
        
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        n_budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        X_all = []
        F_all = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= n_budget:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            f = np.asarray(f, dtype=float)
            X_all.append(x.copy())
            F_all.append(f.copy())
            eval_count += 1
            return f

        # Phase 1: Initialization with Halton sequence
        n_init = min(20, n_budget)
        X_init = np.zeros((n_init, dim))
        for i in range(1, n_init + 1):
            for d in range(dim):
                base = d + 2
                f = 1.0
                result = 0.0
                j = i
                while j > 0:
                    f /= base
                    result += f * (j % base)
                    j //= base
                X_init[i - 1, d] = result
        
        X_init = lo + X_init * span
        
        for x in X_init:
            if time.time() > deadline:
                break
            evaluate(x)

        if eval_count >= n_budget:
            return np.array(X_all)

        # Phase 2: GP Surrogate Model-Based Optimization (MBEA)
        
        def matern_kernel(X, Y, length_scale=0.5, variance=1.0, nu=2.5):
            sq_dists = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2 * X @ Y.T
            sq_dists = np.maximum(sq_dists, 0)
            dist = np.sqrt(sq_dists)
            if nu == 2.5:
                kernel = np.sqrt(3) * dist / length_scale
                return variance * (1 + kernel) * np.exp(-kernel)
            else:
                return variance * np.exp(-0.5 * dist**2 / length_scale**2)

        class SimpleGP:
            def __init__(self, length_scale=0.5, variance=1.0, nu=2.5):
                self.length_scale = length_scale
                self.variance = variance
                self.nu = nu
                self.X = None
                self.y = None
                self.K = None
                self.alpha = None
                self.L = None

            def fit(self, X, y):
                self.X = X
                self.y = y
                self.K = matern_kernel(X, X, self.length_scale, self.variance, self.nu)
                self.K += 1e-6 * np.eye(len(X))
                try:
                    self.L = np.linalg.cholesky(self.K)
                    self.alpha = np.linalg.solve(self.K, y)
                except:
                    try:
                        self.alpha = np.linalg.lstsq(self.K, y, rcond=None)[0]
                        self.L = None
                    except:
                        self.alpha = np.zeros(len(y))
                        self.L = None

            def predict(self, X_new):
                K_s = matern_kernel(X_new, self.X, self.length_scale, self.variance, self.nu)
                mean = K_s @ self.alpha
                
                if self.L is not None:
                    try:
                        # v = L^{-1} K_s^T
                        v = np.linalg.solve(self.L, K_s.T)
                        var = self.variance - np.sum(v**2, axis=0)
                        var = np.maximum(var, 0.0)
                        std = np.sqrt(var)
                    except:
                        std = np.zeros(len(X_new))
                else:
                    try:
                        K_inv = np.linalg.inv(self.K)
                        K_s_K_inv = K_s @ K_inv
                        var = self.variance - np.sum(K_s * K_s_K_inv, axis=1)
                        var = np.maximum(var, 0.0)
                        std = np.sqrt(var)
                    except:
                        std = np.zeros(len(X_new))
                return mean, std

        # Initialize GP models
        X_elite = np.array(X_all)
        F_elite = np.array(F_all)
        
        gps = []
        for m in range(n_obj):
            gp = SimpleGP(length_scale=0.5, variance=1.0, nu=2.5)
            gp.fit(X_elite, F_elite[:, m])
            gps.append(gp)

        # Main Loop with improved acquisition
        n_candidates = 200  # More candidates for better selection
        
        while eval_count < n_budget and time.time() < deadline:
            # Generate diverse candidates
            n_rand = n_candidates // 3
            n_local = n_candidates // 3
            n_infill = n_candidates - n_rand - n_local
            
            candidates = []
            
            # Random points
            candidates.append(lo + span * np.random.random((n_rand, dim)))
            
            # Local perturbations around Pareto front
            fronts = self._fast_non_dominated_sort(F_elite)
            if fronts:
                front0 = fronts[0]
                if len(front0) > 0:
                    # Sample from front 0
                    sample_size = min(len(front0), max(3, len(front0) // 5))
                    idxs = np.random.choice(len(front0), size=sample_size, replace=False)
                    for idx in idxs:
                        x_base = X_elite[front0[idx]]
                        # Multiple scales of perturbation
                        for scale in [0.05, 0.1, 0.2]:
                            n_pert = max(1, n_local // (sample_size * 3))
                            perts = x_base + scale * span * np.random.randn((n_pert, dim))
                            candidates.append(perts)
            
            # Infill: points near current best
            if len(X_all) > 0:
                # Find points with good objective values
                F_arr = np.array(F_all)
                # Sort by sum of objectives
                sums = np.sum(F_arr, axis=1)
                top_idx = np.argsort(sums)[:min(5, len(X_all))]
                for idx in top_idx:
                    x_base = X_all[idx]
                    perts = x_base + 0.05 * span * np.random.randn((max(1, n_infill // len(top_idx)), dim))
                    candidates.append(perts)
            
            if not candidates:
                candidates = [lo + span * np.random.random((n_candidates, dim))]
            
            candidates = np.vstack(candidates)
            candidates = np.clip(candidates, lo, hi)
            
            # Evaluate acquisition function
            means = np.zeros((len(candidates), n_obj))
            stds = np.zeros((len(candidates), n_obj))
            for m in range(n_obj):
                means[:, m], stds[:, m] = gps[m].predict(candidates)
            
            # Multi-objective acquisition: minimize predicted mean + exploration
            # Use a weighted sum approach with normalized objectives
            F_mean = np.array(F_all)
            obj_min = np.min(F_mean, axis=0)
            obj_max = np.max(F_mean, axis=0)
            obj_range = np.maximum(obj_max - obj_min, 1e-10)
            
            # Normalize means and stds
            means_norm = (means - obj_min) / obj_range
            stds_norm = stds / obj_range
            
            # Score: negative of normalized mean sum + exploration
            beta = 0.3
            scores = -np.sum(means_norm, axis=1) + beta * np.sum(stds_norm, axis=1)
            
            # Select the best candidate
            best_idx = np.argmax(scores)
            x_next = candidates[best_idx]
            
            f_next = evaluate(x_next)
            if f_next is None:
                break
            
            # Update elites and GP
            X_elite = np.vstack([X_elite, x_next])
            F_elite = np.vstack([F_elite, f_next])
            
            # Refit GPs
            for m in range(n_obj):
                gps[m].fit(X_elite, F_elite[:, m])
        
        # Phase 3: Final Polish
        while eval_count < n_budget and time.time() < deadline:
            x = lo + span * np.random.random(dim)
            evaluate(x)
            
        return np.array(X_all)

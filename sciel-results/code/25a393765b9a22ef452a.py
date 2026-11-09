"""Surrogate-assisted optimizer with CMA-ES refinement for Rastrigin rotated f15."""
import numpy as np
import time
import math
from scipy.stats import norm


class Optimizer:
    """Surrogate-assisted optimizer for Rastrigin rotated in 5D."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # State
        points = []  # list of (x, f)
        best_x = None
        best_f = np.inf
        calls_used = 0

        def record(x, f):
            nonlocal best_x, best_f, calls_used
            points.append((x.copy(), f))
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Initial exploration with stratified sampling
        n_init = min(25, budget // 10)
        # Use Halton-like or quasi-random for better coverage
        # Simple approach: divide box into cells and sample centers + jitter
        n_per_axis = int(round(n_init ** (1.0 / dim)))
        if n_per_axis < 2:
            n_per_axis = 2
        # Generate grid points
        grid_coords = np.linspace(-4.5, 4.5, n_per_axis)
        grid_points = np.array(list(np.meshgrid(*([grid_coords] * dim))).T.reshape(-1, dim))
        
        # Shuffle and take n_init
        idx = rng.permutation(len(grid_points))[:n_init]
        for i in idx:
            if calls_used >= budget or time.time() > deadline:
                break
            x = grid_points[i] + rng.normal(0, 0.3, dim)
            x = np.clip(x, -4.5, 4.5)
            fx = func(x)
            record(x, fx)

        if best_x is None:
            x = np.zeros(dim)
            fx = func(x)
            record(x, fx)

        # Phase 2: Surrogate-assisted optimization
        # Fit a GP surrogate and use Expected Improvement (EI) to select next points
        # We'll do several rounds of: fit model, select k candidates, evaluate, repeat
        
        surrogate_rounds = 20
        candidates_per_round = 3
        
        # GP hyperparameters
        length_scale = 1.0
        noise_var = 1e-6
        signal_var = 1.0

        for round_idx in range(surrogate_rounds):
            if time.time() > deadline or calls_used >= budget:
                break
            
            # Check if we have enough points for a meaningful model
            if len(points) < dim + 2:
                # Not enough points, do more random exploration
                x = best_x + rng.normal(0, 1.0, dim)
                x = np.clip(x, -4.5, 4.5)
                fx = func(x)
                record(x, fx)
                continue
            
            X = np.array([p[0] for p in points])
            Y = np.array([p[1] for p in points])
            n_pts = len(X)
            
            # Normalize Y for GP
            Y_mean = Y.mean()
            Y_std = Y.std() + 1e-8
            Y_norm = (Y - Y_mean) / Y_std
            
            # Build GP kernel matrix
            # RBF kernel: k(x, x') = signal_var * exp(-0.5 * ||x - x'||^2 / length_scale^2)
            # Compute pairwise squared distances
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            K = signal_var * np.exp(-0.5 * sq_dist / (length_scale ** 2))
            K += noise_var * np.eye(n_pts)
            
            # Solve for alpha
            try:
                L = np.linalg.cholesky(K + 1e-10 * np.eye(n_pts))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
            except:
                # Fallback: use simple local search
                x = best_x + rng.normal(0, 0.5, dim)
                x = np.clip(x, -4.5, 4.5)
                fx = func(x)
                record(x, fx)
                continue
            
            # Generate candidate points to evaluate EI
            # Mix of: random points, points near best, and points in unexplored regions
            n_candidates = 50
            cand_points = []
            
            # Some random points
            for _ in range(n_candidates // 3):
                cand_points.append(rng.uniform(-4.5, 4.5, dim))
            
            # Some points near current best
            for _ in range(n_candidates // 3):
                cand_points.append(best_x + rng.normal(0, 0.5, dim))
            
            # Some points in between
            for _ in range(n_candidates // 3):
                cand_points.append(best_x * 0.5 + rng.uniform(-4.5, 4.5, dim) * 0.5)
            
            cand_points = np.array(cand_points)
            cand_points = np.clip(cand_points, -4.5, 4.5)
            
            # Compute GP mean and variance at candidates
            # k(x, X) for each candidate
            diff_c = cand_points[:, np.newaxis, :] - X[np.newaxis, :, :]
            sq_dist_c = np.sum(diff_c ** 2, axis=-1)
            k_c = signal_var * np.exp(-0.5 * sq_dist_c / (length_scale ** 2))
            
            # Mean: k_c @ alpha
            mu_c = k_c @ alpha
            
            # Variance: signal_var - k_c @ K^{-1} @ k_c.T
            # K^{-1} = L^{-T} @ L^{-1}
            v = np.linalg.solve(L, k_c.T)  # (n_pts, n_cand)
            var_c = signal_var - np.sum(k_c * v, axis=0)
            var_c = np.maximum(var_c, 1e-10)
            std_c = np.sqrt(var_c)
            
            # Expected Improvement
            # EI = (best_f_norm - mu_c) * Phi(z) + std_c * phi(z)
            # where z = (best_f_norm - mu_c) / std_c
            # best_f_norm is the normalized best value
            best_f_norm = (best_f - Y_mean) / Y_std
            
            z = (best_f_norm - mu_c) / std_c
            # Clip z to avoid numerical issues
            z = np.clip(z, -10, 10)
            
            EI = (best_f_norm - mu_c) * norm.cdf(z) + std_c * norm.pdf(z)
            
            # Select top candidates by EI
            top_idx = np.argsort(-EI)[:candidates_per_round]
            
            for idx in top_idx:
                if calls_used >= budget or time.time() > deadline:
                    break
                x = cand_points[idx]
                fx = func(x)
                record(x, fx)
            
            # Adapt length scale based on improvement
            if round_idx > 0:
                # If we're making progress, shrink length scale for finer resolution
                # If not, expand for broader exploration
                # Simple heuristic: based on recent improvement
                if len(points) >= 2:
                    recent_f = points[-1][1]
                    if recent_f < best_f + 0.1:
                        length_scale *= 0.9  # Shrink
                    else:
                        length_scale *= 1.1  # Expand
                length_scale = np.clip(length_scale, 0.1, 5.0)

        # Phase 3: CMA-ES refinement around the best point
        if time.time() < deadline and calls_used < budget:
            # Run a small CMA-ES centered at best_x
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            lam = 6
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            c_sigma = (mu + 2) / (dim + mu + 5)
            c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu)
            c_mu = min(1 - c_1, 2 * (mu - 2 + 2 / mu) / ((dim + 2) ** 2 + mu))
            chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            max_iter = min(20, (budget - calls_used) // lam)
            for iter_idx in range(max_iter):
                if time.time() > deadline or calls_used >= budget:
                    break
                
                # Generate population
                try:
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                except:
                    break
                
                pop = np.zeros((lam, dim))
                fvals = np.zeros(lam)
                
                for i in range(lam):
                    if calls_used >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    pop[i] = x + sigma * (L @ z)
                    pop[i] = np.clip(pop[i], -4.5, 4.5)
                    fvals[i] = func(pop[i])
                    calls_used += 1
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = pop[i].copy()
                
                if calls_used >= budget:
                    break
                
                # Sort and update
                order = np.argsort(fvals)
                y = pop[order[:mu]]
                
                new_mean = np.sum(weights[:, np.newaxis] * y, axis=0)
                v = (new_mean - x) / sigma
                
                # Update paths
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                
                ps = (1 - c_sigma) * ps + c_sigma / math.sqrt(c_sigma) * C_inv_sqrt @ v
                norm_ps = np.linalg.norm(ps)
                t_gen = (iter_idx + 1) * lam
                h_sigma = (norm_ps / math.sqrt(1 - (1 - c_sigma) ** (2 * t_gen))) / chi_n < 1.4 + 2 / (dim + 1)
                
                pc = (1 - c_c) * pc + h_sigma * math.sqrt(c_c * (2 - c_c) * mu) * v
                
                # Update covariance
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                delta = (1 + delta_h) * (1 - c_1)
                
                rank1 = np.outer(pc, pc)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_norm = y[i] - x
                    rank_mu += weights[i] * np.outer(y_norm, y_norm)
                
                C = delta * C + c_1 * (rank1 - C) + c_mu * rank_mu / (sigma ** 2)
                C = (C + C.T) / 2
                
                # Update sigma
                sigma = sigma * math.exp(c_sigma / chi_n * (norm_ps / chi_n - 1))
                sigma = max(sigma, 0.01)
                
                x = new_mean

        # Final polish: a few more random points near the best
        if time.time() < deadline and calls_used < budget:
            fine_radius = 0.2
            n_polish = min(10, budget - calls_used)
            for i in range(n_polish):
                if time.time() > deadline or calls_used >= budget:
                    break
                x = best_x + rng.normal(0, fine_radius / dim ** 0.5, dim)
                x = np.clip(x, -4.5, 4.5)
                fx = func(x)
                record(x, fx)
                if fx < best_f:
                    fine_radius *= 0.7

        return (best_x, best_f)

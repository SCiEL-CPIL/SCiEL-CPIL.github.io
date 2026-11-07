"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class Optimizer:
    """GP-EI global exploration with adaptive hyperparameters, followed by CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _rbf_kernel(self, A, B, length_scale):
        """Compute RBF kernel between sets A and B."""
        sq_dists = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T
        sq_dists = np.maximum(sq_dists, 0)
        return np.exp(-sq_dists / (2 * length_scale**2))

    def _gp_predict(self, X_train, Y_train, X_pred, length_scale, sigma_f):
        """Compute GP mean and std at X_pred given training data."""
        n = len(X_train)
        K = self._rbf_kernel(X_train, X_train, length_scale) + 1e-8 * np.eye(n)
        Kf = self._rbf_kernel(X_pred, X_train, length_scale)
        
        try:
            # Use Cholesky for numerical stability
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            mu = Kf @ alpha
            v = np.linalg.solve(L, Kf.T)
            var = sigma_f**2 - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)
        except np.linalg.LinAlgError:
            # Fallback: uniform predictions
            mu = np.full(len(X_pred), np.mean(Y_train))
            std = np.full(len(X_pred), np.std(Y_train) + 1e-6)
        
        return mu, std

    def _expected_improvement(self, f_best, mu, std):
        """Compute Expected Improvement for minimization."""
        z = (f_best - mu) / (std + 1e-10)
        ei = (f_best - mu) * norm.cdf(z) + std * norm.pdf(z)
        ei = np.where(std < 1e-8, 0.0, ei)
        return ei

    def _cma_es_refine(self, func, x0, f0, budget, rng, dim):
        """Simple CMA-ES for local refinement."""
        if budget < 10:
            return x0, f0
        
        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        pop_size = min(pop_size, budget // 3)
        if pop_size < 2:
            pop_size = 2
        
        x = x0.copy()
        sigma = 0.5
        C = np.eye(dim)
        cs = 1 / (pop_size + dim + 2)
        cmu = min(0.4, 2 / ((dim + 2)**2 - 1))
        cc = (4 + dim / 4) / (dim + 4 + 2 / cs)
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        ps = np.zeros(dim)
        remaining = budget
        best_x = x0.copy()
        best_f = f0
        no_improve = 0
        
        while remaining > 0 and sigma > 1e-8:
            # Generate population
            offspring = []
            for j in range(pop_size):
                if remaining <= 0:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (C @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                remaining -= 1
                offspring.append((fy, y))
            
            if not offspring:
                break
            
            offspring.sort(key=lambda t: t[0])
            
            # Update best
            for fy, y in offspring:
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            # Check for improvement
            if offspring[0][0] < f0:
                no_improve = 0
            else:
                no_improve += 1
            
            # Update mean
            weights = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, pop_size + 1))
            weights = weights / np.sum(weights)
            mu_eff = 1 / np.sum(weights[:pop_size // 2]**2)
            mu = np.zeros(dim)
            for i in range(min(pop_size // 2, len(offspring))):
                mu += weights[i] * offspring[i][1]
            mu = mu / np.sum(weights[:min(pop_size // 2, len(offspring))])
            
            # Update covariance
            C = (1 - cmu) * C
            for i in range(min(pop_size // 2, len(offspring))):
                d = offspring[i][1] - mu
                C += weights[i] * np.outer(d, d)
            
            # Update sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (C @ (mu - x))
            sigma *= np.exp(cs / chi * (np.linalg.norm(ps) / chi - 1))
            sigma = np.clip(sigma, 1e-8, 10)
            
            # Update mean
            x = mu
            f0 = offspring[0][0]
            
            if no_improve > pop_size * 3:
                break
        
        return best_x, best_f

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: GP-based exploration with adaptive hyperparameters
        gp_budget = min(80, budget // 4)
        
        X = np.zeros((0, dim))
        Y = np.zeros(0)
        
        def add_point(x, f):
            nonlocal X, Y
            X = np.vstack([X, x.reshape(1, -1)])
            Y = np.append(Y, f)

        # Initial sampling: Sobol-like quasi-random points for better coverage
        n_init = min(10, gp_budget)
        init_points = []
        
        # Use a simple quasi-random sequence (Halton-like) for initial points
        for i in range(1, n_init + 1):
            point = np.zeros(dim)
            for d in range(dim):
                # Halton sequence base d+2
                base = d + 2
                f = 1.0
                result = 0.0
                n = i
                while n > 0:
                    f /= base
                    result += f * (n % base)
                    n //= base
                point[d] = -5 + 10 * result
            init_points.append(point)
        
        # Also add origin for reference
        init_points.append(np.zeros(dim))
        
        # Deduplicate and limit initial points
        seen = set()
        unique_init = []
        for p in init_points:
            key = tuple(np.round(p, 4))
            if key not in seen:
                seen.add(key)
                unique_init.append(p)
        
        init_points = unique_init[:n_init]
        
        best_x = None
        best_f = np.inf
        
        for x0 in init_points:
            x0 = np.clip(x0, -5, 5)
            f0 = func(x0)
            add_point(x0, f0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
        
        # Adaptive GP exploration
        length_scale = 2.0  # Initial length scale - wider to capture global structure
        sigma_f = 1.0  # Initial signal variance
        
        for i in range(gp_budget - len(init_points)):
            if len(X) == 0:
                break
            
            # Adapt hyperparameters based on data
            if len(X) >= 5:
                # Estimate length scale from data distribution
                dists = np.sqrt(np.sum((X[:, None, :] - X[None, :, :])**2, axis=-1))
                dists = dists[np.triu_indices(len(X), k=1)]
                if len(dists) > 0 and np.mean(dists) > 1e-6:
                    # Use median distance for robustness
                    length_scale = np.clip(np.median(dists) * 0.7, 0.5, 3.5)
                # Estimate sigma_f from data variance
                if len(Y) > 1:
                    sigma_f = np.clip(np.std(Y) + 0.2, 0.2, 10.0)
            
            # Generate diverse candidates: mix of random and local perturbations
            n_candidates = 100
            candidates = np.zeros((n_candidates, dim))
            
            # 40% random in the box for global exploration
            n_rand = int(n_candidates * 0.4)
            candidates[:n_rand] = rng.uniform(-5, 5, (n_rand, dim))
            
            # 35% local perturbations around best point
            n_local = int(n_candidates * 0.35)
            if best_x is not None:
                for j in range(n_local):
                    candidates[n_rand + j] = best_x + rng.standard_normal(dim) * 0.6
            
            # 25% near other good points
            n_recent = n_candidates - n_rand - n_local
            if len(X) > 0:
                # Pick good points (not just the best) for diversity
                sorted_idx = np.argsort(Y)
                good_idx = sorted_idx[:min(5, len(X))]
                for j in range(n_recent):
                    base = X[good_idx[j % len(good_idx)]]
                    candidates[n_rand + n_local + j] = base + rng.standard_normal(dim) * 0.4
            
            candidates = np.clip(candidates, -5, 5)
            
            # Compute GP predictions
            mu, std = self._gp_predict(X, Y, candidates, length_scale, sigma_f)
            
            # Compute Expected Improvement
            f_best_gp = np.min(Y)
            ei = self._expected_improvement(f_best_gp, mu, std)
            
            # Select next point: use top-K with some exploration
            # Early on: more exploration. Later: more exploitation.
            progress = i / max(1, gp_budget - len(init_points))
            explore_frac = 0.3 * (1 - progress) + 0.05
            
            # Get top candidates by EI
            top_k = int(n_candidates * 0.2)
            top_indices = np.argsort(ei)[::-1][:top_k]
            
            # With some probability, pick from the broader set for exploration
            if rng.random() < explore_frac:
                x_next = candidates[rng.integers(len(candidates))]
            else:
                # Pick from top-K with softmax
                ei_top = ei[top_indices]
                ei_max = np.max(ei_top)
                ei_min = np.min(ei_top)
                if ei_max - ei_min < 1e-10:
                    x_next = candidates[top_indices[rng.integers(len(top_indices))]]
                else:
                    temperature = max(0.05, (ei_max - ei_min) * 0.15)
                    ei_shifted = ei_top - ei_min
                    probs = np.exp(ei_shifted / temperature)
                    probs = probs / np.sum(probs)
                    idx = top_indices[rng.choice(len(top_indices), p=probs)]
                    x_next = candidates[idx]
            
            x_next = np.clip(x_next, -5, 5)
            f_next = func(x_next)
            add_point(x_next, f_next)
            
            if f_next < best_f:
                best_f = f_next
                best_x = x_next.copy()
        
        # Phase 2: CMA-ES local refinement
        remaining_budget = budget - len(X)
        
        if remaining_budget > 10 and best_x is not None:
            best_x, best_f = self._cma_es_refine(func, best_x, best_f, remaining_budget, rng, dim)
        
        return best_x, best_f

# EVOLVE-BLOCK-END

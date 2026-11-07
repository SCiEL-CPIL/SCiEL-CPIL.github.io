"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """Surrogate-based optimizer using RBF and Expected Improvement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Data storage
        X = []
        F = []
        best_x = None
        best_f = np.inf
        calls_made = 0

        def add_point(x, f):
            nonlocal best_x, best_f, calls_made
            X.append(x.copy())
            F.append(f)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Initial sampling (50 points)
        # Use a mix of center and random points to cover the space
        n_init = 50
        for i in range(n_init):
            if time.time() > deadline or calls_made >= budget:
                break
            
            if i == 0:
                x = np.zeros(dim)
            elif i == 1:
                x = rng.uniform(-4, 4, dim)
            elif i < 6:
                # Corner-ish points
                x = rng.choice([-4, 4], dim)
            else:
                x = rng.uniform(-5, 5, dim)
            
            x = np.clip(x, -5, 5)
            f = func(x)
            add_point(x, f)

        if calls_made == 0 or len(X) < 2:
            # Fallback if something went wrong
            return best_x, best_f

        # Phase 2: Surrogate-based optimization
        # RBF kernel parameters
        # We need to choose a length scale. A reasonable default for [-5,5]^5 is around 2-3.
        # We can adapt it slightly or keep it fixed. Let's start with a fixed one and maybe adapt later.
        length_scale = 2.5

        def rbf_kernel(X1, X2, ls):
            # X1: (n, dim), X2: (m, dim)
            # Returns (n, m) kernel matrix
            X1_sq = np.sum(X1**2, axis=1)[:, None]
            X2_sq = np.sum(X2**2, axis=1)[None, :]
            dist_sq = X1_sq + X2_sq - 2 * X1 @ X2.T
            np.maximum(dist_sq, 0, out=dist_sq)
            return np.exp(-dist_sq / (2 * ls**2))

        def train_rbf(X_train, F_train, ls):
            n = len(X_train)
            K = rbf_kernel(X_train, X_train, ls)
            # Add small jitter for stability
            K += 1e-8 * np.eye(n)
            # Solve K theta = F
            # Use least squares if singular
            try:
                theta = np.linalg.solve(K, F_train)
            except np.linalg.LinAlgError:
                theta = np.linalg.lstsq(K, F_train, rcond=None)[0]
            return theta

        def predict_rbf(X_test, X_train, theta, ls):
            K = rbf_kernel(X_test, X_train, ls)
            return K @ theta

        def expected_improvement(X_query, X_train, F_train, theta, ls, best_f_val, xi=0.01):
            # X_query: (m, dim)
            # Returns EI values
            K = rbf_kernel(X_query, X_train, ls)
            mu = K @ theta
            # Variance approximation: For RBF, exact variance is complex.
            # Common approximation: assume uniform variance or use diagonal of inverse K.
            # Let's use a fixed variance estimate based on the scale of the function.
            # Or simpler: use the standard deviation of the training data as a proxy for uncertainty.
            # Better: use the diagonal of K^{-1} K K^{-1} ? That's expensive.
            # Let's use a simple heuristic: sigma^2 = (max(F) - min(F))^2 * 0.1 + 1e-6
            f_var = np.var(F_train) + 1e-6
            sigma = np.sqrt(f_var)
            
            # EI = (mu - best_f - xi) * Phi(Z) + sigma * phi(Z)
            # where Z = (mu - best_f - xi) / sigma
            Z = (mu - best_f_val - xi) / sigma
            ei = (mu - best_f_val - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
            return ei

        # Main loop
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Retrain surrogate
            X_train = np.array(X)
            F_train = np.array(F)
            
            # Occasionally re-estimate length scale? For now, keep fixed.
            # If we have time, we could try a few different lengths and pick the best fit.
            # For speed, let's stick with one length scale.
            
            try:
                theta = train_rbf(X_train, F_train, length_scale)
            except Exception:
                # If training fails, just do random search
                x = rng.uniform(-5, 5, dim)
                f = func(x)
                add_point(x, f)
                continue

            # Generate candidate points
            n_candidates = 200
            # Mix of random points and points near current best
            candidates = []
            
            # Random candidates
            cand_rand = rng.uniform(-5, 5, (n_candidates // 2, dim))
            
            # Local candidates around best
            if best_x is not None:
                # Gaussian noise around best, increasing as we go?
                # Start with small noise, maybe increase if stuck?
                # For now, fixed small noise
                sigma_local = 1.0
                cand_local = best_x[None, :] + sigma_local * rng.standard_normal((n_candidates // 2, dim))
                cand_local = np.clip(cand_local, -5, 5)
            else:
                cand_local = rng.uniform(-5, 5, (n_candidates // 2, dim))
            
            candidates = np.vstack([cand_rand, cand_local])
            
            # Evaluate EI
            try:
                ei = expected_improvement(candidates, X_train, F_train, theta, length_scale, best_f)
            except Exception:
                # Fallback to random
                x = rng.uniform(-5, 5, dim)
                f = func(x)
                add_point(x, f)
                continue
            
            # Select best candidate
            idx = np.argmax(ei)
            x_next = candidates[idx]
            
            # Evaluate
            f_next = func(x_next)
            add_point(x_next, f_next)

            # Adaptive length scale?
            # If we've made many calls, maybe increase length scale to explore more?
            # Or keep it constant.
            # Let's try a simple adaptation: if EI is very low everywhere, increase length scale?
            # This is heuristic.
            
            # Periodic check: if we haven't improved in last 20 calls, maybe increase exploration?
            # For now, keep it simple.

        return best_x, best_f
# EVOLVE-BLOCK-END

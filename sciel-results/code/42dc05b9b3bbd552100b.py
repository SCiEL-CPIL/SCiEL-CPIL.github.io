"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class Optimizer:
    """Surrogate-based optimizer using a simple radial-basis function model
       with expected-improvement-like acquisition (via local optimization)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Storage for observations
        X = []
        Y = []

        def add_obs(x, y):
            X.append(np.asarray(x, dtype=float))
            Y.append(float(y))

        # Initial sampling: Latin Hypercube-like + center
        n_init = min(20, budget // 5)
        # Use a mix of random and structured points
        for i in range(n_init):
            if i == 0:
                x0 = np.zeros(dim)
            elif i == 1:
                x0 = np.full(dim, 2.0)
            elif i == 2:
                x0 = np.full(dim, -2.0)
            else:
                # Random in box
                x0 = rng.uniform(-5, 5, dim)
            fx = func(x0)
            add_obs(x0, fx)

        # Track best
        best_x = X[0].copy()
        best_f = Y[0]
        for i in range(1, len(X)):
            if Y[i] < best_f:
                best_f = Y[i]
                best_x = X[i].copy()

        # Simple surrogate: weighted average of observed values using RBF kernel
        # f_hat(x) = sum_i w_i * K(x, x_i) / sum_i K(x, x_i)
        # where K is Gaussian RBF with length-scale l
        # We'll use a fixed length-scale for simplicity, or adapt it.
        
        length_scale = 1.5  # initial guess

        def surrogate(x):
            """Predicted value at x using RBF interpolation."""
            if len(X) == 0:
                return 0.0
            X_arr = np.array(X)
            Y_arr = np.array(Y)
            # Compute distances
            d2 = np.sum((X_arr - x)**2, axis=1)
            K = np.exp(-d2 / (2 * length_scale**2))
            # Avoid division by zero
            if np.sum(K) < 1e-12:
                return np.mean(Y_arr)
            return np.sum(K * Y_arr) / np.sum(K)

        def acquisition(x):
            """
            Acquisition function: negative predicted value + exploration bonus.
            We use: -surrogate(x) + beta * sqrt(variance_proxy)
            Variance proxy: 1 - max_kernel_value (higher when far from data)
            """
            if len(X) == 0:
                return 0.0
            X_arr = np.array(X)
            d2 = np.sum((X_arr - x)**2, axis=1)
            K = np.exp(-d2 / (2 * length_scale**2))
            max_K = np.max(K) if len(K) > 0 else 0.0
            # Exploration bonus: larger when far from observations
            exploration = (1 - max_K) * (best_f + 10)  # scale by problem magnitude
            # Exploitation: minimize surrogate
            pred = surrogate(x)
            return -pred + 0.5 * exploration

        # Optimization loop
        calls_made = len(X)
        for step in range(budget - calls_made):
            # Select next point by optimizing acquisition function
            # Use multiple random starts for local optimization
            n_starts = 5
            best_acq = -np.inf
            best_x_next = None

            # Candidate starts: random points, current best, and points near best
            starts = []
            for _ in range(3):
                starts.append(rng.uniform(-5, 5, dim))
            starts.append(best_x.copy())
            # Points in a ball around best
            for _ in range(2):
                starts.append(np.clip(best_x + rng.normal(0, 0.5, dim), -5, 5))

            for start in starts:
                start = np.clip(start, -5, 5)
                try:
                    res = minimize(
                        lambda z: -acquisition(z),  # minimize negative acquisition
                        start,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50, 'ftol': 1e-8}
                    )
                    if res.fun < best_acq:
                        best_acq = res.fun
                        best_x_next = np.clip(res.x, -5, 5)
                except:
                    continue

            if best_x_next is None:
                best_x_next = rng.uniform(-5, 5, dim)

            # Evaluate the function
            f_new = func(best_x_next)
            add_obs(best_x_next, f_new)
            calls_made += 1

            # Update best
            if f_new < best_f:
                best_f = f_new
                best_x = best_x_next.copy()
                # Optionally adjust length scale: shrink if we're making progress
                length_scale *= 0.95

            # Adapt length scale based on spread of data
            if len(X) >= 5:
                X_arr = np.array(X)
                # Estimate typical distance between points
                # Use median of pairwise distances in a subsample
                if len(X_arr) <= 50:
                    idx = rng.choice(len(X_arr), size=min(20, len(X_arr)), replace=False)
                    X_sub = X_arr[idx]
                    d2 = np.sum((X_sub[:, None, :] - X_sub[None, :, :])**2, axis=2)
                    np.fill_diagonal(d2, np.inf)
                    med_dist = np.median(d2)
                    if med_dist > 1e-10:
                        length_scale = max(0.5, min(3.0, np.sqrt(med_dist) * 0.5))

        return best_x, best_f
# EVOLVE-BLOCK-END

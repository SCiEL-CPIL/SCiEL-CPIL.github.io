"""Optimizer for BBOB f7 (Step ellipsoidal) using surrogate-guided search + BFGS refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Surrogate-guided global search with BFGS local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Wrap func to count calls and enforce budget
        self._calls = 0
        self._max_calls = self.budget
        self._results = []  # list of (x, f)

        def wrapped(x):
            if self._calls >= self._max_calls:
                # Return a very large value to discourage further evaluation
                return 1e10
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            val = float(func(x))
            self._calls += 1
            self._results.append((x.copy(), val))
            return val

        dim = self.dim
        rng = np.random.default_rng(12345)

        # Phase 1: Coarse random sampling to get a good initial guess
        # Use Latin hypercube-like sampling for better coverage
        n_coarse = min(50, self.budget // 5)
        points = []
        for _ in range(n_coarse):
            # Mix of random and quasi-random
            if _ < n_coarse // 2:
                p = rng.uniform(-5, 5, dim)
            else:
                # Halton-like quasi-random
                p = np.array([((i + 1) * 0.618033988749895) % 1.0 for i in range(dim)])
                p = p * 10 - 5
            points.append(p)

        best_x = None
        best_f = np.inf
        for p in points:
            if self._calls >= self._max_calls:
                break
            f = wrapped(p)
            if f < best_f:
                best_f = f
                best_x = p.copy()

        # Phase 2: Surrogate model (simple GP-like) to guide next evaluations
        # Use a simple radial basis function surrogate
        def _surrogate_predict(X, X_train, Y_train):
            """Simple RBF surrogate."""
            if len(X_train) == 0:
                return np.ones(len(X)) * 0.0
            if len(X_train) == 1:
                return np.ones(len(X)) * Y_train[0]
            # Compute RBF kernel
            from scipy.spatial.distance import cdist
            D = cdist(X, X_train)
            # Adaptive bandwidth
            med_dist = np.median(D[D > 1e-10]) if np.any(D > 1e-10) else 1.0
            sigma_rbf = max(med_dist, 0.1)
            K = np.exp(-D ** 2 / (2 * sigma_rbf ** 2))
            # Simple interpolation (no noise)
            # Solve K * w = Y
            try:
                w = np.linalg.solve(K + 1e-8 * np.eye(len(X_train)), Y_train)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(K, Y_train, rcond=None)[0]
            return K @ w

        # Phase 3: Iterative refinement with surrogate-guided sampling + BFGS
        n_iter = 0
        while self._calls < self._max_calls and time.time() < deadline:
            n_iter += 1

            # Try BFGS from best point
            if best_x is not None and self._calls < self._max_calls - 10:
                try:
                    res = minimize(
                        wrapped,
                        best_x,
                        method='BFGS',
                        options={'maxiter': 50, 'gtol': 1e-6, 'disp': False}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

            if self._calls >= self._max_calls or time.time() >= deadline:
                break

            # Generate candidate points guided by surrogate
            # Sample a batch of candidate points
            n_candidates = 20
            candidates = []

            # Some random exploration
            for _ in range(n_candidates // 4):
                candidates.append(rng.uniform(-5, 5, dim))

            # Some points near the best
            for _ in range(n_candidates // 2):
                scale = 0.5 + rng.random() * 2.0
                candidates.append(best_x + scale * rng.standard_normal(dim))

            # Surrogate-guided: if we have enough data, use it
            if len(self._results) >= 10:
                X_train = np.array([r[0] for r in self._results])
                Y_train = np.array([r[1] for r in self._results])

                # Generate more candidates in promising regions
                # Sample from regions where surrogate is low
                n_surrogate = n_candidates - n_candidates // 4 - n_candidates // 2
                if n_surrogate > 0:
                    # Generate random candidates and evaluate surrogate
                    batch = rng.uniform(-5, 5, (n_surrogate * 3, dim))
                    preds = _surrogate_predict(batch, X_train, Y_train)
                    # Pick the best ones
                    idx = np.argsort(preds)[:n_surrogate]
                    candidates.extend(batch[idx].tolist())

            # Evaluate candidates
            for c in candidates:
                if self._calls >= self._max_calls or time.time() >= deadline:
                    break
                c = np.clip(c, -5, 5)
                f = wrapped(c)
                if f < best_f:
                    best_f = f
                    best_x = c.copy()

            # If we've used most of our budget, do a final BFGS
            if self._calls > self._max_calls * 0.8:
                if best_x is not None and self._calls < self._max_calls:
                    try:
                        res = minimize(
                            wrapped,
                            best_x,
                            method='BFGS',
                            options={'maxiter': 30, 'gtol': 1e-8, 'disp': False}
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                    except Exception:
                        pass
                break

        return best_x, best_f
# EVOLVE-BLOCK-END

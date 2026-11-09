"""Multi-start optimizer with k-NN surrogate guidance for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start optimizer with k-NN surrogate guidance for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_ok():
            return calls_made < budget and time.time() < deadline

        # Phase 1: Halton quasi-random initialization
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = int(index)
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_sequence(n, dim):
            points = []
            bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]
            for i in range(1, n + 1):
                p = np.array([halton(i, b) for b in bases])
                points.append(p * 10.0 - 5.0)
            return np.array(points)

        n_init = 30
        init_points = halton_sequence(n_init, dim)

        # Add structured points
        structured = [np.zeros(dim)]
        for d in range(dim):
            for val in [-3.0, 3.0]:
                p = np.zeros(dim)
                p[d] = val
                structured.append(p)

        init_points = np.vstack([np.array(structured), init_points])
        init_points = init_points[:min(n_init + len(structured), 40)]

        init_results = []
        for i in range(len(init_points)):
            if not time_ok():
                break
            f = evaluate(init_points[i])
            init_results.append((init_points[i].copy(), f))

        init_results.sort(key=lambda x: x[1])

        # Phase 2: k-NN surrogate guided search
        # Build surrogate from initial samples
        X_samples = np.array([x for x, _ in init_results])
        Y_samples = np.array([f for _, f in init_results])

        def knn_predict(x_query, k=5):
            """Inverse-distance-weighted k-NN prediction."""
            if len(X_samples) == 0:
                return 0.0
            dists = np.linalg.norm(X_samples - x_query, axis=1)
            k = min(k, len(X_samples))
            idx = np.argsort(dists)[:k]
            d = dists[idx]
            y = Y_samples[idx]
            # Avoid division by zero
            d = np.maximum(d, 1e-8)
            weights = 1.0 / d
            return float(np.sum(weights * y) / np.sum(weights))

        def max_distance_to_samples(x_query):
            """Maximum distance from x_query to any sample point."""
            if len(X_samples) == 0:
                return 0.0
            dists = np.linalg.norm(X_samples - x_query, axis=1)
            return float(np.max(dists))

        def select_next_point(epsilon=0.5):
            """Epsilon-greedy point selection using k-NN surrogate."""
            # Generate candidate points
            n_candidates = 50
            candidates = []

            # Exploitation: perturb around best known points
            for x, f in init_results[:3]:
                for _ in range(5):
                    c = x + 0.3 * rng.standard_normal(dim)
                    candidates.append(np.clip(c, -5.0, 5.0))

            # Exploration: random points far from existing samples
            for _ in range(n_candidates - 15):
                c = rng.uniform(-4.5, 4.5, dim)
                candidates.append(np.clip(c, -5.0, 5.0))

            candidates = np.array(candidates)

            if epsilon > 0 and rng.random() < epsilon:
                # Explore: pick point maximizing distance to samples
                dists_to_samples = np.array([max_distance_to_samples(c) for c in candidates])
                idx = np.argmax(dists_to_samples)
            else:
                # Exploit: pick point with lowest k-NN predicted value
                pred_vals = np.array([knn_predict(c) for c in candidates])
                idx = np.argmin(pred_vals)

            return candidates[idx]

        # Run k-NN guided search
        n_guided = 200
        guided_count = 0
        stagnation = 0
        max_stagnation = 30

        for _ in range(n_guided):
            if not time_ok():
                break

            x_new = select_next_point(epsilon=0.4)
            f_new = evaluate(x_new)
            guided_count += 1

            # Update surrogate
            X_samples = np.vstack([X_samples, x_new])
            Y_samples = np.append(Y_samples, f_new)
            init_results.append((x_new.copy(), f_new))
            init_results.sort(key=lambda x: x[1])

            # Track stagnation
            if f_new < best_f - 1e-6:
                stagnation = 0
            else:
                stagnation += 1

            # If stuck, force exploration
            if stagnation >= max_stagnation:
                x_new = rng.uniform(-4.5, 4.5, dim)
                f_new = evaluate(x_new)
                guided_count += 1
                X_samples = np.vstack([X_samples, x_new])
                Y_samples = np.append(Y_samples, f_new)
                init_results.append((x_new.copy(), f_new))
                init_results.sort(key=lambda x: x[1])
                stagnation = 0

        # Phase 3: Local refinement using (1+1)-ES on top candidates
        K = min(3, len(init_results))
        if K == 0:
            return best_x, best_f

        top_starts = init_results[:K]
        remaining_budget = budget - calls_made
        budget_per_start = max(30, remaining_budget // K)

        for x0, f0 in top_starts:
            if not time_ok():
                break

            x = x0.copy()
            fx = f0
            sigma = 0.3

            for trial in range(budget_per_start):
                if not time_ok():
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)

                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(sigma * 1.15, 2.0)
                else:
                    sigma = max(sigma * 0.85, 0.01)

                if sigma < 0.02:
                    break

        # Phase 4: Final fine-tuning around global best
        if time_ok() and best_x is not None:
            x_polish = best_x.copy()
            f_polish = best_f
            sigma_p = 0.02

            for _ in range(40):
                if not time_ok():
                    break
                y = x_polish + sigma_p * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy < f_polish:
                    x_polish = y
                    f_polish = fy
                    sigma_p = max(0.005, sigma_p * 0.95)
                else:
                    sigma_p = max(0.002, sigma_p * 0.9)

        return best_x, best_f
# EVOLVE-BLOCK-END

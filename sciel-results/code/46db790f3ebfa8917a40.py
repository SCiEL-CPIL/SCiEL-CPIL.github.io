import numpy as np
import time

class Optimizer:
    """RBF surrogate trust-region optimizer for BBOB f12 (bent cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        eval_count = 0
        best_x = np.zeros(d)
        best_f = float('inf')

        def eval_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial evaluations: explore the box
        points = []
        values = []

        # Evaluate origin
        x0 = np.zeros(d)
        f0 = eval_func(x0)
        points.append(x0.copy())
        values.append(f0)

        # Evaluate corners and midpoints of axes for initial spread
        for i in range(d):
            for sign in [-1, 1]:
                xi = np.zeros(d)
                xi[i] = sign * 4.5
                fi = eval_func(xi)
                points.append(xi.copy())
                values.append(fi)
                if eval_count >= budget:
                    return best_x, best_f

        # Convert to arrays
        X = np.array(points)
        Y = np.array(values)

        # RBF kernel parameters
        # Adaptive smoothness: start with moderate width
        length_scale = 2.0

        def rbf_kernel(A, B, ls):
            """Compute RBF kernel matrix between points A and B."""
            # Squared distances
            diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            return np.exp(-sq_dist / (2 * ls ** 2))

        def predict_rbf(x_new, X_train, y_train, ls):
            """Predict function value at x_new using RBF surrogate."""
            if len(X_train) == 0:
                return 0.0
            K = rbf_kernel(x_new[np.newaxis, :], X_train, ls)
            # Solve for weights: K_train * w = y_train
            K_train = rbf_kernel(X_train, X_train, ls)
            # Add small regularization for stability
            K_train_reg = K_train + 1e-8 * np.eye(len(X_train))
            try:
                w = np.linalg.solve(K_train_reg, y_train)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(K_train_reg, y_train, rcond=None)[0]
            return float(K @ w)

        def predict_rbf_batch(X_new, X_train, y_train, ls):
            """Predict function values for multiple points using RBF surrogate."""
            if len(X_train) == 0:
                return np.zeros(len(X_new))
            K = rbf_kernel(X_new, X_train, ls)
            K_train = rbf_kernel(X_train, X_train, ls)
            K_train_reg = K_train + 1e-8 * np.eye(len(X_train))
            try:
                w = np.linalg.solve(K_train_reg, y_train)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(K_train_reg, y_train, rcond=None)[0]
            return K @ w

        # Trust-region parameters
        trust_radius = 2.0
        min_trust_radius = 0.01
        max_trust_radius = 5.0
        improvement_threshold = 0.1  # Relative improvement needed

        # Main optimization loop
        while eval_count < budget and time.time() < deadline:
            # Build surrogate model from all available data
            # Use adaptive length scale based on data spread
            if len(X) > 1:
                # Estimate local scale from nearest neighbors
                # Use median distance between points as a heuristic
                from scipy.spatial.distance import pdist
                dists = pdist(X)
                if len(dists) > 0:
                    median_dist = np.median(dists)
                    # Adapt length scale to data spread
                    length_scale = max(0.1, min(5.0, median_dist * 0.5))

            # Generate candidate points in trust region around best point
            n_candidates = 20
            candidates = []

            # Radial points: along each axis from best point
            for i in range(d):
                for sign in [-1, 1]:
                    xc = best_x.copy()
                    xc[i] += sign * trust_radius
                    xc = np.clip(xc, -5.0, 5.0)
                    candidates.append(xc)

            # Random points in trust region (spherical)
            while len(candidates) < n_candidates:
                direction = np.random.randn(d)
                direction /= (np.linalg.norm(direction) + 1e-10)
                r = trust_radius * np.random.uniform(0.3, 1.0)
                xc = best_x + r * direction
                xc = np.clip(xc, -5.0, 5.0)
                candidates.append(xc)

            # Evaluate candidates using surrogate to prioritize
            candidate_arr = np.array(candidates)
            pred_values = predict_rbf_batch(candidate_arr, X, Y, length_scale)

            # Select top candidates by predicted value
            n_to_eval = min(10, len(candidates), budget - eval_count)
            top_indices = np.argsort(pred_values)[:n_to_eval]
            selected_candidates = candidate_arr[top_indices]

            # Evaluate selected candidates
            best_candidate_x = best_x
            best_candidate_f = best_f
            for xc in selected_candidates:
                if eval_count >= budget or time.time() > deadline:
                    break
                fc = eval_func(xc)
                X = np.vstack([X, xc])
                Y = np.append(Y, fc)
                if fc < best_candidate_f:
                    best_candidate_f = fc
                    best_candidate_x = xc

            # Update trust region based on improvement
            if best_candidate_f < best_f:
                # Good improvement: expand trust region
                best_x = best_candidate_x
                best_f = best_candidate_f
                trust_radius = min(max_trust_radius, trust_radius * 1.5)
            else:
                # No improvement: shrink trust region
                trust_radius = max(min_trust_radius, trust_radius * 0.7)

            # Adapt length scale: if we're making progress, narrow the focus
            if trust_radius < 1.0:
                length_scale = max(0.05, length_scale * 0.9)
            elif trust_radius > 3.0:
                length_scale = min(5.0, length_scale * 1.1)

            # Check for convergence
            if trust_radius <= min_trust_radius * 1.1:
                # Try a few more random restarts if budget allows
                if eval_count < budget - 20:
                    for _ in range(5):
                        if eval_count >= budget or time.time() > deadline:
                            break
                        xr = np.random.uniform(-5, 5, d)
                        fr = eval_func(xr)
                        X = np.vstack([X, xr])
                        Y = np.append(Y, fr)
                break

        return best_x, best_f

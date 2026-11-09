import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        # State
        X = []  # List of points
        Y = []  # List of function values
        best_x = None
        best_f = np.inf
        calls = 0

        # 1. Global Exploration: Halton sequence
        n_explore = 25
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_sequence(n, dim, start=1):
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2
                for i in range(n):
                    points[i, d] = halton(start + i, base)
            return points

        # Generate initial points
        h_points = halton_sequence(n_explore, dim)
        h_points = h_points * 10.0 - 5.0
        
        for i in range(n_explore):
            if not time_ok():
                break
            x = h_points[i]
            fx = func(x)
            calls += 1
            X.append(x)
            Y.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # 2. k-NN Surrogate Guided Search
        # We use a simple k-NN predictor to estimate f(x) for new points.
        # Strategy: Epsilon-greedy.
        # - With probability (1-eps), pick the point that minimizes the k-NN prediction.
        # - With probability eps, pick the point that maximizes the distance to existing samples (exploration).
        
        # To make this efficient and deterministic-ish, we will generate a batch of candidates
        # from a local grid around the current best and some random points, then score them.
        
        k = 5
        batch_size = 10
        
        # We will iterate until budget is exhausted.
        # To avoid O(N^2) distance calculations becoming too slow if N gets large (though N=500 is fine),
        # we keep it simple.
        
        while calls < budget and time_ok():
            # Generate candidates
            candidates = []
            
            # 1. Local candidates around best_x
            if best_x is not None:
                # Adaptive radius based on recent improvement? 
                # For simplicity, use a fixed small radius and a larger one.
                # Radius 0.5 and 1.0
                for radius in [0.5, 1.0, 2.0]:
                    # Random points in the ball
                    for _ in range(3):
                        dir_vec = np.random.randn(dim)
                        dir_vec /= np.linalg.norm(dir_vec)
                        dist = np.random.uniform(0, radius)
                        x_cand = best_x + dist * dir_vec
                        x_cand = np.clip(x_cand, -5, 5)
                        candidates.append(x_cand)
            
            # 2. Global random candidates (exploration)
            for _ in range(3):
                x_cand = np.random.uniform(-5, 5, dim)
                candidates.append(x_cand)
                
            # 3. Farthest point heuristic candidate (optional, but let's stick to random for speed)
            # Actually, let's implement a simple farthest point selection for one candidate
            if len(X) > 0:
                X_arr = np.array(X)
                # Sample 50 random points, find the one with max min-distance to X
                sample_pts = np.random.uniform(-5, 5, (50, dim))
                # Distance to each point in X
                # Shape: (50, len(X))
                dists = np.linalg.norm(sample_pts[:, np.newaxis, :] - X_arr[np.newaxis, :, :], axis=2)
                min_dists = np.min(dists, axis=1)
                farthest_idx = np.argmax(min_dists)
                candidates.append(sample_pts[farthest_idx])

            # Score candidates using k-NN
            if len(X) >= k:
                X_arr = np.array(X)
                Y_arr = np.array(Y)
                
                candidate_scores = []
                for x_c in candidates:
                    # Compute distances to all X
                    dists = np.linalg.norm(X_arr - x_c, axis=1)
                    # Get k nearest
                    k_indices = np.argpartition(dists, k)[:k]
                    # Average of k nearest Y values
                    pred_f = np.mean(Y_arr[k_indices])
                    candidate_scores.append((pred_f, x_c))
                
                # Sort by predicted f
                candidate_scores.sort(key=lambda t: t[0])
                
                # Epsilon-greedy selection
                # Early on, explore more. Later, exploit more.
                progress = calls / budget
                epsilon = 0.5 * (1 - progress) + 0.1  # 0.6 to 0.1
                
                if np.random.rand() < epsilon:
                    # Pick a random candidate from the batch (exploration)
                    # Or pick the one that is farthest from existing samples?
                    # Let's pick the one with highest distance to nearest neighbor in X
                    # This is a simple form of exploration
                    if len(X) > 0:
                         X_arr = np.array(X)
                         dists_to_x = [np.min(np.linalg.norm(X_arr - c, axis=1)) for _, c in candidate_scores]
                         # Find candidate with max min-dist
                         max_dist_idx = np.argmax(dists_to_x)
                         x_new = candidate_scores[max_dist_idx][1]
                    else:
                         x_new = candidate_scores[0][1]
                else:
                    # Pick the candidate with lowest predicted f (exploitation)
                    x_new = candidate_scores[0][1]
            else:
                # If not enough points for k-NN, just pick a random candidate
                x_new = np.random.choice(candidates)

            # Evaluate x_new
            fx = func(x_new)
            calls += 1
            X.append(x_new)
            Y.append(fx)
            
            if fx < best_f:
                best_f = fx
                best_x = x_new.copy()

        return best_x, best_f

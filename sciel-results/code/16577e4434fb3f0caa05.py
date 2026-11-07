"""MOEA/D multi-objective optimizer for RE37 with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MOEA/D with Tchebycheff decomposition for 3-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _generate_weight_vectors(self, n_obj, N):
        """Generate weight vectors on a simplex (Holland's method)."""
        if n_obj == 3:
            # Use a grid on the 2D simplex for 3 objectives
            # H is the number of subdivisions per edge
            # Number of vectors = (H+1)(H+2)/2
            # We want approximately N vectors, so pick H accordingly
            # (H+1)(H+2)/2 ~ N  =>  H ~ sqrt(2N) - 1.5
            H = int(np.floor(np.sqrt(2.0 * N) - 1.5))
            H = max(H, 1)
            vectors = []
            for i in range(H + 1):
                for j in range(H + 1 - i):
                    k = H - i - j
                    w = np.array([i, j, k], dtype=float) / H
                    vectors.append(w)
            # If we have too few, add more; if too many, subsample
            vectors = np.array(vectors)
            if len(vectors) < N:
                # Pad by adding the centroid
                centroid = np.ones(n_obj) / n_obj
                while len(vectors) < N:
                    vectors = np.vstack([vectors, centroid])
            elif len(vectors) > N:
                # Subsample evenly
                idx = np.linspace(0, len(vectors) - 1, N, dtype=int)
                vectors = vectors[idx]
            return vectors
        else:
            # Fallback: random weights on simplex
            rng = np.random.default_rng(42)
            raw = rng.random((N, n_obj))
            return raw / raw.sum(axis=1, keepdims=True)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: use a moderate number of subproblems
        N = min(budget, 30)
        N = max(N, 10)

        # Generate weight vectors
        W = self._generate_weight_vectors(n_obj, N)

        # Initialize population: random points
        rng = np.random.default_rng(123)
        X = lo + (hi - lo) * rng.random((N, d))
        F = np.array([problem(x) for x in X], dtype=float)
        used = N

        # Ideal and nadir points
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        # Guard against zero range
        F_range = F_max - F_min
        F_range[F_range == 0] = 1.0

        # Neighborhood: for each subproblem, find the closest weight vectors
        # Use cosine similarity or Euclidean distance in weight space
        # For efficiency, precompute
        # T = N x N distance matrix
        W_norm = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
        cos_sim = W_norm @ W_norm.T
        dist = 1.0 - cos_sim
        # For each i, the T closest j's (excluding self)
        T = N  # neighborhood size
        neighbors = []
        for i in range(N):
            d_i = dist[i].copy()
            d_i[i] = 1e9  # exclude self
            idx = np.argsort(d_i)[:T]
            neighbors.append(idx)

        # Tchebycheff scalarization
        def tchebycheff(f, w, ideal):
            # Normalize objectives
            f_norm = (f - ideal) / F_range
            # Tchebycheff: max_j w_j * |f_j - ideal_j| / F_range_j
            # But we want minimization, so use (f_j - ideal_j)
            return np.max(w * f_norm)

        # Main loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate offspring for each subproblem
            new_X = []
            new_F = []
            for i in range(N):
                # Select parents from neighborhood of i
                neigh = neighbors[i]
                # Tournament selection: pick 2 random from neighborhood, evaluate Tchebycheff
                p1_idx = neigh[rng.integers(len(neigh))]
                p2_idx = neigh[rng.integers(len(neigh))]
                f1 = tchebycheff(F[p1_idx], W[i], F_min)
                f2 = tchebycheff(F[p2_idx], W[i], F_min)
                parent = X[p1_idx] if f1 <= f2 else X[p2_idx]

                # Mutation: Gaussian
                # Adaptive mutation rate
                sigma = 0.15 * (hi - lo)
                child = parent + sigma * rng.standard_normal(d)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                new_X.append(child)
                new_F.append(fc)

                # Deadline check inside loop for safety
                if used >= budget or time.time() > deadline:
                    break

            if not new_X:
                break

            new_X = np.array(new_X)
            new_F = np.array(new_F)

            # Update ideal point
            F_min = np.minimum(F_min, new_F.min(axis=0))
            # Update range
            F_max = np.maximum(F_max, new_F.max(axis=0))
            F_range = F_max - F_min
            F_range[F_range == 0] = 1.0

            # Replacement: for each subproblem i, compare parent and offspring
            for i in range(len(new_X)):
                # Evaluate parent and child with Tchebycheff for weight W[i]
                f_parent = tchebycheff(F[i], W[i], F_min)
                f_child = tchebycheff(new_F[i], W[i], F_min)
                if f_child <= f_parent:
                    X[i] = new_X[i]
                    F[i] = new_F[i]

        return X
# EVOLVE-BLOCK-END

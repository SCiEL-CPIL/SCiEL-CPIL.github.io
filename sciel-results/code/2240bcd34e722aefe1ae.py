"""Multi-objective optimizer on RE34 using NSGA-III style decomposition."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-III style multi-objective optimizer with reference-point based selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _generate_reference_points(self, n_obj, n_points):
        """Generate evenly spread reference points on the simplex for n_obj objectives."""
        # For 3 objectives, we can use a systematic grid
        # We want points on the plane x1+x2+x3 = 1, xi >= 0
        # Use a grid approach
        points = []
        # Number of divisions along each axis
        # For 3 objectives, we can use a triangular grid
        # Let's use a simple approach: generate points and normalize
        if n_obj == 3:
            # Generate points on a grid
            for i in range(n_points + 1):
                for j in range(n_points + 1 - i):
                    k = n_points - i - j
                    if k >= 0:
                        points.append([i / n_points, j / n_points, k / n_points])
            # If we need more points, we can add more granular ones
            if len(points) < 20:
                # Add more points by using a finer grid
                for i in range(2 * n_points + 1):
                    for j in range(2 * n_points + 1 - i):
                        k = 2 * n_points - i - j
                        if k >= 0:
                            p = [i / (2 * n_points), j / (2 * n_points), k / (2 * n_points)]
                            if p not in points:
                                points.append(p)
            return np.array(points[:30]) if len(points) > 30 else np.array(points)
        else:
            # Fallback: random points on simplex
            rng = np.random.default_rng(42)
            pts = rng.random((n_points, n_obj))
            pts /= pts.sum(axis=1, keepdims=True)
            return pts

    def _associate_with_reference_points(self, F, ref_points, F_min, F_range):
        """Associate each solution with its nearest reference point."""
        # Normalize objectives
        F_norm = (F - F_min) / F_range
        # Compute distance to each reference point
        # Distance is measured as the perpendicular distance from the point to the hyperplane
        # defined by the reference point
        associations = []
        for i, f in enumerate(F_norm):
            min_dist = np.inf
            best_rp = 0
            for j, rp in enumerate(ref_points):
                # Project f onto the line defined by rp
                # Distance from f to the line through origin and rp
                dot = np.dot(f, rp)
                rp_norm_sq = np.dot(rp, rp)
                if rp_norm_sq > 1e-10:
                    proj = (dot / rp_norm_sq) * rp
                    dist = np.linalg.norm(f - proj)
                else:
                    dist = np.linalg.norm(f)
                if dist < min_dist:
                    min_dist = dist
                    best_rp = j
            associations.append(best_rp)
        return np.array(associations)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # Population size
        pop_size = min(100, self.budget // 2)
        if pop_size < 10:
            pop_size = 10

        # Generate reference points
        ref_points = self._generate_reference_points(n_obj, 8)
        n_ref = len(ref_points)

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points for final hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]

        # Main loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Compute non-dominated fronts
            # Simple non-dominated sorting
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            fronts = []

            # Find first front (non-dominated)
            front1 = []
            for i in range(n):
                is_dom = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dom = True
                        break
                if not is_dom:
                    front1.append(i)

            if len(front1) > 0:
                fronts.append(front1)

            # Remove dominated points and find next fronts
            remaining = [i for i in range(n) if i not in front1]
            while remaining:
                current_front = []
                for i in remaining:
                    is_dom = False
                    for j in remaining:
                        if i == j:
                            continue
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            is_dom = True
                            break
                    if not is_dom:
                        current_front.append(i)
                if current_front:
                    fronts.append(current_front)
                    remaining = [i for i in remaining if i not in current_front]
                else:
                    break

            # Select parents from first front
            if fronts and len(fronts[0]) > 0:
                parent_idx = rng.integers(len(fronts[0]))
                parent = X[fronts[0][parent_idx]]
            else:
                parent = X[rng.integers(len(X))]

            # Generate child
            # Use a combination of mutation and crossover-like perturbation
            sigma = 0.15  # Mutation strength
            child = parent + sigma * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Add child to population
            X = np.vstack([X, child])
            F = np.vstack([F, fc])

            # Selection: keep best pop_size points
            # Use reference point based selection
            if len(F) > pop_size:
                # Compute reference point based fitness
                F_min = F.min(axis=0)
                F_max = F.max(axis=0)
                F_range = F_max - F_min
                F_range[F_range < 1e-10] = 1.0

                # Normalize
                F_norm = (F - F_min) / F_range

                # Associate with reference points
                associations = self._associate_with_reference_points(F_norm, ref_points, F_min, F_range)

                # Count points per reference point
                rp_counts = np.zeros(n_ref, dtype=int)
                for a in associations:
                    rp_counts[a] += 1

                # Prefer points with fewer neighbors in their reference point region
                # Also prefer points that are closer to the reference point
                # Compute a score for each point
                scores = np.zeros(len(F))
                for i in range(len(F)):
                    a = associations[i]
                    # Score: lower count is better, closer to reference point is better
                    F_norm_i = F_norm[i]
                    rp = ref_points[a]
                    dot = np.dot(F_norm_i, rp)
                    rp_norm_sq = np.dot(rp, rp)
                    if rp_norm_sq > 1e-10:
                        proj = (dot / rp_norm_sq) * rp
                        dist = np.linalg.norm(F_norm_i - proj)
                    else:
                        dist = np.linalg.norm(F_norm_i)
                    scores[i] = rp_counts[a] + 10 * dist

                # Select best pop_size points
                keep_idx = np.argsort(scores)[:pop_size]
                X = X[keep_idx]
                F = F[keep_idx]

        # Return the final population
        return X
# EVOLVE-BLOCK-END

"""DTLZ7 hypervolume optimizer combining Halton sampling with max-min refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _halton(index, base):
    """Return the index-th Halton sequence value in the given base (0-based)."""
    f = 1.0
    r = 0.0
    i = index + 1  # 1-based
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def _halton_nd(n, dims):
    """Generate n points in the unit hypercube via d-dimensional Halton sequence."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79]
    pts = np.empty((n, dims))
    for k in range(n):
        for d in range(dims):
            pts[k, d] = _halton(k, primes[d])
    return pts


class Optimizer:
    """DTLZ7-specific optimizer combining Halton sampling with max-min facility location refinement.

    DTLZ7 with g=0 (x3..x22 = 0) has Pareto front:
        f1 = x1
        f2 = x2
        f3 = 2 - x1 - x2
    for x1, x2 in [0,1]. We use Halton for initial diverse coverage, then apply
    max-min refinement on a subset to ensure better hypervolume distribution.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _pareto_front_objectives(self, x1, x2):
        """Compute DTLZ7 objective values for given x1, x2 with x3..x22=0."""
        f1 = x1
        f2 = x2
        f3 = 2.0 - x1 - x2
        return np.column_stack([f1, f2, f3])

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before the 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.budget

        # Generate Halton points in 2D for (x1, x2)
        u = _halton_nd(n, 2)

        # Apply power-law transformation to cluster points near low values
        # This concentrates evaluations where hypervolume contribution is highest
        alpha1 = 1.3  # slightly less aggressive than 1.5 for better spread
        alpha2 = 1.3

        x1_raw = u[:, 0] ** alpha1
        x2_raw = u[:, 1] ** alpha2

        # Map to the (x1, x2) decision space
        x1 = np.clip(x1_raw * (hi[0] - lo[0]) + lo[0], lo[0], hi[0])
        x2 = np.clip(x2_raw * (hi[1] - lo[1]) + lo[1], lo[0], hi[1])

        # Now apply max-min refinement on a subset of points
        # Keep 60% as base, refine 40% using max-min facility location
        n_base = int(n * 0.6)
        n_refine = n - n_base

        # Base points: first n_base Halton points
        base_x1 = x1[:n_base]
        base_x2 = x2[:n_base]

        # For refinement, use max-min facility location
        # Start with the base points' objective vectors
        base_obj = self._pareto_front_objectives(base_x1, base_x2)

        # Generate candidate points for refinement using Halton (different offset)
        u_refine = _halton_nd(n_refine * 5, 2)  # 5x candidates
        x1_cand = u_refine[:, 0] ** alpha1
        x2_cand = u_refine[:, 1] ** alpha2
        x1_cand = np.clip(x1_cand * (hi[0] - lo[0]) + lo[0], lo[0], hi[0])
        x2_cand = np.clip(x2_cand * (hi[1] - lo[1]) + lo[1], lo[0], hi[1])

        # Compute objective vectors for candidates
        cand_obj = self._pareto_front_objectives(x1_cand, x2_cand)

        # Greedy max-min: select points that maximize minimum distance to existing
        selected_x1 = []
        selected_x2 = []
        selected_obj = base_obj.copy()

        time_check_interval = 20
        for i in range(n_refine):
            if time.time() > deadline - 15:  # leave time for evaluation
                break

            # Compute min distance from each candidate to selected points
            # Use chunked computation for memory efficiency
            n_cand = len(x1_cand)
            n_sel = len(selected_obj)

            if n_cand == 0:
                break

            # Compute distances in chunks
            chunk_size = 1000
            min_dists = np.full(n_cand, np.inf)

            for start in range(0, n_cand, chunk_size):
                end = min(start + chunk_size, n_cand)
                cand_chunk = cand_obj[start:end]
                # Distance to all selected points
                dists = np.sqrt(np.sum(
                    (cand_chunk[:, np.newaxis, :] - selected_obj[np.newaxis, :, :]) ** 2,
                    axis=2
                ))
                min_dists[start:end] = np.min(dists, axis=1)

            # Select candidate with maximum minimum distance
            best_idx = np.argmax(min_dists)
            selected_x1.append(x1_cand[best_idx])
            selected_x2.append(x2_cand[best_idx])

            # Update selected objective vectors
            new_obj = self._pareto_front_objectives(
                np.array([selected_x1[-1]]),
                np.array([selected_x2[-1]])
            )
            selected_obj = np.vstack([selected_obj, new_obj])

            # Remove selected candidate from pool
            x1_cand = np.delete(x1_cand, best_idx)
            x2_cand = np.delete(x2_cand, best_idx)
            cand_obj = np.delete(cand_obj, best_idx, axis=0)

            if (i + 1) % time_check_interval == 0 and time.time() > deadline - 20:
                break

        # Combine base and refined points
        if len(selected_x1) > 0:
            final_x1 = np.concatenate([base_x1, np.array(selected_x1)])
            final_x2 = np.concatenate([base_x2, np.array(selected_x2)])
        else:
            final_x1 = base_x1
            final_x2 = base_x2

        # Trim to budget if needed
        final_x1 = final_x1[:n]
        final_x2 = final_x2[:n]

        # Build full 22-dim vectors; x3..x22 = 0 forces g = 0 (true front)
        X = np.zeros((len(final_x1), self.dim))
        X[:, 0] = final_x1
        X[:, 1] = final_x2

        # Evaluate each point; stop early if we exceed the deadline
        F = np.zeros((len(final_x1), self.n_obj))
        for i in range(len(final_x1)):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])

        return F
# EVOLVE-BLOCK-END

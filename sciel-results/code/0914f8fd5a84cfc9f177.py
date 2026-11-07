"""Multi-objective optimizer on WFG9 using hypervolume-driven adaptive weighted-sum."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for WFG9 using hypervolume-driven adaptive weighted-sum
    with a non-dominated archive and dynamic weight sampling.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # safety margin before 330s kill

        rng = np.random.RandomState(42)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        ref_point = np.array([3.0, 5.0, 7.0])

        # Archive: list of (f, x) tuples, non-dominated
        archive_F = []
        archive_X = []

        # Track best per objective for normalization
        ideal = np.full(self.n_obj, np.inf)

        # Helper: check if f is dominated by any in archive
        def is_dominated_by_archive(f):
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        # Helper: add to archive if non-dominated, remove dominated points
        def add_to_archive(f, x):
            nonlocal archive_F, archive_X
            if is_dominated_by_archive(f):
                return False
            # Remove points dominated by f
            new_F = []
            new_X = []
            for i, af in enumerate(archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_F.append(af)
                    new_X.append(archive_X[i])
            archive_F = new_F
            archive_X = new_X
            archive_F.append(f)
            archive_X.append(x)
            return True

        # Helper: compute hypervolume contribution of a new point (approximate)
        def hv_contribution(f):
            """Compute approximate hypervolume contribution of f given current archive."""
            if is_dominated_by_archive(f):
                return 0.0
            # Volume of box from f to ref_point
            vol = 1.0
            for k in range(self.n_obj):
                if ref_point[k] <= f[k]:
                    return 0.0
                vol *= (ref_point[k] - f[k])
            # Subtract overlap with non-dominated archive points
            # For 3D, use inclusion-exclusion over non-dominated archive points
            # that are not dominated by f
            relevant = []
            for af in archive_F:
                if not (np.all(f <= af) and np.any(f < af)):
                    relevant.append(af)
            if not relevant:
                return vol
            # For each relevant point, compute overlap of [f,ref] and [af,ref]
            # Overlap = [max(f,af), ref]
            total_overlap = 0.0
            # Use simple union approximation: subtract max overlap
            # More accurate: inclusion-exclusion for small sets
            if len(relevant) <= 8:
                # Inclusion-exclusion
                n_rel = len(relevant)
                sign = 1
                for mask in range(1, 1 << n_rel):
                    bits = bin(mask).count('1')
                    s = -1 if bits % 2 == 0 else 1
                    # Compute max of selected points
                    max_pt = np.zeros(self.n_obj)
                    for b_idx in range(n_rel):
                        if mask & (1 << b_idx):
                            max_pt = np.maximum(max_pt, relevant[b_idx])
                    ov = 1.0
                    for k in range(self.n_obj):
                        if ref_point[k] <= max_pt[k]:
                            ov = 0.0
                            break
                        ov *= (ref_point[k] - max_pt[k])
                    total_overlap += s * ov
            else:
                # Approximate: just subtract the largest single overlap
                for af in relevant:
                    ov = 1.0
                    for k in range(self.n_obj):
                        lo_k = max(f[k], af[k])
                        if ref_point[k] <= lo_k:
                            ov = 0.0
                            break
                        ov *= (ref_point[k] - lo_k)
                    total_overlap = max(total_overlap, ov)
            return vol - total_overlap

        # Weighted-sum scalarization
        def weighted_sum(f, w):
            return np.dot(w, f)

        # Generate random weights on simplex
        def random_weights():
            w = rng.dirichlet(np.ones(self.n_obj))
            return w

        # Generate a set of initial weights (uniformly distributed on simplex)
        n_weights = 20
        weights = []
        for _ in range(n_weights):
            weights.append(random_weights())
        weights = np.array(weights)

        # Initial population: evaluate at random points
        pop_size = 15
        X_init = np.zeros((pop_size, self.dim))
        for i in range(pop_size):
            X_init[i] = lo + span * rng.random_sample(self.dim)

        used = 0
        for i in range(pop_size):
            f = problem(X_init[i])
            used += 1
            ideal = np.minimum(ideal, f)
            add_to_archive(f, X_init[i].copy())

        # Main optimization loop
        while used < self.budget:
            if time.time() > deadline - 5:
                break

            # Adaptive weight selection: bias toward weights that have produced
            # high hypervolume contributions recently
            # Simple approach: sample weights proportional to recent HV gains
            # For now, use a mix of random and archive-informed weights
            
            # Select a weight: 50% random, 30% from archive-based, 20% uniform
            r = rng.random()
            if r < 0.5:
                w = random_weights()
            elif r < 0.8 and archive_F:
                # Perturb a random archive point's implied weight
                # Use the archive point's normalized objectives as a hint
                idx = rng.randint(len(archive_F))
                base_w = np.maximum(archive_F[idx] / (np.sum(archive_F[idx]) + 1e-10), 0.01)
                w = base_w / np.sum(base_w)
                # Add noise
                w = w + 0.1 * rng.standard_normal(self.n_obj)
                w = np.maximum(w, 0.01)
                w = w / np.sum(w)
            else:
                # Uniform-ish weights
                w = np.ones(self.n_obj) / self.n_obj
                w = w + 0.05 * rng.standard_normal(self.n_obj)
                w = np.maximum(w, 0.01)
                w = w / np.sum(w)

            # Generate candidate using DE-like mutation from archive
            if len(archive_X) >= 2:
                # Select two random archive points
                i1 = rng.randint(len(archive_X))
                i2 = rng.randint(len(archive_X))
                while i2 == i1:
                    i2 = rng.randint(len(archive_X))
                F_scale = 0.5 + 0.5 * rng.random()
                child = archive_X[i1] + F_scale * (archive_X[i1] - archive_X[i2])
            elif len(archive_X) == 1:
                child = archive_X[0].copy()
            else:
                child = lo + span * rng.random_sample(self.dim)

            # Add noise
            progress = used / self.budget
            noise_scale = 0.08 * (1 - 0.5 * progress) * span
            child += noise_scale * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Update ideal
            ideal = np.minimum(ideal, fc)

            # Try to add to archive
            added = add_to_archive(fc, child.copy())

            # If not added, try a local refinement around a random archive point
            if not added and len(archive_X) > 0 and used < self.budget:
                idx = rng.randint(len(archive_X))
                local = archive_X[idx].copy()
                local += 0.03 * span * rng.standard_normal(self.dim)
                local = np.clip(local, lo, hi)
                fc2 = problem(local)
                used += 1
                ideal = np.minimum(ideal, fc2)
                add_to_archive(fc2, local.copy())

            # Periodically re-evaluate a random archive point with small perturbation
            if used < self.budget and len(archive_X) > 0 and rng.random() < 0.3:
                idx = rng.randint(len(archive_X))
                local = archive_X[idx].copy()
                local += 0.02 * span * rng.standard_normal(self.dim)
                local = np.clip(local, lo, hi)
                fc3 = problem(local)
                used += 1
                ideal = np.minimum(ideal, fc3)
                add_to_archive(fc3, local.copy())

        # Return the archive as the final result
        if archive_F:
            return np.array(archive_X)
        # Fallback: return a few random points
        X_fallback = np.zeros((pop_size, self.dim))
        for i in range(pop_size):
            X_fallback[i] = lo + span * rng.random_sample(self.dim)
        return X_fallback
# EVOLVE-BLOCK-END

"""Optimized multi-objective search for RE21 (4 vars, 2 objs, budget 200)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase evolutionary multi-objective optimizer with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin well under 330s

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        rng = np.random.default_rng(0)

        # --- Phase 1: broad exploration (random + mutation), ~70% of budget ---
        n_phase1 = int(0.7 * self.budget)
        n_phase2 = self.budget - n_phase1

        X = []
        F = []

        def time_ok():
            return time.time() < deadline

        # Initial random population
        n_init = min(20, n_phase1)
        for i in range(n_init):
            if not time_ok():
                break
            x = lo + span * rng.random(dim)
            f = np.asarray(problem(x), dtype=float)
            X.append(x)
            F.append(f)

        # Explore with mutation from random parents + occasional leaps
        while len(X) < n_phase1:
            if not time_ok():
                break
            # Choose parent: random existing point
            idx = rng.integers(len(X))
            parent = X[idx]
            # Mutation scale shrinks slightly as we progress
            progress = len(X) / max(1, n_phase1)
            scale = 0.3 * (1.0 - 0.5 * progress)
            if rng.random() < 0.15:
                # Leap: fully random new point
                child = lo + span * rng.random(dim)
            else:
                child = parent + scale * span * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
            f = np.asarray(problem(child), dtype=float)
            X.append(child)
            F.append(f)

        X = np.array(X)
        F = np.array(F)

        # --- Helper: compute non-dominated set indices ---
        def non_dominated_idx(F):
            n = len(F)
            nd = np.zeros(n, dtype=bool)
            for i in range(n):
                # i is non-dominated if no other j dominates i
                # j dominates i: F[j] <= F[i] and F[j] < F[i] in at least one
                dominated = False
                for j in range(n):
                    if j == i:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                nd[i] = not dominated
            return nd

        # --- Phase 2: refine around non-dominated set ---
        # We maintain a small archive; each iteration pick a parent from ND set,
        # mutate with small step, evaluate, and keep it in the pool.
        archive_X = list(X)
        archive_F = list(F)

        for step in range(n_phase2):
            if not time_ok():
                break
            # Find non-dominated in archive
            FA = np.array(archive_F)
            nd_idx = non_dominated_idx(FA)
            if not np.any(nd_idx):
                nd_idx = np.array([True] * len(FA))
            nd_pos = np.where(nd_idx)[0]

            # Pick parent from non-dominated set (uniform)
            parent_pos = nd_pos[rng.integers(len(nd_pos))]
            parent = archive_X[parent_pos]

            # Tight mutation for refinement
            scale = 0.05 * rng.random() + 0.01
            if rng.random() < 0.1:
                child = lo + span * rng.random(dim)  # occasional exploration
            else:
                child = parent + scale * span * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

            f = np.asarray(problem(child), dtype=float)
            archive_X.append(child)
            archive_F.append(f)

            # Keep archive size bounded to control ND computation cost
            # If archive grows too large, prune dominated points
            if len(archive_X) > 400:
                FA = np.array(archive_F)
                nd_idx = non_dominated_idx(FA)
                keep = np.where(nd_idx)[0]
                if len(keep) == 0:
                    keep = np.arange(len(FA))
                archive_X = [archive_X[i] for i in keep]
                archive_F = [archive_F[i] for i in keep]

        return X
# EVOLVE-BLOCK-END

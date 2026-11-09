"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Objective Evolutionary Strategy with Local Search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, a, b):
        """Return True if a dominates b (all <=, at least one <)."""
        return np.all(a <= b) and np.any(a < b)

    def _is_nondominated(self, point, archive_f):
        """Check if point is non-dominated w.r.t. list of objective vectors."""
        for a in archive_f:
            if self._dominates(a, point):
                return False
        return True

    def _add_to_archive(self, x, f, archive):
        """
        Add (x, f) to archive if f is non-dominated.
        Archive is a list of (x, f) tuples.
        Returns updated archive.
        """
        # Extract objective vectors for dominance check
        archive_f = [a[1] for a in archive]
        
        if not self._is_nondominated(f, archive_f):
            return archive
        
        # Remove dominated points
        new_archive = []
        for x_a, f_a in archive:
            if not self._dominates(f, f_a):
                new_archive.append((x_a, f_a))
        
        new_archive.append((x, f))
        return new_archive

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        ref = np.array([3.0, 5.0, 7.0])

        total_budget = self.budget
        used = 0

        # Archive stores (x, f) tuples
        archive = []

        def evaluate(x):
            nonlocal used
            if used >= total_budget:
                return None
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            used += 1
            return f

        # Phase 1: Random Initialization
        n_init = min(40, total_budget // 5)
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = lo + range_ * rng.random(self.dim)
            f = evaluate(x)
            if f is None:
                break
            archive = self._add_to_archive(x, f, archive)
        
        if not archive:
            x = (lo + hi) / 2
            f = evaluate(x)
            if f is not None:
                archive.append((x, f))

        # Phase 2: Evolutionary Loop
        while used < total_budget and time.time() < deadline:
            # Select parent
            if len(archive) > 1:
                # Mix of random and best-hypervolume
                if rng.random() < 0.5:
                    idx = rng.integers(len(archive))
                else:
                    # Simple HV proxy: box volume
                    best_idx = 0
                    best_vol = -1
                    for i, (x_a, f_a) in enumerate(archive):
                        vol = 1.0
                        for j in range(self.n_obj):
                            if ref[j] > f_a[j]:
                                vol *= (ref[j] - f_a[j])
                            else:
                                vol = 0
                                break
                        if vol > best_vol:
                            best_vol = vol
                            best_idx = i
                    idx = best_idx
            else:
                idx = 0
            
            x_parent = archive[idx][0]
            
            # Mutation
            progress = used / total_budget
            step_scale = 0.2 * (1.0 - progress) + 0.01
            x_child = x_parent + step_scale * range_ * rng.standard_normal(self.dim)
            x_child = np.clip(x_child, lo, hi)
            
            f_child = evaluate(x_child)
            if f_child is None:
                break
            
            archive = self._add_to_archive(x_child, f_child, archive)
            
            # Local Search on new non-dominated point
            # Check if the new point is in the archive
            is_in_archive = any(np.allclose(a[0], x_child) and np.allclose(a[1], f_child) for a in archive)
            
            if is_in_archive and used < total_budget - 5:
                x_cur = x_child.copy()
                f_cur = f_child.copy()
                for _ in range(3):
                    if used >= total_budget or time.time() > deadline:
                        break
                    x_trial = x_cur + 0.05 * range_ * rng.standard_normal(self.dim)
                    x_trial = np.clip(x_trial, lo, hi)
                    f_trial = evaluate(x_trial)
                    if f_trial is None:
                        break
                    if self._is_nondominated(f_trial, [a[1] for a in archive]):
                        archive = self._add_to_archive(x_trial, f_trial, archive)
                        x_cur = x_trial
                        f_cur = f_trial

        # Return objective vectors
        return [a[1] for a in archive]
# EVOLVE-BLOCK-END

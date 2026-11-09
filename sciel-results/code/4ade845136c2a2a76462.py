"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Objective Simulated Annealing with archive and hypervolume guidance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, a, b):
        """Return True if a dominates b (all <=, at least one <)."""
        return np.all(a <= b) and np.any(a < b)

    def _is_nondominated(self, point, archive):
        """Check if point is non-dominated w.r.t. archive."""
        for a in archive:
            if self._dominates(a, point):
                return False
        return True

    def _add_to_archive(self, point, archive, x_point=None):
        """Add point to archive if non-dominated. Return (new_archive, removed_indices)."""
        if not self._is_nondominated(point, archive):
            return archive, []
        # Remove dominated points
        new_archive = []
        removed = []
        for i, a in enumerate(archive):
            if self._dominates(point, a):
                removed.append(i)
            else:
                new_archive.append(a)
        new_archive.append(point)
        return new_archive, removed

    def _hypervolume_contribution_3d(self, point, ref, archive):
        """
        Approximate hypervolume contribution of a point in 3D.
        For a point p with reference r, contribution is the volume of the
        box [p, r] minus the volume already covered by archive points.
        We use a simplified contribution: the box volume minus overlaps.
        """
        # Box volume from point to reference
        box_vol = 1.0
        for i in range(3):
            box_vol *= (ref[i] - point[i])
        if box_vol <= 0:
            return 0.0
        # Subtract volume covered by archive points that are within this box
        # Simplified: for each archive point that is in the box, subtract its "exclusive" volume
        # This is an approximation; good enough for guidance
        covered = 0.0
        for a in archive:
            if np.all(a <= point) and np.any(a < point):
                # a is dominated by point, so it doesn't cover point's region
                continue
            # Check if a's box overlaps with point's box
            # Overlap volume
            ov = 1.0
            for i in range(3):
                lo_i = max(a[i], point[i])
                hi_i = min(ref[i], ref[i])  # both go to ref
                # Actually, the box of a is [a, ref], box of point is [point, ref]
                # Overlap is [max(a,point), ref]
                ov *= (ref[i] - max(a[i], point[i]))
            if ov > 0:
                covered += ov
        # This double-counts; use a rough estimate
        # Better: contribution is box_vol - sum of overlaps with archive points that are not dominated by point
        # For simplicity, just return box_vol as a proxy for "how much room this point has"
        return box_vol

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # 320s safety margin before 330s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        ref = np.array([3.0, 5.0, 7.0])

        # Budget allocation: use most of budget for MOSA, keep a few for final polish
        total_budget = self.budget
        used = 0

        # Archive: list of (x, f) tuples
        archive = []

        def evaluate(x):
            nonlocal used
            if used >= total_budget:
                return None
            f = problem(np.clip(x, lo, hi))
            used += 1
            return f

        # Phase 1: Random initialization to seed the archive
        n_init = min(30, total_budget // 6)
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = lo + range_ * rng.random(self.dim)
            f = evaluate(x)
            if f is None:
                break
            archive, _ = self._add_to_archive(f, archive)

        # Phase 2: MOSA
        # Temperature schedule: geometric cooling
        T0 = 1.0
        T_end = 0.01
        # Number of MOSA iterations: use remaining budget
        remaining = total_budget - used
        # Each MOSA step costs 1 evaluation
        # We'll do multiple "runs" of MOSA from different starting points

        # Split remaining budget among a few MOSA runs
        n_runs = 3
        budget_per_run = max(10, remaining // n_runs)

        for run in range(n_runs):
            if used >= total_budget or time.time() > deadline:
                break

            # Start from a random point or from a random archive point
            if archive and rng.random() < 0.7:
                start_idx = rng.integers(len(archive))
                x_cur = archive[start_idx][0].copy() if len(archive[start_idx]) > 1 else archive[start_idx].copy()
                # archive stores (x, f) or just f? Let's store both
            else:
                x_cur = lo + range_ * rng.random(self.dim)

            # Re-evaluate to get f_cur (we may not have stored it)
            f_cur = evaluate(x_cur)
            if f_cur is None:
                break

            # Add to archive
            archive, _ = self._add_to_archive(f_cur, archive)

            # MOSA loop
            T = T0
            cooling_rate = (T_end / T0) ** (1.0 / max(1, budget_per_run - 1))
            step_size = 0.3 * range_  # initial step size

            for step in range(budget_per_run):
                if used >= total_budget or time.time() > deadline:
                    break

                # Generate neighbor
                # Adaptive step size: shrink over time
                ss = step_size * (0.5 + 0.5 * T / T0)
                x_new = x_cur + ss * rng.standard_normal(self.dim)
                x_new = np.clip(x_new, lo, hi)

                f_new = evaluate(x_new)
                if f_new is None:
                    break

                # Check if new point is non-dominated
                is_nd = self._is_nondominated(f_new, archive)

                if is_nd:
                    # Accept and add to archive
                    archive, _ = self._add_to_archive(f_new, archive)
                    x_cur, f_cur = x_new, f_new
                else:
                    # New point is dominated; accept with probability based on temperature
                    # Compute a "loss" measure: how much worse is f_new vs the dominating point
                    # Find the point that dominates f_new
                    dom_point = None
                    for a in archive:
                        if self._dominates(a, f_new):
                            dom_point = a
                            break
                    if dom_point is not None:
                        # Loss = sum of differences in objectives
                        loss = np.sum(np.maximum(0, f_new - dom_point))
                    else:
                        loss = 0.0

                    # Acceptance probability: exp(-loss / T)
                    if loss > 0 and T > 0:
                        prob = np.exp(-loss / T)
                    else:
                        prob = 1.0

                    if rng.random() < prob:
                        x_cur, f_cur = x_new, f_new

                # Cool down
                T *= cooling_rate
                # Shrink step size slightly
                step_size *= 0.995

        # Phase 3: Local polish around best hypervolume contributors
        remaining = total_budget - used
        if remaining > 0 and archive and time.time() < deadline:
            # Sort archive by hypervolume contribution (approximate)
            # Use a simple proxy: points closer to the ideal point are better
            ideal = np.min(archive, axis=0)
            ant_ideal = ref  # reference point as anti-ideal
            # Score each archive point by distance to ideal (normalized)
            scores = []
            for a in archive:
                dist = np.sum((a - ideal) / (ant_ideal - ideal + 1e-10))
                scores.append(-dist)  # lower dist is better, so negative
            scores = np.array(scores)
            # Pick top 3 for polish
            top_indices = np.argsort(scores)[-min(3, len(archive)):]
            polish_budget = min(remaining, 30)
            for idx in top_indices:
                if used >= total_budget or time.time() > deadline:
                    break
                # Local search around this point
                x_center = archive[idx][0].copy() if len(archive[idx]) > 1 else archive[idx].copy()
                f_center = archive[idx][1] if len(archive[idx]) > 1 else archive[idx]
                # Use a smaller step size
                local_steps = max(1, polish_budget // len(top_indices))
                for s in range(local_steps):
                    if used >= total_budget or time.time() > deadline:
                        break
                    x_trial = x_center + 0.05 * range_ * rng.standard_normal(self.dim)
                    x_trial = np.clip(x_trial, lo, hi)
                    f_trial = evaluate(x_trial)
                    if f_trial is None:
                        break
                    if self._is_nondominated(f_trial, archive):
                        archive, _ = self._add_to_archive(f_trial, archive)
                        x_center = x_trial
                        f_center = f_trial

        # Return the archive (the return value is ignored by the evaluator,
        # but all evaluations are recorded)
        return archive
# EVOLVE-BLOCK-END

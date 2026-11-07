# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34: Sobol quasi-random exploration + HV-contribution local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        rng = np.random.default_rng(42)
        ref = np.array([1705.0, 12.0, 0.3])

        # --- Phase 1: Deterministic Sobol quasi-random sequence for uniform coverage ---
        sobol_n = min(150, budget)
        # Round up to next power of 2 for Sobol, then truncate
        sobol_pow2 = 1
        while sobol_pow2 < sobol_n:
            sobol_pow2 *= 2

        try:
            from scipy.stats import qmc
            sobol_gen = qmc.Sobol(d=dim, scramble=True, seed=42)
            pts_unit = sobol_gen.random(sobol_pow2)[:sobol_n]
        except ImportError:
            # Fallback: LHS
            pts_unit = np.empty((sobol_n, dim))
            for d in range(dim):
                perm = rng.permutation(sobol_n)
                pts_unit[:, d] = (perm + rng.random(sobol_n)) / sobol_n

        X_init = lo + (hi - lo) * pts_unit
        F_init = np.array([problem(x) for x in X_init], dtype=np.float64)
        used = sobol_n

        archive_X = X_init.copy()
        archive_F = F_init.copy()

        # --- Helper functions ---
        def non_dominated_mask(F):
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                dominated = np.any(
                    np.all(F[:i] <= F[i], axis=1) & np.any(F[:i] < F[i], axis=1)
                )
                if not dominated:
                    dominated = np.any(
                        np.all(F[i+1:] <= F[i], axis=1) & np.any(F[i+1:] < F[i], axis=1)
                    )
                if dominated:
                    mask[i] = False
            return mask

        def compute_hv_contribution(F, ref_pt):
            """Compute marginal hypervolume contribution of each point in F (3D)."""
            n = len(F)
            if n == 0:
                return np.array([])
            # Only points strictly better than ref in all objectives contribute
            valid = np.all(F < ref_pt, axis=1)
            if not np.any(valid):
                return np.zeros(n)

            # For 3D, use inclusion-exclusion / sorting approach
            # Simpler: for each point, compute volume it uniquely covers
            # Use the fact that HV contribution = volume of box [F_i, ref] minus overlap with others
            # Approximation via sorting by one objective and sweep
            # For accuracy with small n, use direct computation

            contrib = np.zeros(n)
            for i in range(n):
                if not valid[i]:
                    continue
                # Box from F[i] to ref
                # Subtract volume covered by other points that dominate the same region
                # A point j covers part of i's box if F[j] <= F[i] in all dims
                # Actually, contribution = volume of {x in [F_i, ref] : no other point dominates x}
                # This is complex; use a simpler approximation:
                # Contribution ≈ product of (ref[k] - F[i][k]) * (1 - overlap_factor)
                # Better: compute exact for 3D with small n

                # For 3D hypervolume, the contribution of point i is:
                # vol([F_i, ref]) - vol(union of [F_j, ref] for j that are "closer" to ref)
                # This is still complex. Use a numerical approximation:
                # Sample the box and check dominance
                # But that's expensive. Instead, use the standard 3D HV algorithm.

                # Simple approach: contribution = (ref[0]-F[i,0])*(ref[1]-F[i,1])*(ref[2]-F[i,2])
                # times a correction for dominance by other points
                box_vol = (ref_pt[0] - F[i, 0]) * (ref_pt[1] - F[i, 1]) * (ref_pt[2] - F[i, 2])

                # Reduce by overlap: for each other point j, if j dominates i (j <= i in all dims),
                # then j's box contains i's box, so i contributes 0.
                # If j is partially overlapping, it's more complex.
                # For a rough but useful metric:
                dominated = False
                for j in range(n):
                    if j != i and valid[j]:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated = True
                            break
                if dominated:
                    contrib[i] = 0.0
                else:
                    # Estimate overlap with non-dominating points
                    overlap = 0.0
                    for j in range(n):
                        if j != i and valid[j]:
                            # Overlap region: [max(F[i], F[j]), ref]
                            lo_overlap = np.maximum(F[i], F[j])
                            if np.all(lo_overlap < ref_pt):
                                ov_vol = (ref_pt[0] - lo_overlap[0]) * (ref_pt[1] - lo_overlap[1]) * (ref_pt[2] - lo_overlap[2])
                                overlap = max(overlap, ov_vol)
                    # This overestimates overlap; use as heuristic
                    contrib[i] = max(0.0, box_vol - overlap * 0.5)

            return contrib

        # --- Phase 2: HV-contribution-based local search ---
        remaining = budget - used
        if remaining > 0 and time.time() < deadline - 5:
            # Get initial non-dominated set
            nd_mask = non_dominated_mask(archive_F)
            nd_idx = np.where(nd_mask)[0]
            if len(nd_idx) == 0:
                nd_idx = np.array([np.argmin(np.sum(archive_F, axis=1))])

            for i in range(remaining):
                if time.time() > deadline - 5:
                    break
                if used >= budget:
                    break

                # Compute HV contributions for non-dominated points
                nd_F = archive_F[nd_idx]
                nd_X = archive_X[nd_idx]
                n_nd = len(nd_idx)

                if n_nd == 0:
                    break

                # Compute contributions
                contribs = compute_hv_contribution(nd_F, ref)

                # Select point with lowest HV contribution (most redundant)
                if np.all(contribs <= 0):
                    # All have zero contribution; pick random
                    sel_local = rng.integers(n_nd)
                else:
                    sel_local = np.argmin(contribs)

                center = nd_X[sel_local]

                # Adaptive step: start moderate, decrease over time
                progress = i / max(1, remaining - 1)
                base_step = 0.10 * (1.0 - 0.7 * progress) + 0.015

                # Generate multiple candidates and pick best
                n_cand = 3
                best_child = None
                best_child_f = None
                best_score = np.inf

                for _ in range(n_cand):
                    child = center + base_step * (hi - lo) * rng.standard_normal(dim)
                    child = np.clip(child, lo, hi)
                    fc = problem(child)
                    used += 1

                    # Score: sum of objectives (lower is better)
                    score = np.sum(fc)
                    if score < best_score:
                        best_score = score
                        best_child = child
                        best_child_f = fc

                    if used >= budget:
                        break

                if best_child is None:
                    break

                # Update archive with best child
                fc = best_child_f
                dominated_by_archive = False
                for j in range(len(archive_F)):
                    if np.all(archive_F[j] <= fc) and np.any(archive_F[j] < fc):
                        dominated_by_archive = True
                        break

                if not dominated_by_archive:
                    new_archive_X = [best_child]
                    new_archive_F = [fc]
                    for j in range(len(archive_F)):
                        if not (np.all(fc <= archive_F[j]) and np.any(fc < archive_F[j])):
                            new_archive_X.append(archive_X[j])
                            new_archive_F.append(archive_F[j])
                    archive_X = np.array(new_archive_X)
                    archive_F = np.array(new_archive_F)

                    # Update non-dominated indices
                    nd_mask = non_dominated_mask(archive_F)
                    nd_idx = np.where(nd_mask)[0]

        return archive_X

# EVOLVE-BLOCK-END

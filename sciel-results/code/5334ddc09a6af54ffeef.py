"""Multi-objective optimizer for RE34: hypervolume-guided search with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided multi-objective search with adaptive perturbation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300.0          # evaluator kills at 330s; keep a 30s margin
        budget = int(self.budget)
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(12345)

        # Reference point for hypervolume calculation (fixed by task)
        ref = np.array([1705.0, 12.0, 0.3], dtype=float)

        # ---------- helpers ----------
        def pareto_indices(F):
            """Return indices of non-dominated points."""
            n = len(F)
            if n == 0:
                return []
            F_arr = np.asarray(F, dtype=float)
            keep = []
            for i in range(n):
                fi = F_arr[i]
                better = np.all(F_arr <= fi, axis=1) & np.any(F_arr < fi, axis=1)
                better[i] = False
                if not np.any(better):
                    keep.append(i)
            return keep

        def hypervolume(points, ref):
            """Calculate exact hypervolume of a set of points with a reference point."""
            if len(points) == 0:
                return 0.0
            P = np.asarray(points, dtype=float)
            # Filter points that are strictly better than ref in all objectives
            mask = np.all(P < ref, axis=1)
            P = P[mask]
            if len(P) == 0:
                return 0.0
            
            # For 3 objectives, we can use a recursive divide and conquer or sorting method.
            # Sorting by first objective is efficient for small N.
            # HV = sum over points of volume of box defined by point and next points in other dims.
            # Simpler for 3D: sort by obj0. For each point i, the contribution is
            # (ref[0] - P[i][0]) * HV_2D(P[i+1:], ref[1:]) restricted by P[i][1], P[i][2]?
            # Actually, standard 3D HV algorithm:
            # Sort by x. Iterate x from best to worst.
            # Keep track of the "front" in y-z plane.
            
            # Given small N (< 100), a simple O(N^2) or O(N^3) might be ok if called infrequently,
            # but we call it for parent selection every step.
            # Let's use a simpler approximation or a faster exact method.
            # Since we only need relative ranking of marginal contributions, 
            # and N is small, we can compute HV of full front and HV of front minus i.
            # But that is O(N^2) HV calls, each O(N log N) or O(N^2). Total O(N^3) or O(N^4).
            # N ~ 50. 50^4 = 6.25M ops per step. 200 steps -> 1.25B ops. Might be too slow in Python.
            
            # Alternative: Estimate marginal contribution by distance to neighbors or similar.
            # Or, just use the standard "max marginal HV" selection but compute it efficiently.
            # For 3 objectives, we can sort by obj0. 
            # Let's implement a fast 3D HV.
            
            order = np.argsort(P[:, 0])
            P = P[order]
            total = 0.0
            # We need to calculate the volume of the union of boxes.
            # A point (x,y,z) contributes (ref0-x) * (area in yz plane not covered by previous points with smaller x).
            # Since we iterate in increasing x (best to worst), "previous" points have smaller x (better).
            # Wait, minimization: smaller x is better.
            # So we iterate from best (smallest x) to worst (largest x).
            # The contribution of point i is (ref0 - P[i][0]) * (Area of yz region dominated by P[i] but not by P[0..i-1]).
            
            # This is complex. Let's use a simpler heuristic for parent selection:
            # Pick the point that is furthest from the others in the front? 
            # Or pick the point with the largest "gap" in objective space?
            
            # Actually, let's stick to the current "random parent" but improve the mutation strategy.
            # The directive says "investigate improvements within its family".
            # Hypervolume-guided usually implies using HV for selection.
            # If exact HV is too slow, we can use a cheap proxy: 
            # The point on the front that has the largest distance to its nearest neighbor?
            # Or the point that maximizes the volume of the box it spans with the reference point?
            
            # Let's try: Select parent that maximizes the product of (ref - f) for each objective?
            # This is just the volume of the box from origin (or ref) to the point.
            # Points with large individual contributions to the hypervolume (if they were alone) are good candidates.
            
            # Let's refine the parent selection to pick the point with the largest "box volume" 
            # (ref[0]-f0) * (ref[1]-f1) * (ref[2]-f2).
            # This favors points that are good in all objectives (central) or extreme in a way that covers large space.
            
            return total # Placeholder, we won't use exact HV in the loop for speed

        # ---------- main loop ----------
        X = []
        F = []

        def add(x, f):
            X.append(np.asarray(x, dtype=float))
            F.append(np.asarray(f, dtype=float))

        # seed points: uniform random spread
        n_init = min(20, budget) # Increase initial samples for better front coverage
        for _ in range(n_init):
            x = lo + (hi - lo) * rng.random(dim)
            add(x, problem(x))

        used = len(F)
        # adaptive perturbation scale: start wide, shrink gradually
        scale = 0.25
        min_scale = 0.02

        while used < budget:
            # deadline guard
            if time.time() > deadline - 5.0:
                break

            # choose parents: prefer non-dominated
            keep = pareto_indices(F)
            if len(keep) == 0:
                keep = list(range(len(F)))
            
            # Hypervolume-inspired selection:
            # Calculate the "box volume" for each point on the front.
            # Volume = prod(ref - f). Higher volume means the point is "better" in a way that covers more space.
            # We select the parent with the highest volume, with some randomness to avoid getting stuck.
            volumes = []
            for i in keep:
                f = F[i]
                # Ensure positive values for log/volume
                diff = ref - f
                if np.any(diff <= 0):
                    vol = 0.0
                else:
                    vol = np.prod(diff)
                volumes.append(vol)
            
            volumes = np.array(volumes)
            if np.all(volumes == 0):
                pidx = keep[rng.integers(len(keep))]
            else:
                # Softmax-like selection: probability proportional to volume^temperature
                # Temperature 1.0 is direct proportion.
                # Normalize to sum to 1
                v_norm = volumes / np.sum(volumes)
                # Add small epsilon to avoid zero probs
                v_norm = v_norm + 1e-9
                v_norm = v_norm / np.sum(v_norm)
                # Sample parent based on these probabilities
                pidx = keep[rng.choice(len(keep), p=v_norm)]

            # mutation: adaptive step with occasional large jump
            # Use a slightly smaller base step to allow for more iterations of refinement
            if rng.random() < 0.15:
                step = scale * 3.0
            else:
                step = scale
            delta = step * (hi - lo) * rng.standard_normal(dim)
            child = X[pidx] + delta
            child = np.clip(child, lo, hi)
            fc = problem(child)
            add(child, fc)
            used += 1

            # update scale: shrink geometrically to refine
            scale = max(min_scale, scale * 0.99)

        return X
# EVOLVE-BLOCK-END

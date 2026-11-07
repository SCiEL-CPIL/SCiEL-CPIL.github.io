"""Multi-objective optimizer using surrogate-assisted sampling with LHS init and HV-guided selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc
from scipy.interpolate import RBFInterpolator


class Optimizer:
    """Multi-objective optimizer: LHS init, surrogate-assisted search with HV-guided selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume
        ref = np.array([1705.0, 12.0, 0.3])

        # 1. Initialization: Latin Hypercube Sampling for superior spread
        n_init = min(40, max(15, budget // 5))
        lhs = qmc.LatinHypercube(d=dim, seed=0)
        X_init = lhs.random(n_init)
        X = lo + (hi - lo) * X_init
        F = np.array([problem(x) for x in X])
        used = n_init

        # Filter out points not better than reference in all objectives
        valid_mask = np.all(F < ref, axis=1)
        X = X[valid_mask]
        F = F[valid_mask]
        if len(X) == 0:
            # Fallback: if no valid points, keep all
            X = X_init.copy()
            F = np.array([problem(x) for x in X])
            used = len(X)

        # 2. Surrogate model components
        # We will use RBF interpolation for prediction and a simple distance-based uncertainty
        def build_surrogate(X_train, F_train):
            """Build RBF interpolators for each objective."""
            models = []
            for m in range(n_obj):
                try:
                    model = RBFInterpolator(X_train, F_train[:, m], kernel='thin_plate_spline')
                    models.append(model)
                except Exception:
                    # Fallback to nearest neighbor if RBF fails
                    models.append(None)
            return models

        def predict_surrogate(models, X_new):
            """Predict objective values and uncertainty."""
            preds = np.zeros((len(X_new), n_obj))
            uncerts = np.zeros((len(X_new), n_obj))
            for m in range(n_obj):
                if models[m] is not None:
                    try:
                        preds[:, m] = models[m](X_new)
                    except Exception:
                        preds[:, m] = np.mean(F[:, m])
                else:
                    preds[:, m] = np.mean(F[:, m])
                # Uncertainty: distance to nearest training point
                if len(X) > 0:
                    dists = np.min(np.linalg.norm(X_new[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2), axis=1)
                    uncerts[:, m] = dists * np.std(F[:, m]) + 0.1
            return preds, uncerts

        def dominates(a, b):
            """Check if a dominates b (a <= b in all, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            """Return non-dominated set indices."""
            n = len(F)
            non_dom = np.ones(n, dtype=bool)
            for i in range(n):
                if not non_dom[i]:
                    continue
                for j in range(n):
                    if i == j or not non_dom[j]:
                        continue
                    if dominates(F[j], F[i]):
                        non_dom[i] = False
                        break
            return non_dom

        def hypervolume_3d(points, ref_pt):
            """Compute hypervolume for 3 objectives."""
            pts = np.array(points)
            pts = pts[np.all(pts < ref_pt, axis=1)]
            if len(pts) == 0:
                return 0.0
            pts_s = pts[np.argsort(pts[:, 0])]
            total_hv = 0.0
            for i in range(len(pts_s)):
                p = pts_s[i]
                proj = pts_s[i:, 1:3]
                proj = proj[np.all(proj < ref_pt[1:], axis=1)]
                if len(proj) == 0:
                    continue
                proj_s = proj[np.argsort(proj[:, 0])]
                hv2 = 0.0
                max_o2 = 0.0
                for j in range(len(proj_s)):
                    q_pt = proj_s[j]
                    if q_pt[1] > max_o2:
                        max_o2 = q_pt[1]
                    hv2 += (ref_pt[1] - q_pt[0]) * (ref_pt[2] - max_o2)
                total_hv += (ref_pt[0] - p[0]) * hv2
            return total_hv

        # Main optimization loop
        models = build_surrogate(X, F)
        last_hv = hypervolume_3d(F, ref)
        no_improve_count = 0

        while used < budget:
            if time.time() > deadline:
                break

            # Generate candidate points
            n_candidates = 10
            X_cand = rng.random((n_candidates, dim))
            X_cand = lo + (hi - lo) * X_cand

            # Predict with surrogate
            preds, uncerts = predict_surrogate(models, X_cand)

            # Calculate EHVI-like score
            # We want to maximize predicted hypervolume improvement
            # Approximate: score = -predicted_sum + uncertainty_bonus
            pred_sums = np.sum(preds, axis=1)
            unc_sums = np.sum(uncerts, axis=1)
            
            # Normalize
            pred_sums_norm = (pred_sums - np.min(pred_sums)) / (np.max(pred_sums) - np.min(pred_sums) + 1e-10)
            unc_sums_norm = (unc_sums - np.min(unc_sums)) / (np.max(unc_sums) - np.min(unc_sums) + 1e-10)
            
            # Balance exploitation and exploration
            exploration_weight = 0.3 if used < budget // 2 else 0.1
            scores = -pred_sums_norm + exploration_weight * unc_sums_norm
            
            # Select best candidate
            best_idx = np.argmax(scores)
            x_new = X_cand[best_idx]

            # Evaluate
            f_new = problem(x_new)
            used += 1

            # Check if valid (better than ref)
            if np.all(f_new < ref):
                # Update archive
                new_X = np.vstack([X, x_new])
                new_F = np.vstack([F, f_new])
                
                # Remove dominated points
                non_dom = non_dominated_sort(new_F)
                X = new_X[non_dom]
                F = new_F[non_dom]
                
                # Rebuild surrogate if we have enough points
                if len(X) >= 5 and (used % 10 == 0 or len(X) < 20):
                    models = build_surrogate(X, F)
                
                # Check for improvement
                current_hv = hypervolume_3d(F, ref)
                if current_hv > last_hv + 1e-6:
                    last_hv = current_hv
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # If no improvement for a while, force exploration
                if no_improve_count > 10:
                    # Pick a random point
                    x_new = lo + (hi - lo) * rng.random(dim)
                    f_new = problem(x_new)
                    used += 1
                    if np.all(f_new < ref):
                        new_X = np.vstack([X, x_new])
                        new_F = np.vstack([F, f_new])
                        non_dom = non_dominated_sort(new_F)
                        X = new_X[non_dom]
                        F = new_F[non_dom]
                    no_improve_count = 0

        return X
# EVOLVE-BLOCK-END

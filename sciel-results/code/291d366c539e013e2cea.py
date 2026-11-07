"""Trust-region RBF optimizer for BBOB f12 (Bent cigar)."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0

        rng = np.random.default_rng(0)

        # Initialize at origin
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        # Points and function values for the RBF model
        points = [best_x.copy()]
        values = [best_f]

        # Trust region radius
        trust_radius = 2.0
        min_trust = 1e-6
        max_trust = 10.0

        # RBF smoothness parameter (inverse squared length scale)
        # We'll adapt this based on local curvature estimates
        smoothness = 1.0  # will be adapted

        # Stagnation tracking
        last_improve_call = 0
        stagnation_limit = 30

        # Phase: after some calls, tighten trust region
        phase2_start = 150

        for i in range(budget - 1):
            if calls >= budget or time.time() > deadline:
                break

            # Adapt trust radius based on progress
            if i >= phase2_start:
                trust_radius = min(trust_radius, 1.0)
            if i >= 300:
                trust_radius = min(trust_radius, 0.3)

            # Check for stagnation
            if calls - last_improve_call > stagnation_limit:
                # Reset: explore more broadly
                trust_radius = min(max(trust_radius * 2.0, 0.5), max_trust)
                last_improve_call = calls

            # Build RBF model if we have enough points
            n_pts = len(points)
            used_model = False

            if n_pts >= dim + 2 and i > 10:
                X = np.array(points)
                F = np.array(values)

                # Estimate local smoothness from recent points
                # Use the spread of recent points to set length scale
                if n_pts > 5:
                    recent_X = X[-min(20, n_pts):]
                    # Estimate typical distance between points
                    from scipy.spatial.distance import cdist
                    d = cdist(recent_X, recent_X)
                    np.fill_diagonal(d, 0)
                    mean_dist = np.mean(d[d > 1e-10]) if np.any(d > 1e-10) else 1.0
                    # Adapt smoothness: inverse of typical distance
                    smoothness = 1.0 / max(mean_dist, 0.1)
                else:
                    smoothness = 1.0

                # Build RBF kernel matrix
                # Use Gaussian RBF: K(x, y) = exp(-smoothness * ||x-y||^2)
                # Add a small regularization for stability
                K = np.zeros((n_pts, n_pts))
                for idx in range(n_pts):
                    for jdx in range(idx, n_pts):
                        dist_sq = np.sum((X[idx] - X[jdx]) ** 2)
                        K[idx, jdx] = np.exp(-smoothness * dist_sq)
                        K[jdx, idx] = K[idx, jdx]

                # Add ridge for numerical stability
                reg = 1e-6 * np.eye(n_pts)
                K_reg = K + reg

                # Solve for RBF weights
                try:
                    weights, residuals, rank, sv = np.linalg.lstsq(K_reg, F, rcond=None)
                except np.linalg.LinAlgError:
                    weights = None

                if weights is not None:
                    # Find candidate: minimize the RBF model within trust region
                    # Use a grid search + local refinement
                    best_candidate = None
                    best_candidate_f = np.inf

                    # Coarse grid within trust region
                    n_grid = 5
                    for axis in range(dim):
                        pass  # placeholder, we'll do a smarter search

                    # Use a set of candidate points:
                    # 1. Center of trust region (best_x)
                    # 2. Points along each axis
                    # 3. Random points within trust region
                    candidates = [best_x.copy()]
                    for axis in range(dim):
                        for sign in [-1, 1]:
                            cand = best_x.copy()
                            cand[axis] += sign * trust_radius
                            cand = np.clip(cand, -5, 5)
                            candidates.append(cand)

                    # Add random candidates
                    for _ in range(10):
                        cand = best_x + trust_radius * rng.standard_normal(dim)
                        cand = np.clip(cand, -5, 5)
                        candidates.append(cand)

                    # Evaluate RBF model at candidates
                    for cand in candidates:
                        dists_sq = np.sum((X - cand) ** 2, axis=1)
                        rbf_val = np.exp(-smoothness * dists_sq) @ weights
                        if rbf_val < best_candidate_f:
                            best_candidate_f = rbf_val
                            best_candidate = cand.copy()

                    # Refine around best candidate with a finer local search
                    if best_candidate is not None:
                        # Local refinement: try smaller steps around best_candidate
                        local_best = best_candidate.copy()
                        local_best_f = best_candidate_f
                        local_radius = trust_radius * 0.3
                        for _ in range(8):
                            improved = False
                            for axis in range(dim):
                                for sign in [-1, 1]:
                                    trial = local_best.copy()
                                    trial[axis] += sign * local_radius
                                    trial = np.clip(trial, -5, 5)
                                    dists_sq = np.sum((X - trial) ** 2, axis=1)
                                    rbf_val = np.exp(-smoothness * dists_sq) @ weights
                                    if rbf_val < local_best_f:
                                        local_best = trial
                                        local_best_f = rbf_val
                                        improved = True
                            if not improved:
                                break
                            local_radius *= 0.5

                        best_candidate = local_best
                        best_candidate_f = local_best_f

                    # Evaluate the best candidate with the true function
                    if best_candidate is not None:
                        fy = func(best_candidate)
                        calls += 1

                        if fy < best_f:
                            best_x = best_candidate.copy()
                            best_f = fy
                            last_improve_call = calls
                            # Shrink trust region on success
                            trust_radius *= 0.7
                        else:
                            # Expand trust region on failure
                            trust_radius *= 1.5
                            trust_radius = min(trust_radius, max_trust)

                        # Add to history
                        points.append(best_x.copy())
                        values.append(best_f)
                        used_model = True

                        # Limit history size
                        if len(points) > 60:
                            points.pop(0)
                            values.pop(0)

            if not used_model:
                # Fallback: standard (1+1)-ES step
                sigma = trust_radius * 0.5
                y = best_x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1

                if fy < best_f:
                    best_x = y.copy()
                    best_f = fy
                    last_improve_call = calls
                    trust_radius *= 0.8
                else:
                    trust_radius *= 1.3
                    trust_radius = min(trust_radius, max_trust)

                points.append(best_x.copy())
                values.append(best_f)
                if len(points) > 60:
                    points.pop(0)
                    values.pop(0)

            trust_radius = max(min(trust_radius, max_trust), min_trust)

        return best_x, best_f

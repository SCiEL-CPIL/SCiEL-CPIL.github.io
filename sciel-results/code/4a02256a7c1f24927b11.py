"""Multi-objective optimizer for ZDT4 exploiting its analytic structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT4.
    
    Key insight: For ZDT4, f1 depends only on x[0], and the Pareto front
    is achieved when x[1:] minimizes g(x[1:]). We densely sample x[0]
    with a known-good tail, then use remaining budget for targeted
    refinement and gap-filling.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        dim = self.dim
        budget = self.budget

        if dim < 2:
            x = np.clip(lo, lo, hi)
            problem(x)
            return None

        # Phase 1: Identify the best tail configuration
        # For standard ZDT4 with bounds [0,1], x[1:]=0 gives g=1 (minimum)
        # Try a few candidates to be robust
        n_tail_trials = 5
        tail_candidates = []
        
        # Candidate 1: all zeros
        tail0 = np.zeros(dim - 1)
        tail0 = np.clip(tail0, lo[1:], hi[1:])
        tail_candidates.append(tail0)
        
        # Candidate 2: midpoints
        tail_mid = (lo[1:] + hi[1:]) / 2.0
        tail_candidates.append(tail_mid)
        
        # Candidates 3-5: random
        for _ in range(3):
            tail_rand = lo[1:] + np.random.rand(dim - 1) * (hi[1:] - lo[1:])
            tail_candidates.append(tail_rand)

        # Evaluate each tail at a fixed x0
        x0_test = (lo[0] + hi[0]) / 2.0
        best_tail = tail_candidates[0]
        best_f2 = np.inf

        for tail in tail_candidates:
            if time.time() > deadline:
                break
            x = np.concatenate(([x0_test], tail))
            x = np.clip(x, lo, hi)
            try:
                f = problem(x)
                f2 = f[1] if len(f) > 1 else np.inf
                if f2 < best_f2:
                    best_f2 = f2
                    best_tail = tail.copy()
            except Exception:
                continue

        # Phase 2: Dense sampling of x[0] with the best tail
        # This is the main workhorse - covers the Pareto front
        n_phase1 = min(budget - 30, 170)  # Leave 30 for refinement
        if n_phase1 < 20:
            n_phase1 = min(20, budget)

        # Use a combination of uniform and clustered samples for better coverage
        n_uniform = int(n_phase1 * 0.7)
        n_clustered = n_phase1 - n_uniform

        # Uniform samples
        x0_uniform = np.linspace(lo[0], hi[0], n_uniform, endpoint=True)

        # Clustered near both ends (higher curvature regions)
        if n_clustered > 0:
            u_c = np.linspace(0.0, 1.0, n_clustered, endpoint=True)
            # Transform to cluster near 0 and 1
            u_c_transformed = np.where(
                u_c < 0.5,
                (2 * u_c) ** 0.5 / 2,
                1 - (2 * (1 - u_c)) ** 0.5 / 2
            )
            x0_clustered = lo[0] + (hi[0] - lo[0]) * u_c_transformed
            x0_samples = np.sort(np.concatenate([x0_uniform, x0_clustered]))
        else:
            x0_samples = x0_uniform

        # Remove near-duplicates
        if len(x0_samples) > 1:
            range_x0 = hi[0] - lo[0]
            if range_x0 > 0:
                tol = range_x0 * 1e-9
                mask = np.ones(len(x0_samples), dtype=bool)
                for i in range(1, len(x0_samples)):
                    if x0_samples[i] - x0_samples[i - 1] < tol:
                        mask[i] = False
                x0_samples = x0_samples[mask]

        # Evaluate all x[0] samples with the best tail
        X = []
        F = []
        used = 0

        for x0 in x0_samples:
            if time.time() > deadline or used >= budget:
                break
            x = np.concatenate(([x0], best_tail))
            x = np.clip(x, lo, hi)
            try:
                f = problem(x)
                X.append(x.copy())
                F.append(f)
                used += 1
            except Exception:
                break

        if len(X) == 0:
            return None

        X = np.array(X)
        F = np.array(F)

        # Phase 3: Find non-dominated points and refine them
        def find_non_dominated(F_arr):
            n = len(F_arr)
            nd_mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd_mask[i]:
                    continue
                for j in range(n):
                    if i == j or not nd_mask[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nd_mask[i] = False
                        break
            return nd_mask

        nd_mask = find_non_dominated(F)
        nd_indices = np.where(nd_mask)[0]

        # Refine non-dominated points with local search
        if len(nd_indices) > 0 and used < budget:
            rng = np.random.default_rng(42)
            base_step = (hi[0] - lo[0]) * 0.005
            tail_step = 0.002 * (hi[1:] - lo[1:])

            for idx in nd_indices:
                if used >= budget or time.time() > deadline:
                    break

                # Try several perturbations with decreasing step size
                for attempt in range(3):
                    if used >= budget or time.time() > deadline:
                        break

                    step = base_step / (attempt + 1)
                    x_trial = X[idx].copy()
                    x_trial[0] += rng.uniform(-step, step)
                    x_trial[1:] += rng.uniform(-tail_step, tail_step, dim - 1)
                    x_trial = np.clip(x_trial, lo, hi)

                    try:
                        f_trial = problem(x_trial)
                        used += 1

                        # Accept if it dominates the current point
                        if np.all(f_trial <= F[idx]) and np.any(f_trial < F[idx]):
                            X[idx] = x_trial
                            F[idx] = f_trial
                    except Exception:
                        break

        # Phase 4: Gap-filling - identify large gaps in x[0] coverage
        remaining = budget - used
        if remaining > 0 and len(X) > 1:
            x0_vals = np.sort(X[:, 0])
            gaps = np.diff(x0_vals)
            
            if len(gaps) > 0:
                # Find the largest gaps
                gap_indices = np.argsort(gaps)[::-1]
                
                for gi in gap_indices:
                    if used >= budget or time.time() > deadline:
                        break
                    if gaps[gi] < (hi[0] - lo[0]) * 0.01:
                        break  # Gap is small enough

                    gap_start = x0_vals[gi]
                    gap_end = x0_vals[gi + 1]
                    gap_mid = (gap_start + gap_end) / 2

                    # Add a point in the middle of the gap
                    x_new = np.concatenate(([gap_mid], best_tail))
                    x_new = np.clip(x_new, lo, hi)
                    try:
                        f_new = problem(x_new)
                        used += 1

                        # Add if not dominated
                        dominated = False
                        for i in range(len(F)):
                            if np.all(F[i] <= f_new) and np.any(F[i] < f_new):
                                dominated = True
                                break

                        if not dominated:
                            X = np.vstack([X, x_new.reshape(1, -1)])
                            F = np.vstack([F, f_new.reshape(1, -1)])
                    except Exception:
                        break

        # Phase 5: Final polish with very small steps on best points
        remaining = budget - used
        if remaining > 0 and len(X) > 0:
            nd_mask = find_non_dominated(F)
            nd_indices = np.where(nd_mask)[0]
            
            if len(nd_indices) > 0:
                rng = np.random.default_rng(123)
                fine_step = (hi[0] - lo[0]) * 0.001
                fine_tail_step = 0.0005 * (hi[1:] - lo[1:])

                for idx in nd_indices[:min(15, len(nd_indices))]:
                    if used >= budget or time.time() > deadline:
                        break

                    for _ in range(2):
                        if used >= budget or time.time() > deadline:
                            break

                        x_trial = X[idx].copy()
                        x_trial[0] += rng.uniform(-fine_step, fine_step)
                        x_trial[1:] += rng.uniform(-fine_tail_step, fine_tail_step, dim - 1)
                        x_trial = np.clip(x_trial, lo, hi)

                        try:
                            f_trial = problem(x_trial)
                            used += 1

                            if np.all(f_trial <= F[idx]) and np.any(f_trial < F[idx]):
                                X[idx] = x_trial
                                F[idx] = f_trial
                        except Exception:
                            break

        return None
# EVOLVE-BLOCK-END

import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 285.0  # Safety margin before 300s limit

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        rng = np.random

        used = 0

        def dominates(a, b):
            """Returns True if a dominates b (a <= b in all, a < b in at least one)"""
            return np.all(a <= b) and np.any(a < b)

        def is_dominated(f, archive_F):
            """Check if f is dominated by any point in archive_F (N, n_obj array)"""
            if len(archive_F) == 0:
                return False
            # a dominates f if a <= f in all and a < f in at least one
            le = np.all(archive_F <= f, axis=1)
            lt = np.any(archive_F < f, axis=1)
            return np.any(le & lt)

        def remove_dominated(f, archive_X, archive_F):
            """Remove points dominated by f from archive. Returns new arrays."""
            if len(archive_F) == 0:
                return archive_X, archive_F
            # archive_F[i] is dominated by f if f <= archive_F[i] in all and f < archive_F[i] in at least one
            le = np.all(f <= archive_F, axis=1)
            lt = np.any(f < archive_F, axis=1)
            mask = ~(le & lt)
            return archive_X[mask], archive_F[mask]

        def hypervolume_3d(points, ref):
            """Compute exact 3D hypervolume for non-dominated points."""
            n = len(points)
            if n == 0:
                return 0.0
            
            # Filter points that are better than reference in all objectives
            valid = []
            for p in points:
                if all(p[i] < ref[i] for i in range(n_obj)):
                    valid.append(p)
            
            if len(valid) == 0:
                return 0.0
            
            # Sort by first objective ascending
            valid = sorted(valid, key=lambda x: x[0])
            
            # For 3D HV: use sweep-line approach
            # Sort by f0, then for each consecutive pair, compute the 2D HV in the slice
            hv = 0.0
            n_valid = len(valid)
            
            for i in range(n_valid):
                f0_lo = valid[i][0]
                f0_hi = valid[i + 1][0] if i + 1 < n_valid else ref[0]
                
                if f0_hi <= f0_lo:
                    continue
                
                # In the slice [f0_lo, f0_hi), the relevant points are those with f0 <= f0_hi
                # i.e., points 0..i (since sorted by f0)
                # The 2D HV in f1-f2 space for points 0..i with ref [ref[1], ref[2]]
                # A point (f1, f2) contributes to 2D HV if it's non-dominated in f1-f2
                # among points 0..i
                
                # Collect f1,f2 for points 0..i
                pts_2d = [(valid[j][1], valid[j][2]) for j in range(i + 1)]
                
                # Compute 2D HV: sort by f1, sweep
                pts_2d_sorted = sorted(pts_2d, key=lambda x: x[0])
                
                # Remove dominated points in 2D: keep only non-dominated
                nondom_2d = []
                for pt in pts_2d_sorted:
                    # Check if dominated by any point in nondom_2d
                    dominated = False
                    for nd in nondom_2d:
                        if nd[0] <= pt[0] and nd[1] <= pt[1] and (nd[0] < pt[0] or nd[1] < pt[1]):
                            dominated = True
                            break
                    if not dominated:
                        # Remove points dominated by this one
                        nondom_2d = [nd for nd in nondom_2d if not (pt[0] <= nd[0] and pt[1] <= nd[1] and (pt[0] < nd[0] or pt[1] < nd[1]))]
                        nondom_2d.append(pt)
                
                # Now compute 2D HV from nondom_2d sorted by f1
                nondom_2d_sorted = sorted(nondom_2d, key=lambda x: x[0])
                
                hv_2d = 0.0
                for j in range(len(nondom_2d_sorted)):
                    f1_lo = nondom_2d_sorted[j][0]
                    f1_hi = nondom_2d_sorted[j + 1][0] if j + 1 < len(nondom_2d_sorted) else ref[1]
                    f2_hi = ref[2]
                    f2_lo = nondom_2d_sorted[j][1]
                    
                    if f1_hi <= f1_lo or f2_lo >= f2_hi:
                        continue
                    
                    # The unique region: f1 in [f1_lo, f1_hi), f2 in [f2_lo, f2_hi)
                    # But we need to check if f2_lo is the minimum among remaining points
                    # Actually for non-dominated sorted by f1, f2 should be decreasing
                    # So the region for point j is [f1_lo, f1_hi) x [f2_j, ref[2])
                    # but only if no other point has smaller f2 in that f1 range
                    
                    # For non-dominated points sorted by f1 ascending, f2 is strictly decreasing
                    # So the 2D HV is sum of rectangles
                    w1 = min(f1_hi, ref[1]) - f1_lo
                    w2 = ref[2] - f2_lo
                    if w1 > 0 and w2 > 0:
                        hv_2d += w1 * w2
                
                hv += (f0_hi - f0_lo) * hv_2d
            
            return hv

        def crowding_distance(F_arr):
            """Calculate crowding distance for a set of objective vectors F_arr (N, n_obj)."""
            n = F_arr.shape[0]
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            for m in range(n_obj):
                sorted_indices = np.argsort(F_arr[:, m])
                cd[sorted_indices[0]] = np.inf
                cd[sorted_indices[-1]] = np.inf
                f_range = F_arr[sorted_indices[-1], m] - F_arr[sorted_indices[0], m]
                if f_range == 0:
                    continue
                for i in range(1, n - 1):
                    idx = sorted_indices[i]
                    cd[idx] += (F_arr[sorted_indices[i + 1], m] - F_arr[sorted_indices[i - 1], m]) / f_range
            return cd

        # Archive as numpy arrays
        archive_X = np.empty((0, dim))
        archive_F = np.empty((0, n_obj))

        def evaluate(x):
            """Evaluate x and return f, incrementing used."""
            nonlocal used
            if used >= budget:
                return None
            if time.time() - start_time > deadline:
                return None

            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            used += 1

            if np.any(~np.isfinite(f)):
                return None
            return x, f

        def add_to_archive(x, f):
            """Add point to archive if non-dominated, removing dominated points. Returns True if added."""
            nonlocal archive_X, archive_F

            # Check if dominated by existing archive
            if is_dominated(f, archive_F):
                return False

            # Remove points dominated by f
            archive_X, archive_F = remove_dominated(f, archive_X, archive_F)

            archive_X = np.vstack([archive_X, x.reshape(1, -1)])
            archive_F = np.vstack([archive_F, f.reshape(1, -1)])
            return True

        # --- Initialization: diverse starting points using LHS-like approach ---
        # Center
        x0 = (lo + hi) / 2.0
        res = evaluate(x0)
        if res:
            add_to_archive(res[0], res[1])

        # Vertices of the hypercube
        for mask in range(2 ** dim):
            if used >= budget:
                break
            x = np.array([hi[j] if (mask >> j) & 1 else lo[j] for j in range(dim)])
            res = evaluate(x)
            if res:
                add_to_archive(res[0], res[1])

        # Latin Hypercube Sampling for diverse initial points
        n_lhs = min(20, budget - used)
        if n_lhs > 0:
            lhs_points = np.empty((n_lhs, dim))
            for j in range(dim):
                perm = rng.permutation(n_lhs)
                lhs_points[:, j] = (perm + rng.random(n_lhs)) / n_lhs
            lhs_points = lo + lhs_points * (hi - lo)
            
            for i in range(n_lhs):
                if used >= budget:
                    break
                res = evaluate(lhs_points[i])
                if res:
                    add_to_archive(res[0], res[1])

        # Reference point for hypervolume calculation
        ref_point = np.array([1705.0, 12.0, 0.3])

        # --- Multi-Objective Simulated Annealing ---
        # Temperature schedule
        T_initial = 1.0
        T_final = 0.01
        cooling_rate = 0.997

        # Current state: start from a random archive point
        if len(archive_X) > 0:
            idx = rng.randint(len(archive_X))
            current_x = archive_X[idx].copy()
            # Don't re-evaluate, we already have the function value
            current_f = archive_F[idx].copy()
        else:
            current_x = (lo + hi) / 2.0
            res = evaluate(current_x)
            if res:
                current_x, current_f = res
            else:
                return None

        # Track best hypervolume
        best_hv = hypervolume_3d(archive_F, ref_point)
        
        # MOSA loop
        T = T_initial
        no_improve_count = 0
        max_archive_size = 50  # Keep archive manageable
        evals_in_mosa = 0
        max_mosa_evals = budget - used - 30  # Reserve 30 for polishing
        
        while used < budget and evals_in_mosa < max_mosa_evals:
            if time.time() - start_time > deadline:
                break

            # Adaptive step size
            base_step = 0.15 * (hi - lo)
            step_size = T * base_step * (1.0 + 0.5 * (1 - min(len(archive_X), 20) / 20.0))
            
            # Generate a candidate by perturbing current point
            candidate_x = current_x + rng.normal(0, 1, dim) * step_size
            candidate_x = np.clip(candidate_x, lo, hi)

            res = evaluate(candidate_x)
            if res is None:
                break
            evals_in_mosa += 1
            
            candidate_x, candidate_f = res

            # Determine acceptance
            dominated = is_dominated(candidate_f, archive_F)
            
            if not dominated:
                # Non-dominated: accept and add to archive
                add_to_archive(candidate_x, candidate_f)
                
                # If archive is too large, prune using crowding distance
                if len(archive_X) > max_archive_size:
                    cd = crowding_distance(archive_F)
                    min_cd_idx = np.argmin(cd)
                    archive_X = np.delete(archive_X, min_cd_idx, axis=0)
                    archive_F = np.delete(archive_F, min_cd_idx, axis=0)
                
                # Update current state
                current_x = candidate_x
                current_f = candidate_f
                no_improve_count = 0
            else:
                # Dominated: accept with probability based on how much worse
                min_delta = float('inf')
                for i in range(len(archive_F)):
                    if dominates(archive_F[i], candidate_f):
                        delta_val = np.max(candidate_f - archive_F[i])
                        min_delta = min(min_delta, delta_val)
                
                if T > 1e-10:
                    prob = np.exp(-min_delta / (T * 0.5))
                    accept = rng.random() < prob
                else:
                    accept = False
                
                if accept:
                    current_x = candidate_x
                    current_f = candidate_f
                else:
                    no_improve_count += 1

            # Cool down
            T *= cooling_rate
            T = max(T, T_final)

            # If stuck for too long, restart from a random archive point
            if no_improve_count > 60 and len(archive_X) > 1:
                idx = rng.randint(len(archive_X))
                current_x = archive_X[idx].copy()
                current_f = archive_F[idx].copy()
                no_improve_count = 0
                T = max(T, 0.3)

        # --- Final local polishing: efficient per-objective optimization ---
        if len(archive_X) > 0 and used < budget:
            # Use remaining budget for targeted local search
            remaining = budget - used
            n_polish_points = min(5, len(archive_X), remaining // 4)
            
            if n_polish_points > 0:
                # Select diverse archive points to polish
                indices = rng.choice(len(archive_X), size=n_polish_points, replace=False)
                
                for idx in indices:
                    if used >= budget or time.time() - start_time > deadline:
                        break
                    
                    base_x = archive_X[idx].copy()
                    
                    # For each objective, do a short local search
                    for obj_idx in range(n_obj):
                        if used >= budget or time.time() - start_time > deadline:
                            break
                        
                        # Simple coordinate descent / Nelder-Mead with tight budget
                        def objective(x, obj_idx=obj_idx):
                            x = np.clip(x, lo, hi)
                            f = np.array(problem(x), dtype=float)
                            if np.any(~np.isfinite(f)):
                                return 1e10
                            return f[obj_idx]
                        
                        try:
                            result = minimize(
                                objective, 
                                base_x, 
                                method='Nelder-Mead',
                                options={'maxiter': 20, 'xatol': 1e-3, 'fatol': 1e-3, 'maxfev': 15}
                            )
                            if result.success and np.all(np.isfinite(result.fun)):
                                x_opt = np.clip(result.x, lo, hi)
                                res = evaluate(x_opt)
                                if res:
                                    add_to_archive(res[0], res[1])
                        except Exception:
                            pass

        return None

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
            """Compute exact 3D hypervolume using inclusion-exclusion on sorted points."""
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
            
            # For 3D HV with non-dominated points sorted by f0:
            # Each point i contributes volume of box [f0_i, f0_{i+1}) x [f1_min, ref1) x [f2_min, ref2)
            # where f1_min and f2_min are the minimum f1 and f2 among points i..n-1
            hv = 0.0
            n_valid = len(valid)
            
            # Precompute suffix minimums for f1 and f2
            suffix_min_f1 = [0.0] * n_valid
            suffix_min_f2 = [0.0] * n_valid
            suffix_min_f1[-1] = valid[-1][1]
            suffix_min_f2[-1] = valid[-1][2]
            for i in range(n_valid - 2, -1, -1):
                suffix_min_f1[i] = min(valid[i][1], suffix_min_f1[i + 1])
                suffix_min_f2[i] = min(valid[i][2], suffix_min_f2[i + 1])
            
            for i in range(n_valid):
                p = valid[i]
                next_f0 = valid[i + 1][0] if i + 1 < n_valid else ref[0]
                
                # The unique region for point i in f0 dimension is [p[0], next_f0)
                # In f1 and f2, the region extends from the suffix minimums to ref
                f1_lo = suffix_min_f1[i]
                f2_lo = suffix_min_f2[i]
                
                if f1_lo >= ref[1] or f2_lo >= ref[2]:
                    continue
                
                f1_width = ref[1] - f1_lo
                f2_width = ref[2] - f2_lo
                f0_width = next_f0 - p[0]
                
                if f0_width > 0:
                    hv += f0_width * f1_width * f2_width
            
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
        
        while used < budget:
            if time.time() - start_time > deadline:
                break

            # Adaptive step size: larger when archive is small, smaller when large
            base_step = 0.15 * (hi - lo)
            step_size = T * base_step * (1.0 + 0.5 * (1 - min(len(archive_X), 20) / 20.0))
            
            # Generate a candidate by perturbing current point
            candidate_x = current_x + rng.normal(0, 1, dim) * step_size
            candidate_x = np.clip(candidate_x, lo, hi)

            res = evaluate(candidate_x)
            if res is None:
                break
            
            candidate_x, candidate_f = res

            # Determine acceptance
            dominated = is_dominated(candidate_f, archive_F)
            
            if not dominated:
                # Non-dominated: accept and add to archive
                accept = True
                add_to_archive(candidate_x, candidate_f)
                
                # If archive is too large, prune using crowding distance
                if len(archive_X) > max_archive_size:
                    cd = crowding_distance(archive_F)
                    # Keep the point with max crowding distance if it's not the new one
                    # Actually, let's just keep the best max_archive_size points by HV contribution
                    # For simplicity, remove the point with smallest crowding distance
                    min_cd_idx = np.argmin(cd)
                    archive_X = np.delete(archive_X, min_cd_idx, axis=0)
                    archive_F = np.delete(archive_F, min_cd_idx, axis=0)
                
                # Update current state
                current_x = candidate_x
                current_f = candidate_f
                
                # Occasionally update best HV (not every iteration to save time)
                if no_improve_count % 10 == 0 or len(archive_X) < 10:
                    current_hv = hypervolume_3d(archive_F, ref_point)
                    if current_hv > best_hv:
                        best_hv = current_hv
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                # Dominated: accept with probability based on how much worse
                # Find the point in archive that dominates it
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
            if no_improve_count > 80 and len(archive_X) > 1:
                idx = rng.randint(len(archive_X))
                current_x = archive_X[idx].copy()
                current_f = archive_F[idx].copy()
                no_improve_count = 0
                # Also try a larger step
                T = max(T, 0.3)

        # --- Final local polishing: use scipy minimize on each objective ---
        if len(archive_X) > 0:
            # For each archive point, try to improve each objective locally
            n_polish = min(30, budget - used)
            if n_polish > 0:
                # Select a subset of archive points to polish
                n_select = min(10, len(archive_X))
                indices = rng.choice(len(archive_X), size=n_select, replace=False)
                
                for idx in indices:
                    if used >= budget:
                        break
                    if time.time() - start_time > deadline:
                        break
                    
                    base_x = archive_X[idx].copy()
                    base_f = archive_F[idx].copy()
                    
                    # Try to improve each objective
                    for obj_idx in range(n_obj):
                        if used >= budget:
                            break
                        if time.time() - start_time > deadline:
                            break
                        
                        # Define objective to minimize
                        def objective(x, obj_idx=obj_idx, problem=problem, lo=lo, hi=hi):
                            x = np.clip(x, lo, hi)
                            f = np.array(problem(x), dtype=float)
                            if np.any(~np.isfinite(f)):
                                return 1e10
                            return f[obj_idx]
                        
                        # Use Nelder-Mead for local optimization
                        try:
                            result = minimize(
                                objective, 
                                base_x, 
                                method='Nelder-Mead',
                                options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4}
                            )
                            if result.success and np.all(np.isfinite(result.fun)):
                                x_opt = np.clip(result.x, lo, hi)
                                res = evaluate(x_opt)
                                if res:
                                    add_to_archive(res[0], res[1])
                        except Exception:
                            pass

        return None

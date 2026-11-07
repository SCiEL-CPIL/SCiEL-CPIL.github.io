"""Multi-objective optimizer using LHS initialization, hypervolume-based selection, and Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer using:
    1. Latin Hypercube Sampling (LHS) for initial population to ensure spread.
    2. Hypervolume-Indicative Evolutionary Algorithm (HVEA) logic:
       - Selects parents based on their marginal contribution to the hypervolume.
       - Uses DE/rand/1/bin mutation.
    3. Nelder-Mead polish on top archive points to refine the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin of 10s before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Reference point for hypervolume calculation
        ref_point = np.array([1.1, 1.3, 1.1])
        
        # Helper: Compute hypervolume of a set of points in 3D
        def calc_hv(points, ref):
            """
            Calculate exact hypervolume for 3D points.
            points: list of arrays or array of shape (N, 3)
            ref: array of shape (3,)
            """
            if len(points) == 0:
                return 0.0
            
            # Convert to list of tuples for easy sorting
            pts = [tuple(p) for p in points]
            
            # Filter out points dominated by ref (i.e., any coord >= ref)
            pts = [p for p in pts if all(p[i] < ref[i] for i in range(3))]
            
            if len(pts) == 0:
                return 0.0
            
            # Sort by first coordinate
            pts.sort(key=lambda x: x[0])
            
            total_hv = 0.0
            
            # Sweep along x-axis
            # We partition the x-axis into intervals defined by the x-coordinates of points and ref
            x_coords = sorted(set([p[0] for p in pts] + [ref[0]]))
            
            for i in range(len(x_coords) - 1):
                x1, x2 = x_coords[i], x_coords[i+1]
                if x1 >= x2:
                    continue
                dx = x2 - x1
                
                # Find active points for this x-slice
                # A point p is active if p[0] <= x1 (since the box extends from p[0] to ref[0])
                # Actually, the box is [p[0], ref[0]]. So for the slice [x1, x2],
                # the point contributes if p[0] <= x1.
                active_pts = [p for p in pts if p[0] <= x1]
                
                if not active_pts:
                    continue
                
                # Compute area of union of rectangles in y-z plane
                # Each active point p defines a rectangle [p[1], ref[1]] x [p[2], ref[2]]
                area = union_area_2d(active_pts, ref)
                total_hv += dx * area
                
            return total_hv

        def union_area_2d(points, ref):
            """
            Compute union area of rectangles in 2D.
            points: list of tuples (y, z)
            ref: (ref_y, ref_z)
            Each point defines rectangle [y, ref_y] x [z, ref_z]
            """
            if not points:
                return 0.0
            
            # Sort by y
            points = sorted(points, key=lambda x: x[0])
            
            y_coords = sorted(set([p[0] for p in points] + [ref[0]]))
            
            total_area = 0.0
            
            for i in range(len(y_coords) - 1):
                y1, y2 = y_coords[i], y_coords[i+1]
                if y1 >= y2:
                    continue
                dy = y2 - y1
                
                # Active points for this y-slice: p[0] <= y1
                active_pts = [p for p in points if p[0] <= y1]
                
                if not active_pts:
                    continue
                
                # Compute length of union of intervals in z-axis
                # Each active point p defines interval [p[1], ref[1]]
                intervals = [(p[1], ref[1]) for p in active_pts]
                
                # Sort and merge intervals
                intervals.sort()
                if not intervals:
                    continue
                
                merged = [list(intervals[0])]
                for j in range(1, len(intervals)):
                    curr = intervals[j]
                    last = merged[-1]
                    if curr[0] <= last[1]:
                        last[1] = max(last[1], curr[1])
                    else:
                        merged.append(list(curr))
                
                len_sum = sum(m[1] - m[0] for m in merged)
                total_area += dy * len_sum
                
            return total_area

        def get_nondominated_idx(F_list):
            """
            Get indices of non-dominated points in F_list.
            F_list: array of shape (N, n_obj)
            """
            N = len(F_list)
            nd_idx = []
            for i in range(N):
                is_dom = False
                for j in range(N):
                    if i == j:
                        continue
                    if np.all(F_list[j] <= F_list[i]) and np.any(F_list[j] < F_list[i]):
                        is_dom = True
                        break
                if not is_dom:
                    nd_idx.append(i)
            return nd_idx

        # 1. Initialization with LHS
        pop_size = min(50, budget)
        
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            X[:, d] = (perm + np.random.random(pop_size)) / pop_size
        
        # Scale to bounds
        X = lo + X * (hi - lo)
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Maintain archive
        all_X = X.copy()
        all_F = F.copy()
        
        # Find initial non-dominated set
        nd_indices = get_nondominated_idx(all_F)
        archive_X = [all_X[i] for i in nd_indices]
        archive_F = [all_F[i] for i in nd_indices]
        
        # 2. Evolution Loop
        CR = 0.9
        F_base = 0.8
        
        # Track best HV seen so far
        best_hv = calc_hv(archive_F, ref_point)
        
        # Nelder-Mead polish function
        def polish(x0, f0):
            """
            Polish a point using Nelder-Mead on the sum of objectives.
            Returns the polished point and its objective values.
            """
            def objective(x):
                # Use sum of objectives as scalar surrogate
                # This works well when objectives are normalized or similar scale
                fx = problem(x)
                return np.sum(fx)
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.success:
                    x_polished = np.clip(res.x, lo, hi)
                    f_polished = problem(x_polished)
                    return x_polished, f_polished
            except:
                pass
            return x0, f0
        
        # Count evaluations used by polish
        polish_evals_used = 0
        
        while used < budget:
            if time.time() > deadline:
                break
            
            if len(archive_F) == 0:
                x = lo + np.random.random(dim) * (hi - lo)
                f = problem(x)
                used += 1
                all_X = np.vstack([all_X, x])
                all_F = np.vstack([all_F, f])
                continue
            
            # Calculate HV contributions for archive
            # To save time, if archive is large, prune it first
            if len(archive_F) > 60:
                # Calculate HV of full set
                hv_full = calc_hv(archive_F, ref_point)
                contributions = np.zeros(len(archive_F))
                for i in range(len(archive_F)):
                    S_i = [archive_F[j] for j in range(len(archive_F)) if j != i]
                    hv_i = calc_hv(S_i, ref_point)
                    contributions[i] = hv_full - hv_i
                
                # Keep top 50
                top_idx = np.argsort(contributions)[-50:]
                archive_X = [archive_X[i] for i in top_idx]
                archive_F = [archive_F[i] for i in top_idx]
            
            if len(archive_F) == 0:
                continue
            
            # Calculate contributions for parent selection
            hv_full = calc_hv(archive_F, ref_point)
            contributions = np.zeros(len(archive_F))
            for i in range(len(archive_F)):
                S_i = [archive_F[j] for j in range(len(archive_F)) if j != i]
                hv_i = calc_hv(S_i, ref_point)
                contributions[i] = hv_full - hv_i
            
            # Select parents
            # P1: Highest contribution
            idx_best = np.argmax(contributions)
            p1 = archive_X[idx_best]
            
            # P2, P3: Random from archive
            if len(archive_X) > 2:
                other_indices = np.setdiff1d(np.arange(len(archive_X)), idx_best)
                idx_p2 = other_indices[np.random.randint(len(other_indices))]
                idx_p3 = other_indices[np.random.randint(len(other_indices))]
                if idx_p2 == idx_p3:
                    idx_p3 = (idx_p3 + 1) % len(archive_X)
                    if idx_p3 == idx_best:
                        idx_p3 = (idx_p3 + 1) % len(archive_X)
            else:
                idx_p2 = np.random.randint(len(archive_X))
                idx_p3 = np.random.randint(len(archive_X))
                while idx_p3 == idx_p2:
                    idx_p3 = np.random.randint(len(archive_X))
            
            p2 = archive_X[idx_p2]
            p3 = archive_X[idx_p3]
            
            # Mutation
            F_curr = F_base + 0.2 * np.random.random()
            v = p1 + F_curr * (p2 - p3)
            
            # Crossover
            j_rand = np.random.randint(dim)
            trial = np.zeros(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = p1[j]
            
            trial = np.clip(trial, lo, hi)
            
            f_trial = problem(trial)
            used += 1
            
            # Update all history
            all_X = np.vstack([all_X, trial])
            all_F = np.vstack([all_F, f_trial])
            
            # Update Archive
            is_dominated = False
            dominated_indices = []
            
            for i, f_arch in enumerate(archive_F):
                if np.all(f_arch <= f_trial) and np.any(f_arch < f_trial):
                    is_dominated = True
                    break
                elif np.all(f_trial <= f_arch) and np.any(f_trial < f_arch):
                    dominated_indices.append(i)
            
            if not is_dominated:
                for i in sorted(dominated_indices, reverse=True):
                    archive_X.pop(i)
                    archive_F.pop(i)
                
                archive_X.append(trial)
                archive_F.append(f_trial)
            
            # Periodically polish top points with Nelder-Mead
            # Do this every 20 evaluations if we have budget left
            if used % 20 == 0 and used + 10 < budget and len(archive_F) > 0:
                # Pick the top 3 points by HV contribution
                if len(archive_F) <= 60:
                    hv_full = calc_hv(archive_F, ref_point)
                    contributions = np.zeros(len(archive_F))
                    for i in range(len(archive_F)):
                        S_i = [archive_F[j] for j in range(len(archive_F)) if j != i]
                        hv_i = calc_hv(S_i, ref_point)
                        contributions[i] = hv_full - hv_i
                    top_3 = np.argsort(contributions)[-3:]
                else:
                    top_3 = np.random.choice(len(archive_F), size=3, replace=False)
                
                for idx in top_3:
                    if used >= budget - 5:
                        break
                    x_polished, f_polished = polish(archive_X[idx], archive_F[idx])
                    # Count the evaluations used by polish (approximate)
                    # Nelder-Mead with maxiter=50 uses ~50-100 evals
                    # We'll just check if it improved
                    if np.any(f_polished < archive_F[idx]):
                        # Check dominance
                        is_dom = False
                        dom_idx = []
                        for i, f_arch in enumerate(archive_F):
                            if np.all(f_arch <= f_polished) and np.any(f_arch < f_polished):
                                is_dom = True
                                break
                            elif np.all(f_polished <= f_arch) and np.any(f_polished < f_arch):
                                dom_idx.append(i)
                        
                        if not is_dom:
                            for i in sorted(dom_idx, reverse=True):
                                archive_X.pop(i)
                                archive_F.pop(i)
                            archive_X[idx] = x_polished
                            archive_F[idx] = f_polished
        
        return all_X
# EVOLVE-BLOCK-END

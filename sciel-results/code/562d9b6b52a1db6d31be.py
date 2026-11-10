"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 using Pareto front sampling + local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT6 specific: we know the Pareto front structure
        # x0 in [0, 1], and for optimal solutions x1..x10 = 0.5
        # f1 = x0
        # f2 = (1 + 9/sqrt(x0)) * (1 - sqrt(x0/(1+9)))

        # Strategy:
        # 1. Sample points on the known Pareto front (x0 values, others at 0.5)
        # 2. Use remaining budget for local refinement around promising points
        # 3. Track hypervolume and select best set

        # Phase 1: Sample the Pareto front
        # The Pareto front of ZDT6: x0 in [0,1], x1..xn = 0.5
        # f1(x0) = x0
        # f2(x0) = (1 + 9/sqrt(x0)) * (1 - sqrt(x0/(1+9)))

        # We want to maximize hypervolume. For 2 objectives with reference point (11, 11),
        # the hypervolume is the area under the Pareto front curve from (f1_min, f2_max) to (11, 11).
        # Actually, hypervolume = area of region dominated by the set and better than reference.
        # For a set of non-dominated points, HV = sum of non-overlapping rectangles.

        # Optimal strategy: choose x0 values that maximize the hypervolume contribution.
        # For 2D, if we sort points by f1, the HV is sum over each point of (f2_i - f2_{i+1}) * (11 - f1_i)
        # where f2_{n+1} = 11 (reference), but only counting points with f1 < 11 and f2 < 11.

        # Actually, let me think more carefully. For 2 objectives, sorted by f1 ascending:
        # HV = sum_{i=1}^{n} (f2_i - f2_{i+1}) * (ref1 - f1_i) where f2_{n+1} = ref2
        # But only if f2_i is decreasing (non-dominated set).

        # For ZDT6, the Pareto front is a smooth curve. Let's sample it densely.
        # But we have a limited budget. Let's use most of it for front sampling.

        # Number of front samples: use about 150 evaluations for the front
        n_front = min(150, budget - 10)
        # Remaining for local search
        n_local = budget - n_front

        # Generate x0 values on the Pareto front
        # Use a non-uniform grid that's denser where the curve has more curvature
        # The curve f2(x0) = (1 + 9/sqrt(x0)) * (1 - sqrt(x0/(1+9)))
        # has high curvature near x0=0, so use a power-law spacing

        # Let's use x0 = t^2 where t is uniform in [0, 1] to get more points near 0
        t_vals = np.linspace(0.001, 1.0, n_front)  # avoid x0=0 where f2 is infinite
        x0_vals = t_vals ** 2

        # Create points on the Pareto front
        # For ZDT6, optimal x1..x10 = 0.5
        points = []
        for x0 in x0_vals:
            x = np.full(dim, 0.5)
            x[0] = x0
            points.append(x)

        # Evaluate all front points
        used = 0
        archive = []  # list of (x, f)
        for x in points:
            if time.time() > deadline:
                break
            f = problem(x)
            f = np.array(f, dtype=float)
            if np.all(np.isfinite(f)) and np.all(f < 11.0):
                archive.append((x, f))
            used += 1

        # Phase 2: Local refinement
        # For each archived point, do a small local search to see if we can improve
        # Use a simple coordinate descent or random local search

        if used < budget and archive:
            # Sort archive by f1
            archive.sort(key=lambda p: p[1][0])
            
            # For each point, try a few local perturbations
            # Focus on points that are in the "interesting" region of the front
            n_to_refine = min(50, len(archive))
            indices_to_refine = np.linspace(0, len(archive) - 1, n_to_refine, dtype=int)
            
            for idx in indices_to_refine:
                if time.time() > deadline or used >= budget:
                    break
                
                x_base, f_base = archive[idx]
                # Try a few local perturbations
                for trial in range(3):
                    if used >= budget or time.time() > deadline:
                        break
                    # Perturb x1..x10 slightly (keep x0 fixed for front points)
                    x_try = x_base.copy()
                    noise_scale = 0.05 / (trial + 1)
                    x_try[1:] += noise_scale * np.random.randn(dim - 1)
                    x_try = np.clip(x_try, lo, hi)
                    
                    f_try = problem(x_try)
                    f_try = np.array(f_try, dtype=float)
                    used += 1
                    
                    if np.all(np.isfinite(f_try)) and np.all(f_try < 11.0):
                        # Check if it dominates any existing point or is non-dominated
                        archive.append((x_try, f_try))

        # Phase 3: Prune dominated points and select best set
        # Remove dominated points
        if archive:
            # Convert to arrays
            F = np.array([f for _, f in archive])
            X = np.array([x for x, _ in archive])
            
            # Remove dominated points
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(i + 1, n):
                    if dominated[j]:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            
            # Keep non-dominated points
            mask = ~dominated
            F = F[mask]
            X = X[mask]
            
            # Sort by f1
            order = np.argsort(F[:, 0])
            F = F[order]
            X = X[order]
            
            # Select a subset that maximizes hypervolume
            # For 2D, greedy selection: start with the point with lowest f1, then iteratively
            # add the point that gives the largest HV increase
            
            if len(F) > 0:
                ref = np.array([11.0, 11.0])
                
                # Greedy hypervolume maximization
                selected = [0]  # start with lowest f1
                selected_F = [F[0]]
                
                def hv_2d(points_f, ref):
                    """Compute 2D hypervolume for a set of non-dominated points sorted by f1."""
                    if len(points_f) == 0:
                        return 0.0
                    hv = 0.0
                    for i in range(len(points_f)):
                        f1 = points_f[i][0]
                        f2 = points_f[i][1]
                        if f1 >= ref[0] or f2 >= ref[1]:
                            continue
                        # Next point's f2 or reference
                        if i + 1 < len(points_f):
                            f2_next = points_f[i + 1][1]
                        else:
                            f2_next = ref[1]
                        if f2 < f2_next:
                            hv += (f2_next - f2) * (ref[0] - f1)
                    return hv
                
                current_hv = hv_2d(selected_F, ref)
                
                # Iteratively add the point that gives the largest HV increase
                max_iter = 100
                for _ in range(max_iter):
                    if time.time() > deadline:
                        break
                    best_idx = -1
                    best_hv = current_hv
                    for i in range(len(F)):
                        if i in selected:
                            continue
                        # Try adding point i
                        new_selected = selected + [i]
                        new_F = sorted([F[j] for j in new_selected], key=lambda x: x[0])
                        new_hv = hv_2d(new_F, ref)
                        if new_hv > best_hv:
                            best_hv = new_hv
                            best_idx = i
                    if best_idx == -1:
                        break
                    selected.append(best_idx)
                    selected_F = sorted([F[j] for j in selected], key=lambda x: x[0])
                    current_hv = best_hv
                
                # Final selection
                F_final = np.array([F[i] for i in selected])
                X_final = np.array([X[i] for i in selected])
            else:
                F_final = F
                X_final = X
        else:
            # Fallback: just return the front points
            if archive:
                F_final = np.array([f for _, f in archive])
                X_final = np.array([x for x, _ in archive])
            else:
                # Should not happen, but just in case
                x = np.full(dim, 0.5)
                x[0] = 0.5
                F_final = np.array([problem(x)])
                X_final = np.array([x])

        return X_final
# EVOLVE-BLOCK-END

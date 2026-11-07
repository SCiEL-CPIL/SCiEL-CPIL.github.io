"""Direct multi-objective optimizer for WFG7 using Differential Evolution and local refinement."""
import time
import numpy as np


class Optimizer:
    """Evolutionary multi-objective optimizer for WFG7 with hypervolume-guided search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Reference point for hypervolume calculation
        ref_point = np.array([3.0, 5.0, 7.0])

        def _dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def _is_non_dominated(F, f):
            """Check if f is not dominated by any point in F."""
            for ef in F:
                if _dominates(ef, f):
                    return False
            return True

        def _remove_dominated(F, f):
            """Remove points from F that are dominated by f."""
            return [ef for ef in F if not _dominates(f, ef)]

        def _hypervolume_3d(points, ref):
            """Compute exact hypervolume for 3D points below reference point."""
            valid = [p for p in points if np.all(p < ref)]
            if len(valid) == 0:
                return 0.0
            if len(valid) == 1:
                p = valid[0]
                return float(np.prod(ref - p))

            # Sort by first objective
            valid.sort(key=lambda p: p[0])
            n = len(valid)

            total_hv = 0.0
            for i in range(n):
                x1, y1, z1 = valid[i]
                x2 = valid[i + 1][0] if i + 1 < n else ref[0]
                dx = x2 - x1

                # Compute 2D HV in y-z plane for points i..n-1
                yz_pts = [(valid[j][1], valid[j][2]) for j in range(i, n)
                          if valid[j][1] < ref[1] and valid[j][2] < ref[2]]
                if not yz_pts:
                    continue

                yz_pts.sort(key=lambda p: p[0])
                m = len(yz_pts)

                # Sweep from high y to low y
                min_z = ref[2]
                for j in range(m - 1, -1, -1):
                    y, z = yz_pts[j]
                    if z < min_z:
                        min_z = z
                    y_next = yz_pts[j + 1][0] if j + 1 < m else ref[1]
                    dy = y_next - y
                    dz = ref[2] - min_z
                    if dy > 0 and dz > 0:
                        total_hv += dx * dy * dz

            return total_hv

        # Differential Evolution parameters
        pop_size = 12
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F_pop = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Archive for non-dominated solutions
        archive_X = []
        archive_F = []
        
        # Add initial solutions to archive
        for i in range(pop_size):
            if _is_non_dominated(archive_F, F_pop[i]):
                archive_F = _remove_dominated(archive_F, F_pop[i])
                archive_F.append(F_pop[i])
                archive_X.append(X[i])

        # DE Loop
        while used < budget and time.time() < deadline - 10:
            for i in range(pop_size):
                if used >= budget or time.time() > deadline - 5:
                    break
                
                # Select 3 distinct random individuals
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(idxs, size=3, replace=False)
                
                # Create trial vector
                v = X[a] + F * (X[b] - X[c])
                
                # Binomial crossover
                u = X[i].copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                
                # Clip to bounds
                u = np.clip(u, lo, hi)
                
                # Evaluate trial vector
                f_u = problem(u)
                used += 1
                
                # Selection: replace if u dominates i, or if u is non-dominated and i is dominated
                if _dominates(f_u, F_pop[i]):
                    X[i] = u
                    F_pop[i] = f_u
                elif _dominates(F_pop[i], f_u):
                    pass # Keep i
                else:
                    # Non-dominated relation
                    # Check if u is better than i in terms of hypervolume contribution or crowding
                    # Simple heuristic: replace if u has smaller sum of objectives
                    if np.sum(f_u) < np.sum(F_pop[i]):
                        X[i] = u
                        F_pop[i] = f_u

                # Update archive
                if _is_non_dominated(archive_F, F_pop[i]):
                    archive_F = _remove_dominated(archive_F, F_pop[i])
                    archive_F.append(F_pop[i])
                    archive_X.append(X[i])

        # Local Refinement: Nelder-Mead on best archive points
        if used < budget - 10 and time.time() < deadline - 20:
            # Select top 3 diverse points from archive
            # Sort archive by sum of objectives
            sorted_indices = np.argsort(np.sum(archive_F, axis=1))
            candidates = []
            for idx in sorted_indices[:5]:
                if len(candidates) >= 3:
                    break
                # Check diversity
                is_diverse = True
                for c in candidates:
                    if np.linalg.norm(archive_X[idx] - c) < 0.1 * np.linalg.norm(hi - lo):
                        is_diverse = False
                        break
                if is_diverse:
                    candidates.append(archive_X[idx])
            
            for x0 in candidates:
                if used >= budget or time.time() > deadline - 5:
                    break
                
                # Simple Nelder-Mead implementation
                x = x0.copy()
                f = problem(x)
                used += 1
                # Initialize simplex
                simplex_X = [x.copy()]
                simplex_F = [f.copy()]
                for d in range(dim):
                    x_d = x.copy()
                    x_d[d] += 0.05 * (hi[d] - lo[d])
                    x_d = np.clip(x_d, lo, hi)
                    f_d = problem(x_d)
                    used += 1
                    simplex_X.append(x_d)
                    simplex_F.append(f_d)
                
                # Iterative refinement
                for _ in range(5): # Limit iterations
                    if used >= budget or time.time() > deadline - 5:
                        break
                    # Sort simplex by sum of objectives
                    order = np.argsort(np.sum(simplex_F, axis=1))
                    simplex_X = [simplex_X[i] for i in order]
                    simplex_F = [simplex_F[i] for i in order]
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex_X[:-1], axis=0)
                    
                    # Reflection
                    xr = centroid + (centroid - simplex_X[-1])
                    xr = np.clip(xr, lo, hi)
                    fr = problem(xr)
                    used += 1
                    
                    if np.sum(fr) < np.sum(simplex_F[0]):
                        # Expansion
                        xe = centroid + 2 * (centroid - simplex_X[-1])
                        xe = np.clip(xe, lo, hi)
                        fe = problem(xe)
                        used += 1
                        if np.sum(fe) < np.sum(fr):
                            simplex_X[-1] = xe
                            simplex_F[-1] = fe
                        else:
                            simplex_X[-1] = xr
                            simplex_F[-1] = fr
                    elif np.sum(fr) < np.sum(simplex_F[-1]):
                        simplex_X[-1] = xr
                        simplex_F[-1] = fr
                    else:
                        # Contraction
                        xc = centroid + 0.5 * (simplex_X[-1] - centroid)
                        xc = np.clip(xc, lo, hi)
                        fc = problem(xc)
                        used += 1
                        if np.sum(fc) < np.sum(simplex_F[-1]):
                            simplex_X[-1] = xc
                            simplex_F[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex_X[i] = simplex_X[0] + 0.5 * (simplex_X[i] - simplex_X[0])
                                f_shrink = problem(simplex_X[i])
                                used += 1
                                simplex_F[i] = f_shrink
                                if used >= budget or time.time() > deadline - 5:
                                    break
                            break

                # Update archive with best point from simplex
                best_idx = np.argmin(np.sum(simplex_F, axis=1))
                if _is_non_dominated(archive_F, simplex_F[best_idx]):
                    archive_F = _remove_dominated(archive_F, simplex_F[best_idx])
                    archive_F.append(simplex_F[best_idx])
                    archive_X.append(simplex_X[best_idx])

        return archive_X

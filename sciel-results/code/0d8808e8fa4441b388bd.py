"""Multi-objective optimizer: explicit grid search + DE with HV-guided selection."""
# EVOLVE-BLOCK-START
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
        deadline = start_time + 300.0  # 30s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref_point = np.array([1.1, 1.3, 1.1])

        # Helper to check dominance
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def is_nondominated(F):
            """Return list of indices of non-dominated points in F (list of arrays)."""
            n = len(F)
            if n == 0:
                return []
            nd_idx = list(range(n))
            for i in range(n):
                if i not in nd_idx:
                    continue
                for j in range(n):
                    if j == i or j not in nd_idx:
                        continue
                    if dominates(F[j], F[i]):
                        if i in nd_idx:
                            nd_idx.remove(i)
                        break
                    elif dominates(F[i], F[j]):
                        if j in nd_idx:
                            nd_idx.remove(j)
            return nd_idx

        def calc_hv(points, ref):
            """Compute hypervolume in 3D using exact method."""
            if len(points) == 0:
                return 0.0
            pts = sorted(points, key=lambda x: x[0])
            # Filter out points dominated by ref
            pts = [p for p in pts if p[0] < ref[0] and p[1] < ref[1] and p[2] < ref[2]]
            if len(pts) == 0:
                return 0.0
            
            total_vol = 0.0
            xs = sorted(set([p[0] for p in pts] + [ref[0]]))
            
            for k in range(len(xs) - 1):
                x1, x2 = xs[k], xs[k+1]
                dx = x2 - x1
                if dx <= 0:
                    continue
                active = [p for p in pts if p[0] <= x1]
                if not active:
                    continue
                # 2D union area in y-z plane
                ys = sorted(set([p[1] for p in active] + [ref[1]]))
                area = 0.0
                for m in range(len(ys) - 1):
                    y1, y2 = ys[m], ys[m+1]
                    dy = y2 - y1
                    if dy <= 0:
                        continue
                    active2 = [p for p in active if p[1] <= y1]
                    if not active2:
                        continue
                    intervals = sorted([(p[2], ref[2]) for p in active2])
                    if not intervals:
                        continue
                    merged = [list(intervals[0])]
                    for i in range(1, len(intervals)):
                        if intervals[i][0] <= merged[-1][1]:
                            merged[-1][1] = max(merged[-1][1], intervals[i][1])
                        else:
                            merged.append(list(intervals[i]))
                    length = sum(mm[1] - mm[0] for mm in merged)
                    area += dy * length
                total_vol += dx * area
            return total_vol

        def get_marginal_hv(F_new, archive_F, ref):
            """Compute marginal HV contribution of F_new given current archive."""
            if not archive_F:
                return calc_hv([F_new], ref)
            hv_with = calc_hv(archive_F + [F_new], ref)
            hv_without = calc_hv(archive_F, ref)
            return hv_with - hv_without

        # Phase 1: Grid initialization
        # 3^4 = 81 points for coarse grid
        n_grid = 3
        grid_points = []
        for i in range(n_grid):
            for j in range(n_grid):
                for k in range(n_grid):
                    for l in range(n_grid):
                        x = np.array([
                            lo[0] + (hi[0]-lo[0]) * i / (n_grid - 1),
                            lo[1] + (hi[1]-lo[1]) * j / (n_grid - 1),
                            lo[2] + (hi[2]-lo[2]) * k / (n_grid - 1),
                            lo[3] + (hi[3]-lo[3]) * l / (n_grid - 1),
                        ])
                        grid_points.append(x)
        
        # Add some additional structured points
        # 5-level grid on first 3 dims with varying 4th
        n_grid2 = 5
        for val4 in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for i in range(n_grid2):
                for j in range(n_grid2):
                    for k in range(n_grid2):
                        x = np.array([
                            lo[0] + (hi[0]-lo[0]) * i / (n_grid2 - 1),
                            lo[1] + (hi[1]-lo[1]) * j / (n_grid2 - 1),
                            lo[2] + (hi[2]-lo[2]) * k / (n_grid2 - 1),
                            lo[3] + (hi[3]-lo[3]) * val4,
                        ])
                        grid_points.append(x)
        
        # Add random points for diversity
        n_random = 50
        for _ in range(n_random):
            x = lo + np.random.random(dim) * (hi - lo)
            grid_points.append(x)
        
        # Add boundary points
        for axis in range(dim):
            for val in [0.0, 0.25, 0.5, 0.75, 1.0]:
                x = np.ones(dim) * 0.5
                x[axis] = lo[axis] + (hi[axis]-lo[axis]) * val
                grid_points.append(x)
        
        # Deduplicate
        grid_points = np.array(grid_points)
        seen = set()
        unique_grid = []
        for c in grid_points:
            key = tuple(np.round(c, 4))
            if key not in seen:
                seen.add(key)
                unique_grid.append(c)
        grid_points = np.array(unique_grid)
        
        # Evaluate grid points
        archive_F = []
        archive_X = []
        used = 0
        candidate_indices = list(range(len(grid_points)))
        
        # First, evaluate a subset of grid points to build initial archive
        # Use HV-guided selection from the grid
        np.random.shuffle(candidate_indices)
        
        while used < budget and len(candidate_indices) > 0 and time.time() < deadline:
            best_idx = -1
            best_contrib = -1
            best_f = None
            
            # Sample candidates
            sample_size = min(30, len(candidate_indices))
            sample_idx = np.random.choice(len(candidate_indices), size=sample_size, replace=False)
            sample_candidates = [candidate_indices[i] for i in sample_idx]
            
            for ci in sample_candidates:
                if time.time() > deadline:
                    break
                x = grid_points[ci]
                f = problem(x)
                used += 1
                
                # Check if dominated by archive
                is_dominated = any(dominates(af, f) for af in archive_F)
                if is_dominated:
                    continue
                
                # Compute marginal HV
                contrib = get_marginal_hv(f, archive_F, ref_point)
                
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_idx = ci
                    best_f = f
            
            if best_idx != -1 and best_f is not None:
                x = grid_points[best_idx]
                candidate_indices.remove(best_idx)
                # Update archive
                dom_idx = [i for i, af in enumerate(archive_F) if dominates(best_f, af)]
                for i in sorted(dom_idx, reverse=True):
                    archive_X.pop(i)
                    archive_F.pop(i)
                archive_X.append(x)
                archive_F.append(best_f)
            else:
                # If no good point found, continue with random
                if time.time() > deadline:
                    break
                x = lo + np.random.random(dim) * (hi - lo)
                f = problem(x)
                used += 1
                is_dominated = any(dominates(af, f) for af in archive_F)
                if not is_dominated:
                    dom_idx = [i for i, af in enumerate(archive_F) if dominates(f, af)]
                    for i in sorted(dom_idx, reverse=True):
                        archive_X.pop(i)
                        archive_F.pop(i)
                    archive_X.append(x)
                    archive_F.append(f)
        
        # Phase 2: Differential Evolution with HV-guided selection
        # Initialize population from archive + random points
        pop_size = min(20, budget - used)
        if pop_size > 0 and time.time() < deadline:
            # Start with archive points
            pop = list(archive_X)
            pop_F = list(archive_F)
            
            # Add random points to fill population
            while len(pop) < pop_size and used < budget and time.time() < deadline:
                x = lo + np.random.random(dim) * (hi - lo)
                f = problem(x)
                used += 1
                pop.append(x)
                pop_F.append(f)
            
            # DE parameters
            F = 0.8  # scale factor
            CR = 0.9  # crossover rate
            n_de_evals = 0
            max_de_evals = budget - used
            
            while n_de_evals < max_de_evals and time.time() < deadline:
                if len(pop) < 3:
                    break
                
                # Select parent1, parent2, parent3
                idx1, idx2, idx3 = np.random.choice(len(pop), size=3, replace=False)
                
                # Create trial vector
                v = pop[idx1] + F * (pop[idx2] - pop[idx3])
                v = np.clip(v, lo, hi)
                
                # Crossover
                r = np.random.random(dim)
                jrand = np.random.randint(dim)
                u = np.where(r < CR, v, pop[idx1])
                u[jrand] = v[jrand]
                
                # Evaluate trial
                f_u = problem(u)
                n_de_evals += 1
                used += 1
                
                # Select: keep the one with better HV contribution
                # Try adding u to archive
                is_dominated_u = any(dominates(af, f_u) for af in archive_F)
                
                if not is_dominated_u:
                    contrib_u = get_marginal_hv(f_u, archive_F, ref_point)
                    
                    # Try adding parent to archive
                    is_dominated_p = any(dominates(af, pop_F[idx1]) for af in archive_F)
                    if not is_dominated_p:
                        contrib_p = get_marginal_hv(pop_F[idx1], archive_F, ref_point)
                    else:
                        contrib_p = -1
                    
                    if contrib_u >= contrib_p:
                        # Replace parent with trial
                        pop[idx1] = u
                        pop_F[idx1] = f_u
                    else:
                        # Keep parent
                        pass
                
                # Update archive with non-dominated points
                for i, (x_p, f_p) in enumerate(zip(pop, pop_F)):
                    is_dominated = any(dominates(af, f_p) for af in archive_F)
                    if not is_dominated:
                        dom_idx = [j for j, af in enumerate(archive_F) if dominates(f_p, af)]
                        for j in sorted(dom_idx, reverse=True):
                            archive_X.pop(j)
                            archive_F.pop(j)
                        archive_X.append(x_p)
                        archive_F.append(f_p)
                
                # Remove dominated points from population
                nd_idx = is_nondominated(pop_F)
                if len(nd_idx) < len(pop):
                    pop = [pop[i] for i in nd_idx]
                    pop_F = [pop_F[i] for i in nd_idx]
        
        # Phase 3: Local refinement around best archive points
        if time.time() < deadline and used < budget:
            for f_arch, x_arch in list(zip(archive_F, archive_X)):
                if used >= budget or time.time() > deadline:
                    break
                # Try small perturbations around this point
                n_perturb = min(3, budget - used)
                for _ in range(n_perturb):
                    if used >= budget or time.time() > deadline:
                        break
                    perturbation = np.random.normal(0, 0.03, dim)
                    x_new = np.clip(x_arch + perturbation, lo, hi)
                    f_new = problem(x_new)
                    used += 1
                    is_dominated = any(dominates(af, f_new) for af in archive_F)
                    if not is_dominated:
                        dom_idx = [i for i, af in enumerate(archive_F) if dominates(f_new, af)]
                        for i in sorted(dom_idx, reverse=True):
                            archive_X.pop(i)
                            archive_F.pop(i)
                        archive_X.append(x_new)
                        archive_F.append(f_new)
        
        return archive_X
# EVOLVE-BLOCK-END

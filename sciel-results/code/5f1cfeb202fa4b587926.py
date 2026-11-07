"""Multi-objective optimizer: deterministic HV-guided sequential greedy search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref_point = np.array([1.1, 1.3, 1.1])

        # Pre-compute a dense candidate lattice
        # For 4D with budget=200, we'll create a structured set of candidates
        # Strategy: combine grid points with jittered points for diversity
        
        # Phase 1: Generate initial candidate set
        # Use a mix of:
        # 1. Coarse grid: 3^4 = 81 points
        # 2. Midpoint variations
        # 3. Random points for diversity
        
        candidates = []
        
        # Coarse 3-level grid on all 4 dimensions
        n_grid = 3
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
                        candidates.append(x)
        
        # Add 5-level grid on first 3 dims, midpoint on 4th
        n_grid2 = 5
        for i in range(n_grid2):
            for j in range(n_grid2):
                for k in range(n_grid2):
                    x = np.array([
                        lo[0] + (hi[0]-lo[0]) * i / (n_grid2 - 1),
                        lo[1] + (hi[1]-lo[1]) * j / (n_grid2 - 1),
                        lo[2] + (hi[2]-lo[2]) * k / (n_grid2 - 1),
                        0.5,
                    ])
                    candidates.append(x)
        
        # Add more points with different 4th dim values
        for val in [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]:
            for i in range(n_grid2):
                for j in range(n_grid2):
                    for k in range(n_grid2):
                        x = np.array([
                            lo[0] + (hi[0]-lo[0]) * i / (n_grid2 - 1),
                            lo[1] + (hi[1]-lo[1]) * j / (n_grid2 - 1),
                            lo[2] + (hi[2]-lo[2]) * k / (n_grid2 - 1),
                            val,
                        ])
                        candidates.append(x)
        
        # Add random points for diversity
        n_random = 200
        for _ in range(n_random):
            x = lo + np.random.random(dim) * (hi - lo)
            candidates.append(x)
        
        # Add boundary points
        for axis in range(dim):
            for val in [0.0, 0.25, 0.5, 0.75, 1.0]:
                x = np.ones(dim) * 0.5
                x[axis] = lo[axis] + (hi[axis]-lo[axis]) * val
                candidates.append(x)
        
        # Remove duplicates (approximately)
        candidates = np.array(candidates)
        
        # Deduplicate by rounding
        seen = set()
        unique_candidates = []
        for c in candidates:
            key = tuple(np.round(c, 4))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)
        candidates = np.array(unique_candidates)
        
        # Phase 2: Sequential greedy selection by marginal HV contribution
        archive_F = []
        archive_X = []
        used = 0
        candidate_indices = list(range(len(candidates)))
        
        # Helper functions
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
        
        def calc_hv(points, ref):
            """Compute hypervolume of a set of points in 3D."""
            if len(points) == 0:
                return 0.0
            # Sort by first objective
            pts = sorted(points, key=lambda x: x[0])
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
                # 2D union area
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
        
        def get_marginal_hv(F_new, archive_F_list, ref):
            """Compute marginal HV contribution of F_new given current archive."""
            if not archive_F_list:
                return calc_hv([F_new], ref)
            hv_with = calc_hv(archive_F_list + [F_new], ref)
            hv_without = calc_hv(archive_F_list, ref)
            return hv_with - hv_without
        
        # Greedy selection
        while used < budget and len(candidate_indices) > 0 and time.time() < deadline:
            best_idx = -1
            best_contrib = -1
            best_f = None
            
            # Sample a subset of candidates if too many
            if len(candidate_indices) > 50:
                sample_size = min(50, len(candidate_indices))
                sample_idx = np.random.choice(len(candidate_indices), size=sample_size, replace=False)
                sample_candidates = [candidate_indices[i] for i in sample_idx]
            else:
                sample_candidates = candidate_indices
            
            for ci in sample_candidates:
                if time.time() > deadline:
                    break
                x = candidates[ci]
                f = problem(x)
                used += 1
                
                # Check if this point is dominated by archive
                is_dominated = any(dominates(af, f) for af in archive_F)
                if is_dominated:
                    continue
                
                # Compute marginal HV
                contrib = get_marginal_hv(f, archive_F, ref_point)
                
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_idx = ci
                    best_f = f
            
            if best_idx == -1:
                # No non-dominated point found in sample, try more
                remaining = [ci for ci in candidate_indices if ci not in sample_candidates]
                if remaining:
                    sample_size = min(30, len(remaining))
                    sample_idx = np.random.choice(len(remaining), size=sample_size, replace=False)
                    sample_candidates2 = [remaining[i] for i in sample_idx]
                    for ci in sample_candidates2:
                        if time.time() > deadline:
                            break
                        x = candidates[ci]
                        f = problem(x)
                        used += 1
                        is_dominated = any(dominates(af, f) for af in archive_F)
                        if is_dominated:
                            continue
                        contrib = get_marginal_hv(f, archive_F, ref_point)
                        if contrib > best_contrib:
                            best_contrib = contrib
                            best_idx = ci
                            best_f = f
            
            if best_idx != -1 and best_f is not None:
                x = candidates[best_idx]
                # Remove from candidates
                candidate_indices.remove(best_idx)
                # Update archive
                dom_idx = [i for i, af in enumerate(archive_F) if dominates(best_f, af)]
                for i in sorted(dom_idx, reverse=True):
                    archive_X.pop(i)
                    archive_F.pop(i)
                archive_X.append(x)
                archive_F.append(best_f)
            else:
                # If no good point found, do a random evaluation
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
        
        # Final phase: local refinement around best archive points
        if time.time() < deadline and used < budget:
            for f_arch, x_arch in list(zip(archive_F, archive_X)):
                if used >= budget or time.time() > deadline:
                    break
                # Try small perturbations around this point
                n_perturb = min(5, budget - used)
                for _ in range(n_perturb):
                    if used >= budget or time.time() > deadline:
                        break
                    perturbation = np.random.normal(0, 0.05, dim)
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

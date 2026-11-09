"""Multi-objective optimizer on RE21 using Sobol sampling + local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Sobol-based multi-objective optimizer with local polish phase."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref = np.array([3000.0, 0.06], dtype=np.float64)

        archive = []  # list of (x, f) tuples
        used = 0

        def is_valid(f):
            return np.all(f < ref) and np.all(np.isfinite(f))

        def compute_hypervolume(points, ref_point):
            """Compute hypervolume of 2D points relative to ref_point."""
            if len(points) == 0:
                return 0.0
            sorted_pts = sorted(points, key=lambda p: p[0])
            hv = 0.0
            prev_y = ref_point[1]
            for p in sorted_pts:
                if p[0] >= ref_point[0] or p[1] >= ref_point[1]:
                    continue
                if p[1] < prev_y:
                    hv += (ref_point[0] - p[0]) * (prev_y - p[1])
                    prev_y = p[1]
            return hv

        def marginal_hypervolume(point, archive_points, ref_point):
            """Compute marginal hypervolume contribution of a point."""
            hv_with = compute_hypervolume(archive_points + [point], ref_point)
            hv_without = compute_hypervolume(archive_points, ref_point)
            return hv_with - hv_without

        def add_to_archive(x, f):
            if not is_valid(f):
                return
            # Check if dominated by existing archive
            for _, af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return
            # Remove dominated points
            new_archive = []
            for ax, af in archive:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append((ax, af))
            new_archive.append((x, f))
            archive[:] = new_archive

        def evaluate(x):
            nonlocal used
            if used >= self.budget or time.time() > deadline:
                return None
            f = problem(x)
            used += 1
            if is_valid(f):
                add_to_archive(x, f)
            return f

        # Phase 1: Deterministic Sobol sequence sampling
        # Use a quasi-random sequence for better coverage
        n_sobol = min(120, self.budget)
        
        # Simple Sobol-like sequence using golden ratio for 4D
        # This gives good low-discrepancy coverage
        golden = (1 + np.sqrt(5)) / 2.0
        inv_golden = 1.0 / golden
        inv_golden2 = inv_golden * inv_golden
        inv_golden3 = inv_golden2 * inv_golden
        inv_golden4 = inv_golden3 * inv_golden
        
        for i in range(n_sobol):
            if used >= self.budget or time.time() > deadline:
                break
            x = np.zeros(self.dim)
            for d in range(self.dim):
                # Use different golden ratio powers for each dimension
                if d == 0:
                    t = (i * inv_golden) % 1.0
                elif d == 1:
                    t = (i * inv_golden2) % 1.0
                elif d == 2:
                    t = (i * inv_golden3) % 1.0
                else:
                    t = (i * inv_golden4) % 1.0
                x[d] = lo[d] + (hi[d] - lo[d]) * t
            evaluate(x)

        # Phase 2: Local polish on top hypervolume contributors
        if len(archive) > 0 and used < self.budget and time.time() < deadline:
            # Compute marginal hypervolume contributions
            archive_points = [f for _, f in archive]
            contributions = []
            for i, (x_i, f_i) in enumerate(archive):
                contrib = marginal_hypervolume(f_i, archive_points, ref)
                contributions.append(contrib)
            
            # Sort by contribution descending, take top 10
            sorted_indices = np.argsort(contributions)[::-1]
            top_k = min(10, len(archive))
            
            for idx in sorted_indices[:top_k]:
                if used >= self.budget or time.time() > deadline:
                    break
                
                parent_x = archive[idx][0].copy()
                
                # Try local optimization on a scalarized objective
                # Use weighted sum with weights that emphasize hypervolume-critical regions
                # For 2 objectives, try several weight combinations
                weights_to_try = [
                    [0.5, 0.5],
                    [0.3, 0.7],
                    [0.7, 0.3],
                    [0.1, 0.9],
                    [0.9, 0.1],
                ]
                
                for w in weights_to_try:
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    def scalar_obj(x_vec, w=w):
                        f = problem(x_vec)
                        if not np.all(np.isfinite(f)):
                            return 1e10
                        return w[0] * f[0] + w[1] * f[1]
                    
                    # Use Nelder-Mead for local refinement
                    try:
                        result = minimize(
                            scalar_obj,
                            parent_x,
                            method='Nelder-Mead',
                            options={'maxiter': 5, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
                        )
                    except Exception:
                        continue
                    
                    # Evaluate the result
                    x_opt = np.clip(result.x, lo, hi)
                    f_opt = problem(x_opt)
                    used += 1
                    if is_valid(f_opt):
                        add_to_archive(x_opt, f_opt)
                    
                    # Also evaluate the best point found during minimization
                    # (Nelder-Mead doesn't easily expose this, so we just use the result)
        
        return archive

# EVOLVE-BLOCK-END

"""Multi-objective optimizer for RE21 using HypE-style selection with stratified init and local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """HypE-style multi-objective optimizer for RE21 with stratified initialization and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref_point = np.array([3000.0, 0.06])
        range_vec = hi - lo

        # Archive: list of (x, f) pairs that are non-dominated
        archive_X = []
        archive_F = []
        used = 0

        def dominates(a, b):
            """Check if a dominates b (a <= b in all objectives, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def compute_hypervolume_2d(points, ref_pt):
            """Compute exact hypervolume for 2D points with given reference point."""
            valid = [p for p in points if p[0] < ref_pt[0] and p[1] < ref_pt[1]]
            if not valid:
                return 0.0
            valid.sort(key=lambda p: p[0])
            hv = 0.0
            prev_y = ref_pt[1]
            for p in valid:
                if p[1] < prev_y:
                    hv += (ref_pt[0] - p[0]) * (prev_y - p[1])
                    prev_y = p[1]
            return hv

        def add_to_archive(x, f):
            """Add a point to the archive if it's not dominated. Remove dominated points."""
            nonlocal archive_X, archive_F
            if f[0] >= ref_point[0] or f[1] >= ref_point[1]:
                return False
            
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return False
            
            new_X = []
            new_F = []
            for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_X.append(ax)
                    new_F.append(af)
            
            new_X.append(x.copy())
            new_F.append(f.copy())
            archive_X = new_X
            archive_F = new_F
            return True

        def evaluate_and_add(x):
            """Evaluate x and add to archive if non-dominated."""
            nonlocal used
            f = problem(x)
            f = np.array(f, dtype=np.float64)
            used += 1
            add_to_archive(x.copy(), f)
            return f

        def compute_hv_contributions():
            """Compute marginal hypervolume contribution of each archive point."""
            n = len(archive_F)
            if n == 0:
                return np.array([])
            if n == 1:
                f = archive_F[0]
                if f[0] < ref_point[0] and f[1] < ref_point[1]:
                    return np.array([(ref_point[0] - f[0]) * (ref_point[1] - f[1])])
                return np.array([0.0])
            
            # Sort by f[0] ascending
            sorted_idx = np.argsort([f[0] for f in archive_F])
            sorted_F = [archive_F[i] for i in sorted_idx]
            
            contribs = np.zeros(n)
            for i in range(n - 1):
                w = sorted_F[i+1][0] - sorted_F[i][0]
                h = ref_point[1] - sorted_F[i][1]
                contribs[i] = max(0.0, w) * max(0.0, h)
            
            w = ref_point[0] - sorted_F[-1][0]
            h = ref_point[1] - sorted_F[-1][1]
            contribs[-1] = max(0.0, w) * max(0.0, h)
            
            # Map back to original indices
            result = np.zeros(n)
            for i, orig_idx in enumerate(sorted_idx):
                result[orig_idx] = contribs[i]
            return result

        def prune_archive_by_hv(max_size):
            """Prune archive to non-dominated points and limit size by hypervolume contribution."""
            nonlocal archive_X, archive_F
            if len(archive_X) <= max_size:
                return
            
            while len(archive_X) > max_size and time.time() < deadline - 5:
                n = len(archive_X)
                if n <= 1:
                    break
                
                contribs = compute_hv_contributions()
                min_idx = np.argmin(contribs)
                archive_X.pop(min_idx)
                archive_F.pop(min_idx)

        def select_parent_hype():
            """Select a parent using HypE-style marginal hypervolume contribution weighting."""
            n = len(archive_F)
            if n == 0:
                return None
            if n == 1:
                return archive_X[0]
            
            contribs = compute_hv_contributions()
            total = np.sum(contribs)
            if total <= 0:
                return archive_X[np.random.randint(n)]
            
            probs = contribs / total
            idx = np.random.choice(n, p=probs)
            return archive_X[idx]

        # Phase 1: Stratified initialization for better coverage
        n_init = min(40, self.budget // 4)
        # Use Latin hypercube-like stratified sampling in 4D
        n_per_cell = max(1, n_init // (2 ** self.dim))
        for i in range(n_init):
            # Stratified: divide each dimension into 2 bins, pick one point per cell
            cell = i
            x = np.zeros(self.dim)
            for d in range(self.dim):
                bin_idx = (cell >> d) % 2
                frac = (bin_idx + np.random.random()) / 2.0
                x[d] = lo[d] + frac * range_vec[d]
            evaluate_and_add(x)

        # Phase 2: HypE-style evolutionary search
        max_archive_size = 60
        # Adaptive sigma: start wide, narrow down
        while used < self.budget:
            if time.time() > deadline - 15:
                break
            
            if not archive_F:
                x = lo + range_vec * np.random.random(self.dim)
                evaluate_and_add(x)
                continue
            
            # Select parent using HypE weighting
            parent = select_parent_hype()
            if parent is None:
                parent = archive_X[np.random.randint(len(archive_F))]
            
            # Adaptive mutation sigma
            progress = used / self.budget
            # Exponential decay from 0.4*range to 0.02*range
            sigma_scale = 0.4 * np.exp(-3.0 * progress) + 0.02
            sigma = sigma_scale * range_vec
            
            # Gaussian mutation
            child = parent + np.random.standard_normal(self.dim) * sigma
            child = np.clip(child, lo, hi)
            
            evaluate_and_add(child)
            
            # Prune archive periodically
            if len(archive_X) > max_archive_size:
                prune_archive_by_hv(max_archive_size)
            
            # Maintain diversity: inject random points every 15 evaluations
            if used % 15 == 0 and used < self.budget - 10:
                x = lo + range_vec * np.random.random(self.dim)
                evaluate_and_add(x)

        # Phase 3: Local search polish on top-contributing points
        if archive_X and time.time() < deadline - 10:
            contribs = compute_hv_contributions()
            n = len(archive_X)
            top_k = min(8, n)
            top_indices = np.argsort(contribs)[-top_k:]
            top_indices = sorted(top_indices, key=lambda i: -contribs[i])
            
            # Polish each top point with adaptive local search
            for idx in top_indices:
                if used >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break
                
                best_x = archive_X[idx].copy()
                best_f = archive_F[idx].copy()
                
                # Adaptive local search: try multiple scales
                base_sigma = 0.03 * range_vec
                max_local_steps = min(12, self.budget - used)
                
                for step in range(max_local_steps):
                    if used >= self.budget:
                        break
                    if time.time() > deadline - 3:
                        break
                    
                    # Decreasing step size
                    scale = base_sigma * (1.0 - 0.5 * step / max_local_steps)
                    
                    # Perturb a random subset of dimensions (2 of 4)
                    dims_to_perturb = np.random.choice(self.dim, size=2, replace=False)
                    new_x = best_x.copy()
                    for d in dims_to_perturb:
                        new_x[d] += np.random.standard_normal() * scale[d]
                    new_x = np.clip(new_x, lo, hi)
                    
                    new_f = problem(new_x)
                    new_f = np.array(new_f, dtype=np.float64)
                    used += 1
                    
                    if dominates(new_f, best_f):
                        best_x = new_x
                        best_f = new_f
                    elif not dominates(best_f, new_f):
                        add_to_archive(new_x.copy(), new_f)

                # Add the best found point to archive
                add_to_archive(best_x.copy(), best_f.copy())

        return archive_X if archive_X else [lo + range_vec * 0.5]
# EVOLVE-BLOCK-END

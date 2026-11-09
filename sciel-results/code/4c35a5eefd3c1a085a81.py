"""Multi-objective optimizer using HypE strategy with Hypervolume contribution selection."""
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
        deadline = start_time + 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Reference point for hypervolume calculation
        # Task says reference point [3000.0, 0.06]
        # We only care about points strictly better than reference in every objective.
        # Let's determine bounds for reference dynamically or use task's if available?
        # Task says: "reference point [3000.0, 0.06]"
        # However, for local selection, we can use a dynamic reference or the task one.
        # Using task's reference is safer for consistency with scoring.
        ref = np.array([3000.0, 0.06])
        
        # 1. Latin Hypercube Sampling for initialization
        n_init = min(20, self.budget // 2)
        if n_init < 2: n_init = 2
        
        X_init = np.zeros((n_init, self.dim))
        for i in range(self.dim):
            # Divide range into n_init equal segments, pick one point in each
            segments = np.linspace(lo[i], hi[i], n_init + 1)
            # Random offset within each segment
            offsets = np.random.random(n_init) * (segments[1] - segments[0])
            X_init[:, i] = segments[:-1] + offsets
            
        F_init = np.array([problem(x) for x in X_init])
        
        # Archive: list of non-dominated points (X, F)
        # We maintain a list of indices or just the points.
        # For efficiency with small budgets, a simple list is fine.
        
        archive_X = []
        archive_F = []
        
        def is_dominated(f_new, archive_F_list):
            """Check if f_new is dominated by any point in archive_F_list."""
            for f_a in archive_F_list:
                if np.all(f_a <= f_new) and np.any(f_a < f_new):
                    return True
            return False

        def dominates(f_new, archive_F_list):
            """Check which points in archive_F_list are dominated by f_new."""
            dominated_idx = []
            for i, f_a in enumerate(archive_F_list):
                if np.all(f_new <= f_a) and np.any(f_new < f_a):
                    dominated_idx.append(i)
            return dominated_idx

        # Initialize archive
        for i in range(n_init):
            f = F_init[i]
            # Check if f is better than ref
            if np.all(f < ref):
                if not is_dominated(f, archive_F):
                    # Remove points dominated by f
                    dom_idx = dominates(f, archive_F)
                    for idx in sorted(dom_idx, reverse=True):
                        archive_X.pop(idx)
                        archive_F.pop(idx)
                    archive_X.append(X_init[i])
                    archive_F.append(f)
        
        used = n_init
        
        # Hypervolume calculation for 2D case
        # Points must be sorted by f1 ascending.
        # HV = sum over i of (f1[i+1] - f1[i]) * (ref2 - f2[i])
        # where f1[i+1] is min(f1[i+1], ref1) and we only include points with f2 < ref2
        
        def calc_hv(archive_F_list):
            if not archive_F_list:
                return 0.0
            # Filter points strictly better than ref
            pts = [f for f in archive_F_list if np.all(f < ref)]
            if not pts:
                return 0.0
            
            # Sort by first objective
            pts.sort(key=lambda x: x[0])
            
            hv = 0.0
            ref1, ref2 = ref[0], ref[1]
            
            for i in range(len(pts)):
                f1 = pts[i][0]
                f2 = pts[i][1]
                
                # Next point's f1 or ref1
                if i + 1 < len(pts):
                    next_f1 = pts[i+1][0]
                else:
                    next_f1 = ref1
                
                # Width in obj1
                w1 = next_f1 - f1
                if w1 <= 0: continue
                
                # Height in obj2
                h2 = ref2 - f2
                if h2 <= 0: continue
                
                hv += w1 * h2
                
            return hv

        def calc_contribution(f_new, archive_F_list):
            """Calculate HV contribution of f_new given the archive."""
            # HV with new point - HV without
            new_archive = archive_F_list + [f_new]
            # To be efficient, we only need to consider the region affected.
            # But for n<=200, simple recalculation is fast enough.
            hv_with = calc_hv(new_archive)
            hv_without = calc_hv(archive_F_list)
            return hv_with - hv_without

        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Selection: Pick parent with highest HV contribution
            # If archive is empty, pick random
            if not archive_F:
                # Fallback: random point from initial or bounds
                x_parent = lo + (hi - lo) * np.random.random(self.dim)
            else:
                # Sample a few candidates to speed up if archive is large?
                # With budget 200, archive size is likely < 50. Full scan is fine.
                best_idx = 0
                best_contrib = -np.inf
                
                # Optimization: We don't need to check all if we are fast enough.
                # But let's stick to full scan for correctness first.
                for i in range(len(archive_F)):
                    contrib = calc_contribution(archive_F[i], archive_F)
                    if contrib > best_contrib:
                        best_contrib = contrib
                        best_idx = i
                
                x_parent = archive_X[best_idx]
            
            # Mutation: Gaussian
            # Adaptive sigma? Let's use a fixed fraction of range for now, maybe 0.1
            sigma = 0.1 * (hi - lo)
            x_child = x_parent + np.random.normal(0, sigma)
            x_child = np.clip(x_child, lo, hi)
            
            f_child = problem(x_child)
            used += 1
            
            # Update Archive
            if np.all(f_child < ref):
                if not is_dominated(f_child, archive_F):
                    # Remove dominated points
                    dom_idx = dominates(f_child, archive_F)
                    for idx in sorted(dom_idx, reverse=True):
                        archive_X.pop(idx)
                        archive_F.pop(idx)
                    
                    archive_X.append(x_child)
                    archive_F.append(f_child)

        return X_init # Return value is ignored, but must not crash

# EVOLVE-BLOCK-END

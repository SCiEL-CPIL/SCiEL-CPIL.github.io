"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Adaptive Tchebycheff Multi-Objective Optimizer with Gap-Filling Strategy."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        used = 0
        archive_X = []
        archive_F = []
        
        def add_to_archive(x, f):
            nonlocal archive_X, archive_F
            f = np.array(f)
            # Check domination
            dominated = False
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break
            
            if dominated:
                return False
            
            # Remove points dominated by f
            new_archive_X = []
            new_archive_F = []
            for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            
            new_archive_X.append(x)
            new_archive_F.append(f)
            
            archive_X = new_archive_X
            archive_F = new_archive_F
            return True

        # Phase 1: Initial Exploration using Latin Hypercube Sampling
        # Use 20% of budget for initial spread
        n_init = min(40, self.budget // 5)
        lhs_points = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.arange(n_init)
            np.random.shuffle(perm)
            lhs_points[:, d] = (perm + 0.5) / n_init
        
        X_init = lo + lhs_points * span
        for x in X_init:
            if time.time() > deadline or used >= self.budget:
                break
            f = problem(x)
            used += 1
            add_to_archive(x, f)
        
        # Phase 2: Adaptive Tchebycheff Optimization
        # We iteratively identify "gaps" in the current front and optimize towards them.
        # Tchebycheff function: min max_i (w_i * |f_i(x) - z_i^ref|)
        # z^ref is a reference point on the front or slightly outside.
        
        # We will run this loop until budget is exhausted or time runs out.
        # Each run uses ~5-10 evaluations.
        
        max_runs = (self.budget - used) // 8
        if max_runs < 5:
            max_runs = 5
            
        for run_idx in range(max_runs):
            if time.time() > deadline or used >= self.budget:
                break
                
            # Determine reference point for this run
            # Strategy: Pick a point from the current archive that is "far" from others
            # or simply cycle through a set of weight vectors but adapt the reference point
            # to be the ideal point or a point on the current front.
            
            # To fill gaps, we select a weight vector. 
            # We use a dynamic set of weights based on the current front's extent.
            # For simplicity and robustness, we use a predefined set of dense weights
            # but adapt the reference point z_ref to be the current best-known point 
            # in that direction, or the ideal point if unknown.
            
            # Let's use a set of weight vectors that cover the simplex.
            # Since we have 3 objectives, we can use a grid of weights.
            # However, to be adaptive, we pick a random weight vector each time,
            # biased towards regions of the front that are sparsely populated.
            
            # Simple adaptive strategy: 
            # 1. If archive is small, use uniform weights.
            # 2. Otherwise, pick a weight vector that points towards the "middle" 
            #    of the gaps between existing points in the objective space.
            
            if len(archive_F) < 5:
                # Early stage: use standard vertices and midpoints
                weights_list = [
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                    np.array([0.0, 0.0, 1.0]),
                    np.array([0.5, 0.5, 0.0]),
                    np.array([0.5, 0.0, 0.5]),
                    np.array([0.0, 0.5, 0.5]),
                    np.array([1.0/3, 1.0/3, 1.0/3])
                ]
                w = weights_list[run_idx % len(weights_list)]
            else:
                # Later stage: Pick a random weight vector to explore non-convex parts
                # Or pick a weight that corresponds to a direction not well covered.
                # For now, random sampling of the simplex is effective for filling gaps.
                w = np.random.random(3)
                w = w / np.sum(w)
            
            # Determine Reference Point z_ref
            # z_ref should be a point on the ideal front or slightly beyond.
            # We can estimate the ideal point as the minimum of each objective in the archive.
            # If archive is empty, use the bounds' lower limits (assuming minimization).
            if archive_F:
                z_ideal = np.min(archive_F, axis=0)
                # Add a small epsilon to avoid being exactly on a known point, 
                # or use the point itself to refine it.
                # For Tchebycheff, using the ideal point as reference helps find 
                # points that improve the nadir or fill the front.
                # Let's use the ideal point scaled slightly outwards to encourage new points.
                z_ref = z_ideal * 0.95 # Slightly inside the ideal point? 
                # Actually, for minimization, we want to find points that are 
                # "better" than the current front in some direction.
                # A common choice is the ideal point itself.
                z_ref = z_ideal.copy()
            else:
                # Fallback: use the lower bounds
                z_ref = lo.copy()

            # Multi-start for this weight vector
            n_starts = 3
            starts = []
            
            # Start 1: Center
            starts.append((lo + hi) / 2.0)
            
            # Start 2 & 3: Random points or points from archive
            if archive_X:
                # Pick random points from archive
                idxs = np.random.randint(0, len(archive_X), size=min(2, len(archive_X)))
                for idx in idxs:
                    # Perturb slightly
                    x_start = archive_X[idx].copy()
                    x_start += np.random.normal(0, 0.1, self.dim)
                    x_start = np.clip(x_start, lo, hi)
                    starts.append(x_start)
            else:
                for _ in range(2):
                    x_start = lo + np.random.random(self.dim) * span
                    starts.append(x_start)
            
            best_f_this_run = None
            best_x_this_run = None
            
            for x0 in starts:
                if used >= self.budget or time.time() > deadline:
                    break
                    
                # Tchebycheff Objective Function
                def tcheb_obj(x):
                    nonlocal used
                    if used >= self.budget:
                        return 1e10
                    fx = problem(x)
                    used += 1
                    # Calculate Tchebycheff distance
                    # max_i (w_i * |f_i(x) - z_ref_i|)
                    # Since we are minimizing, and z_ref is likely the ideal point (lower than fx),
                    # f_i(x) - z_ref_i is positive.
                    dists = w * (fx - z_ref)
                    return np.max(dists)
                
                try:
                    res = minimize(
                        tcheb_obj, 
                        x0, 
                        method='L-BFGS-B', 
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 10, 'maxfev': 5, 'ftol': 1e-4}
                    )
                    
                    # Evaluate the result to get the actual objective vector
                    if used < self.budget:
                        x_best = res.x
                        f_best = problem(x_best)
                        used += 1
                        
                        # Add to archive if it improves hypervolume (non-dominated)
                        if add_to_archive(x_best, f_best):
                            if best_f_this_run is None:
                                best_f_this_run = f_best
                                best_x_this_run = x_best
                            else:
                                # Keep the one with lower sum or similar metric if both non-dominated
                                if np.sum(f_best) < np.sum(best_f_this_run):
                                    best_f_this_run = f_best
                                    best_x_this_run = x_best
                                    
                except Exception:
                    pass

        # Phase 3: Final Polish with Weighted Sum on the Best Points
        # If budget remains, do a few quick weighted sum optimizations on the 
        # most promising directions.
        
        if archive_F and used < self.budget:
            # Pick a few weight vectors
            final_weights = [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([1.0/3, 1.0/3, 1.0/3])
            ]
            
            for w in final_weights:
                if time.time() > deadline or used >= self.budget:
                    break
                
                # Start from a point in the archive that is close to this weight direction
                # Project archive points onto the weight vector
                if archive_F:
                    scores = np.array([np.dot(w, f) for f in archive_F])
                    best_idx = np.argmin(scores)
                    x0 = archive_X[best_idx].copy()
                else:
                    x0 = (lo + hi) / 2.0
                
                def ws_obj(x):
                    nonlocal used
                    if used >= self.budget:
                        return 1e10
                    fx = problem(x)
                    used += 1
                    return np.dot(w, fx)
                
                try:
                    res = minimize(
                        ws_obj, 
                        x0, 
                        method='L-BFGS-B', 
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 5, 'maxfev': 3, 'ftol': 1e-5}
                    )
                    if used < self.budget:
                        x_best = res.x
                        f_best = problem(x_best)
                        used += 1
                        add_to_archive(x_best, f_best)
                except Exception:
                    pass

        return archive_X

# EVOLVE-BLOCK-END

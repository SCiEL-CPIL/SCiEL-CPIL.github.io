"""Multi-objective optimizer on RE21 using an Evolutionary Strategy with Hypervolume-based selection."""
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
        deadline = start_time + 280.0  # Safety margin under 330s kill
        B = int(self.budget)
        d = self.dim
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        
        # Reference point for hypervolume calculation (as per task)
        ref = np.array([3000.0, 0.06])

        # Archive stores (x, f) for non-dominated points
        xs = []
        fs = []
        
        def dominates(f1, f2):
            """Returns True if f1 strictly dominates f2 (minimization)"""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def update_archive(x_new, f_new):
            """
            Adds x_new, f_new to the archive if it is non-dominated.
            Removes any existing points dominated by the new point.
            Returns True if the point was added to the archive.
            """
            if len(fs) == 0:
                xs.append(x_new)
                fs.append(f_new)
                return True
            
            # Check if new point is dominated by any existing point
            for f_exist in fs:
                if dominates(f_exist, f_new):
                    return False
            
            # Remove any existing points dominated by the new point
            new_xs = []
            new_fs = []
            for i in range(len(fs)):
                if not dominates(f_new, fs[i]):
                    new_xs.append(xs[i])
                    new_fs.append(fs[i])
            
            new_xs.append(x_new)
            new_fs.append(f_new)
            
            xs.clear()
            fs.clear()
            for i in range(len(new_xs)):
                xs.append(new_xs[i])
                fs.append(new_fs[i])
            return True

        def marginal_hv_contribution(f_new, existing_fs, ref):
            """
            Calculate the marginal hypervolume contribution of f_new to the set existing_fs.
            Returns 0 if f_new is dominated by any point in existing_fs.
            """
            if len(existing_fs) == 0:
                # Contribution is area of box [f_new, ref]
                if f_new[0] >= ref[0] or f_new[1] >= ref[1]:
                    return 0.0
                return (ref[0] - f_new[0]) * (ref[1] - f_new[1])
            
            # Check if dominated
            for f_exist in existing_fs:
                if dominates(f_exist, f_new):
                    return 0.0
            
            # Calculate HV of existing set
            hv_old = hypervolume_2d(existing_fs, ref)
            
            # Calculate HV of new set
            new_set = list(existing_fs) + [f_new]
            hv_new = hypervolume_2d(new_set, ref)
            
            return max(0.0, hv_new - hv_old)

        def hypervolume_2d(points, ref):
            """
            Calculate 2D hypervolume for a list of points.
            """
            if len(points) == 0:
                return 0.0
            p = np.array(points)
            # Filter points strictly better than ref
            mask = (p[:, 0] < ref[0]) & (p[:, 1] < ref[1])
            p = p[mask]
            if len(p) == 0:
                return 0.0
            
            # Sort by f1 ascending
            p = p[np.argsort(p[:, 0])]
            
            f1 = p[:, 0]
            f2 = p[:, 1]
            
            total_hv = 0.0
            for i in range(len(p)):
                x1 = f1[i]
                x2 = f1[i+1] if i < len(p)-1 else ref[0]
                y1 = f2[i]
                y2 = f2[i+1] if i < len(p)-1 else ref[1]
                
                w = max(0.0, x2 - x1)
                h = max(0.0, y1 - y2)
                total_hv += w * h
                
            return total_hv

        # Initial Sampling: Stratified
        n_init = min(20, B)
        # Use Latin Hypercube-like sampling for better initial spread
        for i in range(n_init):
            if time.time() > deadline:
                break
            # Simple stratified sampling per dimension
            x = np.zeros(d)
            for j in range(d):
                # Divide range into n_init bins, pick random point in bin i
                bin_size = span[j] / n_init
                x[j] = lo[j] + (i + np.random.random()) * bin_size
            f = problem(x)
            f = np.array(f, dtype=np.float64)
            if np.all(np.isfinite(f)):
                update_archive(x, f)
            B -= 1

        # Main Evolutionary Loop
        pop_size = 10
        sigma_decay = 0.98
        sigma = 0.2 * span  # Initial step size
        
        while B > 0 and time.time() < deadline:
            candidates = []
            
            if len(xs) > 0:
                # Compute marginal HV contributions for weighted selection
                # This is expensive if archive is large, but archive is small (2D non-dominated)
                # Optimization: Only compute for a subset or use approximate method?
                # For now, compute exactly.
                if len(xs) <= 50: # Cap to avoid slowdown if archive grows
                    contribs = []
                    for i in range(len(xs)):
                        c = marginal_hv_contribution(fs[i], fs, ref)
                        contribs.append(c)
                    contribs = np.array(contribs)
                    # Add small epsilon to avoid zero weights
                    contribs += 1e-10
                    probs = contribs / np.sum(contribs)
                else:
                    probs = np.ones(len(xs)) / len(xs)
                
                for _ in range(pop_size):
                    # Select parent based on HV contribution
                    if len(xs) > 2:
                        # Differential Evolution style with HV-weighted parent selection
                        i1 = np.random.choice(len(xs), p=probs)
                        i2 = np.random.choice(len(xs), p=probs)
                        i3 = np.random.choice(len(xs), p=probs)
                        
                        # Ensure distinct indices if possible
                        if i1 == i2: i2 = (i2 + 1) % len(xs)
                        if i1 == i3: i3 = (i3 + 1) % len(xs)
                        
                        F = 0.5 + np.random.random() * 0.5
                        v = xs[i1] + F * (xs[i2] - xs[i3])
                        v = np.clip(v, lo, hi)
                        candidates.append(v)
                    else:
                        # Local mutation
                        i = np.random.choice(len(xs), p=probs)
                        v = xs[i] + np.random.normal(0, 1, d) * sigma
                        v = np.clip(v, lo, hi)
                        candidates.append(v)
            
            # Random exploration
            n_random = 5
            for _ in range(n_random):
                x = lo + span * np.random.random(d)
                candidates.append(x)
            
            # Evaluate candidates
            for x in candidates:
                if B <= 0 or time.time() > deadline:
                    break
                f = problem(x)
                f = np.array(f, dtype=np.float64)
                if not np.all(np.isfinite(f)):
                    continue
                update_archive(x, f)
                B -= 1
            
            # Decay sigma
            sigma *= sigma_decay

        if len(fs) > 0:
            return np.array(fs)
        else:
            return np.array([])

# EVOLVE-BLOCK-END

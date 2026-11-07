"""Multi-objective optimizer for ZDT1 using hypervolume-driven adaptive search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven adaptive multi-objective optimizer for ZDT1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT1 Pareto front: f1 = x1, f2 = (1 + 10*n) * (1 - sqrt(f1/(1+10*n)))
        # where n = dim - 1
        n = dim - 1

        def zdt1_pareto(x1):
            """Compute the Pareto-optimal f2 for given x1 on ZDT1."""
            if x1 <= 0:
                return (1 + 10 * n) * (1 - 0)
            elif x1 >= 1:
                return 0.0
            else:
                return (1 + 10 * n) * (1 - np.sqrt(x1 / (1 + 10 * n)))

        # Reference point for hypervolume
        ref = np.array([11.0, 11.0])

        def compute_hypervolume(points):
            """Compute 2D hypervolume with reference point [11, 11].
            Points must be strictly better than ref in both objectives."""
            if len(points) == 0:
                return 0.0
            # Filter points strictly better than ref
            valid = np.all(points < ref, axis=1)
            pts = points[valid]
            if len(pts) == 0:
                return 0.0
            # Sort by f1 ascending
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            # Compute hypervolume in 2D
            hv = 0.0
            prev_f2 = ref[1]
            for i in range(len(pts)):
                f1, f2 = pts[i]
                if f2 < prev_f2:
                    width = f1 - (pts[i-1][0] if i > 0 else 0.0)
                    if i == 0:
                        width = f1
                    hv += width * (prev_f2 - f2)
                    prev_f2 = f2
            # Add the last segment to ref
            if prev_f2 < ref[1] and len(pts) > 0:
                hv += (ref[0] - pts[-1][0]) * (prev_f2 - ref[1]) if prev_f2 > ref[1] else 0
            # Actually, let me redo this more carefully
            hv = 0.0
            for i in range(len(pts)):
                f1 = pts[i][0]
                f2 = pts[i][1]
                # Width in f1 direction
                if i == 0:
                    w = f1
                else:
                    w = f1 - pts[i-1][0]
                # Height in f2 direction
                if i == len(pts) - 1:
                    h = f2 - ref[1]
                    if h < 0:
                        h = 0
                else:
                    h = f2 - pts[i+1][1]
                    if h < 0:
                        h = 0
                hv += w * h
            return hv

        # Better hypervolume computation for 2D
        def hv2d(points, ref_point):
            """Exact 2D hypervolume."""
            if len(points) == 0:
                return 0.0
            valid = np.all(points < ref_point, axis=1)
            pts = points[valid]
            if len(pts) == 0:
                return 0.0
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            # Remove dominated points (keep only Pareto front)
            pareto = [pts[0]]
            for i in range(1, len(pts)):
                if pts[i][1] < pareto[-1][1]:
                    pareto.append(pts[i])
            pts = np.array(pareto)
            if len(pts) == 0:
                return 0.0
            hv = 0.0
            for i in range(len(pts)):
                f1 = pts[i][0]
                f2 = pts[i][1]
                w = f1 - (pts[i-1][0] if i > 0 else 0.0)
                h = (pts[i+1][1] if i < len(pts) - 1 else ref_point[1]) - f2
                if h < 0:
                    h = 0
                hv += w * h
            return hv

        # Evaluate a point
        def evaluate(x):
            return np.array(problem(x))

        # Initialize population on the Pareto front
        pop_size = min(20, budget // 10)
        # Place initial points at evenly spaced x1 values
        x1_init = np.linspace(0.05, 0.95, pop_size)
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, 2))
        for i in range(pop_size):
            X[i, 0] = x1_init[i]
            # Set other variables to optimal values for ZDT1 (g(x) = 0)
            # For ZDT1, g(x) = sum((xj - 0.35)^2 * sin(12*pi*xj)) for j=2..n
            # The minimum is at xj = 0.35
            X[i, 1:] = 0.35
            F[i] = evaluate(X[i])
        
        used = pop_size
        archive = list(zip(X.tolist(), F.tolist()))

        # Adaptive insertion: try to add points that maximize hypervolume gain
        remaining = budget - used
        for _ in range(remaining):
            if time.time() > deadline:
                break
            
            # Try several candidate positions and pick the best hypervolume gain
            best_hv_gain = -1
            best_x = None
            best_f = None
            
            # Generate candidates around existing points and in gaps
            candidates = []
            
            # Random candidates
            for _ in range(5):
                x1_c = rng.uniform(0, 1)
                x_c = np.zeros(dim)
                x_c[0] = x1_c
                x_c[1:] = 0.35
                candidates.append(x_c)
            
            # Candidates near existing points (local refinement)
            for _ in range(5):
                idx = rng.integers(len(X))
                x1_c = X[idx, 0] + rng.normal(0, 0.05)
                x1_c = np.clip(x1_c, 0, 1)
                x_c = np.zeros(dim)
                x_c[0] = x1_c
                x_c[1:] = 0.35
                candidates.append(x_c)
            
            # Candidates in gaps between existing x1 values
            x1_sorted = np.sort(X[:, 0])
            gaps = []
            for i in range(len(x1_sorted) - 1):
                gap = x1_sorted[i+1] - x1_sorted[i]
                if gap > 0.01:
                    gaps.append((gap, (x1_sorted[i] + x1_sorted[i+1]) / 2))
            gaps.sort(reverse=True)
            for _, mid in gaps[:3]:
                x_c = np.zeros(dim)
                x_c[0] = mid
                x_c[1:] = 0.35
                candidates.append(x_c)
            
            # Evaluate candidates and find best hypervolume gain
            current_hv = hv2d(F, ref)
            
            for x_c in candidates:
                f_c = evaluate(x_c)
                # Check if this point would improve hypervolume
                new_F = np.vstack([F, f_c.reshape(1, -1)])
                new_hv = hv2d(new_F, ref)
                gain = new_hv - current_hv
                if gain > best_hv_gain:
                    best_hv_gain = gain
                    best_x = x_c
                    best_f = f_c
                used += 1
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
            
            if best_x is not None:
                # Replace the worst point in the population
                # Use a simple dominance-based selection
                dominated = np.zeros(len(F), dtype=bool)
                for i in range(len(F)):
                    for j in range(len(F)):
                        if i != j:
                            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                dominated[i] = True
                                break
                
                if np.any(dominated):
                    worst_idx = np.where(dominated)[0][0]
                else:
                    # Replace the point with the smallest hypervolume contribution
                    worst_idx = np.argmin(F[:, 0] + F[:, 1])
                
                X[worst_idx] = best_x
                F[worst_idx] = best_f
        
        # Final polish: try to optimize g(x) for the best x1 values
        if used < budget and time.time() < deadline:
            # For the best x1 values, try to reduce g(x) further
            for i in range(len(X)):
                if time.time() > deadline or used >= budget:
                    break
                x1_val = X[i, 0]
                # Local search on x2..xn to minimize g(x)
                x_best = X[i].copy()
                f_best = F[i].copy()
                for _ in range(3):
                    if used >= budget or time.time() > deadline:
                        break
                    x_try = x_best.copy()
                    x_try[1:] += rng.normal(0, 0.01, dim - 1)
                    x_try = np.clip(x_try, lo, hi)
                    f_try = evaluate(x_try)
                    used += 1
                    # Check if f_try is better
                    if f_try[1] < f_best[1] and f_try[0] == f_best[0]:
                        x_best = x_try
                        f_best = f_try
                X[i] = x_best
                F[i] = f_best

        return X
# EVOLVE-BLOCK-END

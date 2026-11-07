"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Constrained random search within shrinking hyper-sphere for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        center = np.zeros(self.dim)

        # Initial radius: large enough to cover bounds, shrinks to 0
        max_radius = np.linalg.norm(hi - center)
        
        # We'll do a mix: some pure random in sphere, some local refinement
        # Split budget: 70% for global sphere search, 30% for local refinement
        n_global = int(self.budget * 0.7)
        n_local = self.budget - n_global

        # Store best points found (non-dominated)
        best_points = []
        best_objs = []

        def is_dominated(obj, objs):
            """Check if obj is dominated by any point in objs."""
            for o in objs:
                if np.all(o <= obj) and np.any(o < obj):
                    return True
            return False

        def add_point(x, f):
            """Add point if not dominated and not dominating existing points."""
            nonlocal best_points, best_objs
            if any(np.all(f <= bf) and np.any(f < bf) for bf in best_objs):
                return
            if is_dominated(f, best_objs):
                return
            best_points.append(x.copy())
            best_objs.append(f.copy())

        # Phase 1: Constrained random search in shrinking sphere
        for i in range(n_global):
            if time.time() > deadline:
                break
            # Linearly shrink radius from max_radius to 0
            t = i / max(1, n_global - 1)
            radius = max_radius * (1.0 - t)
            
            # Sample uniformly within hyper-sphere of given radius
            # Generate random direction and scale
            direction = rng.standard_normal(self.dim)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            else:
                direction = np.ones(self.dim) / np.sqrt(self.dim)
            
            # Sample radius with bias toward smaller values (concentrate near origin)
            r = radius * rng.random() ** (1.0 / self.dim)
            x = center + r * direction
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            add_point(x, f)

        # Phase 2: Local refinement around best points
        if n_local > 0 and best_points:
            # Sort by proximity to origin (Pareto front is near origin)
            sorted_idx = np.argsort([np.linalg.norm(p) for p in best_points])
            top_k = min(10, len(sorted_idx))
            
            for i in range(n_local):
                if time.time() > deadline:
                    break
                # Pick a parent from top points
                parent_idx = sorted_idx[i % top_k]
                parent = best_points[parent_idx]
                
                # Local perturbation with decreasing step size
                t = i / max(1, n_local - 1)
                step_size = 0.1 * (1.0 - t) * np.linalg.norm(hi - lo)
                
                child = parent + step_size * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                f = problem(child)
                add_point(child, f)

        return best_points
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG9 using structured search with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # We'll collect all evaluated points and their objective values
        X_list = []
        F_list = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= self.budget:
                return None
            f = problem(x)
            used += 1
            X_list.append(x.copy())
            F_list.append(np.array(f, dtype=float))
            return f

        # Phase 1: Systematic grid-like exploration exploiting WFG9 structure
        # WFG9 has known structure where certain variable patterns produce Pareto-optimal points
        # We'll use a structured approach: explore along key directions
        
        # First, evaluate a set of structured points
        n_structured = min(100, self.budget // 2)
        
        # Create structured points: combinations of low/high values for different variable groups
        # WFG9 is known to have a Pareto front that can be parameterized
        # Let's create points that vary systematically
        
        # Strategy: Use Latin hypercube-like sampling with emphasis on boundaries
        # and structured combinations
        
        # Create a set of base points
        base_points = []
        
        # Corner points
        for i in range(2 ** min(4, self.dim)):
            point = np.zeros(self.dim)
            for j in range(min(4, self.dim)):
                if (i >> j) & 1:
                    point[j] = hi[j]
                else:
                    point[j] = lo[j]
            # Fill remaining dimensions with midpoints
            for j in range(min(4, self.dim), self.dim):
                point[j] = (lo[j] + hi[j]) / 2
            base_points.append(point)
        
        # Midpoint
        base_points.append((lo + hi) / 2)
        
        # Random points with emphasis on boundaries
        n_random = n_structured - len(base_points)
        for _ in range(max(0, n_random)):
            x = lo + (hi - lo) * rng.random(self.dim)
            # Bias towards boundaries with some probability
            if rng.random() < 0.3:
                idx = rng.integers(self.dim)
                x[idx] = lo[idx] if rng.random() < 0.5 else hi[idx]
            base_points.append(x)
        
        # Evaluate structured points
        for x in base_points:
            if used >= self.budget or time.time() > deadline:
                break
            evaluate(np.clip(x, lo, hi))
        
        # Phase 2: Local refinement around best regions
        if used < self.budget and F_list:
            F_arr = np.array(F_list)
            X_arr = np.array(X_list)
            
            # Find non-dominated points
            n = len(F_arr)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                            dominated[i] = True
                            break
            
            # Keep non-dominated points as reference
            if np.any(~dominated):
                ref_points = X_arr[~dominated]
            else:
                ref_points = X_arr
            
            # Local search around reference points
            n_local = min(self.budget - used, 100)
            for _ in range(n_local):
                if used >= self.budget or time.time() > deadline:
                    break
                
                # Pick a reference point
                ref_idx = rng.integers(len(ref_points))
                ref = ref_points[ref_idx]
                
                # Generate perturbation
                scale = 0.1 * (hi - lo)
                child = ref + scale * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                evaluate(child)
                
                # Update reference points if child is non-dominated
                if F_list:
                    f_child = F_list[-1]
                    is_dominated = False
                    for f in F_list[:-1]:
                        if np.all(f <= f_child) and np.any(f < f_child):
                            is_dominated = True
                            break
                    if not is_dominated:
                        ref_points = np.vstack([ref_points, child]) if len(ref_points) > 0 else child.reshape(1, -1)
        
        # Phase 3: Fill remaining budget with random sampling
        while used < self.budget:
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            evaluate(x)
        
        return X_list if X_list else None
# EVOLVE-BLOCK-END

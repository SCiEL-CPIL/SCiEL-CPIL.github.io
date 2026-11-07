"""Rewritten optimizer using deterministic lattice search + coordinate-wise line search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lattice search followed by coordinate-wise line search for Schaffers F7 ill-conditioned."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        call_count = 0

        def safe_call(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            try:
                val = func(np.asarray(x, dtype=np.float64))
                call_count += 1
                if not np.isfinite(val):
                    return np.inf
                return float(val)
            except Exception:
                call_count += 1
                return np.inf

        # Phase 1: Deterministic lattice search to find the best basin
        # Use a grid over [-5, 5]^d with varying resolution
        # For dim=5, we need to be careful about number of points
        
        best_x = None
        best_f = np.inf

        # Start with a coarse grid
        # Use a stratified grid: try center, corners, and midpoints
        # Then refine around the best point
        
        # Initial diverse set of points
        candidates = []
        
        # Center
        candidates.append(np.zeros(d))
        
        # Box corners
        for mask in range(2 ** d):
            pt = np.array([5.0 if (mask >> i) & 1 else -5.0 for i in range(d)])
            candidates.append(pt)
        
        # Midpoints of each edge
        for i in range(d):
            for sign in [-1, 1]:
                pt = np.zeros(d)
                pt[i] = sign * 2.5
                candidates.append(pt)
        
        # Random-ish deterministic points
        rng = np.random.default_rng(123)
        for _ in range(20):
            candidates.append(rng.uniform(-4.5, 4.5, d))
        
        # Evaluate all candidates
        for c in candidates:
            if call_count >= budget or time.time() > deadline:
                break
            f = safe_call(c)
            if f < best_f:
                best_f = f
                best_x = c.copy()
        
        # Phase 2: Coordinate-wise line search (exploiting axis-aligned structure)
        # Schaffers F7 is ill-conditioned along axes, so coordinate search should work well
        
        if best_x is None:
            # Fallback: random search
            for _ in range(budget - call_count):
                if time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, d)
                f = safe_call(x)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            return best_x, best_f

        # Iterative coordinate descent with line search
        step_sizes = [10.0, 5.0, 2.5, 1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001]
        
        for step in step_sizes:
            if call_count >= budget or time.time() > deadline:
                break
            
            improved = True
            while improved:
                improved = False
                for i in range(d):
                    if call_count >= budget or time.time() > deadline:
                        break
                    
                    # Try positive and negative directions
                    for direction in [1, -1]:
                        new_x = best_x.copy()
                        new_x[i] += direction * step
                        new_x = np.clip(new_x, -5, 5)
                        
                        f = safe_call(new_x)
                        if f < best_f:
                            best_f = f
                            best_x = new_x
                            improved = True
        
        # Phase 3: Fine-tuning with very small steps
        fine_steps = [0.0001, 0.00005, 0.00002, 0.00001, 0.000005, 0.000002, 0.000001]
        for step in fine_steps:
            if call_count >= budget or time.time() > deadline:
                break
            
            for i in range(d):
                if call_count >= budget or time.time() > deadline:
                    break
                for direction in [1, -1]:
                    if call_count >= budget or time.time() > deadline:
                        break
                    new_x = best_x.copy()
                    new_x[i] += direction * step
                    new_x = np.clip(new_x, -5, 5)
                    
                    f = safe_call(new_x)
                    if f < best_f:
                        best_f = f
                        best_x = new_x

        # Phase 4: If we still have budget, do a local random search around best
        if best_x is not None and call_count < budget:
            local_sigma = 0.01
            for _ in range(100):
                if call_count >= budget or time.time() > deadline:
                    break
                x = np.clip(best_x + local_sigma * rng.standard_normal(d), -5, 5)
                f = safe_call(x)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    local_sigma *= 1.2
                else:
                    local_sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END

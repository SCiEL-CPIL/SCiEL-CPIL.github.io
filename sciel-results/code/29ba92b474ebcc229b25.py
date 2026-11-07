"""Combined optimizer: structured global search with periodic awareness + adaptive (1+1)-ES refinement."""
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer for 5D Rastrigin combining structured search and local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(7)

        calls = [0]
        best_x = None
        best_f = np.inf

        def call_func(x):
            nonlocal best_x, best_f
            calls[0] += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Structured coarse search exploiting Rastrigin's periodic structure.
        # Rastrigin's global minimum is at the origin (before shift/rotation).
        # We evaluate a diverse set of candidates: origin, integer points, and random samples.
        candidates = []
        
        # Origin
        candidates.append(np.zeros(dim))
        
        # Integer-valued points along each axis
        for d in range(dim):
            for v in range(-4, 5):
                pt = np.zeros(dim)
                pt[d] = float(v)
                candidates.append(pt)
        
        # Half-integer offsets
        for d in range(dim):
            for offset in [0.5, -0.5]:
                pt = np.zeros(dim)
                pt[d] = offset
                candidates.append(pt)
        
        # 2D slices with integer values
        for d1 in range(dim):
            for d2 in range(d1 + 1, dim):
                for v1 in range(-2, 3):
                    for v2 in range(-2, 3):
                        pt = np.zeros(dim)
                        pt[d1] = float(v1)
                        pt[d2] = float(v2)
                        candidates.append(pt)
        
        # 3D slices (sparse)
        for d1 in range(dim):
            for d2 in range(d1 + 1, dim):
                for d3 in range(d2 + 1, dim):
                    for v1 in range(-1, 2):
                        for v2 in range(-1, 2):
                            for v3 in range(-1, 2):
                                pt = np.zeros(dim)
                                pt[d1] = float(v1)
                                pt[d2] = float(v2)
                                pt[d3] = float(v3)
                                candidates.append(pt)
        
        # Random uniform points for diversity
        for _ in range(40):
            candidates.append(rng.uniform(-5, 5, dim))
        
        # Shuffle candidates for diversity
        for i in range(len(candidates) - 1, 0, -1):
            j = rng.integers(0, i + 1)
            candidates[i], candidates[j] = candidates[j], candidates[i]
        
        # Evaluate up to 60 candidates in Phase 1
        phase1_budget = 60
        for x in candidates:
            if calls[0] >= phase1_budget or time.time() > deadline:
                break
            x_clipped = np.clip(x, -5, 5)
            call_func(x_clipped)
        
        # Phase 2: Multi-start local refinement with adaptive (1+1)-ES
        # Use multiple starting points to handle multi-modality
        starts = [best_x.copy()] if best_x is not None else [np.zeros(dim)]
        
        # Add diverse starting points from the candidate pool
        for _ in range(5):
            starts.append(rng.uniform(-4, 4, dim))
        
        for start_x in starts:
            if calls[0] >= budget or time.time() > deadline:
                break
            
            x = start_x.copy()
            if best_x is not None and np.array_equal(start_x, best_x):
                fx = best_f
            else:
                fx = call_func(x)
                if calls[0] >= budget or time.time() > deadline:
                    break
            
            sigma = 0.5
            no_improve = 0
            max_no_improve = 40
            
            while calls[0] < budget and time.time() < deadline:
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = call_func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.12
                    no_improve = 0
                else:
                    sigma *= 0.93
                    no_improve += 1
                
                if sigma < 0.003:
                    break
                if no_improve >= max_no_improve:
                    break
        
        # Phase 3: Fine-tuning with very small steps near the best point
        if calls[0] < budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            
            while calls[0] < budget and time.time() < deadline:
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = call_func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.08
                else:
                    sigma *= 0.88
                
                if sigma < 0.0005:
                    break
        
        return best_x, best_f

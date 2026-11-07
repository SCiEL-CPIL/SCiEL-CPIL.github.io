"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Rastrigin-aware coordinate local search with multi-basin global loop."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        call_count = 0
        max_calls = self.budget
        best_f = float('inf')
        best_x = np.zeros(self.dim)
        rng = np.random.default_rng(42)
        
        def counted_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= max_calls:
                raise RuntimeError("Budget exceeded")
            call_count += 1
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def coordinate_local_search(x0, max_iter=50):
            """Fast coordinate-wise local search with Rastrigin-aware jumps."""
            nonlocal call_count, best_f, best_x, deadline
            dim = self.dim
            x = x0.copy()
            f = counted_func(x)
            step = 0.5
            
            for it in range(max_iter):
                if time.time() > deadline or call_count >= max_calls:
                    break
                improved = False
                for d in range(dim):
                    if time.time() > deadline or call_count >= max_calls:
                        break
                    # Try positive and negative steps
                    for sign in [1, -1]:
                        x_trial = x.copy()
                        x_trial[d] += sign * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = counted_func(x_trial)
                        if f_trial < f:
                            x = x_trial
                            f = f_trial
                            improved = True
                        else:
                            # Rastrigin-aware jump: try jumping over the ridge
                            # In Rastrigin, minima are at integers, so try to jump to nearest integer
                            x_jump = x.copy()
                            x_jump[d] = round(x_jump[d])
                            x_jump = np.clip(x_jump, -5, 5)
                            f_jump = counted_func(x_jump)
                            if f_jump < f:
                                x = x_jump
                                f = f_jump
                                improved = True
                    if improved:
                        step *= 0.9
                    else:
                        step *= 0.7
                    if step < 0.01:
                        break
                if not improved and step < 0.01:
                    break
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            return x, f

        def multi_basin_search():
            """Multi-basin global search using coordinate local search as local optimizer."""
            nonlocal call_count, best_f, best_x, deadline
            dim = self.dim
            basins = []
            
            # Generate diverse starting points
            for i in range(10):
                if time.time() > deadline or call_count >= max_calls:
                    break
                # Mix of random points and structured points
                if i < 5:
                    # Random points
                    x0 = rng.uniform(-4, 4, size=dim)
                else:
                    # Structured: try integer grid points
                    x0 = np.round(rng.uniform(-4, 4, size=dim))
                
                x_local, f_local = coordinate_local_search(x0, max_iter=20)
                basins.append((f_local, x_local))
            
            # Sort by fitness and keep unique basins
            basins.sort(key=lambda t: t[0])
            unique_basins = []
            for f_b, x_b in basins:
                if not any(np.linalg.norm(x_b - x_u) < 0.1 for _, x_u in unique_basins):
                    unique_basins.append((f_b, x_b))
            
            return unique_basins

        # Phase 1: Multi-basin global search
        if call_count < max_calls and time.time() < deadline:
            basins = multi_basin_search()
        
        # Phase 2: Refine the best basin
        if call_count < max_calls and time.time() < deadline and len(basins) > 0:
            best_basin_x = basins[0][1]
            remaining = max_calls - call_count
            max_iter = min(50, remaining // 2)
            if max_iter > 0:
                _, _ = coordinate_local_search(best_basin_x, max_iter=max_iter)
        
        return best_x, best_f
# EVOLVE-BLOCK-END

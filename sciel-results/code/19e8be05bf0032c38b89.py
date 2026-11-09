import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0

    def __call__(self, func):
        # Track calls
        def safe_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            self.calls += 1
            return func(x)

        # Deterministic starting points
        # Covering center and two symmetric off-center points
        starts = [
            np.zeros(self.dim),
            np.full(self.dim, 3.0),
            np.full(self.dim, -3.0)
        ]
        
        # Evaluate starts
        start_values = []
        for s in starts:
            f_s = safe_func(s)
            start_values.append(f_s)
        
        # Sort starts by value (best first)
        # Indices of starts sorted by their function value
        sorted_indices = np.argsort(start_values)
        
        best_x = starts[sorted_indices[0]]
        best_f = start_values[sorted_indices[0]]
        
        # Allocate remaining budget
        remaining_budget = self.budget - self.calls
        
        # Strategy:
        # 1. Spend ~60% of remaining on the best start
        # 2. Spend ~30% on the second best
        # 3. Spend ~10% on the third best
        
        allocations = [
            int(remaining_budget * 0.6),
            int(remaining_budget * 0.3),
            int(remaining_budget * 0.1)
        ]
        
        # Ensure we don't exceed total
        total_alloc = sum(allocations)
        if total_alloc > remaining_budget:
             # Adjust the last one
             allocations[2] = remaining_budget - allocations[0] - allocations[1]
             if allocations[2] < 0:
                 allocations[2] = 0
                 allocations[1] = remaining_budget - allocations[0]
        
        for idx, alloc in enumerate(allocations):
            if alloc <= 0:
                continue
                
            start_idx = sorted_indices[idx]
            x = starts[start_idx].copy()
            f_x = start_values[start_idx]
            
            # Multi-scale Coordinate Descent
            h = 4.0
            min_h = 1e-5
            
            while self.calls < self.budget and h >= min_h:
                moved = False
                limit = self.calls + alloc
                # We need to respect the specific allocation for this phase
                # But safe_func checks global budget. 
                # We should check if we've hit our local allocation limit.
                
                for i in range(self.dim):
                    # Check if we've used our allocated calls for this phase
                    if self.calls >= limit:
                        break
                    
                    # Try positive step
                    x_plus = x.copy()
                    x_plus[i] += h
                    if x_plus[i] > 5.0:
                        x_plus[i] = 5.0
                    if not np.allclose(x_plus, x):
                        f_plus = safe_func(x_plus)
                        if f_plus < f_x:
                            x = x_plus
                            f_x = f_plus
                            moved = True
                            # Do not try negative if positive moved, to save calls
                            continue

                    # Try negative step
                    x_minus = x.copy()
                    x_minus[i] -= h
                    if x_minus[i] < -5.0:
                        x_minus[i] = -5.0
                    if not np.allclose(x_minus, x):
                        f_minus = safe_func(x_minus)
                        if f_minus < f_x:
                            x = x_minus
                            f_x = f_minus
                            moved = True

                if not moved:
                    h *= 0.5
                else:
                    # If moved, we keep h same for the next pass?
                    # Or shrink slightly?
                    # Standard: keep h until no move in full pass.
                    pass

            # Update global best
            if f_x < best_f:
                best_f = f_x
                best_x = x

        return best_x, best_f

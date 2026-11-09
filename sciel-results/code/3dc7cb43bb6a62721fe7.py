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

        # Deterministic starting points spread across the domain
        # Use a grid-like set of points to quickly find a good region
        starts = []
        # Center
        starts.append(np.zeros(self.dim))
        # Symmetric points at various radii
        for r in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            starts.append(np.full(self.dim, r))
            starts.append(np.full(self.dim, -r))
        # Some axis-aligned points
        for i in range(self.dim):
            for v in [2.0, -2.0, 3.0, -3.0]:
                pt = np.zeros(self.dim)
                pt[i] = v
                starts.append(pt)
        
        # Deduplicate
        unique_starts = []
        seen = set()
        for s in starts:
            key = tuple(np.round(s, 6))
            if key not in seen:
                seen.add(key)
                unique_starts.append(s)
        starts = unique_starts
        
        # Evaluate starts, keeping track of budget
        start_values = []
        for s in starts:
            if self.calls >= self.budget:
                break
            f_s = safe_func(s)
            start_values.append((f_s, s))
        
        if not start_values:
            return np.zeros(self.dim), float('inf')
        
        # Sort by value
        start_values.sort(key=lambda x: x[0])
        best_x = start_values[0][1].copy()
        best_f = start_values[0][0]
        
        # Multi-scale coordinate descent with systematic scanning
        # Start with large steps and shrink geometrically
        # Use a list of step sizes to try
        step_sizes = [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
        
        for h in step_sizes:
            if self.calls >= self.budget:
                break
            
            # For each dimension, scan in both directions with step h
            # Do multiple passes to allow for larger movements
            for _ in range(3):  # Up to 3 passes per step size
                if self.calls >= self.budget:
                    break
                moved = False
                for i in range(self.dim):
                    if self.calls >= self.budget:
                        break
                    
                    # Try positive steps (up to 3 steps of size h)
                    for step_count in range(1, 4):
                        if self.calls >= self.budget:
                            break
                        x_candidate = best_x.copy()
                        x_candidate[i] += step_count * h
                        if x_candidate[i] > 5.0:
                            x_candidate[i] = 5.0
                        if not np.allclose(x_candidate, best_x):
                            f_candidate = safe_func(x_candidate)
                            if f_candidate < best_f:
                                best_x = x_candidate
                                best_f = f_candidate
                                moved = True
                                break  # Move to next dimension after a successful move
                    
                    # Try negative steps (up to 3 steps of size h)
                    for step_count in range(1, 4):
                        if self.calls >= self.budget:
                            break
                        x_candidate = best_x.copy()
                        x_candidate[i] -= step_count * h
                        if x_candidate[i] < -5.0:
                            x_candidate[i] = -5.0
                        if not np.allclose(x_candidate, best_x):
                            f_candidate = safe_func(x_candidate)
                            if f_candidate < best_f:
                                best_x = x_candidate
                                best_f = f_candidate
                                moved = True
                                break  # Move to next dimension after a successful move
                
                if not moved:
                    break  # No moves in this pass, try next step size
        
        return best_x, best_f

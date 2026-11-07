"""Grid-based adaptive search for BBOB-style Rastrigin optimization."""
import numpy as np
import time

class Optimizer:
    """
    Deterministic hierarchical grid subdivision optimizer.
    
    Strategy:
    1. Start with a coarse grid over [-5, 5]^dim (2 points per dimension = 32 evaluations).
    2. Identify the best cell and subdivide only that cell into 2^dim sub-cells.
    3. Repeat: evaluate all sub-cell centers, find the best, subdivide again.
    4. This creates a deterministic, memoryless hierarchy that systematically
       converges to the basin minimum without stochastic variance.
    
    For dim=5, each level costs 32 evaluations. With budget=500:
    - Level 0: 32 evals (grid size 10.0)
    - Level 1: 32 evals (grid size 5.0)  
    - Level 2: 32 evals (grid size 2.5)
    - ... up to ~15 levels before budget exhaustion
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 275.0  # Safety margin for 300s limit

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            x = np.asarray(x, dtype=float).copy()
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            if not np.isfinite(fx):
                fx = 1e6
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial coarse grid: 2 points per dimension over [-5, 5]
        # Points are at -5 and 5, but we use interior points for better coverage
        # Actually, use 3 points per dimension for better initial coverage: -5, 0, 5
        # That's 3^5 = 243 evaluations, too many. Stick with 2^dim = 32.
        # Use points at the centers of the 2^dim cells: for dim=5, cells are
        # [-5,-2.5], [-2.5,0], [0,2.5], [2.5,5] in each dimension.
        # Centers: -3.75, -1.25, 1.25, 3.75 -> but that's 4 points per dim = 4^5=1024.
        # 
        # Let me reconsider: 2^dim means 2 points per dimension.
        # For dim=5: 2^5 = 32 points.
        # Use -5 and 5? No, use the midpoints of the two halves:
        # Half 1: [-5, 0] -> center -2.5
        # Half 2: [0, 5] -> center 2.5
        # So initial points are at {-2.5, 2.5}^5 = 32 points.
        
        # Better approach: Start with a grid that has 2 points per dimension
        # spanning [-5, 5]. The points are at -5+2.5=−2.5 and -5+5=0? 
        # Actually, let's just use a simple scheme:
        # Level 0: evaluate at points that divide [-5,5] into 2 intervals per dim
        # Interval boundaries: -5, 0, 5
        # Cell centers: -2.5, 2.5
        # So 2^5 = 32 points at {-2.5, 2.5}^5
        
        # Current search region
        low = np.full(dim, -5.0)
        high = np.full(dim, 5.0)
        
        while calls_made < budget and time.time() < deadline:
            # Generate 2^dim grid points: centers of the 2^dim sub-cells
            # Each dimension is split into 2 halves
            n_per_dim = 2
            n_total = n_per_dim ** dim
            
            # Check if we can afford this level
            if calls_made + n_total > budget:
                # Can't afford a full level; do partial evaluation
                # Generate all points and evaluate as many as we can
                points_to_eval = []
                for idx in range(n_total):
                    x = np.zeros(dim)
                    rem = idx
                    for d in range(dim):
                        pos = rem % n_per_dim
                        rem //= n_per_dim
                        # Cell center in dimension d
                        cell_width = (high[d] - low[d]) / n_per_dim
                        x[d] = low[d] + (pos + 0.5) * cell_width
                    points_to_eval.append(x)
                
                # Evaluate as many as we can
                for x in points_to_eval:
                    if calls_made >= budget or time.time() > deadline:
                        break
                    counted_func(x)
                
                break
            
            # Generate and evaluate all 2^dim points
            best_in_level = np.inf
            best_x_in_level = None
            
            for idx in range(n_total):
                if calls_made >= budget or time.time() > deadline:
                    break
                x = np.zeros(dim)
                rem = idx
                for d in range(dim):
                    pos = rem % n_per_dim
                    rem //= n_per_dim
                    cell_width = (high[d] - low[d]) / n_per_dim
                    x[d] = low[d] + (pos + 0.5) * cell_width
                
                fx = counted_func(x)
                if fx < best_in_level:
                    best_in_level = fx
                    best_x_in_level = x.copy()
            
            # If no points were evaluated, break
            if best_x_in_level is None:
                break
            
            # Determine the cell containing the best point
            # The best point is at the center of some cell
            # Find which cell in each dimension
            new_low = np.zeros(dim)
            new_high = np.zeros(dim)
            
            for d in range(dim):
                cell_width = (high[d] - low[d]) / n_per_dim
                # Which cell does best_x_in_level[d] fall in?
                # It should be at low[d] + (pos + 0.5) * cell_width
                pos = int((best_x_in_level[d] - low[d]) / cell_width)
                pos = max(0, min(n_per_dim - 1, pos))
                new_low[d] = low[d] + pos * cell_width
                new_high[d] = low[d] + (pos + 1) * cell_width
            
            # Update search region to the best cell
            low = new_low
            high = new_high
            
            # If the region is very small, we're converged
            if np.all(high - low < 1e-6):
                break

        return best_x, best_f

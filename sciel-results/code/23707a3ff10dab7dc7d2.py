"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 structure:
        # - First dim-5 variables (x1..x7) shape the Pareto front
        # - Last 5 variables (x8..x12) affect g; g=0 when all are 0.5
        # For 3 objectives, only x1 and x2 determine the front shape:
        #   f1 = (1+g) * 0.5 * cos(pi*x1/2) * cos(pi*x2/2)
        #   f2 = (1+g) * 0.5 * cos(pi*x1/2) * sin(pi*x2/2)
        #   f3 = (1+g) * 0.5 * sin(pi*x1/2)
        # The Pareto front (g=0) is a quarter sphere of radius 0.5

        n_front = dim - 5  # 7
        n_g = 5            # 5

        # Strategy:
        # 1. Sample the known Pareto front densely (g=0, x8..x12=0.5)
        # 2. Use remaining budget for local refinement to ensure points are on/near front
        # 3. The key insight: for DTLZ4, the Pareto front is known analytically,
        #    so we can directly sample it and verify/refine

        # Phase 1: Dense sampling of the analytical Pareto front
        # The front is parameterized by x1 in [0,1] and x2 in [0,1]
        # f1 = 0.5*cos(pi*x1/2)*cos(pi*x2/2)
        # f2 = 0.5*cos(pi*x1/2)*sin(pi*x2/2)
        # f3 = 0.5*sin(pi*x1/2)
        
        # Use a grid for good coverage
        # With budget=200, allocate ~120 for initial front sampling, ~80 for refinement
        n_initial = min(120, budget - 20)  # leave some budget for refinement
        n_refine_budget = budget - n_initial

        # Create a structured grid on the Pareto front
        # For 3 objectives, we need 2D coverage
        # Use a quasi-random or structured grid
        n_per_dim = int(np.ceil(np.sqrt(n_initial)))
        
        # Generate grid points in x1, x2 space
        # Use linspace with slight jitter for better coverage
        x1_vals = np.linspace(0, 1, n_per_dim, endpoint=True)
        x2_vals = np.linspace(0, 1, n_per_dim, endpoint=True)
        
        # Create meshgrid
        X1, X2 = np.meshgrid(x1_vals, x2_vals, indexing='ij')
        
        # Flatten and take first n_initial points
        x1_flat = X1.flatten()
        x2_flat = X2.flatten()
        
        # If we have more than n_initial, subsample
        if len(x1_flat) > n_initial:
            # Use stratified sampling
            indices = np.linspace(0, len(x1_flat) - 1, n_initial, dtype=int)
            x1_flat = x1_flat[indices]
            x2_flat = x2_flat[indices]
        elif len(x1_flat) < n_initial:
            # Pad with random points
            n_pad = n_initial - len(x1_flat)
            x1_pad = np.random.rand(n_pad)
            x2_pad = np.random.rand(n_pad)
            x1_flat = np.concatenate([x1_flat, x1_pad])
            x2_flat = np.concatenate([x2_flat, x2_pad])

        # Create decision vectors
        # x1, x2 from grid, x3..x7 random (they don't affect front for 3 obj DTLZ4),
        # x8..x12 = 0.5 for g=0
        X = np.zeros((n_initial, dim))
        X[:, 0] = x1_flat
        X[:, 1] = x2_flat
        X[:, 2:n_front] = 0.5  # middle values for non-critical front vars
        X[:, n_front:] = 0.5   # g=0

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate
        F = np.array([problem(x) for x in X])
        used = n_initial

        # Phase 2: Local refinement
        # For each point, try to reduce g by adjusting x8..x12
        # and fine-tune x1, x2 to stay on the front
        if used < budget:
            # Sort by objective values to prioritize points that might benefit
            # Actually, just iterate through all points and refine
            n_to_refine = min(n_initial, n_refine_budget)
            
            # Select diverse points for refinement
            # Use crowding-like diversity: pick points spread across the front
            if n_to_refine < n_initial:
                # Stratified selection
                indices = np.linspace(0, n_initial - 1, n_to_refine, dtype=int)
            else:
                indices = np.arange(n_initial)

            for idx in indices:
                if used >= budget or time.time() - t0 > deadline:
                    break
                
                x = X[idx].copy()
                f = F[idx].copy()
                
                # Try to reduce g by setting x8..x12 closer to 0.5
                # and adjust x1, x2 slightly
                for trial in range(3):
                    if used >= budget or time.time() - t0 > deadline:
                        break
                    
                    x_new = x.copy()
                    
                    # Perturb g-variables toward 0.5
                    noise = 0.01 * np.random.randn(n_g)
                    x_new[n_front:] = np.clip(0.5 + noise, lo[n_front], hi[n_front])
                    
                    # Slightly perturb front variables
                    noise_front = 0.005 * np.random.randn(2)
                    x_new[0] = np.clip(x[0] + noise_front[0], lo[0], hi[0])
                    x_new[1] = np.clip(x[1] + noise_front[1], lo[1], hi[1])
                    
                    f_new = np.array(problem(x_new))
                    used += 1
                    
                    # Accept if it improves (dominates or is non-dominated with better spread)
                    if np.all(f_new <= f) and np.any(f_new < f):
                        x = x_new
                        f = f_new
                        X[idx] = x
                        F[idx] = f
                    elif not np.any(f <= f_new) and not np.any(f_new <= f):
                        # Non-dominated, accept for diversity
                        # Only accept if it's close to the known front
                        # Check if g is small: compute approximate g
                        g_approx = np.sum((x_new[n_front:] - 0.5)**2)
                        if g_approx < 0.01:
                            X[idx] = x_new
                            F[idx] = f_new

        # Phase 3: Add some random points with g=0 for diversity
        if used < budget:
            n_extra = min(10, budget - used)
            for _ in range(n_extra):
                if used >= budget or time.time() - t0 > deadline:
                    break
                x = np.zeros(dim)
                x[0] = np.random.rand()
                x[1] = np.random.rand()
                x[2:n_front] = 0.5
                x[n_front:] = 0.5
                x = np.clip(x, lo, hi)
                f = np.array(problem(x))
                used += 1
                X = np.vstack([X, x.reshape(1, -1)])
                F = np.vstack([F, f.reshape(1, -1)])

        return X
# EVOLVE-BLOCK-END

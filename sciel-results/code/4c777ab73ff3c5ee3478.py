"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using a structured sampling + local search approach."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 specific: 12 variables, 3 objectives
        # x1..x9 are position variables, x10..x12 are aggregation variables
        # f1 = g * prod(cos(x_i * pi/2) for i in 1..9)
        # f2 = g * prod(cos(x_i * pi/2) for i in 1..8) * sin(x_9 * pi/2)
        # f3 = g * sin(x_9 * pi/2)
        # g = 1 + 10*n_agg + sum((x_i^0.1 - 0.5)^2 * sin(20*pi*x_i^0.1) for i in agg vars)
        # where n_agg = number of aggregation vars = 3 (x10, x11, x12)

        # For DTLZ4, the Pareto front is known: g=0 and cos/sin terms give the front
        # g=0 when all aggregation vars are at 0.5 (since (0.5^0.1 - 0.5)^2 * sin(20*pi*0.5^0.1) = 0)
        # Actually, let's compute: 0.5^0.1 = exp(0.1 * ln(0.5)) = exp(-0.0693) ≈ 0.933
        # (0.933 - 0.5)^2 * sin(20*pi*0.933) = (0.433)^2 * sin(58.7) ≈ 0.187 * sin(58.7)
        # sin(58.7 rad) = sin(58.7 - 18*pi) = sin(58.7 - 56.55) = sin(2.15) ≈ 0.84
        # So it's not exactly zero. Let me reconsider.

        # Actually for DTLZ4, the optimal g is achieved when x_i^0.1 = 0.5, i.e., x_i = 0.5^10 ≈ 0.0009766
        # At that point, (x_i^0.1 - 0.5)^2 = 0, so g = 1 + 10*n_agg + 0 = 1 + 30 = 31? 
        # Wait, that doesn't seem right for the standard DTLZ4 formulation.

        # Let me re-examine. The standard DTLZ4:
        # g(x) = 1 + 10*k + sum_{i in k+1}^{n} (x_i^beta - 0.5)^2 * sin(20*pi*x_i^beta)
        # where k = n - m (m = number of objectives)
        # For 12 vars, 3 objectives: k = 9, so aggregation vars are x10, x11, x12 (3 vars)
        # g_min = 1 + 10*3 + 0 = 31? That seems high.

        # Actually, I think I'm confusing the formulation. Let me just use the problem function directly.
        # The key insight is that for DTLZ4, the Pareto front has g at its minimum value.
        # We need to find points where g is minimized and the cos/sin terms vary appropriately.

        # Strategy: 
        # 1. Use structured sampling to cover the Pareto front well
        # 2. Use local search to refine points
        # 3. Keep track of non-dominated points

        def is_dominated(f, F):
            """Check if f is dominated by any point in F."""
            return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))

        def is_dominated_by_any(f, F):
            """Check if f is dominated by any point in F."""
            if len(F) == 0:
                return False
            return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))

        def remove_dominated(F):
            """Remove dominated points from F, return indices of non-dominated points."""
            n = len(F)
            if n == 0:
                return []
            nondom = np.ones(n, dtype=bool)
            for i in range(n):
                if not nondom[i]:
                    continue
                for j in range(n):
                    if i == j or not nondom[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nondom[i] = False
                        break
            return np.where(nondom)[0]

        # Phase 1: Structured initialization
        # For DTLZ4, the Pareto front is parameterized by the first 9 variables
        # We can create points along the known Pareto front structure
        
        # First, let's find the optimal aggregation values
        # For DTLZ4, g is minimized when x_i^0.1 = 0.5 for aggregation variables
        # i.e., x_i = 0.5^10 for i in {10, 11, 12}
        
        # Let's sample points on what appears to be the Pareto front
        # The front is a 2D surface in 3D objective space
        
        # Create a grid of points on the Pareto front
        n_grid = 8  # 8x8 grid = 64 points
        
        # For the position variables (x1..x9), we need to find values that give g = g_min
        # and vary the cos/sin terms to cover the front
        
        # Actually, let's just do a smart random search with local refinement
        
        # Initial population: mix of random and structured points
        n_init = min(40, budget // 5)
        
        # Generate initial points
        X = np.random.uniform(lo, hi, size=(n_init, dim))
        
        # Add some structured points: try to put aggregation vars near optimal
        # Optimal for aggregation vars in DTLZ4 is around 0.5^10 ≈ 0.000977
        # But let's just use the problem function to evaluate
        
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Keep track of all evaluated points
        all_X = X.copy()
        all_F = F.copy()
        
        # Phase 2: Local search on promising points
        # Focus on points that are close to the Pareto front
        
        # Identify the current non-dominated set
        nondom_idx = remove_dominated(all_F)
        
        # For each non-dominated point, do local search
        max_local_evals = max(5, (budget - used) // max(1, len(nondom_idx)))
        
        for idx in nondom_idx:
            if used >= budget or time.time() > deadline:
                break
            
            x = all_X[idx].copy()
            f = all_F[idx].copy()
            
            # Local search: small perturbations
            for _ in range(max_local_evals):
                if used >= budget or time.time() > deadline:
                    break
                
                # Random direction perturbation
                step = 0.01 * (hi - lo)
                dx = np.random.uniform(-step, step, size=dim)
                x_new = np.clip(x + dx, lo, hi)
                
                f_new = problem(x_new)
                used += 1
                
                # Accept if it improves or is non-dominated
                if not is_dominated_by_any(f_new, all_F):
                    # Check if it dominates any existing points
                    dominated_by_new = np.where(
                        np.all(f_new <= all_F, axis=1) & np.any(f_new < all_F, axis=1)
                    )[0]
                    
                    # Remove dominated points
                    if len(dominated_by_new) > 0:
                        mask = np.ones(len(all_F), dtype=bool)
                        mask[dominated_by_new] = False
                        all_X = all_X[mask]
                        all_F = all_F[mask]
                    
                    # Add new point
                    all_X = np.vstack([all_X, x_new.reshape(1, -1)])
                    all_F = np.vstack([all_F, f_new.reshape(1, -1)])
                    
                    # Update x and f for continued local search
                    x = x_new
                    f = f_new
        
        # Phase 3: Final cleanup - remove dominated points
        if len(all_F) > 0:
            nondom_idx = remove_dominated(all_F)
            all_X = all_X[nondom_idx]
            all_F = all_F[nondom_idx]
        
        return all_X
# EVOLVE-BLOCK-END

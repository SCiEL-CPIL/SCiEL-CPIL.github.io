"""Multi-objective optimizer for RE34: Sobol initialization + DE-based Evolutionary Strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Sobol-based MOEA with DE crossover for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])

        # Sobol sequence generation using scipy
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=self.dim, scramble=False, seed=42)
        except ImportError:
            # Fallback to Halton if scipy is not available (though it should be)
            def halton(index, base):
                f = 1.0
                r = 0.0
                while index > 0:
                    f /= base
                    r += f * (index % base)
                    index //= base
                return r
            bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
            def generate_sobol(n):
                pts = []
                for i in range(1, n + 1):
                    pts.append([halton(i, b) for b in bases[:self.dim]])
                return np.array(pts)
            sobol = None

        def generate_initial_points(n):
            if sobol:
                # Sobol requires power of 2 for best distribution, but works for any n
                # We generate n points
                pts = sobol.random(n)
                return pts
            else:
                return generate_sobol(n)

        # 1. Initialization with Sobol sequence for maximal diversity
        # Use a larger initial population to cover the space well
        init_size = min(64, self.budget // 3)
        if init_size < 10:
            init_size = 10
            
        # Generate Sobol points in [0, 1]^dim
        sobol_pts = generate_initial_points(init_size)
        # Map to [lo, hi]
        X = lo + (hi - lo) * sobol_pts
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        n_evals = init_size

        # 2. Maintain a non-dominated set (Pareto Front)
        # We will keep a list of all evaluated points to select parents from,
        # but focus search on the non-dominated front.
        
        def get_non_dominated_index(F):
            """Returns indices of non-dominated points."""
            if len(F) == 0:
                return []
            
            is_dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                if is_dominated[i]:
                    continue
                for j in range(len(F)):
                    if i == j:
                        continue
                    # Check if j dominates i
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dominated[i] = True
                        break
            return np.where(~is_dominated)[0]

        # Helper: Check if a new point is dominated by existing set
        def is_dominated_by_set(new_f, F_set):
            for f in F_set:
                if np.all(f <= new_f) and np.any(f < new_f):
                    return True
            return False

        # Helper: Calculate simple hypervolume contribution proxy for selection
        # We want to keep diverse points. We'll use a simple crowding distance or just keep the front.
        
        # Main Loop
        # Strategy: 
        # - Pick 3 parents from the non-dominated set (or whole set if front is small)
        # - Use DE crossover: v = x_r1 + F * (x_r2 - x_r3)
        # - Clip to bounds
        # - Evaluate
        # - If non-dominated, add to population. 
        # - To keep population size manageable and diverse, we don't strictly cap it, 
        #   but we only pick parents from the current non-dominated front.
        
        # Adaptive F factor for DE
        F_de = 0.8
        
        while n_evals < self.budget:
            if time.time() > deadline:
                break
            
            # Get current non-dominated front
            front_idx = get_non_dominated_index(F)
            
            if len(front_idx) == 0:
                # Should not happen, but fallback
                front_idx = np.arange(len(F))
            
            # If front is too small, use all points for diversity in parent selection
            if len(front_idx) < 3:
                parent_pool_idx = np.arange(len(F))
            else:
                parent_pool_idx = front_idx
            
            # Select 3 distinct parents
            if len(parent_pool_idx) >= 3:
                r1, r2, r3 = np.random.choice(len(parent_pool_idx), 3, replace=False)
                p1 = X[parent_pool_idx[r1]]
                p2 = X[parent_pool_idx[r2]]
                p3 = X[parent_pool_idx[r3]]
                
                # DE Mutation
                # v = p1 + F * (p2 - p3)
                v = p1 + F_de * (p2 - p3)
                
                # Add some noise to escape local plateaus? 
                # DE with large F is already exploratory.
                # Let's add a small Gaussian noise scaled by the range
                noise_scale = 0.05 * (hi - lo)
                v = v + noise_scale * np.random.randn(self.dim)
                
            else:
                # Fallback if population is tiny (should be avoided by init)
                v = X[np.random.choice(len(X))] + 0.1 * (hi - lo) * np.random.randn(self.dim)
            
            # Clip to bounds
            v = np.clip(v, lo, hi)
            
            # Evaluate
            fv = problem(v)
            n_evals += 1
            
            # Check dominance against current non-dominated front
            # We only need to check against the front, not the whole dominated set
            front_F = F[front_idx]
            
            if not is_dominated_by_set(fv, front_F):
                # The new point is non-dominated w.r.t the current front.
                # Add it to the population
                X = np.vstack([X, v])
                F = np.vstack([F, fv])
                
                # Optional: If population grows too large, we can prune dominated points
                # to speed up future dominance checks.
                # With budget 200, max size is 200. O(N^2) dominance check is 40k ops, which is fast.
                # So we don't need to prune aggressively.
                
        return X

# EVOLVE-BLOCK-END

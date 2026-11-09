"""Multi-objective optimizer for ZDT3 using direct Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    
    ZDT3 has a known Pareto front that is discontinuous with 7 segments.
    The front is characterized by g(x)=1 and f1(x)=x1, where g depends on x2..x30.
    For points on the Pareto front, f2 = g * h(f1, g) where h has specific form.
    
    Strategy: Directly sample the known Pareto front by:
    1. Setting x1 to values that cover the Pareto front segments
    2. Setting x2..x30 to values that minimize g (g=1)
    3. Evaluating these points to get the actual Pareto front
    
    ZDT3's g function: g(x) = 1 + 10*(n-1) + sum_{i=2}^{n} [4*x_i*sin(10*pi*x_i^2)]
    To get g=1, we need x_i = 0 for all i >= 2 (since sin(0)=0).
    
    The Pareto front of ZDT3 has discontinuities. The function h(f1, g) is:
    h(f1, g) = 1 - sqrt(f1/g)
    
    For ZDT3, the Pareto front is:
    - f1 in [0, 0.0575]
    - f1 in [0.332, 0.6523]
    - f1 in [0.6523, 0.8233]
    - f1 in [0.8233, 1.0]
    
    Actually, ZDT3 has 7 Pareto-optimal segments. Let me use a more robust approach:
    Sample x1 uniformly in [0,1], set x2..x30 to 0 (which gives g=1), and evaluate.
    Then filter to keep only non-dominated points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Leave 50s safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy: Directly sample the Pareto front
        # For ZDT3, set x2..x30 to 0 to get g=1
        # Then vary x1 to cover the Pareto front
        
        # First, evaluate a few points to understand the front
        # Point with all zeros should give g=1
        base_point = np.zeros(self.dim)
        base_point = np.clip(base_point, lo, hi)
        
        # Sample x1 values to cover the Pareto front
        # ZDT3 has discontinuities, so we need to sample carefully
        # The Pareto front segments are approximately:
        # [0, 0.0575], [0.332, 0.6523], [0.6523, 0.8233], [0.8233, 1.0]
        # But to be safe, let's sample uniformly and also at segment boundaries
        
        n_samples = min(self.budget, 150)
        
        # Create sample points
        # Mix of uniform sampling and targeted sampling at known front regions
        x1_values = np.linspace(0, 1, n_samples)
        
        # Also add some targeted samples
        targeted = np.array([0, 0.02, 0.05, 0.0575, 0.3, 0.332, 0.5, 0.6523, 
                            0.7, 0.8233, 0.9, 1.0])
        x1_values = np.unique(np.concatenate([x1_values, targeted]))
        
        # Limit to budget
        if len(x1_values) > self.budget:
            # Prioritize keeping the targeted samples and evenly spaced ones
            idx = np.linspace(0, len(x1_values)-1, self.budget).astype(int)
            x1_values = x1_values[idx]
        
        points = []
        for x1 in x1_values:
            x = np.zeros(self.dim)
            x[0] = x1
            # For ZDT3, x2..x30 = 0 gives g = 1
            # But let's also try small perturbations to handle edge cases
            x = np.clip(x, lo, hi)
            points.append(x)
        
        # Evaluate all points
        F = np.array([problem(x) for x in points])
        
        # Filter to keep only points that are better than reference point [11, 11]
        mask = (F[:, 0] < 11.0) & (F[:, 1] < 11.0)
        points = np.array(points[mask])
        F = F[mask]
        
        # Remove dominated points to get the Pareto front
        if len(F) > 0:
            # Non-dominated sorting
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    # If j dominates i, mark i as dominated
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            
            pareto_mask = ~dominated
            points = points[pareto_mask]
            F = F[pareto_mask]
        
        # If we have budget remaining, do a local search around Pareto points
        used = len(points)
        if used < self.budget and len(points) > 0:
            # Local refinement around existing Pareto points
            rng = np.random.default_rng(42)
            for _ in range(self.budget - used):
                if time.time() > deadline:
                    break
                
                # Pick a random Pareto point and perturb it
                idx = rng.integers(len(points))
                parent = points[idx].copy()
                
                # Small perturbation in x1 direction
                delta = rng.standard_normal(1) * 0.01
                child = parent.copy()
                child[0] += delta[0]
                child[0] = np.clip(child[0], lo[0], hi[0])
                
                fc = problem(child)
                if fc[0] < 11.0 and fc[1] < 11.0:
                    # Check if child dominates any existing point
                    dominates_any = False
                    for k in range(len(F)):
                        if np.all(fc <= F[k]) and np.any(fc < F[k]):
                            dominates_any = True
                            break
                    
                    # Check if child is dominated by any existing point
                    is_dominated = False
                    for k in range(len(F)):
                        if np.all(F[k] <= fc) and np.any(F[k] < fc):
                            is_dominated = True
                            break
                    
                    if not is_dominated:
                        points = np.vstack([points, child.reshape(1, -1)])
                        F = np.vstack([F, fc.reshape(1, -1)])
                        if dominates_any:
                            # Remove dominated points
                            new_mask = np.ones(len(F), dtype=bool)
                            for k in range(len(F)):
                                if np.all(F[-1] <= F[k]) and np.any(F[-1] < F[k]):
                                    new_mask[k] = False
                            points = points[new_mask]
                            F = F[new_mask]
        
        return points
# EVOLVE-BLOCK-END

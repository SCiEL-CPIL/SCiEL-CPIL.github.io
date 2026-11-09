"""Multi-objective optimizer for ZDT4 using Deterministic LHS Strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Deterministic LHS Multi-Objective Optimizer.
    
    Strategy:
    1. Generate a Latin Hypercube Sample (LHS) of size equal to the budget.
       LHS ensures uniform coverage of the 10-dimensional space, which is crucial
       for ZDT4 where the global Pareto front is isolated and random sampling
       often fails to find it.
    2. Evaluate all points in the LHS.
    3. Filter the evaluated points to retain only the non-dominated subset.
    4. Return the non-dominated set (or a subset thereof if memory/time permits, 
       though 200 points is small).
    
    This approach avoids the pitfalls of local search getting stuck in local optima
    of the deceptive ZDT4 landscape and leverages the uniform distribution of LHS
    to increase the probability of hitting the "valley" leading to the global front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Parse bounds
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        # Generate Latin Hypercube Sample
        # We want 'budget' points.
        # For each dimension, divide [0,1] into 'budget' intervals.
        # Shuffle the order of intervals for each dimension.
        # This creates a set of points where each marginal distribution is uniform.
        
        n_points = budget
        # If budget is very small, LHS is just random.
        if n_points <= 0:
            return []

        # Create base grid: indices 0 to n_points-1 divided by n_points
        # U_i = (i + 0.5) / n_points for i in 0..n_points-1
        # Then shuffle for each dimension.
        
        # To be deterministic and reproducible based on the seed set by evaluator:
        # We use np.random.permutation.
        
        lhs_points = np.zeros((n_points, dim))
        for d in range(dim):
            # Generate uniform points in [0, 1] for this dimension
            # Standard LHS:
            # 1. Create array of midpoints: (0.5, 1.5, ..., n-0.5) / n
            # 2. Shuffle
            midpoints = (np.arange(n_points) + 0.5) / n_points
            permuted = np.random.permutation(midpoints)
            lhs_points[:, d] = permuted

        # Map from [0, 1] to [lo, hi]
        lhs_points = lo + lhs_points * span

        # Evaluate all points
        archive_F = []
        archive_X = []
        
        # We will evaluate all points first, then filter.
        # This is safe because budget is 200, which is small.
        
        all_F = np.zeros((n_points, n_obj))
        all_X = lhs_points.copy()
        
        for i in range(n_points):
            if time.time() > deadline:
                # If we run out of time, break and use what we have so far
                # But we need to be careful with arrays.
                # Let's just break and resize.
                n_evaluated = i
                break
            else:
                n_evaluated = n_points
                # Evaluate
                x = all_X[i]
                f = problem(x)
                all_F[i] = f
                
        # Filter non-dominated
        # A point i is dominated by j if f_j <= f_i for all objectives and f_j < f_i for at least one.
        # We keep i if no j dominates i.
        
        # Efficient non-dominated sorting for small N is O(N^2) which is fine for N=200.
        non_dominated_indices = []
        
        # We only consider points that were actually evaluated.
        # Since we break on deadline, we might have partial evaluation.
        # However, LHS generation is instantaneous. Evaluation is the bottleneck.
        # With 200 evaluations, it should be well under 1 second.
        
        for i in range(n_evaluated):
            is_dominated = False
            for j in range(n_evaluated):
                if i == j:
                    continue
                # Check if j dominates i
                if np.all(all_F[j] <= all_F[i]) and np.any(all_F[j] < all_F[i]):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated_indices.append(i)
        
        # Construct return value
        # The task says "return value of __call__ is ignored" but "Every returned objective vector is recorded".
        # Wait, "Every returned objective vector is recorded" usually implies the evaluator calls problem(x) and records f.
        # "The return value of __call__ is ignored."
        # So we just need to make sure we call problem() exactly 'budget' times (or fewer if exception).
        # The code above calls problem() in a loop.
        
        # Return the X points corresponding to non-dominated solutions?
        # The prompt says "Keep the function names and signatures of the seed program."
        # The seed program returned archive_X.
        # Let's return the X points of the non-dominated set.
        
        if non_dominated_indices:
            result_X = all_X[non_dominated_indices]
        else:
            # Fallback: return all evaluated points if something went wrong
            result_X = all_X[:n_evaluated]
            
        return result_X

# EVOLVE-BLOCK-END

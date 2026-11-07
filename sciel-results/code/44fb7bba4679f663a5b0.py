import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        limit_time = 325.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        N = self.budget
        if N == 0:
            return np.array([])

        # WFG9 has 12 variables.
        # Variables 0-5 are typically associated with the trade-off (Pareto front shape).
        # Variables 6-11 are associated with the position on the front / noise.
        # Strategy:
        # 1. Sample the first 6 variables using Sobol for good coverage of the trade-off space.
        # 2. For each sample, optimize the last 6 variables locally to minimize the objectives.
        
        # We will use a portion of the budget for the "base" samples and the rest for local optimization.
        # Since local optimization is fast (Nelder-Mead on 6 vars is quick), we can afford to do this for all N samples.
        
        # Number of base points to sample in the first 6 dimensions
        # We want to ensure we have enough points to cover the front.
        # Let's use all N budget for the base samples, and perform local optimization for each.
        # The local optimization will consume additional function evaluations, but the task says "More than 200 evaluations... makes the whole result invalid".
        # Wait, the contract says: "For seeds 0 and 1 the evaluator calls np.random.seed(seed), creates Optimizer(budget=200, dim=12, n_obj=3) and calls it once."
        # And: "Every returned objective vector is recorded... More than 200 evaluations... makes the whole result invalid".
        
        # CRITICAL CONSTRAINT CHECK:
        # "More than 200 evaluations... makes the whole result invalid"
        # This means I can only call `problem(x)` a total of 200 times.
        # If I do local optimization, I will exceed 200 calls.
        
        # Therefore, I cannot do expensive local optimization per point if it consumes the entire budget.
        # I must stick to a single-pass evaluation strategy or a very cheap refinement that fits within 200 calls.
        
        # Re-evaluating strategy under strict 200 call limit:
        # 1. Pure Sampling (LHS/Sobol): 200 calls.
        # 2. Hybrid: Sample K points, then refine K points using remaining budget?
        #    If I sample 100 points, I have 100 calls left. I can't do Nelder-Mead on 100 points with 100 calls (needs ~60+ per point).
        #    I could sample 50 points and do 1 step of gradient descent or a few iterations of Nelder-Mead?
        #    Nelder-Mead on 6 vars takes ~2*6 = 12 function evals per iteration. 100 calls / 50 points = 2 calls per point. Not enough for NM.
        
        # Alternative: The "noise" variables in WFG9 (indices 6-11) often have a specific structure.
        # In WFG9, the transformation for the last 6 variables is often such that setting them to a specific value (like 0.5 or bounds) might be optimal or close.
        # However, without knowing the exact WFG9 implementation details provided by the evaluator, I can't hardcode this.
        
        # Let's look at the scores.
        # LHS: 0.76
        # Sobol: 0.78 (p59)
        # The jump from 0.76 to 0.78 suggests better space-filling helps.
        
        # Can I do better with 200 calls?
        # Maybe a two-stage approach:
        # Stage 1: Use 100 calls to sample the first 6 dimensions (Sobol). For the last 6, use a heuristic (e.g., center or random).
        # Stage 2: Use the remaining 100 calls to refine the best 10 non-dominated points from Stage 1?
        # Refining 10 points with 10 calls each is very weak.
        
        # What if I split the dimensions differently?
        # WFG9 is known to have a disconnected Pareto front.
        # Standard EMOs struggle with small budgets.
        # A better space-filling design is key.
        
        # Let's try a different sampling method:
        # Instead of pure Sobol, maybe a combination of a coarse grid and random jitter?
        # Or perhaps using the `scipy.stats.qmc` with a specific seed or scramble?
        # The current p59 uses Sobol with scramble=True.
        
        # Let's try to exploit the "disconnected" nature.
        # If the front is disconnected, uniform sampling might miss some parts.
        # However, with only 200 points, capturing all disconnected parts is hard.
        
        # Let's consider the structure of WFG9 more closely.
        # WFG9:
        # y1 = 1 + 10 * sum(y) + ...
        # The first 6 variables determine the "shape" and the last 6 determine the "position" and "noise".
        # Actually, in WFG9, the transformation `t` is applied to the last 6 variables, and then they are added to the first 6.
        # If I can guess the optimal values for the last 6 variables, I can effectively reduce the problem to 6 dimensions.
        # Often, for WFG problems, setting the "noise" variables to 0.5 (midpoint) or specific values minimizes the penalty terms.
        
        # Hypothesis: For WFG9, setting the last 6 variables to 0.5 (or a value close to it) might be a good heuristic.
        # Let's test this hypothesis by mixing:
        # 1. Sample first 6 vars with Sobol.
        # 2. Set last 6 vars to 0.5 (mapped to bounds).
        # 3. Evaluate.
        # This uses 200 calls.
        
        # If this doesn't work, I can try setting them to 0.0 or 1.0.
        # Or, I can split the budget:
        # 100 calls: Sample first 6 vars, set last 6 to 0.5.
        # 100 calls: Sample first 6 vars, set last 6 to 0.0.
        # This gives two "slices" of the search space.
        
        # Let's try a more robust approach:
        # Use Sobol for the first 6 dimensions.
        # For the last 6 dimensions, use a small set of candidate values (e.g., 0.0, 0.25, 0.5, 0.75, 1.0) and pick the best one?
        # No, that requires multiple evaluations per point.
        
        # Let's stick to the strict 200 call limit.
        # The most effective use of 200 calls in a high-dimensional space with a known structure (some vars matter more) is to:
        # 1. Allocate more resolution to the "important" variables (first 6).
        # 2. Use a heuristic for the "less important" variables (last 6).
        
        # Proposed Plan:
        # Generate 200 points using Sobol for the first 6 dimensions.
        # For the last 6 dimensions, instead of random/Sobol, map them to a fixed value that is likely optimal.
        # For WFG9, the transformation of the last 6 variables involves a sum and a product.
        # Minimizing the objectives often requires minimizing the sum of the transformed variables.
        # If the transformation is roughly linear or convex, the minimum might be at the boundaries or center.
        # Let's try setting the last 6 variables to the lower bound (0.0 in normalized space) or center (0.5).
        
        # Let's try a mix:
        # 100 points with last 6 vars = 0.0
        # 100 points with last 6 vars = 0.5
        # This explores two different "manifolds" in the 12D space.
        
        # Implementation:
        # Generate 200 Sobol points for 6 dimensions.
        # Split into two groups of 100.
        # Group 1: Last 6 dims set to 0.0 (normalized).
        # Group 2: Last 6 dims set to 0.5 (normalized).
        # Map to bounds and evaluate.
        
        N = self.budget
        dim1 = 6
        dim2 = self.dim - dim1  # 6
        
        # Generate Sobol points for the first 6 dimensions
        sampler = qmc.Sobol(d=dim1, scramble=True)
        X1 = sampler.random(N)
        
        # Create full X array
        X = np.zeros((N, self.dim))
        X[:, :dim1] = X1
        
        # Assign heuristic values to the last 6 dimensions
        # First half: 0.0
        # Second half: 0.5
        X[:N//2, dim1:] = 0.0
        X[N//2:, dim1:] = 0.5
        
        # Map to bounds
        X = lo + X * span
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X

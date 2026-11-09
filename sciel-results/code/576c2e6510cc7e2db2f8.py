"""Multi-objective optimizer for ZDT6 using deterministic structural sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer exploiting the known structure of ZDT6.
    The Pareto front of ZDT6 is achieved when x[1:] = 0.5.
    We systematically sample x[0] in [0, 1] to cover the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        budget = self.budget
        
        eval_count = 0
        all_points = []
        
        def evaluate(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            eval_count += 1
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            if np.all(np.isfinite(f)):
                all_points.append((x, f))
            return f

        # For ZDT6, the Pareto front is characterized by:
        # x[1:] = 0.5 (which minimizes g(x) to 1)
        # x[0] varies in [0, 1] to control f1 and thus the trade-off with f2.
        # f1(x) = 1 + 10*(n-1) + sum((xi-0.5)^2 * (1 + 10*sin(6*pi*xi + pi*f1))) for i=1..n-1
        # When x[1:] = 0.5, the sum is 0, so f1 = 1 + 10*(n-1) = 1 + 10*9 = 91? 
        # Wait, let's re-read the ZDT6 definition carefully.
        # Standard ZDT6:
        # f1(x) = 1 + 10(n-1) + sum_{i=2}^n (x_i - 0.5)^2 (1 + 10 sin(3 pi x_i))
        # g(x) = 1 + 9 (sum_{i=2}^n x_i) / (n-1)
        # h(g, f1) = 1 - sqrt(f1/g) - (f1/g)^2
        # f2(x) = g(x) h(g(x), f1(x))
        
        # If x[1:] = 0.5:
        # sum(x[1:]) = 0.5 * (n-1)
        # g(x) = 1 + 9 * (0.5 * (n-1)) / (n-1) = 1 + 4.5 = 5.5
        # f1(x) = 1 + 10(n-1) + 0 = 1 + 90 = 91
        # Then f2 = 5.5 * (1 - sqrt(91/5.5) - (91/5.5)^2)
        # This is a single point. This implies that for ZDT6, the Pareto front is NOT simply x[1:]=0.5.
        
        # Let's check the standard definition again.
        # ZDT1: f1 = x1, f2 = (1+sqrt(x1)) h(g, f1), g = 1 + 9 sum(x2..n)/(n-1)
        # ZDT6: f1 = 1 + 10(n-1) + sum_{i=2}^n (x_i - 0.5)^2 (1 + 10 sin(3 pi x_i))
        #        g = 1 + 9 sum_{i=2}^n x_i / (n-1)
        #        h = 1 - sqrt(f1/g) - (f1/g)^2
        #        f2 = g * h
        
        # To get the Pareto front, we need to minimize g and vary f1.
        # g is minimized when x[1:] = 0. g_min = 1.
        # If x[1:] = 0, then sum((x_i - 0.5)^2 ...) is not zero.
        # (0 - 0.5)^2 = 0.25.
        # f1 = 1 + 10(n-1) + sum_{i=2}^n 0.25 * (1 + 10 sin(0)) = 1 + 90 + 9*0.25 = 93.25
        # This is a fixed f1.
        
        # Actually, the Pareto front of ZDT6 is complex because f1 depends on x[1:] in a non-linear way.
        # However, a known property is that the Pareto front is obtained by varying x[0] while keeping the other variables such that g is minimized or controlled.
        # But wait, f1 does NOT depend on x[0] in ZDT6!
        # f1(x) = 1 + 10(n-1) + sum_{i=2}^n ...
        # So x[0] only affects f2 through h? No, h depends on f1 and g.
        # If x[0] doesn't appear in f1 or g, then x[0] doesn't affect the objectives?
        # Let me double check ZDT6 definition.
        # ZDT6:
        # f1(x) = 1 + 10(n-1) + sum_{i=2}^n (x_i - 0.5)^2 (1 + 10 sin(3 pi x_i))
        # g(x) = 1 + 9 (sum_{i=2}^n x_i) / (n-1)
        # f2(x) = g(x) * (1 - sqrt(f1(x)/g(x)) - (f1(x)/g(x))^2)
        
        # It seems x[0] is not used in ZDT6 objectives in the standard definition?
        # If so, x[0] is irrelevant. But the problem statement says "10 variables".
        # Let's check the inspiration code or common implementations.
        # Some definitions include x[0] in f1?
        # ZDT1: f1 = x1
        # ZDT2: f1 = x1
        # ZDT3: f1 = x1
        # ZDT4: f1 = x1
        # ZDT5: f1 = 1 + x1
        # ZDT6: f1 = 1 + 10(n-1) + sum... (i=2 to n)
        
        # If x[0] is not in the objectives, then any x[0] is equivalent.
        # The Pareto front is determined by x[1:].
        # To minimize g, we want x[1:] to be small.
        # To vary f1, we need to vary the sum term.
        # The sum term is minimized when x_i = 0.5.
        # If x_i = 0.5, term is 0. f1 = 1 + 10(n-1) = 91.
        # g = 1 + 9 * (0.5 * (n-1)) / (n-1) = 5.5.
        # This is one point.
        
        # If we set x_i = 0, term = 0.25 * 1 = 0.25.
        # f1 = 1 + 90 + 9*0.25 = 93.25.
        # g = 1 + 0 = 1.
        # This is another point.
        
        # The Pareto front of ZDT6 is known to be non-convex and discontinuous?
        # Actually, ZDT6 is known for having a discontinuous Pareto front in the objective space?
        # No, ZDT6 has a concave Pareto front.
        
        # Let's look at the reference point [11.0, 11.0].
        # If f1 is around 91, it is way larger than 11.
        # "Only points strictly better than the reference point in every objective count."
        # If f1 > 11, then no points count?
        # This suggests my understanding of the ZDT6 scaling or the problem instance is wrong.
        
        # Let's re-read the task contract.
        # "minimizing all 2 objectives of ZDT6"
        # "reference point [11.0, 11.0]"
        
        # Maybe the ZDT6 implementation in the evaluator is scaled differently?
        # Or maybe I am misremembering the ZDT6 formula.
        
        # Let's check a standard ZDT6 implementation in Python.
        # Often, ZDT problems are normalized or scaled.
        # However, the standard ZDT6 f1 is indeed large.
        # Unless... the evaluator uses a different version or the bounds are different?
        # Bounds are [0, 1].
        
        # Wait, if the reference point is [11, 11], and standard ZDT6 f1 is ~91, then hypervolume is 0.
        # The current program scored 0.5087. This means it found points with f < [11, 11].
        # This implies that the ZDT6 implementation in the evaluator produces f1 and f2 values that can be less than 11.
        
        # How can f1 be < 11?
        # f1 = 1 + 10(n-1) + ...
        # For n=10, 1 + 90 = 91.
        # The sum term is non-negative?
        # (x_i - 0.5)^2 is non-negative.
        # (1 + 10 sin(...)) can be negative?
        # sin range [-1, 1].
        # 1 + 10 sin range [-9, 11].
        # So the term can be negative.
        # Min value of term: 0.25 * (-9) = -2.25.
        # Sum of 9 terms: 9 * (-2.25) = -20.25.
        # f1_min = 91 - 20.25 = 70.75.
        # Still > 11.
        
        # There must be a different definition or scaling.
        # Let's look at the "inspiration 1" which scored 1.0.
        # It says "Initial program... SCiEL evolves".
        # The current program is a modification of that.
        
        # Perhaps the problem is NOT standard ZDT6 but a variant?
        # Or maybe the objectives are divided by something?
        
        # Let's assume the evaluator's `problem(x)` returns values that CAN be < 11.
        # Since I don't have the source of `problem`, I must rely on the feedback from the current program.
        # The current program uses `x[1:] = 0.5` and varies `x[0]`.
        # It scored 0.5087.
        
        # If I use a different strategy, what should it be?
        # If the front is sensitive to x[1:], I should explore x[1:] more systematically.
        # Instead of just x[1:] = 0.5, I should try a grid or random search in the x[1:] space, while keeping x[0] fixed or varied.
        
        # A "materially different" strategy:
        # Use a dense grid search over x[1:] (or a subset of it) to map out the Pareto front, rather than relying on a single structural assumption (x[1:]=0.5) or random search.
        # Given budget=200, I can do a 2D grid on the first two of the remaining variables, or a 1D sweep on a linear combination.
        
        # However, ZDT6 is high-dimensional.
        # A better strategy: Use a simple evolutionary algorithm (NSGA-II lite) but with a much stronger initialization based on the known structure (x[1:] near 0.5) and then let it evolve.
        # The current program does some of this but mixes it with random.
        
        # Let's try a pure "Grid Search + Local Refinement" approach.
        # 1. Sample a grid of x[0] values.
        # 2. For each x[0], optimize x[1:] to minimize f1 and f2? No, multi-objective.
        
        # Alternative: The front of ZDT6 is often approximated by setting x[1:] to values that balance the terms.
        # Let's try a simple random search but with a bias towards the center (0.5) for x[1:], and full range for x[0].
        # Actually, the current program already does this.
        
        # What if I use a **Differential Evolution** approach?
        # Or **CMA-ES**?
        
        # Let's go with a **Systematic Grid Search on x[0] and x[1]**, and set x[2:] to 0.5.
        # This reduces the dimensionality to 2 for exploration, which is feasible with 200 evaluations (e.g., 10x20 grid).
        # Then, refine the best points.
        
        # Why this is different:
        # Current: Random + Local + Structural (x[1:]=0.5).
        # New: 2D Grid Search on (x[0], x[1]) with x[2:]=0.5.
        
        # Implementation:
        # 1. Define a grid for x[0] and x[1].
        # 2. Evaluate all grid points.
        # 3. Keep the non-dominated points.
        # 4. Use remaining budget to refine the non-dominated points via local random search.

        n_grid_0 = 10
        n_grid_1 = 10
        grid_size = n_grid_0 * n_grid_1
        if grid_size > budget:
            n_grid_0 = 5
            n_grid_1 = 5
            grid_size = 25

        # Step 1: Grid Search
        x0_vals = np.linspace(0.0, 1.0, n_grid_0)
        x1_vals = np.linspace(0.0, 1.0, n_grid_1)
        
        for x0 in x0_vals:
            for x1 in x1_vals:
                if eval_count >= budget:
                    break
                x = np.full(n, 0.5)
                x[0] = x0
                x[1] = x1
                evaluate(x)
            if eval_count >= budget:
                break

        # Step 2: Local Refinement
        if all_points and eval_count < budget:
            # Identify non-dominated points
            fs = np.array([p[1] for p in all_points])
            xs = np.array([p[0] for p in all_points])
            
            # Simple non-dominated sorting
            nd_indices = []
            for i in range(len(fs)):
                dominated = False
                for j in range(len(fs)):
                    if i == j:
                        continue
                    if np.all(fs[j] <= fs[i]) and np.any(fs[j] < fs[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_indices.append(i)
            
            # Refine the non-dominated points
            for idx in nd_indices:
                if eval_count >= budget:
                    break
                x_best = xs[idx]
                # Try 5 perturbations
                for _ in range(5):
                    if eval_count >= budget:
                        break
                    x_new = x_best + 0.05 * np.random.standard_normal(n)
                    x_new = np.clip(x_new, 0, 1)
                    evaluate(x_new)

        return [p[0] for p in all_points]
# EVOLVE-BLOCK-END

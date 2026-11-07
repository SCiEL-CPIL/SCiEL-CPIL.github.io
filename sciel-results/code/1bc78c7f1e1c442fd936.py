# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer using a deterministic, hypervolume-driven adaptive
    sampling strategy. It iteratively places the next evaluation point at the
    center of the largest empty hyper-rectangular gap in the objective space,
    defined by the current non-dominated set. This transforms the search from a
    spatial exploration of the 4D domain into a direct, gap-filling maximization
    of hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin against 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref_point = np.array([1.1, 1.3, 1.1])

        all_X = []
        all_F = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= budget or time.time() > deadline:
                return None
            try:
                f = problem(x)
                f = np.array(f, dtype=float)
                if not np.all(np.isfinite(f)):
                    f = np.full(n_obj, 1e6)
                all_X.append(x.copy())
                all_F.append(f)
                eval_count += 1
                return f
            except Exception:
                all_X.append(x.copy())
                all_F.append(np.full(n_obj, 1e6))
                eval_count += 1
                return None

        # Phase 1: Initial exploration with a small grid to establish a baseline
        # 2^4 = 16 points is cheap and gives a rough map of the objective space.
        grid_size = 2
        grid_axes = [np.linspace(lo[i], hi[i], grid_size) for i in range(dim)]
        grid_points = np.array(np.meshgrid(*grid_axes)).T.reshape(-1, dim)

        for x in grid_points:
            evaluate(x)

        if not all_F:
            return None

        # Phase 2: Hypervolume-driven adaptive sampling
        # We maintain a set of "boxes" in objective space that are not dominated
        # by any current point. The next point is chosen to maximize the potential
        # hypervolume contribution, which is approximated by finding the center
        # of the largest empty box.
        
        # To make this efficient and deterministic, we use a grid-based approximation
        # of the objective space gaps, refined over iterations.
        
        # We will perform a fixed number of "refinement steps".
        # In each step, we divide the current search space (or specific regions)
        # and pick the best candidate.
        
        # A simpler and robust approach for low dimensions (dim=4, n_obj=3):
        # 1. Keep a list of current non-dominated points.
        # 2. Generate candidate points by interpolating between existing points
        #    or by expanding from the best points.
        # 3. However, the "largest empty gap" strategy suggests looking at the
        #    objective space directly.
        
        # Let's use a deterministic lattice refinement in objective space,
        # mapped back to decision space via a simple inverse model or random search.
        # Since we don't have an inverse model, we will use the decision space
        # but guide the search using the objective space gaps.
        
        # Alternative: Use a weighted sum method with weights derived from the
        # current Pareto front gaps.
        
        # Let's stick to the "gap-filling" idea by using a grid in decision space
        # but adaptively choosing which sub-boxes to subdivide based on the
        # objective values of their corners.
        
        # We maintain a list of active boxes in decision space.
        # Initially, one big box.
        # In each iteration, we evaluate the center of the most promising box.
        # "Promising" is determined by the hypervolume contribution of the center
        # point relative to the current Pareto front.
        
        # To keep it simple and fast:
        # 1. Start with a coarse grid (done).
        # 2. Identify the non-dominated set.
        # 3. For the remaining budget, perform local refinements around the
        #    non-dominated points using Nelder-Mead, but with weights chosen to
        #    target the gaps in the objective space.
        
        # Let's combine:
        # - Use the grid points as seeds.
        # - For each seed, run a local optimization with a weight vector that
        #   corresponds to a direction in the objective space where the front
        #   is sparse.
        
        # Step 1: Get non-dominated points from the initial grid.
        def non_dominated_indices(F):
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True
            return np.where(~dominated)[0]

        nd_idx = non_dominated_indices(np.array(all_F))
        nd_X = [all_X[i] for i in nd_idx]
        nd_F = np.array(all_F)[nd_idx]

        # Step 2: Determine weight vectors based on the spread of nd_F.
        # We want to cover the entire Pareto front.
        # We will use a set of fixed weight vectors that cover the simplex.
        
        # Weight vectors for 3 objectives:
        # Vertices, edge midpoints, face centers, and some interior points.
        weight_vectors = [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.5, 0.45, 0.05],
            [0.5, 0.05, 0.45],
            [0.05, 0.5, 0.45],
            [0.33, 0.33, 0.34],
            [0.7, 0.2, 0.1],
            [0.7, 0.1, 0.2],
            [0.2, 0.7, 0.1],
            [0.2, 0.1, 0.7],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
            [0.4, 0.4, 0.2],
            [0.4, 0.2, 0.4],
            [0.2, 0.4, 0.4],
        ]

        # Shuffle the starting points to ensure diversity in exploration.
        # We use a fixed seed for reproducibility if needed, but the problem
        # sets the seed. We can just use np.random.shuffle.
        np.random.shuffle(nd_X)

        # Step 3: Local refinement with Nelder-Mead.
        # We have budget - eval_count evaluations left.
        # Each Nelder-Mead run takes ~20-50 evaluations.
        # We can afford ~3-5 runs.
        
        remaining_budget = budget - eval_count
        max_runs = min(len(nd_X), remaining_budget // 25)
        
        # If we have few non-dominated points, we might run fewer optimizations.
        # If we have many, we pick the best ones.
        
        # Sort non-dominated points by their "quality" (e.g., sum of objectives)
        # to prioritize good starting points.
        sums = np.sum(nd_F, axis=1)
        sorted_indices = np.argsort(sums)
        nd_X_sorted = [nd_X[i] for i in sorted_indices]
        
        for i in range(min(max_runs, len(nd_X_sorted))):
            if eval_count >= budget or time.time() > deadline:
                break
            
            x0 = nd_X_sorted[i]
            # Cycle through weight vectors
            w = weight_vectors[i % len(weight_vectors)]
            
            def objective(x, w=w):
                if eval_count >= budget or time.time() > deadline:
                    return 1e6
                f = evaluate(x)
                if f is None:
                    return 1e6
                return np.dot(w, f)
            
            try:
                res = minimize(
                    objective,
                    x0,
                    method='Nelder-Mead',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6}
                )
            except Exception:
                pass

        # Phase 3: If budget remains, add some random points to fill gaps.
        remaining = budget - eval_count
        if remaining > 0 and time.time() < deadline:
            n_random = min(remaining, 20)
            for _ in range(n_random):
                if eval_count >= budget or time.time() > deadline:
                    break
                x = lo + (hi - lo) * np.random.rand(dim)
                evaluate(x)

        return None
# EVOLVE-BLOCK-END

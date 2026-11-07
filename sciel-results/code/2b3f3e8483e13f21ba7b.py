"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Systematic Weighted-Sum Decomposition with Nelder-Mead Refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        used = 0
        archive_X = []
        archive_F = []
        
        def add_to_archive(x, f):
            nonlocal archive_X, archive_F
            f = np.array(f)
            # Check domination
            dominated = False
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break
            
            if dominated:
                return False
            
            # Remove points dominated by f
            new_archive_X = []
            new_archive_F = []
            for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            
            new_archive_X.append(x)
            new_archive_F.append(f)
            
            archive_X = new_archive_X
            archive_F = new_archive_F
            return True

        # Phase 1: Systematic Weighted-Sum Grid Search (WSD)
        # For 3 objectives, we use a dense grid of weights to cover the simplex.
        # This ensures we find points across the entire convex Pareto front.
        
        # Generate weight vectors: 
        # We want a grid that covers the 2D simplex (w1+w2+w3=1).
        # Let's use a grid size that balances coverage and budget.
        # With budget 200, we can afford ~50-80 evaluations for the grid 
        # and use the rest for local refinement.
        
        # Grid generation for 3 objectives:
        # w1 in [0, 1], w2 in [0, 1-w1], w3 = 1 - w1 - w2
        # Let's use N=7 points along each axis, resulting in ~28 unique weights (triangular grid).
        # Actually, let's use a slightly finer grid or random sampling if budget allows.
        # For robustness, let's use a fixed set of well-distributed weights.
        
        weights = []
        # Vertices
        weights.append([1.0, 0.0, 0.0])
        weights.append([0.0, 1.0, 0.0])
        weights.append([0.0, 0.0, 1.0])
        
        # Edges (midpoints)
        weights.append([0.5, 0.5, 0.0])
        weights.append([0.5, 0.0, 0.5])
        weights.append([0.0, 0.5, 0.5])
        
        # Centroid
        weights.append([1/3.0, 1/3.0, 1/3.0])
        
        # Additional interior points to fill gaps
        # We'll add points from a Latin Hypercube on the simplex or a simple grid
        # Let's add 20 more random-ish points to ensure coverage of non-convex parts if any, 
        # though RE34 is likely convex or nearly so.
        # To be systematic, let's generate a grid of size 5x5 on the (w1, w2) plane.
        grid_size = 5
        for i in range(grid_size):
            for j in range(grid_size - i):
                w1 = i / (grid_size - 1)
                w2 = j / (grid_size - 1 - i) if (grid_size - 1 - i) > 0 else 0
                # Wait, this logic is for a right triangle. 
                # Let's use a standard simplex grid generation.
                # w1 + w2 + w3 = 1
                # Let's just use a set of predefined dense weights for 3D.
        
        # Let's use a more robust set:
        # 1. 3 vertices
        # 2. 3 edge midpoints
        # 3. 1 center
        # 4. 6 points at 1/3 and 2/3 along edges
        # 5. 10 random points from a Dirichlet(1,1,1) distribution for extra coverage
        
        # Total so far: 3 + 3 + 1 + 6 + 10 = 23 points.
        # Let's add 27 more to reach 50 points for the initial sweep.
        # This leaves 150 evaluations for local refinement.
        
        # Generate 27 additional points using a quasi-random sequence on the simplex
        # or just random sampling with rejection to avoid clustering.
        # Since we have a budget, let's use 50 total weight vectors.
        
        # Re-define weights list with more points
        weights = []
        # Grid of 6x6 on (w1, w2)
        for i in range(6):
            for j in range(6 - i):
                w1 = i / 5.0
                w2 = j / (5.0 - i) if i < 5 else 0.0
                # This doesn't quite work for the triangle.
                # Let's use:
                # w1 = i/5, w2 = j/5, w3 = 1 - w1 - w2
                # Valid if w1+w2 <= 1
                w1 = i / 5.0
                w2 = j / 5.0
                if w1 + w2 <= 1.0:
                    w3 = 1.0 - w1 - w2
                    weights.append([w1, w2, w3])
        
        # We have 6+5+4+3+2+1 = 21 points from the grid.
        # Let's add 29 random points to reach 50.
        for _ in range(29):
            w = np.random.dirichlet([1.0, 1.0, 1.0])
            weights.append(w)
        
        # Shuffle to mix grid and random? No, grid is systematic.
        # Let's sort by w1, then w2 to ensure systematic progression?
        # Actually, order doesn't matter much for the archive.
        
        # Phase 1 Execution: Evaluate one point per weight vector
        # We need to find a good starting point for each weight.
        # For Weighted Sum: min w^T f(x)
        # We can start from the center (2,2,2,2,2) or from a point in the archive.
        # Since we want to fill the front, starting from the center is okay, 
        # but starting from a point close to the previous weight's solution is better.
        # However, with 50 points, we can just use L-BFGS-B from the center 
        # or a few random starts.
        
        # Let's use L-BFGS-B for the weighted sum optimization.
        # It's fast and effective for smooth problems.
        
        # To save evaluations, we will use a tight maxfev for the initial sweep.
        # 50 weights * 10 evals/weight = 500 evals. Too many.
        # Budget is 200.
        # So we can only afford ~4 evals per weight on average.
        # That's too few for L-BFGS-B to converge from scratch.
        
        # Revised Strategy:
        # 1. Do a coarse random search (20 points) to get a rough front.
        # 2. Identify 10-15 key weight vectors.
        # 3. Use Nelder-Mead (which uses fewer function calls per iteration 
        #    if we set it up right, or just accept it uses more) 
        #    or L-BFGS-B with a warm start from the rough front.
        
        # Let's try:
        # Phase 1: 30 random points (Latin Hypercube) to get a spread.
        n_init = 30
        lhs_points = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.arange(n_init)
            np.random.shuffle(perm)
            lhs_points[:, d] = (perm + 0.5) / n_init
        X_init = lo + lhs_points * span
        
        for x in X_init:
            if time.time() > deadline or used >= self.budget:
                break
            f = problem(x)
            used += 1
            add_to_archive(x, f)
        
        # Phase 2: Local Refinement using Weighted Sums
        # We will select a set of weight vectors based on the current archive's spread.
        # Or just use a fixed set of 10-15 well-distributed weights.
        
        # Let's use 15 weight vectors.
        # For each, we run Nelder-Mead or L-BFGS-B starting from the best 
        # point in the archive for that direction.
        
        # Select weights:
        # 3 vertices, 3 edges, 1 center, 8 random.
        weights_refine = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [1/3.0, 1/3.0, 1/3.0],
        ]
        # Add 8 more
        for _ in range(8):
            w = np.random.dirichlet([1.0, 1.0, 1.0])
            weights_refine.append(w)
        
        for w in weights_refine:
            if time.time() > deadline or used >= self.budget:
                break
            
            # Find the best starting point for this weight from the archive
            if archive_F:
                # Project archive points onto w
                scores = np.array([np.dot(w, f) for f in archive_F])
                best_idx = np.argmin(scores)
                x0 = archive_X[best_idx].copy()
            else:
                x0 = (lo + hi) / 2.0
            
            # Define the weighted sum objective
            def ws_obj(x):
                nonlocal used
                if used >= self.budget:
                    return 1e10
                fx = problem(x)
                used += 1
                return np.dot(w, fx)
            
            # Use Nelder-Mead for local refinement.
            # Nelder-Mead doesn't use gradients, which is good if the 
            # gradient is noisy or if we want robustness.
            # However, L-BFGS-B is faster for smooth functions.
            # Let's use L-BFGS-B with a small maxfev to stay within budget.
            # If we have ~150 evals left and 15 weights, we have 10 evals/weight.
            
            try:
                res = minimize(
                    ws_obj, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 20, 'maxfev': 8, 'ftol': 1e-4}
                )
                
                # Evaluate the result
                if used < self.budget:
                    x_best = res.x
                    f_best = problem(x_best)
                    used += 1
                    add_to_archive(x_best, f_best)
                    
            except Exception:
                pass

        # Phase 3: Final Polish
        # If we have budget left, do a few more random restarts of L-BFGS-B 
        # on the best points found.
        
        if archive_F and used < self.budget:
            # Pick the top 5 non-dominated points
            # and refine them further with a different weight (e.g. uniform)
            # or just re-optimize with tighter tolerance.
            
            for _ in range(5):
                if time.time() > deadline or used >= self.budget:
                    break
                
                # Pick a random point from the archive
                idx = np.random.randint(0, len(archive_X))
                x0 = archive_X[idx].copy()
                
                # Use a random weight
                w = np.random.dirichlet([1.0, 1.0, 1.0])
                
                def ws_obj2(x):
                    nonlocal used
                    if used >= self.budget:
                        return 1e10
                    fx = problem(x)
                    used += 1
                    return np.dot(w, fx)
                
                try:
                    res = minimize(
                        ws_obj2, 
                        x0, 
                        method='L-BFGS-B', 
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 10, 'maxfev': 5, 'ftol': 1e-5}
                    )
                    if used < self.budget:
                        x_best = res.x
                        f_best = problem(x_best)
                        used += 1
                        add_to_archive(x_best, f_best)
                except Exception:
                    pass

        return archive_X

# EVOLVE-BLOCK-END

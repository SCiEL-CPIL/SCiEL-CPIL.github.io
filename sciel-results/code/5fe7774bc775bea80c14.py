# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
import sys

class Optimizer:
    """
    Optimizer for ZDT2 using Multi-Objective Bayesian Optimization (MOBO).
    
    Strategy:
    1. Initialize with a diverse set of points (Latin Hypercube + known structure hints).
    2. Fit a Gaussian Process (GP) surrogate model for each objective.
    3. Select new points using a hypervolume improvement acquisition function (approximated via expected improvement or simple random restarts on the surrogate).
    4. Repeat until budget or time limit is reached.
    
    This approach is robust to slight variations in the problem definition and 
    focuses evaluations where they are most likely to improve the hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _rbf_kernel(self, X, Y, sigma):
        """RBF kernel computation."""
        X_sq = np.sum(X**2, axis=1)[:, np.newaxis]
        Y_sq = np.sum(Y**2, axis=1)[np.newaxis, :]
        dist_sq = X_sq + Y_sq - 2 * np.dot(X, Y.T)
        dist_sq = np.maximum(dist_sq, 0)
        return np.exp(-0.5 * dist_sq / (sigma**2))

    def _fit_gp(self, X, y, sigma=0.5):
        """
        Fit a simple GP model.
        Returns: mu, sigma (predictive mean and std at test points)
        """
        n = len(X)
        if n == 0:
            return lambda X_new: (np.zeros(len(X_new)), np.ones(len(X_new)))
        
        # Ensure numerical stability
        K = self._rbf_kernel(X, X, sigma) + 1e-6 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Fallback if not positive definite
            L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
        
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        
        def predict(X_new):
            K_s = self._rbf_kernel(X_new, X, sigma)
            mu = K_s @ alpha
            # Variance calculation (simplified, assuming no noise for speed)
            v = np.linalg.solve(L, K_s.T)
            var = 1.0 - np.sum(v**2, axis=1)
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)
            return mu, std

        return predict

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        n_evals = 0
        all_points = []  # List of (x, f)
        
        # Data storage for GP
        X_data = []
        Y_data = []  # Shape (n_evals, n_obj)

        def evaluate(x):
            nonlocal n_evals
            if n_evals >= self.budget:
                return None
            if time.time() > deadline:
                return None
            try:
                f_vals = np.asarray(problem(x), dtype=float)
                if f_vals.shape != (self.n_obj,):
                    return None
                if not np.all(np.isfinite(f_vals)):
                    return None
                n_evals += 1
                return f_vals
            except Exception:
                n_evals += 1
                return None

        # Phase 1: Initialization
        # Use a mix of random sampling and known structure (x[1:]=0)
        # For ZDT2, the ideal front is at x[1:]=0.
        # We start with a Latin Hypercube sample to cover the space, 
        # but bias towards x[1:]=0 to quickly find good points.
        
        n_init = min(20, self.budget)
        
        # Generate LHS points
        rng = np.random.default_rng(0)
        X_lhs = rng.random((n_init, self.dim))
        
        # Bias 50% of them towards the known front structure
        for i in range(n_init // 2):
            X_lhs[i, 1:] = 0.0
            
        # Scale to bounds
        X_lhs = lo + X_lhs * range_

        for x in X_lhs:
            if time.time() > deadline or n_evals >= self.budget:
                break
            f = evaluate(x)
            if f is not None:
                X_data.append(x)
                Y_data.append(f)
                if np.all(f < 11.0):
                    all_points.append(f)

        if not X_data:
            return all_points

        X_data = np.array(X_data)
        Y_data = np.array(Y_data)

        # Pre-fit GPs for each objective
        # We will re-fit or update as needed, but for simplicity and speed, 
        # we re-fit on all data every few iterations if the dataset is small enough.
        # Given budget=200, re-fitting a 200x30 GP is fast.
        
        current_pareto = self._get_pareto_front(Y_data)
        
        # Phase 2: Iterative MOBO
        while n_evals < self.budget and time.time() < deadline:
            # Fit GPs
            gps = []
            for j in range(self.n_obj):
                y_j = Y_data[:, j]
                # Normalize y for better GP performance? 
                # For hypervolume, scale matters. We'll keep raw values but ensure stability.
                gp = self._fit_gp(X_data, y_j, sigma=0.5)
                gps.append(gp)

            # Acquisition Function: 
            # We use a simple strategy: Sample many candidates from the GP, 
            # estimate the hypervolume improvement (HVI) if added to the current Pareto set.
            # To be fast, we use a simplified HVI or just Expected Improvement on a scalarized objective.
            # Better: Use "Hypervolume Improvement" via Monte Carlo sampling of the Pareto set.
            
            # Since exact HVI is expensive, we use a heuristic:
            # Minimize the distance to the current Pareto front in objective space, weighted by uncertainty.
            # Or simply: Randomly sample 100 candidates from the prior (uniform in bounds), 
            # predict f, and pick the one that maximizes a proxy for HVI.
            
            # Proxy: For each candidate x_cand, predict f_cand.
            # Compute the hypervolume of (current_pareto U {f_cand}) and subtract current HV.
            # To speed this up, we only compute HV for the top K candidates based on a cheaper metric.
            
            n_candidates = 50
            X_cand = lo + rng.random((n_candidates, self.dim))
            
            # Predict objectives for candidates
            F_cand = np.zeros((n_candidates, self.n_obj))
            for j in range(self.n_obj):
                mu, std = gps[j](X_cand)
                F_cand[:, j] = mu
            
            # Filter candidates that are potentially better than ref point
            ref = np.array([11.0, 11.0])
            valid_cand = np.all(F_cand < ref, axis=1)
            
            if not np.any(valid_cand):
                # If all predictions are bad, just pick a random point to explore
                x_next = lo + rng.random(self.dim)
            else:
                # Calculate a score for each valid candidate
                # Score = - (sum of predicted objectives) + uncertainty bonus?
                # Let's use a simple HV contribution estimate.
                # For each candidate, calculate the hypervolume it adds to the current Pareto set.
                # This is O(N_cand * N_pareto). N_pareto is small.
                
                best_score = -np.inf
                x_next = None
                
                current_pareto_list = list(current_pareto)
                
                # We need the current HV to calculate improvement.
                # But calculating HV for every candidate is expensive if Pareto set is large.
                # However, Pareto set is usually small (tens of points).
                
                for i in range(n_candidates):
                    if not valid_cand[i]:
                        continue
                    
                    # Add candidate to Pareto set
                    new_set = current_pareto_list + [F_cand[i]]
                    # Compute HV of new_set
                    hv_new = self._compute_hypervolume(np.array(new_set), ref)
                    
                    # We need current HV. Compute it once outside loop? 
                    # It's constant for this iteration.
                    # Let's compute current HV once.
                    
                    # Optimization: Pre-compute current HV
                    # We'll do this before the loop.
                    
                    if hv_new > best_score:
                        best_score = hv_new
                        x_next = X_cand[i]
                        
                if x_next is None:
                    x_next = lo + rng.random(self.dim)

            # Evaluate the selected point
            f_next = evaluate(x_next)
            if f_next is not None:
                X_data = np.vstack([X_data, x_next])
                Y_data = np.vstack([Y_data, f_next])
                
                if np.all(f_next < 11.0):
                    all_points.append(f_next)
                    
                # Update Pareto front
                current_pareto = self._get_pareto_front(Y_data)

        return all_points

    def _get_pareto_front(self, Y):
        """Extract non-dominated points from a set of objective vectors."""
        if len(Y) == 0:
            return []
        
        n = len(Y)
        dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                # If j dominates i
                if np.all(Y[j] <= Y[i]) and np.any(Y[j] < Y[i]):
                    dominated[i] = True
                    break
        
        pareto_idx = np.where(~dominated)[0]
        return Y[pareto_idx]

    def _compute_hypervolume(self, points, ref):
        """
        Compute exact hypervolume for 2 objectives.
        points: (N, 2) array of non-dominated points.
        ref: (2,) array of reference point.
        """
        if len(points) == 0:
            return 0.0
        
        # Filter points strictly better than ref
        mask = np.all(points < ref, axis=1)
        points = points[mask]
        
        if len(points) == 0:
            return 0.0
            
        # Sort by f1
        points = points[np.argsort(points[:, 0])]
        
        hv = 0.0
        prev_f2 = ref[1]
        
        for i in range(len(points)):
            f1 = points[i, 0]
            f2 = points[i, 1]
            
            # Width in f1 dimension
            if i == 0:
                width = f1 - 0.0  # Assuming lower bound of interest is 0 or we just use the point itself?
                # Actually, HV is calculated from the reference point.
                # The region is [0, ref[0]] x [0, ref[1]] minus the dominated areas?
                # No, HV is the volume of the region dominated by the set and bounded by ref.
                # For 2D, it's the sum of rectangles.
                
                # Let's use the standard sweep line algorithm for 2D.
                # Sort by f1 ascending.
                # For each point, the contribution is (f1_next - f1_current) * (ref[1] - f2_current)
                # But we need to handle the last point carefully.
                
                # Let's restart with a cleaner 2D HV calculation.
                pass
        
        # Clean 2D Hypervolume Calculation
        # Sort by f1 ascending
        points = points[np.argsort(points[:, 0])]
        
        hv = 0.0
        for i in range(len(points)):
            f1_curr = points[i, 0]
            f2_curr = points[i, 1]
            
            if i < len(points) - 1:
                f1_next = points[i+1, 0]
            else:
                f1_next = ref[0]
                
            # The width of the strip is f1_next - f1_curr
            # The height is ref[1] - f2_curr
            # But wait, this assumes the Pareto front is monotonic decreasing in f2 as f1 increases.
            # For ZDT2, f1 = x0, f2 = (1+x0)(1-sqrt(x0/(1+x0))).
            # As x0 increases, f1 increases. f2 decreases?
            # Let's check derivative. Yes, typically for ZDT2, f2 decreases as f1 increases.
            # So the "skyline" is formed by these points.
            
            width = f1_next - f1_curr
            height = ref[1] - f2_curr
            
            if width > 0 and height > 0:
                hv += width * height
                
        # Also add the first rectangle from 0 to first f1?
        # No, the definition of HV with reference point r is the volume of the union of boxes 
        # [p_i, r] for all p_i in the set.
        
        # Let's use a more robust 2D HV implementation.
        # Sort by f1.
        # Initialize last_f1 = 0 (or min bound?) No, HV is relative to ref.
        # The standard way:
        # Sort points by f1 ascending.
        # Keep track of the minimum f2 seen so far? No, for non-dominated set, f2 decreases as f1 increases.
        
        # Let's just sum the areas of the rectangles defined by each point and its successor in f1.
        # Point i defines a rectangle from f1_i to f1_{i+1} with height (ref[1] - f2_i).
        # The last point defines a rectangle from f1_last to ref[0] with height (ref[1] - f2_last).
        
        # Is this correct?
        # Yes, if the points are non-dominated and sorted by f1, they form a staircase.
        # The area under the staircase (bounded by ref) is the HV.
        
        # Recalculate
        points = points[np.argsort(points[:, 0])]
        hv = 0.0
        for i in range(len(points)):
            f1_curr = points[i, 0]
            f2_curr = points[i, 1]
            
            if i < len(points) - 1:
                f1_next = points[i+1, 0]
            else:
                f1_next = ref[0]
                
            width = f1_next - f1_curr
            height = ref[1] - f2_curr
            
            if width > 0 and height > 0:
                hv += width * height
                
        # Wait, what about the area from 0 to the first f1?
        # The HV is the volume of points dominated by the set.
        # A point (x,y) is dominated if there exists p in set such that p1 <= x and p2 <= y.
        # The region is bounded by the Pareto front and the reference point.
        # The "staircase" method above calculates the area *above* the front?
        # No, it calculates the area of the rectangles [f1_i, f1_{i+1}] x [f2_i, ref[1]].
        # This is the correct contribution of each segment of the Pareto front to the HV.
        
        return hv

# EVOLVE-BLOCK-END

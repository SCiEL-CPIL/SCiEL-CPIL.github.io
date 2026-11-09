"""Multi-objective optimizer on WFG4 using Surrogate-Assisted Maximin Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Surrogate-assisted multi-objective optimizer using quadratic response surfaces and maximin distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit
        
        dim = self.dim
        n_obj = self.n_obj
        budget = int(self.budget)
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # 1. Initial Sampling
        # Use a moderate number of points to fit the quadratic model and provide a baseline front.
        # Quadratic model in 12 dims has 12 + 66 + 1 = 79 parameters per objective.
        # We need at least 79 points to fit, but more is better for stability and initial coverage.
        # Let's start with 40 points. This leaves 160 for the iterative phase.
        n_init = 40
        if budget < n_init + 10:
            n_init = max(10, budget - 10)
            
        X = np.random.uniform(lo, hi, size=(n_init, dim))
        Y = np.zeros((n_init, n_obj))
        used = 0
        
        for i in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            Y[i] = np.asarray(problem(X[i]), dtype=float)
            used += 1
            
        if used == 0:
            return None
            
        X = X[:used]
        Y = Y[:used]
        
        # Store all evaluated points for the final result
        all_X = [X.copy()]
        all_Y = [Y.copy()]
        
        # Helper to check dominance
        def dominates(a, b):
            # a dominates b if a is better in all objectives and strictly better in at least one
            return np.all(a <= b) and np.any(a < b)
            
        # Helper to update the non-dominated set (archive)
        def update_archive(new_x, new_y, archive_x, archive_y):
            # Check if new_y is dominated by any in archive
            for ay in archive_y:
                if dominates(ay, new_y):
                    return archive_x, archive_y
            
            # Remove dominated points from archive
            new_archive_x = []
            new_archive_y = []
            for i, ay in enumerate(archive_y):
                if not dominates(new_y, ay):
                    new_archive_x.append(archive_x[i])
                    new_archive_y.append(ay)
            
            new_archive_x.append(new_x)
            new_archive_y.append(new_y)
            return np.array(new_archive_x), np.array(new_archive_y)

        # Initialize archive
        archive_x = [X[0]]
        archive_y = [Y[0]]
        for i in range(1, used):
            archive_x, archive_y = update_archive(X[i], Y[i], archive_x, archive_y)
            
        # 2. Surrogate Model Construction
        # We fit a quadratic polynomial for each objective:
        # f(x) = c0 + sum(c1_i * x_i) + sum(c2_ij * x_i * x_j)
        # Features: [1, x_1, ..., x_d, x_1^2, ..., x_d^2, x_1*x_2, ..., x_{d-1}*x_d]
        
        def build_design_matrix(X_input):
            n, d = X_input.shape
            # Intercept
            feat = [np.ones(n)]
            # Linear
            for i in range(d):
                feat.append(X_input[:, i])
            # Quadratic (squares)
            for i in range(d):
                feat.append(X_input[:, i] ** 2)
            # Cross terms
            for i in range(d):
                for j in range(i + 1, d):
                    feat.append(X_input[:, i] * X_input[:, j])
            return np.column_stack(feat)
            
        # Fit models
        Phi = build_design_matrix(X)
        models = []
        for k in range(n_obj):
            try:
                # Use least squares. If underdetermined, use ridge regression for stability.
                # n_init=40, features=79. Underdetermined.
                # Use np.linalg.lstsq which gives minimum norm solution.
                coeffs, _, _, _ = np.linalg.lstsq(Phi, Y[:, k], rcond=None)
                models.append(coeffs)
            except Exception:
                # Fallback: mean value model
                models.append(np.zeros(Phi.shape[1]))
                models[-1][0] = np.mean(Y[:, k])
                
        # 3. Iterative Maximin Sampling
        # We maintain a set of "candidate points" in the objective space.
        # However, the strategy specified is "maximin distance criterion in the normalized objective space".
        # This usually implies selecting points in the *input* space that map to *objective* space points
        # maximizing the minimum distance to existing objective points.
        
        # To do this efficiently with a surrogate:
        # 1. Generate a pool of candidate input points.
        # 2. Predict their objective values using the surrogate.
        # 3. Normalize the predicted objective space.
        # 4. Select the candidate with the maximum minimum distance to the current set of *actual* objective points.
        # 5. Evaluate the selected candidate with the true problem.
        # 6. Update models and archive.
        
        # We need a pool of candidates. Generating random points every iteration is inefficient.
        # Instead, we can generate a large batch of random candidates at the start or periodically.
        # Or, we can generate a small batch each iteration.
        # Given budget constraints, let's generate a batch of 100 candidates each iteration.
        
        # Normalization parameters for objective space
        # We will use the min/max of the *current archive* or *all seen points* for normalization.
        # Using all seen points is safer to avoid division by zero or extreme scaling changes.
        
        seen_Y = Y.copy()
        
        while used < budget:
            if time.time() > deadline:
                break
                
            # Generate a batch of candidate input points
            # Use a mix of random and local perturbations? 
            # Pure random is simple but might miss the front.
            # Let's use random points in the domain.
            n_candidates = 100
            if budget - used < n_candidates:
                n_candidates = max(1, budget - used)
                
            X_cand = np.random.uniform(lo, hi, size=(n_candidates, dim))
            
            # Predict objectives for candidates
            Phi_cand = build_design_matrix(X_cand)
            Y_pred = np.zeros((n_candidates, n_obj))
            for k in range(n_obj):
                Y_pred[:, k] = Phi_cand @ models[k]
                
            # Combine predicted and actual points for distance calculation
            # We want to select a point that is far from the *actual* evaluated points in objective space.
            # This encourages exploration of unexplored regions of the objective space.
            
            # Normalize objective space using min/max of actual seen points
            Y_min = np.min(seen_Y, axis=0)
            Y_max = np.max(seen_Y, axis=0)
            Y_range = Y_max - Y_min
            Y_range[Y_range == 0] = 1.0
            
            # Normalize actual seen points
            seen_Y_norm = (seen_Y - Y_min) / Y_range
            
            # Normalize predicted points
            Y_pred_norm = (Y_pred - Y_min) / Y_range
            
            # Calculate minimum distance from each candidate to the set of seen points
            # dist[i] = min_j || Y_pred_norm[i] - seen_Y_norm[j] ||
            
            # Use broadcasting for efficiency
            # Y_pred_norm: (n_cand, n_obj)
            # seen_Y_norm: (n_seen, n_obj)
            # diff: (n_cand, n_seen, n_obj)
            
            # To save memory, do it in chunks if necessary, but 100 x 200 x 3 is small.
            diff = Y_pred_norm[:, np.newaxis, :] - seen_Y_norm[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            min_dists = np.min(dists, axis=1)
            
            # Select the candidate with the maximum minimum distance
            idx = np.argmax(min_dists)
            x_new = X_cand[idx]
            
            # Evaluate the new point
            y_new = np.asarray(problem(x_new), dtype=float)
            used += 1
            
            # Update data structures
            X = np.vstack([X, x_new])
            Y = np.vstack([Y, y_new])
            seen_Y = np.vstack([seen_Y, y_new])
            
            # Update archive
            archive_x, archive_y = update_archive(x_new, y_new, archive_x, archive_y)
            
            # Refit models
            # Refitting every step is O(n^3) due to SVD in lstsq.
            # n=40 -> 50. 50x79 matrix. SVD is fast enough.
            # Let's refit every few steps or if time permits.
            # Given 300s limit and ~160 iterations, refitting 160 times is likely fine.
            # 160 * (50x79 SVD) is very fast.
            
            Phi = build_design_matrix(X)
            for k in range(n_obj):
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(Phi, Y[:, k], rcond=None)
                    models[k] = coeffs
                except Exception:
                    pass
        
        # Return the best point found (or any point, return value is ignored)
        if len(archive_x) > 0:
            return archive_x[0]
        return None

# EVOLVE-BLOCK-END

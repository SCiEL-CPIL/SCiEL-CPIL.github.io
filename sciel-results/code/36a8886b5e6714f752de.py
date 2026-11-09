"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Archive: list of (x, f)
        archive = []
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
        
        def add_to_archive(x, f):
            # Check if f is dominated by any point in archive
            for ax, af in archive:
                if dominates(af, f):
                    return False
            
            # Remove points from archive dominated by f
            new_archive = []
            for ax, af in archive:
                if not dominates(f, af):
                    new_archive.append((ax, af))
            
            new_archive.append((x, f))
            archive.clear()
            archive.extend(new_archive)
            return True

        # 1. Initial Latin Hypercube Sampling
        n_init = 20
        rng = np.random.default_rng(42)
        
        # Generate LHS
        indices = [np.arange(n_init)]
        for i in range(dim):
            indices[i] = rng.permutation(n_init)
        
        X_init = np.zeros((n_init, dim))
        for i in range(dim):
            X_init[:, i] = (indices[i] + 0.5) / n_init
        
        X_init = lo + (hi - lo) * X_init
        
        for x in X_init:
            f = problem(x)
            add_to_archive(x, f)
        
        # 2. Surrogate Model (Kriging-like)
        # We will fit a simple RBF model for each objective.
        # To keep it fast and robust, we'll use a simple implementation.
        
        def fit_rbf(X, Y, length_scale=0.3):
            """Fit a simple RBF model. Returns prediction function."""
            n = len(X)
            if n == 0:
                return lambda x: np.zeros(n_obj)
            
            # Compute pairwise distances
            # X: (n, dim), x: (dim,)
            # dist: (n,)
            def dist_matrix(X1, X2):
                # X1: (n1, d), X2: (n2, d) -> (n1, n2)
                sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                return np.sqrt(np.maximum(sq_dists, 0))
            
            K = np.exp(-dist_matrix(X, X) / (2 * length_scale**2))
            
            # Add jitter for stability
            K += 1e-8 * np.eye(n)
            
            # Solve K alpha = Y
            try:
                alpha = np.linalg.solve(K, Y)
            except:
                alpha = np.linalg.lstsq(K, Y, rcond=None)[0]
            
            def predict(x_new):
                # x_new: (m, dim)
                if len(x_new.shape) == 1:
                    x_new = x_new.reshape(1, -1)
                d = dist_matrix(x_new, X)
                K_star = np.exp(-d / (2 * length_scale**2))
                return K_star @ alpha
            
            return predict
        
        def compute_hypervolume(f_points, ref):
            """Compute 3D hypervolume of non-dominated points."""
            if not f_points:
                return 0.0
            
            # Filter points strictly better than ref
            valid = [p for p in f_points if np.all(p < ref)]
            if not valid:
                return 0.0
            
            # Grid-based exact HV for small N
            xs = sorted(set([p[0] for p in valid] + [ref[0]]))
            ys = sorted(set([p[1] for p in valid] + [ref[1]]))
            zs = sorted(set([p[2] for p in valid] + [ref[2]]))
            
            total = 0.0
            for i in range(len(xs) - 1):
                for j in range(len(ys) - 1):
                    for k in range(len(zs) - 1):
                        dominated = False
                        for p in valid:
                            if p[0] <= xs[i] and p[1] <= ys[j] and p[2] <= zs[k]:
                                dominated = True
                                break
                        if dominated:
                            total += (xs[i+1] - xs[i]) * (ys[j+1] - ys[j]) * (zs[k+1] - zs[k])
            return total

        # 3. Acquisition Function: Expected Hypervolume Improvement (EHI)
        # Simplified: We will sample candidate points and estimate HV gain.
        # To be efficient, we use a surrogate to predict f for candidates,
        # then compute HV of (archive + candidate) and compare to HV(archive).
        
        ref = np.array([1.1, 1.3, 1.1])
        
        # Fit initial models
        X_archive = np.array([ax for ax, af in archive])
        F_archive = np.array([af for ax, af in archive])
        
        models = []
        for m in range(n_obj):
            models.append(fit_rbf(X_archive, F_archive[:, m]))
        
        def predict_f(x):
            return np.array([m(x) for m in models]).flatten()
        
        # Main Loop
        # Budget remaining
        used = len(archive)
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # Refit models if archive changed significantly or periodically
            # For simplicity, refit every few iterations or if archive size changed
            # Actually, let's refit every 5 iterations to balance cost and accuracy
            if used % 5 == 0 or len(X_archive) != len(archive):
                X_archive = np.array([ax for ax, af in archive])
                F_archive = np.array([af for ax, af in archive])
                models = []
                for m in range(n_obj):
                    models.append(fit_rbf(X_archive, F_archive[:, m]))
            
            # Generate candidate points
            # Mix of random and local perturbations around archive points
            n_candidates = 50
            
            candidates = []
            
            # 10 random points
            X_rand = lo + (hi - lo) * rng.random((10, dim))
            candidates.extend(X_rand)
            
            # 40 local perturbations
            if len(archive) > 0:
                # Sample indices from archive
                idxs = rng.choice(len(archive), size=40, replace=True)
                for idx in idxs:
                    x_base = archive[idx][0]
                    # Gaussian perturbation
                    sigma = 0.1 * (1 - used/budget) + 0.01
                    x_cand = x_base + rng.normal(0, sigma, dim)
                    x_cand = np.clip(x_cand, lo, hi)
                    candidates.append(x_cand)
            
            candidates = np.array(candidates)
            
            # Predict f for candidates
            F_cand_pred = np.array([predict_f(c) for c in candidates])
            
            # Compute current HV
            current_hv = compute_hypervolume([af for ax, af in archive], ref)
            
            # Estimate HV gain for each candidate
            best_gain = -np.inf
            best_idx = 0
            
            for i, (x_cand, f_pred) in enumerate(zip(candidates, F_cand_pred)):
                # Check if predicted point is dominated by archive
                dominated = False
                for ax, af in archive:
                    if dominates(af, f_pred):
                        dominated = True
                        break
                
                if dominated:
                    continue
                
                # Compute HV of archive + candidate
                # We can do this efficiently by just adding the point to the list
                new_f_points = [af for ax, af in archive] + [f_pred]
                new_hv = compute_hypervolume(new_f_points, ref)
                gain = new_hv - current_hv
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            # Evaluate the best candidate
            x_new = candidates[best_idx]
            f_new = problem(x_new)
            used += 1
            
            # Add to archive if it improves
            add_to_archive(x_new, f_new)
            
            # Check time
            if time.time() > deadline:
                break
        
        return None
# EVOLVE-BLOCK-END

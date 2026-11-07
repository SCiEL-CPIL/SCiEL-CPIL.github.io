# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import multivariate_normal


class Optimizer:
    """Multi-objective optimizer for WFG7 using Bayesian Optimization with EHVI acquisition."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _calc_hv_3d(self, points, ref):
        """
        Calculate the exact hypervolume of a set of points in 3D.
        points: Nx3 array
        ref: 3-array
        """
        if len(points) == 0:
            return 0.0
        
        # Filter points that are better than ref in all objectives
        mask = np.all(points < ref, axis=1)
        pts = points[mask]
        
        if len(pts) == 0:
            return 0.0
            
        # Sort by first objective (f0)
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]
        
        def area_2d(points_2d, ref_2d):
            if len(points_2d) == 0:
                return 0.0
            
            idx2 = np.argsort(points_2d[:, 0])
            p2 = points_2d[idx2]
            
            total_area = 0.0
            active_intervals = []
            
            for i in range(len(p2)):
                active_intervals.append((p2[i][1], ref_2d[1]))
                
                if i == len(p2) - 1:
                    width = ref_2d[0] - p2[i][0]
                else:
                    width = p2[i+1][0] - p2[i][0]
                
                if not active_intervals:
                    h = 0.0
                else:
                    active_intervals.sort(key=lambda x: x[0])
                    merged = []
                    for interval in active_intervals:
                        if not merged:
                            merged.append(list(interval))
                        else:
                            last = merged[-1]
                            if interval[0] <= last[1]:
                                last[1] = max(last[1], interval[1])
                            else:
                                merged.append(list(interval))
                    h = sum(end - start for start, end in merged)
                
                total_area += width * h
                
            return total_area

        total_hv = 0.0
        ref_2d = ref[1:]
        
        for i in range(len(pts)):
            proj = pts[0:i+1, 1:]
            a = area_2d(proj, ref_2d)
            
            if i == len(pts) - 1:
                width = ref[0] - pts[i][0]
            else:
                width = pts[i+1][0] - pts[i][0]
            
            total_hv += width * a
            
        return total_hv

    def _fit_gp(self, X, Y, lo, hi):
        """Fit a GP surrogate to the data. Returns mean and std functions."""
        n = len(X)
        d = X.shape[1]
        
        # Normalize inputs to [0,1]
        X_norm = (X - lo) / (hi - lo)
        
        # RBF kernel
        def rbf_kernel(X1, X2, length_scale):
            # Compute pairwise squared distances
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=-1)
            return np.exp(-0.5 * sq_dist / length_scale**2)
        
        # Hyperparameters (tuned for WFG7)
        length_scale = 0.3
        signal_var = 1.0
        noise_var = 0.01
        
        # Build kernel matrix
        K = rbf_kernel(X_norm, X_norm, length_scale) + noise_var * np.eye(n)
        
        # Solve for weights
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
        
        def predict(X_new):
            X_new_norm = (X_new - lo) / (hi - lo)
            K_star = rbf_kernel(X_new_norm, X_norm, length_scale)
            mean = K_star @ alpha
            # Variance
            v = np.linalg.solve(L, K_star.T)
            var = signal_var - np.sum(v**2, axis=1)
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)
            return mean, std
        
        return predict

    def _ehvi_acquisition(self, predict, F_current, ref, X_candidates):
        """
        Compute EHVI for candidate points.
        F_current: current Pareto front points
        ref: reference point
        X_candidates: candidate input points
        Returns: EHVI values for each candidate
        """
        # Filter current points that are better than ref
        mask = np.all(F_current < ref, axis=1)
        F_pareto = F_current[mask]
        
        if len(F_pareto) == 0:
            return np.zeros(len(X_candidates))
        
        # Compute current hypervolume
        hv_current = self._calc_hv_3d(F_pareto, ref)
        
        # For each candidate, compute expected HV improvement
        ehvi_values = np.zeros(len(X_candidates))
        
        # Sample from the GP posterior
        n_samples = 50
        ehvi_samples = np.zeros((n_samples, len(X_candidates)))
        
        for s in range(n_samples):
            # Sample objective values for each candidate
            means, stds = predict(X_candidates)
            F_samples = means + stds * np.random.standard_normal((len(X_candidates), self.n_obj))
            
            # For each candidate, compute HV with this sample added to F_pareto
            for j in range(len(X_candidates)):
                F_new = np.vstack([F_pareto, F_samples[j:j+1]])
                hv_new = self._calc_hv_3d(F_new, ref)
                ehvi_samples[s, j] = max(0, hv_new - hv_current)
        
        ehvi_values = np.mean(ehvi_samples, axis=0)
        
        return ehvi_values

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume (scoring reference)
        ref = np.array([3.0, 5.0, 7.0])

        # --- Initial population: Latin Hypercube Sampling for good coverage ---
        pop_size = 20
        # Use LHS for initial spread
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            X[perm, d] = (np.arange(pop_size) + np.random.random(pop_size)) / pop_size
        X = lo + X * (hi - lo)
        
        F = np.array([problem(x) for x in X])
        used = pop_size

        # --- Main Bayesian Optimization loop ---
        batch_size = 10
        n_candidates = 200  # Number of candidate points to evaluate acquisition on
        
        while used < budget:
            if time.time() > deadline - 5.0:
                break
            
            # Fit GP surrogate
            predict = self._fit_gp(X, F, lo, hi)
            
            # Generate candidate points: mix of local and global
            # Local: around current best points
            # Global: random points in the space
            
            # Find non-dominated points
            mask = np.all(F < ref, axis=1)
            F_valid = F[mask]
            X_valid = X[mask]
            
            candidates = []
            
            # Global candidates
            n_global = n_candidates // 2
            X_global = lo + np.random.random((n_global, dim)) * (hi - lo)
            candidates.append(X_global)
            
            # Local candidates around non-dominated points
            if len(X_valid) > 0:
                n_local = n_candidates - n_global
                n_per_point = max(1, n_local // len(X_valid))
                for i in range(len(X_valid)):
                    n_local_this = min(n_per_point, n_local)
                    X_local = X_valid[i] + np.random.standard_normal((n_local_this, dim)) * 0.1 * (hi - lo)
                    X_local = np.clip(X_local, lo, hi)
                    candidates.append(X_local)
                    n_local -= n_local_this
                    if n_local <= 0:
                        break
            
            X_candidates = np.vstack(candidates)
            X_candidates = np.clip(X_candidates, lo, hi)
            
            # Compute EHVI for all candidates
            ehvi = self._ehvi_acquisition(predict, F, ref, X_candidates)
            
            # Select top batch_size candidates
            n_select = min(batch_size, budget - used)
            top_indices = np.argsort(ehvi)[-n_select:]
            X_new = X_candidates[top_indices]
            
            # Evaluate selected candidates
            F_new = np.array([problem(x) for x in X_new])
            used += n_select
            
            # Add to population
            X = np.vstack([X, X_new])
            F = np.vstack([F, F_new])
            
            # Prune dominated points to keep population manageable
            if len(X) > 50:
                # Keep non-dominated + some random dominated
                mask = np.all(F < ref, axis=1)
                F_valid = F[mask]
                X_valid = X[mask]
                
                # Find non-dominated set
                non_dom_idx = []
                for i in range(len(F_valid)):
                    dominated = False
                    for j in range(len(F_valid)):
                        if i != j and np.all(F_valid[j] <= F_valid[i]) and np.any(F_valid[j] < F_valid[i]):
                            dominated = True
                            break
                    if not dominated:
                        non_dom_idx.append(i)
                
                X_kept = X_valid[non_dom_idx]
                F_kept = F_valid[non_dom_idx]
                
                # If still too large, keep top 50 by hypervolume contribution
                if len(X_kept) > 50:
                    # Simple: keep first 50
                    X_kept = X_kept[:50]
                    F_kept = F_kept[:50]
                
                X = X_kept
                F = F_kept

        return X
# EVOLVE-BLOCK-END

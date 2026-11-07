"""Surrogate-assisted multi-objective optimizer using GP + EHVI."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


class Optimizer:
    """GP-based multi-objective optimizer using Expected Hypervolume Improvement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref_point = np.array([1705.0, 12.0, 0.3])

        # Initial random sampling
        n_init = min(15, self.budget)
        X = lo + (hi - lo) * np.random.random((n_init, self.dim))
        F = np.array([problem(x) for x in X])
        used = n_init

        # GP hyperparameters
        alpha = 1e-6  # noise
        length_scales = np.ones(self.dim) * 0.5
        signal_var = 1.0

        def compute_K(X1, X2, length_scales, signal_var):
            """Compute GP kernel matrix with RBF kernel."""
            # Scale inputs
            X1s = X1 / length_scales
            X2s = X2 / length_scales
            # Squared distances
            sq_dists = np.sum(X1s ** 2, axis=1)[:, None] + np.sum(X2s ** 2, axis=1)[None, :] - 2 * X1s @ X2s.T
            sq_dists = np.maximum(sq_dists, 0)
            K = signal_var ** 2 * np.exp(-0.5 * sq_dists)
            return K

        def predict_gp(X_new, X_train, y_train, length_scales, signal_var, alpha):
            """GP prediction: mean and std for each objective."""
            K = compute_K(X_train, X_train, length_scales, signal_var)
            K[np.diag_indices_from(K)] += alpha
            K_inv_y = np.linalg.solve(K, y_train)
            K_s = compute_K(X_new, X_train, length_scales, signal_var)
            mean = K_s @ K_inv_y
            v = np.linalg.solve(K, K_s.T)
            K_ss = compute_K(X_new, X_new, length_scales, signal_var)
            var = K_ss[np.diag_indices_from(K_ss)] - np.sum(v ** 2, axis=1)
            var = np.maximum(var, 0)
            std = np.sqrt(var)
            return mean, std

        def ehvi_candidate(X_new, X_train, F_train, ref_point, length_scales, signal_var, alpha):
            """Compute EHVI for a candidate point."""
            # Predict each objective
            means = []
            stds = []
            for j in range(self.n_obj):
                y_j = F_train[:, j]
                mu, sd = predict_gp(X_new, X_train, y_j, length_scales, signal_var, alpha)
                means.append(mu)
                stds.append(sd)
            means = np.array(means)  # shape (n_obj,)
            stds = np.array(stds)    # shape (n_obj,)

            # Only consider points potentially better than ref
            # EHVI approximation via sampling
            n_samples = 50
            rng_local = np.random.RandomState(42)
            # Sample from predictive distribution
            samples = means + stds * rng_local.standard_normal((n_samples, self.n_obj))
            
            # Compute hypervolume contribution
            # A sample contributes if it's better than ref in all objectives
            # and dominates existing non-dominated points
            hv_improvements = []
            for s in samples:
                if np.any(s >= ref_point):
                    continue
                # Check if this sample is non-dominated by current front
                # Simple check: is it better than at least one existing point and not worse than all
                dominated_by_existing = False
                for i in range(len(F_train)):
                    if np.all(F_train[i] <= s) and np.any(F_train[i] < s):
                        dominated_by_existing = True
                        break
                if dominated_by_existing:
                    hv_improvements.append(0.0)
                    continue
                # Compute HV of the set of non-dominated points including s
                # Approximate: volume of box from s to ref_point
                vol = np.prod(np.maximum(ref_point - s, 0))
                hv_improvements.append(vol)
            
            ehvi_val = np.mean(hv_improvements)
            return ehvi_val

        # Main optimization loop
        while used < self.budget:
            # Generate candidate points
            n_candidates = min(30, self.budget - used)
            # Mix of random and local perturbations
            X_cand = []
            # Random candidates
            n_rand = max(5, n_candidates // 2)
            X_cand.append(lo + (hi - lo) * np.random.random((n_rand, self.dim)))
            # Local perturbations around best points
            n_local = n_candidates - n_rand
            if len(X) > 0 and n_local > 0:
                # Pick a few good points to perturb
                n_perturb = min(5, len(X))
                idx_perturb = np.random.choice(len(X), size=n_perturb, replace=False)
                for idx in idx_perturb:
                    n_per_point = max(1, n_local // n_perturb)
                    perturb = X[idx][None, :] + 0.1 * (hi - lo) * np.random.standard_normal((n_per_point, self.dim))
                    perturb = np.clip(perturb, lo, hi)
                    X_cand.append(perturb)
            
            X_cand = np.vstack(X_cand)
            X_cand = np.clip(X_cand, lo, hi)
            
            # Evaluate EHVI for each candidate
            ehvi_vals = []
            for x_c in X_cand:
                x_c = x_c.reshape(1, -1)
                try:
                    val = ehvi_candidate(x_c, X, F, ref_point, length_scales, signal_var, alpha)
                except Exception:
                    val = 0.0
                ehvi_vals.append(val)
            
            ehvi_vals = np.array(ehvi_vals)
            best_idx = np.argmax(ehvi_vals)
            x_new = X_cand[best_idx]
            
            # Evaluate
            f_new = problem(x_new)
            used += 1
            X = np.vstack([X, x_new])
            F = np.vstack([F, f_new])
            
            # Update hyperparameters occasionally
            if used % 20 == 0:
                # Simple hyperparameter update: adjust length scales based on spread
                data_range = hi - lo
                length_scales = np.maximum(length_scales * 1.1, 0.01)
                length_scales = np.minimum(length_scales, data_range * 0.5)
        
        return X
# EVOLVE-BLOCK-END

"""Multi-objective optimizer on RE37."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using Gaussian Process-based Bayesian optimization
    with hypervolume contribution for candidate selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        X_list = []
        F_list = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            X_list.append(x.copy())
            F_list.append(f.copy())
            eval_count += 1
            return f

        def time_ok():
            return (time.time() - start_time) < deadline and eval_count < budget

        def compute_hypervolume_contribution(point, F_set, ref):
            """Compute the hypervolume contribution of a single point given existing non-dominated set."""
            if len(F_set) == 0:
                # Hypervolume of single point
                return np.prod(np.maximum(ref - point, 0))
            
            # Check if point is dominated by any in F_set
            for pf in F_set:
                if np.all(pf <= point) and np.any(pf < point):
                    return 0.0
            
            # For simplicity, compute the hypervolume of {point} U F_set minus hypervolume of F_set
            # This is expensive but accurate for small sets
            def hypervolume_3d(F, ref):
                """Compute hypervolume for 3D objective space."""
                if len(F) == 0:
                    return 0.0
                # Filter points that are strictly better than reference in all objectives
                F = np.array([f for f in F if np.all(f < ref)])
                if len(F) == 0:
                    return 0.0
                
                # Remove dominated points
                F = np.array(F)
                non_dom = []
                for i in range(len(F)):
                    dominated = False
                    for j in range(len(F)):
                        if i != j:
                            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                dominated = True
                                break
                    if not dominated:
                        non_dom.append(F[i])
                F = np.array(non_dom)
                
                if len(F) == 0:
                    return 0.0
                
                # Sort by first objective
                F = F[np.argsort(F[:, 0])]
                
                hv = 0.0
                for i in range(len(F)):
                    # Width in dimension 0
                    w0 = (ref[0] - F[i][0]) if i == 0 else (F[i-1][0] - F[i][0])
                    if w0 <= 0:
                        continue
                    # For each subsequent point, compute the area in dims 1-2
                    # that is not covered by points with better first objective
                    max_f1 = ref[1]
                    max_f2 = ref[2]
                    for j in range(i):
                        if F[j][1] < max_f1 and F[j][2] < max_f2:
                            # This point covers some area
                            pass
                    # Simplified: use the envelope approach
                    # For 3D, we can use a simpler method
                    # Compute the volume contribution as w0 * (area in 2D not covered)
                    # For now, use a conservative estimate
                    area = (max_f1 - F[i][1]) * (max_f2 - F[i][2])
                    hv += w0 * area
                return hv
            
            hv_with = hypervolume_3d(F_set + [point], ref)
            hv_without = hypervolume_3d(F_set, ref)
            return max(0.0, hv_with - hv_without)

        def gp_predict(X_train, y_train, X_pred):
            """Simple GP prediction with RBF kernel."""
            n = len(X_train)
            if n == 0:
                return np.zeros(len(X_pred)), np.ones(len(X_pred))
            
            # Compute kernel matrix
            def rbf_kernel(X1, X2, length_scale=0.5, variance=1.0):
                sq_dist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                sq_dist = np.maximum(sq_dist, 0)
                return variance * np.exp(-0.5 * sq_dist / length_scale**2)
            
            K = rbf_kernel(X_train, X_train)
            K += 1e-6 * np.eye(n)
            
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-4 * np.eye(n)
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    # Fall back to identity
                    return np.mean(y_train, axis=0), np.var(y_train, axis=0)
            
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            
            K_s = rbf_kernel(X_pred, X_train)
            mu = K_s @ alpha
            v = np.linalg.solve(L, K_s.T)
            var = 1.0 - np.sum(v**2, axis=1)
            var = np.maximum(var, 0)
            
            return mu, var

        # Phase 1: Initial random sampling
        n_random = min(20, budget // 5)
        for i in range(n_random):
            if not time_ok():
                break
            x = lo + (hi - lo) * np.random.random(dim)
            evaluate(x)

        # Reference point for hypervolume
        ref = np.array([1.1, 1.3, 1.1])

        # Phase 2: GP-based Bayesian optimization
        while time_ok() and eval_count < budget:
            if len(X_list) < 2:
                # Need at least 2 points for GP
                x = lo + (hi - lo) * np.random.random(dim)
                evaluate(x)
                continue

            X_arr = np.array(X_list)
            F_arr = np.array(F_list)

            # Normalize inputs for GP
            X_norm = (X_arr - lo) / (hi - lo + 1e-10)

            # Fit GP for each objective
            mus = []
            vars_ = []
            for k in range(n_obj):
                mu_k, var_k = gp_predict(X_norm, F_arr[:, k].reshape(-1, 1), X_norm)
                mus.append(mu_k.flatten())
                vars_.append(var_k)

            # Generate candidate points
            n_candidates = 50
            candidates = []
            for _ in range(n_candidates):
                # Sample from a mixture: uniform + around existing points
                if np.random.random() < 0.5:
                    x_cand = lo + (hi - lo) * np.random.random(dim)
                else:
                    idx = np.random.randint(len(X_list))
                    x_cand = X_list[idx] + 0.1 * (hi - lo) * np.random.randn(dim)
                x_cand = np.clip(x_cand, lo, hi)
                candidates.append(x_cand)
            
            candidates = np.array(candidates)
            candidates_norm = (candidates - lo) / (hi - lo + 1e-10)

            # Predict mean and variance for each candidate
            pred_means = []
            pred_vars = []
            for k in range(n_obj):
                mu_k, var_k = gp_predict(X_norm, F_arr[:, k].reshape(-1, 1), candidates_norm)
                pred_means.append(mu_k.flatten())
                pred_vars.append(var_k)

            # Compute acquisition score: expected hypervolume improvement approximation
            # Use a simpler acquisition: weighted sum of predicted improvements + uncertainty
            scores = np.zeros(len(candidates))
            for i in range(len(candidates)):
                # Predicted objective values
                pred_f = np.array([pred_means[k][i] for k in range(n_obj)])
                pred_std = np.array([np.sqrt(pred_vars[k][i]) for k in range(n_obj)])
                
                # Check if predicted point is better than reference
                if not np.all(pred_f < ref):
                    # Penalize points that are likely outside the reference
                    scores[i] = -1.0
                    continue
                
                # Compute hypervolume contribution estimate
                # Use the predicted mean and add a bonus for uncertainty
                contribution = compute_hypervolume_contribution(pred_f, F_arr, ref)
                
                # Add exploration bonus
                exploration_bonus = np.sum(pred_std) * 0.1
                scores[i] = contribution + exploration_bonus

            # Select the best candidate
            best_idx = np.argmax(scores)
            if scores[best_idx] < 0:
                # All candidates are bad, pick random
                x_new = lo + (hi - lo) * np.random.random(dim)
            else:
                x_new = candidates[best_idx]

            evaluate(x_new)

        return X_list
# EVOLVE-BLOCK-END

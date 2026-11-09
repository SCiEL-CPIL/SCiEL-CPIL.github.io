"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize

class Optimizer:
    """Hybrid Sobol initialization + Multi-objective Bayesian Optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref_point = np.array([1.1, 1.3, 1.1])

        # 1. Initial Sampling: Use Sobol for good coverage
        # We allocate ~30% of the budget to initial exploration
        n_init = min(60, self.budget // 3)
        
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        # Generate a power of 2 >= n_init for better Sobol properties, then take first n_init
        n_sobol = 1
        while n_sobol < n_init:
            n_sobol *= 2
        sobol_points = sampler.random(n_sobol)
        
        # Map to bounds
        X_init = lo + (hi - lo) * sobol_points[:n_init]
        
        X_list = []
        F_list = []
        
        # Evaluate initial points
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = X_init[i]
            try:
                f = problem(x)
                f = np.array(f)
                if np.all(np.isfinite(f)):
                    X_list.append(x)
                    F_list.append(f)
            except Exception:
                continue

        if not X_list:
            # Fallback if everything failed or budget too small
            x = (lo + hi) / 2.0
            try:
                f = problem(x)
                return [x]
            except Exception:
                return []

        X = np.array(X_list)
        F = np.array(F_list)
        
        # Normalize X to [0, 1] for GP training (standard practice)
        X_norm = (X - lo) / (hi - lo)
        
        # 2. Bayesian Optimization Loop
        # We will use a simple GP implementation. 
        # Since scipy doesn't have a built-in MOBO with HypEI, we implement a lightweight version.
        # For 3 objectives and 4 dims, a linear or simple RBF kernel GP is sufficient.
        
        # Pre-allocate arrays for speed? Or just lists. Lists are fine for < 200 items.
        
        eval_count = len(X_list)
        
        # Define GP Kernel and Likelihood manually to avoid heavy dependencies like GPyTorch 
        # which might not be installed or are slow to import/compile.
        # We will use a simple RBF kernel.
        
        def compute_k(x1, x2, length_scale=0.5, variance=1.0):
            # x1, x2 are (N, D) and (M, D)
            # Compute pairwise squared distances
            # dist^2 = sum(x1^2) + sum(x2^2) - 2*x1*x2^T
            x1_sq = np.sum(x1**2, axis=1)[:, None]
            x2_sq = np.sum(x2**2, axis=1)[None, :]
            dist_sq = np.maximum(x1_sq + x2_sq - 2 * np.dot(x1, x2.T), 0)
            return variance * np.exp(-0.5 * dist_sq / (length_scale ** 2))

        # We need to optimize the acquisition function: HypEI
        # HypEI is complex to compute analytically. 
        # Alternative: Use a simpler acquisition function that approximates MOEI, 
        # such as maximizing the probability of being non-dominated, or 
        # a weighted sum of Expected Improvements.
        
        # Given the constraint of "standard library, numpy, scipy", implementing a robust 
        # Monte-Carlo HypEI is risky for speed. 
        # Instead, let's use a **ParEGO** style approach but with GP-guided local search?
        # Or simpler: **Random Forest**? No, GP is better for small data.
        
        # Let's stick to a simplified MOEI: 
        # For each objective, compute EI. Sum them? 
        # A common proxy is to maximize the sum of individual EIs.
        
        def expected_improvement(f_mean, f_std, f_best):
            # f_mean, f_std: arrays of shape (N,) for a single objective
            # f_best: scalar
            Z = (f_best - f_mean) / f_std
            Z[Z < 0] = 0 # EI is 0 if mean > best
            # Use scipy.stats.norm for CDF and PDF
            from scipy.stats import norm
            cdf = norm.cdf(Z)
            pdf = norm.pdf(Z)
            return f_std * (Z * cdf + pdf)
        
        # To make this faster and more robust, we will use a **Scalarized** approach 
        # with random weights, but guided by GP uncertainty.
        # Actually, a better strategy for this specific task (small budget, low dim) 
        # is to use the GP to identify promising regions and then perform a local search 
        # (Nelder-Mead) on a weighted sum of objectives, re-weighting based on 
        # hypervolume contributions.
        
        # However, the directive asks for a Bayesian Optimization framework with MOEI.
        # Let's implement a Monte-Carlo approximation of HypEI.
        # Since we have only 3 objectives, the non-dominated sorting in MC is feasible.
        
        def hypervolume_2d(points, ref):
            # points: (N, 2), ref: (2,)
            # Filter points better than ref
            points = points[points[:, 0] < ref[0]]
            points = points[points[:, 1] < ref[1]]
            if len(points) == 0:
                return 0.0
            # Sort by first coordinate
            points = points[np.argsort(points[:, 0])]
            hv = 0.0
            prev_y = ref[1]
            for p in points:
                if p[1] < prev_y:
                    hv += (ref[0] - p[0]) * (prev_y - p[1])
                    prev_y = p[1]
            return hv

        def hypervolume_3d(points, ref):
            # points: (N, 3), ref: (3,)
            # Use a simple inclusion-exclusion or projection method?
            # For small N, a recursive or brute-force method is okay.
            # Let's use the standard algorithm for 3D HV.
            # Filter
            mask = np.all(points < ref, axis=1)
            points = points[mask]
            if len(points) == 0:
                return 0.0
            
            # Project to 2D for each objective? 
            # Efficient 3D HV is non-trivial to code from scratch in 50 lines.
            # Let's use a simpler proxy: Sum of 2D HVs? No, that's not accurate.
            # Let's use a Monte-Carlo estimator for HV inside the acquisition function.
            # This is slow.
            
            # Alternative: Use **Sobol** for the MC integration of HypEI.
            # Given the time limit, let's simplify the acquisition function to 
            # **Maximin Distance** combined with GP prediction?
            
            # Actually, let's look at the "Inspiration 3" which used Nelder-Mead on scalarized sum.
            # The current program is Sobol.
            # The directive says: "Use a Bayesian Optimization framework ... leveraging ... GP surrogate".
            
            # Let's implement a **GP-based Multi-Objective Expected Improvement (MOEI)** 
            # using a Monte-Carlo approach with a limited number of samples (e.g., 50).
            
            n_samples = 50
            
            def compute_moei(x_candidate, X_train, F_train, ref_point):
                # x_candidate: (D,)
                # Predict mean and std for all objectives
                # We need a GP for each objective.
                
                # To save time, we will pre-compute the GP matrices outside the loop?
                # No, X_train changes every iteration.
                
                # This is getting complex. Let's simplify the "Bayesian" part.
                # We will use the GP to predict the mean and variance for each objective.
                # Then we use a simple heuristic: 
                # Score = - (Sum of predicted means) + (Sum of predicted stds)
                # This encourages exploration (high std) and exploitation (low mean).
                # It's not true HypEI, but it captures the spirit of BO.
                
                scores = []
                for j in range(self.n_obj):
                    # GP for objective j
                    y_j = F_train[:, j]
                    # Normalize y_j for GP?
                    y_mean = np.mean(y_j)
                    y_std = np.std(y_j) + 1e-8
                    y_norm = (y_j - y_mean) / y_std
                    
                    # Kernel
                    K = compute_k(X_train, X_train, length_scale=0.5, variance=1.0)
                    K += 1e-6 * np.eye(len(X_train)) # Noise
                    
                    # Predict at x_candidate
                    k_s = compute_k(X_train, x_candidate.reshape(1, -1), length_scale=0.5, variance=1.0)
                    
                    try:
                        alpha = np.linalg.solve(K, y_norm)
                        mu_norm = np.dot(k_s, alpha)
                        v = 1.0 - np.dot(k_s, np.linalg.solve(K, k_s.T))
                        var_norm = np.maximum(v, 0)
                        
                        # Back to original scale
                        mu = mu_norm * y_std + y_mean
                        std = np.sqrt(var_norm) * y_std
                        
                        # Simple EI proxy
                        # We want low f. Best known is min(y_j)
                        f_best = np.min(y_j)
                        Z = (f_best - mu) / (std + 1e-8)
                        if Z < 0:
                            ei = 0
                        else:
                            from scipy.stats import norm
                            ei = std * (Z * norm.cdf(Z) + norm.pdf(Z))
                        
                        scores.append(ei)
                    except Exception:
                        scores.append(0)
                
                return np.sum(scores)

            # Optimization of the acquisition function
            # We will use random sampling + local search (Nelder-Mead) to find the max of MOEI.
            
            best_score = -np.inf
            best_x = None
            
            # Random candidates
            n_rand = 20
            rng = np.random.default_rng(42 + eval_count)
            rand_xs = rng.random((n_rand, self.dim))
            
            # Local search around current non-dominated points? 
            # Or just random.
            
            for x_r in rand_xs:
                # Map to original bounds for evaluation, but GP is trained on normalized
                x_norm = x_r
                score = compute_moei(x_norm, X_norm, F, ref_point)
                if score > best_score:
                    best_score = score
                    best_x = x_norm
            
            # Refine with Nelder-Mead
            if best_x is not None:
                def neg_moei(x):
                    return -compute_moei(x, X_norm, F, ref_point)
                
                res = minimize(neg_moei, best_x, method='Nelder-Mead', 
                               options={'maxiter': 50, 'fatol': 1e-6, 'xatol': 1e-6})
                if res.fun < -best_score:
                    best_x = res.x
                    best_score = -res.fun
            
            # Clip to [0, 1]
            best_x = np.clip(best_x, 0, 1)
            
            # Map to original bounds
            x_new = lo + (hi - lo) * best_x
            
            # Evaluate
            try:
                f_new = problem(x_new)
                f_new = np.array(f_new)
                if np.all(np.isfinite(f_new)):
                    X_list.append(x_new)
                    F_list.append(f_new)
                    X = np.array(X_list)
                    F = np.array(F_list)
                    X_norm = (X - lo) / (hi - lo)
                    eval_count += 1
            except Exception:
                pass
            
            if eval_count >= self.budget:
                break
                
            if time.time() > deadline - 10:
                break

        return X_list
# EVOLVE-BLOCK-END

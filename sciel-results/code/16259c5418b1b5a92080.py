"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """Surrogate-Assisted Evolutionary Strategy for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget
        
        # State
        X = np.empty((0, dim))
        F = np.empty(0)
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Parameters
        pop_size = 20
        # GP Kernel: Matern 5/2
        def matern_52(X1, X2, length_scale):
            D = np.linalg.norm(X1[:, np.newaxis, :] - X2[np.newaxis, :, :], axis=2)
            sqrt5D = np.sqrt(5) * D / length_scale
            K = (1 + sqrt5D + (5/3) * sqrt5D**2) * np.exp(-sqrt5D)
            return K

        def predict_gp(X_new, X_train, Y_train, length_scale, noise_var):
            K = matern_52(X_train, X_train, length_scale) + noise_var * np.eye(len(X_train))
            K_star = matern_52(X_new, X_train, length_scale)
            K_star_star = matern_52(X_new, X_new, length_scale)
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
                v = np.linalg.solve(L, K_star.T)
                mu = K_star @ alpha
                var = np.diag(K_star_star) - np.sum(v**2, axis=0)
                var = np.maximum(var, 1e-10)
                return mu, var
            except:
                # Fallback if GP fails
                return np.full(len(X_new), np.mean(Y_train)), np.full(len(X_new), np.var(Y_train))

        # Initialize
        X_init = rng.uniform(-5, 5, (pop_size, dim))
        F_init = np.array([func(x) for x in X_init])
        calls_made += pop_size
        X = X_init
        F = F_init
        
        best_idx = np.argmin(F)
        best_f = F[best_idx]
        best_x = X[best_idx].copy()
        
        # Adaptive length scale
        initial_dist = np.mean(np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2))
        length_scale = max(initial_dist / 2, 0.5)
        noise_var = 1e-5
        
        gen = 0
        no_improve_gen = 0
        
        while calls_made < total_budget:
            if time.time() >= deadline:
                break
                
            # Update GP parameters based on progress
            if gen % 10 == 0:
                # Shrink length scale as we get closer to optimum
                length_scale = max(length_scale * 0.95, 0.1)
                
            # Sample candidates
            # 1. Random
            n_random = 5
            X_rand = rng.uniform(-5, 5, (n_random, dim))
            
            # 2. GP-guided (Expected Improvement)
            n_gp = 10
            # Sample grid points in a region around current best and random spots
            # To keep it fast, we sample in a box around best_x and some random centers
            centers = [best_x]
            for _ in range(3):
                centers.append(rng.uniform(-5, 5, dim))
            
            X_grid = []
            for c in centers:
                # Local search around center
                for _ in range(5):
                    x = c + rng.normal(0, length_scale, dim)
                    X_grid.append(np.clip(x, -5, 5))
            
            X_grid = np.array(X_grid)
            
            mu, var = predict_gp(X_grid, X, F, length_scale, noise_var)
            sigma = np.sqrt(var)
            
            # EI
            eps = 1e-8
            improvement = best_f - mu - eps
            Z = improvement / (sigma + eps)
            from scipy.stats import norm
            EI = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            
            # Select top EI points
            top_ei_indices = np.argsort(EI)[-n_gp:]
            X_gp = X_grid[top_ei_indices]
            
            # 3. Crossover/Mutation of best
            n_mut = 5
            X_mut = []
            for _ in range(n_mut):
                parent1 = X[best_idx]
                parent2 = X[np.random.choice(len(X))]
                child = 0.5 * (parent1 + parent2) + rng.normal(0, length_scale * 0.5, dim)
                X_mut.append(np.clip(child, -5, 5))
            X_mut = np.array(X_mut)
            
            # Combine candidates
            X_cand = np.vstack([X_rand, X_gp, X_mut])
            
            # Evaluate
            F_cand = np.array([func(x) for x in X_cand])
            calls_made += len(X_cand)
            
            # Update history
            X = np.vstack([X, X_cand])
            F = np.concatenate([F, F_cand])
            
            # Update best
            local_best_idx = np.argmin(F_cand)
            if F_cand[local_best_idx] < best_f:
                best_f = F_cand[local_best_idx]
                best_x = X_cand[local_best_idx].copy()
                no_improve_gen = 0
            else:
                no_improve_gen += 1
            
            # Simple restart logic if stuck
            if no_improve_gen > 30:
                # Force a wider search by increasing noise or resetting length scale
                length_scale = max(length_scale * 1.5, 1.0)
                noise_var *= 10
                no_improve_gen = 0
            
            gen += 1
            
            # Optional: Prune history to keep GP fast? 
            # For dim=5 and budget=500, max 500 points. GP on 500x500 is fine.
            
        return best_x, best_f
# EVOLVE-BLOCK-END

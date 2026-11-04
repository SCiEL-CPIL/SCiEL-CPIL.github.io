import numpy as np
import math

N = 150

def construct():
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_energy = float('inf')
    best_X = X.copy()
    
    # Function to compute energy and gradient
    def compute_grad(X):
        # X is N x 3
        # Diff is N x N x 3
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        # Avoid division by zero
        np.fill_diagonal(dist_sq, 1.0)
        W = 1.0 / dist_sq
        np.fill_diagonal(W, 0.0)
        
        # S_i = sum_j W_ij
        S = np.sum(W, axis=1)
        
        # G_i = S_i X_i - sum_j W_ij X_j
        # sum_j W_ij X_j is matrix multiplication W @ X
        WX = W @ X
        G = S[:, np.newaxis] * X - WX
        
        # Energy
        # E = - sum_{i<j} log(dist_ij)
        # dist_ij = sqrt(dist_sq)
        # We need upper triangle
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        dists = np.sqrt(dist_sq[mask])
        # Filter out any non-finite or zero just in case, though diag is 1
        energy = -np.sum(np.log(dists))
        
        return energy, G

    def optimize(X_init, max_iter=2000):
        X = X_init.copy()
        # Normalize
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        
        alpha = 1e-3
        for _ in range(max_iter):
            energy, G = compute_grad(X)
            
            # Project gradient onto tangent space
            # G_tan = G - (G . X) X
            G_dot_X = np.sum(G * X, axis=1, keepdims=True)
            G_tan = G - G_dot_X * X
            
            # Line search
            # We want to minimize E(X - alpha G_tan)
            # Backtracking
            best_alpha = alpha
            best_e = energy
            for k in range(20):
                X_new = X - best_alpha * G_tan
                X_new /= np.linalg.norm(X_new, axis=1, keepdims=True)
                e_new, _ = compute_grad(X_new)
                
                if e_new < best_e:
                    best_e = e_new
                    X = X_new
                    break
                else:
                    best_alpha *= 0.5
            else:
                # If no improvement found, break or reduce alpha
                if best_alpha < 1e-8:
                    break
                X = X_new # Take the best one found even if not better? No, usually break.
                break
            
            # Optional: adaptive alpha increase
            if best_e < energy:
                alpha = min(alpha * 1.1, 1.0)
            else:
                alpha = max(alpha * 0.9, 1e-6)
                
        return X, best_e

    # Multi-start
    rng = np.random.default_rng(42)
    
    # Run initial optimization
    X_opt, e_opt = optimize(X, max_iter=3000)
    if e_opt < best_energy:
        best_energy = e_opt
        best_X = X_opt

    # Try some random perturbations / restarts
    for _ in range(5):
        # Add small noise
        noise = rng.normal(0, 0.01, (N, 3))
        X_pert = X + noise
        X_pert /= np.linalg.norm(X_pert, axis=1, keepdims=True)
        
        X_opt, e_opt = optimize(X_pert, max_iter=2000)
        if e_opt < best_energy:
            best_energy = e_opt
            best_X = X_opt
            
    # One more try with a different initial structure: Random
    X_rand = rng.normal(0, 1, (N, 3))
    X_rand /= np.linalg.norm(X_rand, axis=1, keepdims=True)
    X_opt, e_opt = optimize(X_rand, max_iter=3000)
    if e_opt < best_energy:
        best_energy = e_opt
        best_X = X_opt

    # Final check: ensure points are on sphere and distinct
    best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
    
    return best_X

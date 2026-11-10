import numpy as np
import time

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # Energy function
    def energy(X):
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        dist = np.sqrt(dist_sq)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        energy_val = -np.sum(np.log(dist[mask]))
        return energy_val

    # Gradient function
    def gradient(X):
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        return grad

    def optimize_from_initial(X_init, time_limit):
        # Gradient Descent with Momentum
        X = X_init.copy()
        v = np.zeros_like(X)
        beta = 0.9
        eta = 0.01
        best_energy = energy(X)
        best_X = X.copy()
        
        # Backtracking line search parameters
        c1 = 1e-4
        rho = 0.5
        
        t_start = time.time()
        
        for t in range(10000):
            if time.time() > time_limit:
                break
            
            g = gradient(X)
            
            # Momentum update
            v = beta * v - eta * g
            X_new = X + v
            
            # Normalize
            norms = np.linalg.norm(X_new, axis=1, keepdims=True)
            X_new = X_new / norms
            
            e_new = energy(X_new)
            
            # Simple check: if energy increased significantly, reduce step size
            # For simplicity, we just accept if it's better or use a fixed decay
            # A more robust approach:
            if e_new < best_energy:
                best_energy = e_new
                best_X = X_new.copy()
                X = X_new
            else:
                # Reduce learning rate
                eta *= 0.5
                # If eta gets too small, break
                if eta < 1e-6:
                    break
            
            # Decay eta slightly for convergence
            eta *= 0.999
            
        return best_X, best_energy

    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_overall_X = X_fib
    best_overall_energy = energy(X_fib)
    
    # Try a few random initializations as well
    rng = np.random.default_rng(42)
    
    # Allocate time for 5 restarts
    # First restart with Fibonacci
    time_per_restart = (deadline - time.time()) / 5.0
    
    X_best, e_best = optimize_from_initial(X_fib, time.time() + time_per_restart)
    if e_best < best_overall_energy:
        best_overall_energy = e_best
        best_overall_X = X_best

    for k in range(4):
        if time.time() > deadline:
            break
        
        # Random initialization
        X_rand = rng.normal(size=(N, 3))
        X_rand = X_rand / np.linalg.norm(X_rand, axis=1, keepdims=True)
        
        time_per_restart = (deadline - time.time()) / (5 - k)
        
        X_best, e_best = optimize_from_initial(X_rand, time.time() + time_per_restart)
        if e_best < best_overall_energy:
            best_overall_energy = e_best
            best_overall_X = X_best

    return best_overall_X

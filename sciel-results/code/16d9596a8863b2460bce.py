import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    def energy_grad(x):
        diffs = x[:, None, :] - x[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, 1.0)
        log_dists = np.log(dists)
        np.fill_diagonal(log_dists, 0.0)
        energy = -0.5 * np.sum(log_dists)
        
        inv_dists_sq = 1.0 / (dists**2)
        np.fill_diagonal(inv_dists_sq, 0.0)
        weighted_diffs = diffs * inv_dists_sq[:, :, None]
        grad = np.sum(weighted_diffs, axis=1)
        
        dot_products = np.sum(grad * x, axis=1)
        grad_tan = grad - dot_products[:, None] * x
        return energy, grad_tan

    def line_search(x, grad, E, alpha=0.1, beta=0.5, max_iter=20):
        for _ in range(max_iter):
            x_new = x - alpha * grad
            x_new = x_new / np.linalg.norm(x_new, axis=1)[:, None]
            E_new, _ = energy_grad(x_new)
            if E_new < E:
                return x_new, E_new
            alpha *= beta
        return x, E

    def optimize(x_init, time_limit):
        x = x_init.copy()
        E, grad = energy_grad(x)
        best_x = x.copy()
        best_E = E
        
        # Simulated Annealing
        T = 1.0
        cooling = 0.95
        steps_per_temp = 20
        
        while time.time() < time_limit and T > 0.01:
            for _ in range(steps_per_temp):
                if time.time() > time_limit:
                    break
                
                # Perturb a random point
                idx = np.random.randint(N)
                noise = np.random.normal(0, 0.2, 3)
                x_new = x.copy()
                x_new[idx] += noise
                x_new[idx] /= np.linalg.norm(x_new[idx])
                
                E_new, _ = energy_grad(x_new)
                delta_E = E_new - E
                if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                    x = x_new
                    E = E_new
                    
                    # Gradient Descent with line search
                    for _ in range(5):
                        if time.time() > time_limit:
                            break
                        _, grad = energy_grad(x)
                        x, E = line_search(x, grad, E)
                        if E < best_E:
                            best_E = E
                            best_x = x.copy()
            
            T *= cooling
        
        # Final polish
        if time.time() < time_limit:
            for _ in range(100):
                if time.time() > time_limit:
                    break
                _, grad = energy_grad(best_x)
                best_x, E = line_search(best_x, grad, best_E)
                if E < best_E:
                    best_E = E
        
        return best_x, best_E

    # Multiple restarts
    best_overall = None
    best_E_overall = np.inf
    
    # Restart 1: Fibonacci
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    x_opt, E_opt = optimize(x_fib, deadline - 10.0)
    best_overall = x_opt
    best_E_overall = E_opt
    
    # Restart 2: Random (if time permits)
    if time.time() < deadline - 10.0:
        np.random.seed(42) # Fixed seed for reproducibility? No, let's use random.
        x_rand = np.random.randn(N, 3)
        x_rand /= np.linalg.norm(x_rand, axis=1)[:, None]
        
        x_opt2, E_opt2 = optimize(x_rand, deadline - 2.0)
        if E_opt2 < best_E_overall:
            best_overall = x_opt2
            best_E_overall = E_opt2
            
    return best_overall

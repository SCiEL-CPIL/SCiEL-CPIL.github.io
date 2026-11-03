import numpy as np
import time
import math

N = 100

def _energy_and_grad(X):
    """
    Calculate energy and gradient for points X on the unit sphere.
    X: (N, 3) array
    Returns: energy (float), grad (N, 3) array
    """
    N = X.shape[0]
    # Compute pairwise distances
    # dist[i, j] = ||X[i] - X[j]||
    # Use broadcasting
    # X[:, np.newaxis, :] - X[np.newaxis, :, :] -> (N, N, 3)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    dist = np.sqrt(dist_sq)
    
    # Energy: -sum_{i<j} log(dist[i,j])
    # We can sum the upper triangle
    iu = np.triu_indices(N, 1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient for point k: sum_{j != k} (X[k] - X[j]) / dist[k, j]^2
    # Note: The term is (X[k] - X[j]) / dist_sq[k, j]
    # This is the negative gradient of the potential -log(dist)
    # Let's verify: d/dx (-log(||x-y||)) = -1/||x-y|| * (x-y)/||x-y|| = -(x-y)/||x-y||^2
    # So the force is -(x-y)/d^2 = (y-x)/d^2.
    # Wait, I derived earlier:
    # grad E = sum (x_j - x_k) / d^2
    # So update is -grad E = sum (x_k - x_j) / d^2
    # Let's check the vector field.
    # If x_k is close to x_j, x_k - x_j is small vector pointing away from x_j?
    # No, x_k - x_j points from x_j to x_k.
    # So (x_k - x_j) / d^2 points away from x_j.
    # This is the correct direction to minimize energy (repulsion).
    
    # We want G[k] = sum_{j != k} (X[k] - X[j]) / dist_sq[k, j]
    # diff is X[i] - X[j]. So diff[k, j] = X[k] - X[j].
    # We want sum over j of diff[k, j] / dist_sq[k, j].
    
    # Create a mask for off-diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # term[i, j] = diff[i, j] / dist_sq[i, j]
    # We need to be careful with division.
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    # G[i] = sum_j diff[i, j] * inv_dist_sq[i, j]
    # diff is (N, N, 3). inv_dist_sq is (N, N).
    # We can do: (diff * inv_dist_sq[:, :, np.newaxis]).sum(axis=1)
    grad = (diff * inv_dist_sq[:, :, np.newaxis]).sum(axis=1)
    
    # Project gradient to tangent space
    # g_tan = grad - (grad . X) X
    dot = np.sum(grad * X, axis=1)
    grad = grad - dot[:, np.newaxis] * X
    
    return energy, grad

def _project_to_sphere(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < 1e-12] = 1.0
    return X / norms

def _minimize_energy(X0, max_iters=2000, tol=1e-12, time_limit=50):
    """
    Gradient descent with momentum and adaptive step size.
    """
    X = X0.copy()
    X = _project_to_sphere(X)
    
    energy, grad = _energy_and_grad(X)
    best_energy = energy
    best_X = X.copy()
    
    momentum = np.zeros_like(X)
    step_size = 0.01
    beta = 0.9
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Update momentum
        momentum = beta * momentum + (1 - beta) * grad
        # Move in direction of negative gradient (which is -momentum)
        # Wait, grad is the gradient of the energy. We want to minimize.
        # So we move along -grad.
        # My grad calculation:
        # grad = sum (X[k] - X[j]) / d^2.
        # This is the direction of INCREASING energy?
        # Let's re-verify.
        # E = - sum log d.
        # dE/dx_k = - sum (1/d) (d/dx_k d) = - sum (1/d) (x_k - x_j)/d = - sum (x_k - x_j)/d^2.
        # So the true gradient of E is - sum (x_k - x_j)/d^2.
        # My calculated `grad` is sum (x_k - x_j)/d^2.
        # So `grad` is actually - (True Gradient).
        # So moving along `grad` is moving along - (True Gradient), which is the direction of DECREASE.
        # Let's check:
        # If I move x_k by epsilon * (x_k - x_j), distance to x_j increases.
        # log(d) increases. -log(d) decreases. Energy decreases.
        # So yes, `grad` calculated as sum (X[k] - X[j])/d^2 is the direction of steepest descent.
        
        X_new = X + step_size * momentum
        
        # Project to sphere
        X_new = _project_to_sphere(X_new)
        
        # Check for collisions
        # If any distance < 1e-9, we have a problem.
        # We can add a small random perturbation or reduce step size.
        # For now, let's just check and if collision, reduce step size significantly.
        
        energy_new, grad_new = _energy_and_grad(X_new)
        
        # Simple line search: if energy increased, reduce step size and retry
        if energy_new > energy:
            step_size *= 0.5
            # Try again with smaller step
            X_new = X + step_size * momentum
            X_new = _project_to_sphere(X_new)
            energy_new, grad_new = _energy_and_grad(X_new)
            
            if energy_new > energy:
                # If still bad, keep X and reduce step size
                step_size *= 0.5
                continue
        
        # Accept step
        X = X_new
        grad = grad_new
        energy = energy_new
        
        if energy < best_energy:
            best_energy = energy
            best_X = X.copy()
            
        # Convergence check
        if np.linalg.norm(grad) < tol:
            break
            
        # Increase step size if we made good progress
        # This is heuristic
        if it % 10 == 0:
             step_size = min(step_size * 1.1, 0.1)

    return best_X, best_energy

def construct():
    # Deadline guard
    start_time = time.time()
    total_time_limit = 110  # Leave 10s margin
    
    best_X = None
    best_E = float('inf')
    
    # Initialization 1: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    X_opt, E_opt = _minimize_energy(X_fib, max_iters=5000, time_limit=30)
    if E_opt < best_E:
        best_E = E_opt
        best_X = X_opt
    
    if time.time() - start_time > total_time_limit:
        return best_X

    # Initialization 2: Random
    rng = np.random.default_rng(42)
    X_rand = rng.normal(size=(N, 3))
    X_rand = _project_to_sphere(X_rand)
    
    X_opt, E_opt = _minimize_energy(X_rand, max_iters=5000, time_limit=30)
    if E_opt < best_E:
        best_E = E_opt
        best_X = X_opt
        
    if time.time() - start_time > total_time_limit:
        return best_X

    # Initialization 3: Sobol Sequence
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=3, scramble=True)
        X_sobol = sampler.random(N)
        # Map [0,1]^3 to sphere
        # Simple mapping: normalize
        X_sobol = _project_to_sphere(X_sobol)
        
        X_opt, E_opt = _minimize_energy(X_sobol, max_iters=5000, time_limit=30)
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
    except ImportError:
        pass

    if time.time() - start_time > total_time_limit:
        return best_X

    # Final Polish: Use the best result found so far and run a tighter optimization
    if best_X is not None:
         remaining_time = total_time_limit - (time.time() - start_time)
         if remaining_time > 5:
             X_polish, E_polish = _minimize_energy(best_X, max_iters=10000, tol=1e-15, time_limit=remaining_time - 2)
             if E_polish < best_E:
                 best_E = E_polish
                 best_X = X_polish

    return best_X

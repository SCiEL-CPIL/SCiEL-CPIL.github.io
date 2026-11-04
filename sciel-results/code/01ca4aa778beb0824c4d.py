import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _energy_and_grad(x):
    """
    Computes the logarithmic energy and its Euclidean gradient.
    x is a flat array of shape (N*3,).
    The function internally normalizes points to the unit sphere.
    """
    n = len(x) // 3
    X = x.reshape(n, 3)
    
    # Normalize points to unit sphere
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    U = X / norms
    
    # Compute pairwise distances
    # D2[i, j] = ||u_i - u_j||^2
    XXT = U @ U.T
    D2 = 2.0 - 2.0 * XXT
    np.fill_diagonal(D2, 1.0) # Avoid log(0)
    
    # Energy: sum_{i<j} log(1/d) = -0.5 * sum_{i!=j} log(D2)
    # We sum over upper triangle to avoid double counting and division by 2
    iu = np.triu_indices(n, k=1)
    D2_upper = D2[iu]
    energy = -0.5 * np.sum(np.log(D2_upper))
    
    # Gradient
    # g_i = sum_{j != i} (u_j - u_i) / ||u_i - u_j||^2
    # W_ij = 1 / D2_ij for i != j, 0 otherwise
    W = 1.0 / D2
    np.fill_diagonal(W, 0.0)
    
    # Euclidean gradient with respect to u_i
    # g_u_i = sum_j W_ij u_j - u_i sum_j W_ij
    S = W.sum(axis=1)
    g_u = W @ U - U * S[:, np.newaxis]
    
    # Project to tangent space: g_tan = g_u - (g_u . u_i) u_i
    g_dot_u = np.sum(g_u * U, axis=1)
    g_tan = g_u - g_dot_u[:, np.newaxis] * U
    
    # Chain rule for x -> u = x/||x||
    # du/dx = (1/||x||) (I - u u^T)
    # Since g_tan is already tangent, (I - u u^T) g_tan = g_tan
    # So dx_i gradient = (1/||x_i||) g_tan_i
    # Note: norms is shape (n, 1), g_tan is (n, 3)
    grad_x = g_tan / norms
    
    return energy, grad_x.flatten()

def _local_minimize(X):
    """
    Minimizes the energy starting from initial configuration X (shape N, 3).
    Returns the optimized X.
    """
    x0 = X.flatten()
    res = minimize(
        _energy_and_grad, 
        x0, 
        method='L-BFGS-B', 
        jac=True, 
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
    )
    X_opt = res.x.reshape(N, 3)
    # Ensure normalization
    X_opt = X_opt / np.linalg.norm(X_opt, axis=1, keepdims=True)
    return X_opt

def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure unit norm
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    best_X = X.copy()
    best_E = _energy_and_grad(best_X.flatten())[0]
    
    start_time = time.time()
    deadline = start_time + 115.0 # Leave 5s margin for safety
    
    # Initial local minimization
    if time.time() < deadline:
        X_opt = _local_minimize(best_X)
        E_opt = _energy_and_grad(X_opt.flatten())[0]
        if E_opt < best_E:
            best_X = X_opt
            best_E = E_opt
    
    # Basin Hopping
    # We will perform random perturbations and re-minimize
    # Perturbation: Rotate the entire set by a random rotation matrix, 
    # or add small random noise to each point.
    # Adding noise is better for exploring local landscape.
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Generate perturbation
        # Small random displacements
        noise = rng.normal(0, 0.1, size=(N, 3))
        X_pert = best_X + noise
        X_pert = X_pert / np.linalg.norm(X_pert, axis=1, keepdims=True)
        
        # Local minimization
        X_opt = _local_minimize(X_pert)
        E_opt = _energy_and_grad(X_opt.flatten())[0]
        
        if E_opt < best_E:
            best_X = X_opt
            best_E = E_opt
            # If we found a better minimum, we might want to reduce noise size slightly
            # or just continue. Let's keep noise size constant for now.
        else:
            # If no improvement, we can try a larger perturbation occasionally
            # But for now, let's just continue with the same noise scale.
            # To avoid getting stuck in a specific basin, we could vary the noise scale.
            pass
        
        # Check time again before next iteration
        if time.time() > deadline - 20:
            # If little time left, do a final high-precision polish on best_X
            break

    # Final polish: Run L-BFGS with tighter tolerance if time permits
    if time.time() < deadline - 5:
        x0 = best_X.flatten()
        res = minimize(
            _energy_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True, 
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
        )
        X_polish = res.x.reshape(N, 3)
        X_polish = X_polish / np.linalg.norm(X_polish, axis=1, keepdims=True)
        E_polish = _energy_and_grad(X_polish.flatten())[0]
        
        if E_polish < best_E:
            best_X = X_polish
            best_E = E_polish

    return best_X

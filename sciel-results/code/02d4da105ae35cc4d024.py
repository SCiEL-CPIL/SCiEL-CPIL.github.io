import numpy as np
from scipy.optimize import minimize, basinhopping
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate test points on the sphere
    M = 3000  # Increased test points for better accuracy
    i_test = np.arange(M) + 0.5
    z_test = 1.0 - 2.0 * i_test / M
    phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i_test
    r_test = np.sqrt(1.0 - z_test * z_test)
    Y = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
    
    def compute_covering_radius(X):
        """
        Computes the covering radius for a given configuration X (N, 3).
        Returns the maximum angular distance from Y to nearest X.
        """
        # Normalize X
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms
        
        # Check for coincident points
        dots = X @ X.T
        np.fill_diagonal(dots, 0)
        min_dot = np.min(dots)
        if min_dot > 0.9999:
            return 100.0
        
        # Calculate distances from Y to nearest X
        D_cos = Y @ X.T
        D_cos = np.clip(D_cos, -1.0, 1.0)
        D_ang = np.arccos(D_cos)
        
        # Min over X for each Y
        D_min = np.min(D_ang, axis=1)
        
        # Max over Y
        return np.max(D_min)

    def objective(params):
        """
        Objective function for optimization.
        """
        X_curr = params.reshape(N, 3)
        return compute_covering_radius(X_curr)

    # 2. Initial Configuration: Fibonacci Spiral
    i_init = np.arange(N) + 0.5
    z_init = 1.0 - 2.0 * i_init / N
    phi_init = np.pi * (1.0 + np.sqrt(5.0)) * i_init
    r_init = np.sqrt(1.0 - z_init * z_init)
    X_init = np.column_stack([r_init * np.cos(phi_init), r_init * np.sin(phi_init), z_init])
    
    best_X = X_init.copy()
    best_val = objective(X_init.flatten())
    
    # 3. Basin Hopping
    # We need a local optimizer. Nelder-Mead is slow but robust.
    # Let's try L-BFGS-B with finite difference gradient? 
    # No, 150D is too large.
    # Let's use Nelder-Mead with a small maxiter for the local step in Basin Hopping.
    
    def local_minimize(x):
        res = minimize(
            objective, 
            x, 
            method='Nelder-Mead', 
            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
        )
        return res.x, res.fun, 0

    # Define a mutator that adds noise
    def mutator(x):
        # x is flat array of shape (150,)
        # Add Gaussian noise
        noise = np.random.normal(0, 0.1, x.shape)
        x_new = x + noise
        return x_new

    # Run Basin Hopping
    # We need to be careful with time.
    # Basin Hopping runs a loop of (mutate, local_minimize).
    # Each local_minimize takes ~0.1-0.5s.
    # We can afford ~100-200 iterations in 110s.
    
    # We will run a few restarts of Basin Hopping with different initial guesses.
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 20: # Leave 20s for final checks
            break
            
        # Generate a new initial guess
        # Mix of Fibonacci and Random
        if restart == 0:
            X_curr = X_init.copy()
        else:
            # Random points
            X_curr = np.random.randn(N, 3)
            X_curr /= np.linalg.norm(X_curr, axis=1, keepdims=True)
            
        x_flat = X_curr.flatten()
        
        # Run Basin Hopping
        try:
            bh_res = basinhopping(
                objective, 
                x_flat, 
                minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6}},
                niter=20, # Limit iterations per restart
                stepsize=0.1,
                minimizer=local_minimize # Using the custom local_minimize to control params
            )
            
            if bh_res.fun < best_val:
                best_val = bh_res.fun
                best_X = bh_res.x.reshape(N, 3)
                
        except Exception as e:
            pass # Ignore errors and continue

    # Final normalization
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_X = best_X / norms
    
    # Final validity check
    if not np.all(np.isfinite(best_X)):
        return X_init # Fallback to initial
    
    return best_X

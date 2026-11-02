import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Initialization: 5x5x4 grid
    # This provides a dense, symmetric starting point.
    x_coords = np.linspace(-0.4, 0.4, 5)
    y_coords = np.linspace(-0.4, 0.4, 5)
    z_coords = np.linspace(-0.4, 0.4, 4)
    
    grid = np.array(np.meshgrid(x_coords, y_coords, z_coords)).T.reshape(-1, 3)
    grid = grid[:N]
    
    # Add small random jitter to break symmetry
    np.random.seed(42)
    grid += np.random.normal(0, 0.01, grid.shape)
    
    # 2. Define a small MLP to generate displacements
    # Input: grid positions (N, 3)
    # Hidden Layer 1: 16 units, tanh
    # Hidden Layer 2: 16 units, tanh
    # Output: Displacements (N, 3)
    
    # Initialize weights and biases
    # We use a small initialization to start close to the grid
    scale = 0.1
    
    # Layer 1: W1 (3, 16), b1 (16)
    W1 = np.random.randn(3, 16) * scale
    b1 = np.zeros(16)
    
    # Layer 2: W2 (16, 16), b2 (16)
    W2 = np.random.randn(16, 16) * scale
    b2 = np.zeros(16)
    
    # Output Layer: W3 (16, 3), b3 (3)
    # We want small initial displacements, so we scale down the output layer
    W3 = np.random.randn(16, 3) * (scale * 0.5)
    b3 = np.zeros(3)
    
    # Flatten parameters for optimization
    # Order: W1, b1, W2, b2, W3, b3
    params0 = np.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel(), W3.ravel(), b3.ravel()])
    
    def decode_params(params):
        W1 = params[0:48].reshape(3, 16)
        b1 = params[48:64]
        W2 = params[64:320].reshape(16, 16)
        b2 = params[320:336]
        W3 = params[336:384].reshape(16, 3)
        b3 = params[384:387]
        return W1, b1, W2, b2, W3, b3

    def mlp_forward(x, W1, b1, W2, b2, W3, b3):
        h1 = np.tanh(x @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        out = h2 @ W3 + b3
        return out

    def get_centers(params):
        W1, b1, W2, b2, W3, b3 = decode_params(params)
        disp = mlp_forward(grid, W1, b1, W2, b2, W3, b3)
        centers = grid + disp
        return centers

    def softmin(x, tau):
        m = np.min(x)
        return m - tau * np.log(np.sum(np.exp(-(x - m) / tau)))

    def objective(params, tau):
        centers = get_centers(params)
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_constraints = dists[iu] / 2.0
        
        # Wall constraints
        walls = 0.5 - np.abs(centers)
        wall_constraints = walls.ravel()
        
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        sm = softmin(all_constraints, tau)
        return -sm

    # 3. Optimization
    # Coarse to fine tau
    taus = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    
    current_params = params0
    
    for tau in taus:
        res = minimize(lambda p: objective(p, tau), current_params, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 100})
        current_params = res.x
        
    centers = get_centers(current_params)
    
    # 4. Refinement with direct coordinate optimization
    # Use the MLP result as a starting point for direct optimization
    # This helps to fix any minor issues the MLP might have left
    
    def objective_direct(centers_flat, tau):
        centers = centers_flat.reshape(N, 3)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_constraints = dists[iu] / 2.0
        
        walls = 0.5 - np.abs(centers)
        wall_constraints = walls.ravel()
        
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        sm = softmin(all_constraints, tau)
        return -sm

    taus_direct = [0.001, 0.0005, 0.0002]
    c_opt = centers.ravel()
    for tau in taus_direct:
        res = minimize(lambda c: objective_direct(c, tau), c_opt, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100})
        c_opt = res.x
        
    centers = c_opt.reshape(N, 3)
    
    # Final safety clip
    centers = np.clip(centers, -0.49999, 0.49999)
    
    return centers

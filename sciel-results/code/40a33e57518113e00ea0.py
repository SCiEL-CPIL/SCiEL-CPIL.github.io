import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Define a base grid
    # 4 x 5 x 5 grid
    # We want to start with a valid configuration.
    # Let's start with r=0.1, so centers are in [-0.4, 0.4]
    x_coords = np.linspace(-0.4, 0.4, 4)
    y_coords = np.linspace(-0.4, 0.4, 5)
    z_coords = np.linspace(-0.4, 0.4, 5)
    
    grid = np.array(np.meshgrid(x_coords, y_coords, z_coords)).T.reshape(-1, 3)
    # Ensure shape is (100, 3)
    grid = grid[:N] # Just in case, though 4*5*5=100
    
    # 2. Define the Neural Network (Residual)
    # Input z: (D,) -> Dense -> Dense -> Displacement (300,)
    D = 50
    H = 100
    rng = np.random.default_rng(42)
    
    W1 = rng.normal(0, 1.0 / np.sqrt(D), size=(H, D))
    b1 = rng.normal(0, 0.1, size=(H,))
    W2 = rng.normal(0, 1.0 / np.sqrt(H), size=(300, H))
    b2 = rng.normal(0, 0.1, size=(300,))
    
    # Scale factor for displacement to ensure it starts small
    # We want the network to learn adjustments, not replace the grid
    scale = 0.1 
    
    def centers_from_z(z):
        h = np.tanh(W1 @ z + b1)
        disp = W2 @ h + b2
        disp = np.tanh(disp) * scale # Keep displacement bounded
        centers = grid + disp.reshape(N, 3)
        return centers

    # 3. Objective Function
    def softmin(x, tau=0.01):
        m = np.min(x)
        return m - tau * np.log(np.sum(np.exp(-(x - m) / tau)))

    def objective(z):
        centers = centers_from_z(z)
        # Clip to ensure validity during optimization (though tanh helps)
        # The objective should penalize going out of bounds heavily
        # We calculate the actual constraints
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_constraints = dists[iu] / 2.0
        
        # Wall constraints
        walls = 0.5 - np.abs(centers)
        wall_constraints = walls.ravel()
        
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        # If any constraint is negative, the sphere is invalid.
        # We want to maximize the minimum constraint.
        # softmin approximates the min.
        # If a constraint is negative, softmin will be negative.
        # We want to maximize this.
        
        sm = softmin(all_constraints, tau=0.01)
        return -sm

    # 4. Optimization
    # Start z at 0, so disp is small (b2 is small random, tanh(0)=0 for linear part? 
    # No, tanh(W1@0+b1) = tanh(b1). 
    # To start exactly at grid, we can initialize b1, b2 to 0?
    # If b1=0, b2=0, then h=tanh(0)=0, disp=0.
    # This is a perfect start.
    
    # Let's re-initialize weights to start at 0 displacement
    # This is a "new design" feature: Zero-init for residual branch.
    # But if W is random, gradient might be 0?
    # If h=0, grad of tanh is 1. So gradients flow.
    
    # Let's use the initialized W, but set b1, b2 to 0 for the first trial?
    # Or just use a very small scale.
    
    best_z = np.zeros(D)
    best_val = -np.inf
    
    for trial in range(3):
        if trial == 0:
            z0 = np.zeros(D)
        else:
            z0 = rng.normal(0, 0.1, size=D)
            
        res = minimize(objective, z0, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        val = -res.fun
        if val > best_val:
            best_val = val
            best_z = res.x

    centers = centers_from_z(best_z)
    
    # 5. Local Polish
    # Optimize the 300 coordinates directly to fine-tune
    def objective_direct(centers_flat):
        centers = centers_flat.reshape(N, 3)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_constraints = dists[iu] / 2.0
        walls = 0.5 - np.abs(centers)
        wall_constraints = walls.ravel()
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        sm = softmin(all_constraints, tau=0.005)
        return -sm

    res2 = minimize(objective_direct, centers.ravel(), method='L-BFGS-B', 
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
    centers = res2.x.reshape(N, 3)
    
    # Final safety clip
    centers = np.clip(centers, -0.49999, 0.49999)
    
    return centers

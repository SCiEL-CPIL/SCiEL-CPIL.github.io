import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # Small MLP parameterization: latent code -> 100x3 centers
    # Architecture: 16-dim latent -> 32 hidden (ReLU) -> 32 hidden (ReLU) -> 100*3 output
    # This gives a compact 16-dim parameter space for L-BFGS-B to optimize.
    
    np.random.seed(42)
    
    # Initialize MLP weights
    # Layer 1: 16 -> 32
    W1 = np.random.randn(16, 32) * np.sqrt(2.0 / 16)
    b1 = np.zeros(32)
    # Layer 2: 32 -> 32
    W2 = np.random.randn(32, 32) * np.sqrt(2.0 / 32)
    b2 = np.zeros(32)
    # Layer 3: 32 -> 300 (100 centers * 3 coords)
    W3 = np.random.randn(32, 300) * np.sqrt(2.0 / 32)
    b3 = np.zeros(300)
    
    # Initial latent code
    z0 = np.zeros(16)
    
    # Precompute fixed input features: use a simple basis
    # We'll feed the latent code z through the MLP
    # The MLP output is shaped to (100, 3) and then scaled/shifted to cube
    
    def mlp_forward(z):
        h1 = np.maximum(0, z @ W1 + b1)
        h2 = np.maximum(0, h1 @ W2 + b2)
        out = h2 @ W3 + b3
        return out.reshape(N, 3)
    
    def softmin(x, tau):
        m = np.min(x)
        return m - tau * np.log(np.sum(np.exp(-(x - m) / tau)))
    
    def objective(z, tau):
        centers = mlp_forward(z)
        # Clip to ensure we're in a reasonable range for optimization
        # The MLP output is in some range; we'll use tanh to bound it
        # Actually, let's use a sigmoid/tanh mapping in the output layer
        # For now, let's scale the output to [-0.45, 0.45]
        centers = 0.45 * np.tanh(centers)
        
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
    
    # Coarse-to-fine annealing
    taus = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    z_opt = z0.copy()
    for tau in taus:
        res = minimize(lambda z: objective(z, tau), z_opt, method='L-BFGS-B',
                       options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 100})
        z_opt = res.x
    
    centers = 0.45 * np.tanh(mlp_forward(z_opt))
    
    # Stage 2: Direct coordinate refinement
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
    
    taus_direct = [0.001, 0.0005, 0.0002, 0.0001, 0.00005]
    c_opt = centers.ravel()
    for tau in taus_direct:
        res = minimize(lambda c: objective_direct(c, tau), c_opt, method='L-BFGS-B',
                       options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100})
        c_opt = res.x
    
    centers = c_opt.reshape(N, 3)
    
    # Final safety clip
    centers = np.clip(centers, -0.49999, 0.49999)
    
    return centers

import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Initialization: 5x5x4 grid
    x_coords = np.linspace(-0.4, 0.4, 5)
    y_coords = np.linspace(-0.4, 0.4, 5)
    z_coords = np.linspace(-0.4, 0.4, 4)
    
    grid = np.array(np.meshgrid(x_coords, y_coords, z_coords)).T.reshape(-1, 3)
    grid = grid[:N]
    
    # 2. Define a small MLP parameterization
    # Input: 3 (grid coords)
    # Hidden: 16
    # Output: 3 (displacement)
    
    hidden_dim = 16
    
    # Parameters: W1 (3x16), b1 (16), W2 (16x3), b2 (3)
    # Total params: 3*16 + 16 + 16*3 + 3 = 48 + 16 + 48 + 3 = 115
    
    np.random.seed(42)
    W1 = np.random.normal(0, 0.1, (3, hidden_dim))
    b1 = np.zeros(hidden_dim)
    W2 = np.zeros((hidden_dim, 3)) # Start with zero output weights to ensure initial displacement is 0
    b2 = np.zeros(3)
    
    params_init = np.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()])
    
    def unpack_params(p):
        W1 = p[:3*hidden_dim].reshape(3, hidden_dim)
        b1 = p[3*hidden_dim:3*hidden_dim+hidden_dim]
        W2 = p[3*hidden_dim+hidden_dim:3*hidden_dim+hidden_dim+hidden_dim*3].reshape(hidden_dim, 3)
        b2 = p[3*hidden_dim+hidden_dim+hidden_dim*3:]
        return W1, b1, W2, b2

    def mlp_forward(x, W1, b1, W2, b2):
        h = np.tanh(x @ W1 + b1)
        out = h @ W2 + b2
        return out

    def centers_from_params(p):
        W1, b1, W2, b2 = unpack_params(p)
        disp = mlp_forward(grid, W1, b1, W2, b2)
        return grid + disp

    def softmin(x, tau):
        m = np.min(x)
        return m - tau * np.log(np.sum(np.exp(-(x - m) / tau)))

    def objective_mlp(p, tau):
        centers = centers_from_params(p)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_constraints = dists[iu] / 2.0
        
        walls = 0.5 - np.abs(centers)
        wall_constraints = walls.ravel()
        
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        sm = softmin(all_constraints, tau)
        return -sm

    # 3. Optimization Stage 1: MLP Weights
    # Coarse to fine tau
    taus_mlp = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    p_opt = params_init
    for tau in taus_mlp:
        res = minimize(lambda p: objective_mlp(p, tau), p_opt, method='L-BFGS-B', 
                       options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 100})
        p_opt = res.x
        
    centers = centers_from_params(p_opt)
    
    # 4. Optimization Stage 2: Direct Coordinates
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

    taus_direct = [0.005, 0.002, 0.001, 0.0005, 0.0002]
    c_opt = centers.ravel()
    for tau in taus_direct:
        res = minimize(lambda c: objective_direct(c, tau), c_opt, method='L-BFGS-B', 
                       options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100})
        c_opt = res.x
        
    centers = c_opt.reshape(N, 3)
    
    # Final safety clip
    centers = np.clip(centers, -0.49999, 0.49999)
    
    return centers

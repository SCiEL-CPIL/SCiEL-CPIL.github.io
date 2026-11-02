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
    
    # 2. Random Fourier Feature Parameterization
    # Use random frequencies to allow for a more flexible, non-structured displacement field
    n_features = 64
    # Scale of frequencies: We want features that vary on the scale of the cube (size 1)
    # sigma controls the smoothness. Smaller sigma = smoother.
    sigma = 2.0 
    np.random.seed(123)
    # Sample frequencies from Gaussian
    freqs = np.random.randn(n_features, 3) * sigma
    # Project grid points onto these frequencies
    proj = grid @ freqs.T
    phi = np.hstack([np.cos(proj), np.sin(proj)])
    # Add bias term
    phi = np.hstack([np.ones((N, 1)), phi])
    
    n_params = phi.shape[1]
    W0 = np.zeros((n_params, 3))
    
    def centers_from_W(W_flat):
        W = W_flat.reshape(n_params, 3)
        disp = phi @ W
        centers = grid + disp
        return centers

    def softmin(x, tau):
        m = np.min(x)
        return m - tau * np.log(np.sum(np.exp(-(x - m) / tau)))

    def objective_fourier(W_flat, tau):
        centers = centers_from_W(W_flat)
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

    # 3. Optimization Stage 1: Fourier Weights
    # Coarse to fine tau
    taus_fourier = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    W_opt = W0.ravel()
    for tau in taus_fourier:
        res = minimize(lambda w: objective_fourier(w, tau), W_opt, method='L-BFGS-B', 
                       options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 100})
        W_opt = res.x
        
    centers = centers_from_W(W_opt)
    
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

    taus_direct = [0.002, 0.001, 0.0005, 0.0002, 0.0001]
    c_opt = centers.ravel()
    for tau in taus_direct:
        res = minimize(lambda c: objective_direct(c, tau), c_opt, method='L-BFGS-B', 
                       options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100})
        c_opt = res.x
        
    centers = c_opt.reshape(N, 3)
    
    # Final safety clip
    centers = np.clip(centers, -0.49999, 0.49999)
    
    return centers

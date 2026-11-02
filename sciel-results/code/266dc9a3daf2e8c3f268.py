import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Define a base grid for initialization
    # A 5x4x5 grid provides a good starting point with spacing 0.225
    x_coords = np.linspace(-0.45, 0.45, 5)
    y_coords = np.linspace(-0.45, 0.45, 4)
    z_coords = np.linspace(-0.45, 0.45, 5)
    
    grid = np.array(np.meshgrid(x_coords, y_coords, z_coords)).T.reshape(-1, 3)
    grid = grid[:N]
    
    # 2. Define Fourier Feature Parameterization
    # Increased max_k to 6 for more flexibility in local adjustments
    max_k = 6
    freqs = []
    for kx in range(0, max_k + 1):
        for ky in range(0, max_k + 1):
            for kz in range(0, max_k + 1):
                if kx == 0 and ky == 0 and kz == 0:
                    continue
                freqs.append([np.pi * kx, np.pi * ky, np.pi * kz])
    
    omega = np.array(freqs)
    
    # Precompute Fourier features for the grid points
    proj = grid @ omega.T
    phi = np.hstack([np.cos(proj), np.sin(proj)])
    phi = np.hstack([np.ones((N, 1)), phi])
    
    n_features = phi.shape[1]
    
    # Weights W: (n_features, 3)
    W0 = np.zeros((n_features, 3))
    
    def centers_from_W(W_flat):
        W = W_flat.reshape(n_features, 3)
        disp = phi @ W
        centers = grid + disp
        return centers

    def stable_softmin(x, tau):
        m = np.min(x)
        return m - tau * np.log(np.sum(np.exp(-(x - m) / tau)))

    def objective(W_flat, tau):
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
        
        sm = stable_softmin(all_constraints, tau)
        return -sm

    # 3. Optimization
    # Extended tau schedule for better convergence
    taus = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    W_opt = W0.ravel()
    for tau in taus:
        res = minimize(lambda w: objective(w, tau), W_opt, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 100})
        W_opt = res.x
    
    centers = centers_from_W(W_opt)
    
    # Stage 2: Direct coordinate optimization
    def objective_direct(centers_flat, tau):
        centers = centers_flat.reshape(N, 3)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_constraints = dists[iu] / 2.0
        walls = 0.5 - np.abs(centers)
        wall_constraints = walls.ravel()
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        sm = stable_softmin(all_constraints, tau)
        return -sm

    # Further refine with direct coordinates
    taus_direct = [0.002, 0.001, 0.0005, 0.0002, 0.0001]
    c_opt = centers.ravel()
    for tau in taus_direct:
        res = minimize(lambda c: objective_direct(c, tau), c_opt, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100})
        c_opt = res.x
    
    centers = c_opt.reshape(N, 3)
    
    # Final safety clip
    centers = np.clip(centers, -0.499999, 0.499999)
    
    return centers

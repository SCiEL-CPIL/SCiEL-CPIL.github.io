import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Define a base grid for initialization
    # A 4x5x5 grid provides a good starting point
    x_coords = np.linspace(-0.4, 0.4, 4)
    y_coords = np.linspace(-0.4, 0.4, 5)
    z_coords = np.linspace(-0.4, 0.4, 5)
    
    grid = np.array(np.meshgrid(x_coords, y_coords, z_coords)).T.reshape(-1, 3)
    grid = grid[:N]
    
    # 2. Define Fourier Feature Parameterization
    # We use a deterministic set of frequencies to ensure good coverage.
    # Frequencies are chosen to be integer multiples of pi to match the domain [-0.5, 0.5] roughly.
    # The domain is side 1, so fundamental frequency is pi.
    
    # Let's select a set of frequencies up to max_k
    max_k = 4
    # Generate frequency vectors
    freqs = []
    for kx in range(0, max_k + 1):
        for ky in range(0, max_k + 1):
            for kz in range(0, max_k + 1):
                if kx == 0 and ky == 0 and kz == 0:
                    continue # Skip DC component, handled by bias
                freqs.append([np.pi * kx, np.pi * ky, np.pi * kz])
    
    # We might have too many frequencies if max_k is large. 
    # For max_k=4, there are 5^3 - 1 = 124 frequencies. 
    # This results in 2 * 124 + 1 = 249 features.
    # Parameters: 249 * 3 = 747 parameters. This is manageable for L-BFGS-B.
    
    omega = np.array(freqs)
    
    # Precompute Fourier features for the grid points
    # phi: (N, 2*K + 1)
    proj = grid @ omega.T # (N, K)
    phi = np.hstack([np.cos(proj), np.sin(proj)])
    # Add bias term (1)
    phi = np.hstack([np.ones((N, 1)), phi])
    
    n_features = phi.shape[1]
    
    # Weights W: (n_features, 3)
    # Initialize to 0 so we start at the grid
    W0 = np.zeros((n_features, 3))
    
    def centers_from_W(W_flat):
        W = W_flat.reshape(n_features, 3)
        disp = phi @ W # (N, 3)
        centers = grid + disp
        return centers

    def softmin(x, tau):
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
        
        # Maximize the minimum constraint
        sm = softmin(all_constraints, tau)
        return -sm

    # 3. Optimization
    # Multi-stage optimization with decreasing tau
    # Stage 1: Coarse optimization with larger tau to find a good basin
    taus = [0.1, 0.05, 0.01, 0.005, 0.001]
    
    W_opt = W0.ravel()
    for tau in taus:
        res = minimize(lambda w: objective(w, tau), W_opt, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50})
        W_opt = res.x
    
    centers = centers_from_W(W_opt)
    
    # Stage 2: Direct coordinate optimization (Local Polish)
    # This allows escaping the linear subspace of the Fourier features.
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

    # Further refine with direct coordinates
    taus_direct = [0.005, 0.001, 0.0005]
    c_opt = centers.ravel()
    for tau in taus_direct:
        res = minimize(lambda c: objective_direct(c, tau), c_opt, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
        c_opt = res.x
    
    centers = c_opt.reshape(N, 3)
    
    # Final safety clip
    centers = np.clip(centers, -0.49999, 0.49999)
    
    return centers

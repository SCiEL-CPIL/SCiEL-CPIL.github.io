import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Define a base grid for initialization
    # A 5x5x4 grid provides 100 points exactly.
    # Distribute points evenly in [-0.4, 0.4] to leave room for radius.
    x_coords = np.linspace(-0.4, 0.4, 5)
    y_coords = np.linspace(-0.4, 0.4, 5)
    z_coords = np.linspace(-0.4, 0.4, 4)
    
    grid = np.array(np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')).T.reshape(-1, 3)
    # Shuffle slightly to break perfect grid symmetry which might trap in local minima?
    # Actually, keeping symmetry might help. Let's keep it ordered.
    # The grid is 100 points.
    
    # 2. Define Fourier Feature Parameterization
    # We use a deterministic set of frequencies to ensure good coverage.
    # Higher max_k allows for more complex displacements.
    max_k = 6
    freqs = []
    for kx in range(0, max_k + 1):
        for ky in range(0, max_k + 1):
            for kz in range(0, max_k + 1):
                if kx == 0 and ky == 0 and kz == 0:
                    continue
                freqs.append([np.pi * kx, np.pi * ky, np.pi * kz])
    
    omega = np.array(freqs)
    n_freqs = len(omega)
    
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
        # Numerical stability: shift by min
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
    # Start with larger tau to smooth the landscape, then refine.
    taus = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    W_opt = W0.ravel()
    for tau in taus:
        res = minimize(lambda w: objective(w, tau), W_opt, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
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
    # Use very small taus for final polish
    taus_direct = [0.001, 0.0005, 0.0002, 0.0001]
    c_opt = centers.ravel()
    for tau in taus_direct:
        res = minimize(lambda c: objective_direct(c, tau), c_opt, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
        c_opt = res.x
    
    centers = c_opt.reshape(N, 3)
    
    # Final safety clip
    centers = np.clip(centers, -0.49999, 0.49999)
    
    return centers

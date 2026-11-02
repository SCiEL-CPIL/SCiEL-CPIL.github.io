import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Initialization: 5x5x4 grid
    # Slightly expanded grid to start near the optimal spacing
    x_coords = np.linspace(-0.45, 0.45, 5)
    y_coords = np.linspace(-0.45, 0.45, 5)
    z_coords = np.linspace(-0.45, 0.45, 4)
    
    grid = np.array(np.meshgrid(x_coords, y_coords, z_coords)).T.reshape(-1, 3)
    grid = grid[:N]
    
    # 2. Fourier Feature Parameterization
    # Use frequencies up to k=4 for increased expressiveness
    max_k = 4
    freqs = []
    for kx in range(0, max_k + 1):
        for ky in range(0, max_k + 1):
            for kz in range(0, max_k + 1):
                if kx == 0 and ky == 0 and kz == 0:
                    continue
                freqs.append([np.pi * kx, np.pi * ky, np.pi * kz])
                
    omega = np.array(freqs)
    
    # Precompute features
    proj = grid @ omega.T
    phi = np.hstack([np.cos(proj), np.sin(proj)])
    phi = np.hstack([np.ones((N, 1)), phi])
    
    n_features = phi.shape[1]
    
    # Initialize weights with small random values to break symmetry
    np.random.seed(123)
    W0 = np.random.normal(0, 0.01, (n_features, 3))
    
    def centers_from_W(W_flat):
        W = W_flat.reshape(n_features, 3)
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

    # 3. Optimization: Single stage in latent space with aggressive annealing
    # This keeps the optimization in the lower-dimensional manifold,
    # avoiding the flat directions of direct coordinate optimization.
    W_opt = W0.ravel()
    
    # Aggressive annealing schedule
    taus = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for tau in taus:
        res = minimize(lambda w: objective_fourier(w, tau), W_opt, method='L-BFGS-B', 
                       options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50})
        W_opt = res.x
        
    centers = centers_from_W(W_opt)
    
    # Final safety clip
    centers = np.clip(centers, -0.499999, 0.499999)
    
    return centers

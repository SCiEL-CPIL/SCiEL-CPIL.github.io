import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Define inputs for the MLP
    # We use the index of the sphere as the input feature.
    # Normalized to [-1, 1]
    idx = np.linspace(-1, 1, N).reshape(-1, 1)
    
    # 2. Define MLP architecture
    # Input dim: 1
    # Hidden layer 1: 64 units
    # Hidden layer 2: 64 units
    # Output dim: 3
    
    # Weights and biases
    # W1: (1, 64), b1: (64,)
    # W2: (64, 64), b2: (64,)
    # W3: (64, 3), b3: (3,)
    
    # Initialize weights using He initialization for ReLU
    np.random.seed(42) # Deterministic
    
    W1 = np.random.randn(1, 64) * np.sqrt(2.0 / 1)
    b1 = np.zeros(64)
    W2 = np.random.randn(64, 64) * np.sqrt(2.0 / 64)
    b2 = np.zeros(64)
    W3 = np.random.randn(64, 3) * np.sqrt(2.0 / 64)
    b3 = np.zeros(3)
    
    # Flatten parameters
    def params_to_weights(params):
        W1_f = params[0:64].reshape(1, 64)
        b1_f = params[64:128]
        W2_f = params[128:128+4096].reshape(64, 64)
        b2_f = params[128+4096:128+4096+64]
        W3_f = params[128+4096+64:128+4096+64+192].reshape(64, 3)
        b3_f = params[128+4096+64+192:]
        return W1_f, b1_f, W2_f, b2_f, W3_f, b3_f

    def forward(params):
        W1_f, b1_f, W2_f, b2_f, W3_f, b3_f = params_to_weights(params)
        h1 = np.maximum(0, idx @ W1_f + b1_f)
        h2 = np.maximum(0, h1 @ W2_f + b2_f)
        centers = h2 @ W3_f + b3_f
        # Clip to ensure they stay roughly within bounds during optimization
        # The objective function will penalize going outside, but clipping helps stability
        return np.clip(centers, -0.5, 0.5)

    # Initial parameters
    p0 = np.concatenate([W1.ravel(), b1, W2.ravel(), b2, W3.ravel(), b3])
    
    def softmin(x, tau):
        m = np.min(x)
        return m - tau * np.log(np.sum(np.exp(-(x - m) / tau)))

    def objective(params, tau):
        centers = forward(params)
        
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
    taus = [0.1, 0.05, 0.01, 0.005, 0.001]
    
    p_opt = p0
    for tau in taus:
        res = minimize(lambda p: objective(p, tau), p_opt, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-6, 'maxls': 50})
        p_opt = res.x
    
    centers = forward(p_opt)
    
    # Stage 2: Direct coordinate optimization (Local Polish)
    # This allows escaping the MLP subspace and fine-tuning positions.
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
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
        c_opt = res.x
    
    centers = c_opt.reshape(N, 3)
    
    # Final safety clip
    centers = np.clip(centers, -0.49999, 0.49999)
    
    return centers

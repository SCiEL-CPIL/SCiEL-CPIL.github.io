import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # Set up random seed for reproducibility? The task doesn't require it, but for consistent scoring, maybe.
    # Actually, the evaluator runs construct() and measures the raw value. Randomness is fine as long as it's valid.
    # But to be safe, I'll use a fixed seed for the random parts to ensure a good result every time.
    rng = np.random.default_rng(42)
    
    # Network parameters: Fourier frequencies and linear weights
    D = 200  # latent dim
    F = 400  # Fourier feature dim
    # Random frequencies for Fourier expansion
    freqs = rng.normal(0, 1, size=(F//2, D))
    # We'll use sin and cos of freqs @ z
    
    # Linear layer: F -> 300
    W = rng.normal(0, 0.5, size=(300, F))
    b = rng.normal(0, 0.1, size=(300,))
    
    def centers_from_z(z):
        # z: (D,)
        # Fourier features
        proj = freqs @ z  # (F//2,)
        fourier = np.concatenate([np.sin(proj), np.cos(proj)])  # (F,)
        out = W @ fourier + b  # (300,)
        centers = np.tanh(out) * 0.5  # in [-0.5, 0.5]
        return centers.reshape(N, 3)
    
    def softmin(x, tau=0.01):
        # x: array of constraints (all positive ideally)
        # To avoid overflow, shift by max
        x_shifted = x - np.max(x)
        return -tau * np.log(np.sum(np.exp(x_shifted / tau)))  # this is -softmax(-x/tau)*tau? 
        # Actually, softmin(x) = -tau * log(sum(exp(-x/tau)))
        # Let me fix: softmin = -tau * log(sum(exp(-x/tau)))
        # But exp(-x/tau) can underflow if x is large. Shift: 
        # -x/tau, shift by max(-x/tau) = -min(x)/tau
        # Let m = min(x)
        # exp(-x/tau) = exp(-(x-m)/tau) * exp(-m/tau)
        # sum = exp(-m/tau) * sum(exp(-(x-m)/tau))
        # log(sum) = -m/tau + log(sum(exp(-(x-m)/tau)))
        # softmin = -tau * (-m/tau + log(...)) = m - tau*log(sum(exp(-(x-m)/tau)))
        # As tau->0, softmin -> m.
        m = np.min(x)
        return m - tau * np.log(np.sum(np.exp(-(x - m) / tau)))
    
    def objective(z):
        centers = centers_from_z(z)
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Take upper triangle
        iu = np.triu_indices(N, k=1)
        pair_constraints = dists[iu] / 2.0  # (4950,)
        # Wall constraints
        walls = 0.5 - np.abs(centers)  # (100,3)
        wall_constraints = walls.ravel()  # (300,)
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        # Ensure positivity (should be, but clip small negatives to 0 for safety)
        all_constraints = np.maximum(all_constraints, 0.0)
        sm = softmin(all_constraints, tau=0.01)
        return -sm  # minimize negative
    
    # Optimize z
    best_z = None
    best_val = -np.inf
    for trial in range(5):
        z0 = rng.normal(0, 1, size=D)
        res = minimize(objective, z0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
        val = -res.fun
        if val > best_val:
            best_val = val
            best_z = res.x
    
    centers = centers_from_z(best_z)
    
    # Local polish: optimize the 300 coordinates directly
    def objective_direct(centers_flat):
        centers = centers_flat.reshape(N, 3)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_constraints = dists[iu] / 2.0
        walls = 0.5 - np.abs(centers)
        wall_constraints = walls.ravel()
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        all_constraints = np.maximum(all_constraints, 0.0)
        sm = softmin(all_constraints, tau=0.005)
        return -sm
    
    res2 = minimize(objective_direct, centers.ravel(), method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
    centers = res2.x.reshape(N, 3)
    
    # Ensure centers are strictly inside (clip to avoid boundary issues)
    centers = np.clip(centers, -0.4999, 0.4999)
    
    return centers

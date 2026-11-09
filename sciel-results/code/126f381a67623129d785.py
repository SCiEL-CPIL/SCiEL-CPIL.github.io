"""Rewritten program using lattice-based parameterisation plus spherical refinement."""
import numpy as np
from scipy.optimize import minimize
import math

N = 150

def construct():
    # ------------------------------------------------------------------
    # Step 1: Build a compact set of 150 lattice points inside unit ball
    # We use an FCC-like lattice with a few unit cells, then keep points
    # inside the unit ball and, if needed, trim/pad to exactly N=150.
    # ------------------------------------------------------------------
    # Generate a simple cubic grid of candidate lattice points in [-1.5, 1.5]^3
    # with a spacing that yields roughly 150 points inside the unit sphere.
    spacing = 0.25  # initial guess; we'll adjust by trimming/padding
    coords = np.arange(-1.5, 1.5 + 1e-9, spacing)
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing='ij')
    flat = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    # Keep points strictly inside unit ball
    mask = np.linalg.norm(flat, axis=1) < 1.0
    points_in = flat[mask]
    
    # If we have too many, randomly subsample; if too few, add random points on sphere
    if len(points_in) >= N:
        idx = np.random.default_rng(0).choice(len(points_in), size=N, replace=False)
        lattice_points = points_in[idx]
    else:
        # Pad with random points on the unit sphere
        extra_needed = N - len(points_in)
        rng = np.random.default_rng(1)
        extra = rng.normal(size=(extra_needed, 3))
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        lattice_points = np.vstack([points_in, extra])
    
    # Normalize all points to unit sphere (they should already be on or inside)
    norms0 = np.linalg.norm(lattice_points, axis=1, keepdims=True)
    norms0 = np.where(norms0 == 0, 1.0, norms0)
    x0 = lattice_points / norms0
    
    # ------------------------------------------------------------------
    # Step 2: Define the logarithmic energy objective and its gradient
    # Energy E = sum_{i<j} log(1 / ||x_i - x_j||) = - sum_{i<j} log(||x_i - x_j||)
    # We add a small penalty to keep points on the sphere:
    #   P = lam * sum_i (||x_i||^2 - 1)^2
    # ------------------------------------------------------------------
    lam = 1e4  # sphere constraint penalty

    def energy_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        # Pairwise differences
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist2 = np.einsum('ijk,ijk->ij', diff, diff)
        # Avoid log(0) on diagonal
        np.fill_diagonal(dist2, 1.0)
        
        # Energy term
        # E_log = -0.5 * sum_{i != j} log(sqrt(dist2_ij)) = -0.25 * sum_{i != j} log(dist2_ij)
        log_dist2 = np.log(dist2)
        E_log = -0.25 * (np.sum(log_dist2) - np.log(np.diag(dist2)))
        # Since we set diagonal to 1.0, log(diag)=0, so:
        E_log = -0.25 * np.sum(log_dist2)
        
        # Penalty term
        r2 = np.einsum('ij,ij->i', x, x)
        P = lam * np.sum((r2 - 1.0) ** 2)
        total = E_log + P
        
        # Gradient of E_log w.r.t x_k:
        # dE_log/dx_k = -0.5 * sum_{j != k} (x_k - x_j) / ||x_k - x_j||^2
        # Using dist2_safe with diagonal set to 1.0 to avoid div-by-zero
        inv_dist2 = np.zeros_like(dist2)
        np.fill_diagonal(inv_dist2, 0.0)
        np.divide(1.0, dist2, out=inv_dist2, where=(dist2 > 1e-12))
        
        # grad_E = -0.5 * sum_j (x_k - x_j) / dist2_kj
        grad_E = -0.5 * np.einsum('ij,ijk->ik', inv_dist2, diff)
        
        # Gradient of penalty: dP/dx_k = 4*lam*(r2_k - 1)*x_k
        grad_P = 4.0 * lam * (r2 - 1.0)[:, np.newaxis] * x
        
        grad_total = grad_E + grad_P
        return total, grad_total.ravel()
    
    # ------------------------------------------------------------------
    # Step 3: Optimize with L-BFGS-B
    # ------------------------------------------------------------------
    res = minimize(
        energy_and_grad,
        x0.ravel(),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
    )
    x_opt = res.x.reshape(N, 3)
    
    # ------------------------------------------------------------------
    # Step 4: Final projection onto unit sphere and validation
    # ------------------------------------------------------------------
    norms_final = np.linalg.norm(x_opt, axis=1, keepdims=True)
    norms_final = np.where(norms_final < 1e-12, 1.0, norms_final)
    x_final = x_opt / norms_final
    
    # Ensure no two points are closer than 1e-9
    diff_final = x_final[:, np.newaxis, :] - x_final[np.newaxis, :, :]
    dist_final = np.linalg.norm(diff_final, axis=2)
    np.fill_diagonal(dist_final, np.inf)
    min_dist = np.min(dist_final)
    if min_dist < 1e-9:
        # Perturb the closest pair slightly
        i, j = np.unravel_index(np.argmin(dist_final), dist_final.shape)
        x_final[j] += 1e-8 * np.random.randn(3)
        x_final[j] /= np.linalg.norm(x_final[j])
    
    # Ensure all values are finite
    if not np.all(np.isfinite(x_final)):
        # Fallback: Fibonacci sphere
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        x_final = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return x_final

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    """
    pts = pts.reshape(-1, 3)
    n = pts.shape[0]
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    dist = dist[mask]
    # Energy: sum log(1/d) = -sum log(d)
    energy = -np.sum(np.log(dist))
    
    # Gradient: d/dx_i [log(1/|x_i - x_j|)] = (x_i - x_j) / |x_i - x_j|^2
    # For each point i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # But we need to be careful with the factor of 2 from double counting
    # Actually, let's compute it properly:
    # E = sum_{i<j} -log(|x_i - x_j|)
    # dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2  (for j < i, contributes (x_i - x_j)/d^2)
    #          + sum_{j > i} (-1) * (x_i - x_j) / |x_i - x_j|^2  (for j > i, derivative of -log(d) w.r.t. x_i is -(x_i-x_j)/d^2 * (-1)... wait)
    # Let me redo: E = sum_{i<j} -log(|x_i - x_j|)
    # dE/dx_i = sum_{j<i} -1 * (x_i - x_j) / |x_i - x_j|^2  (from term (i,j) where i<j... no)
    # For pair (i,j) with i<j: term is -log(|x_i - x_j|)
    # d/dx_i of -log(|x_i - x_j|) = -(x_i - x_j) / |x_i - x_j|^2
    # d/dx_j of -log(|x_i - x_j|) = -(-1) * (x_i - x_j) / |x_i - x_j|^2 = (x_i - x_j) / |x_i - x_j|^2
    # So grad_i = sum_{j>i} -(x_i - x_j)/|x_i-x_j|^2 + sum_{j<i} (x_j - x_i)/|x_j-x_i|^2
    #            = sum_{j>i} (x_j - x_i)/|x_i-x_j|^2 + sum_{j<i} (x_j - x_i)/|x_i-x_j|^2
    #            = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    
    # Compute gradient efficiently
    # inv_dist_sq = 1 / dist^2 for all pairs
    diff_sq = np.sum(diff ** 2, axis=-1)
    inv_dist_sq = np.zeros_like(diff_sq)
    np.fill_diagonal(inv_dist_sq, 0)
    np.divide(1.0, diff_sq, out=inv_dist_sq, where=diff_sq > 1e-15)
    
    # grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    # = sum_j (x_j - x_i) * inv_dist_sq[i,j]
    # = (sum_j x_j * inv_dist_sq[i,j]) - x_i * sum_j inv_dist_sq[i,j]
    row_sums = np.sum(inv_dist_sq, axis=1)
    grad = inv_dist_sq @ pts - pts * row_sums[:, np.newaxis]
    
    # Note: this gradient is for the full space. We need to project onto the sphere tangent space
    # in the optimizer, but L-BFGS-B doesn't handle constraints well. 
    # Instead, we'll use a penalty or just optimize in Cartesian and renormalize.
    # Actually, for simplicity, let's just return the Cartesian gradient and let the 
    # optimizer handle it. We'll renormalize after each step.
    
    return energy, grad


def _project_and_normalize(pts):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start from Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    x0 = x0.flatten()
    
    best_pts = x0.copy()
    best_energy = _energy_and_grad(best_pts.reshape(-1, 3))[0]
    
    def objective(x):
        pts = _project_and_normalize(x.reshape(-1, 3))
        return _energy_and_grad(pts)[0]
    
    def grad(x):
        pts = _project_and_normalize(x.reshape(-1, 3))
        e, g = _energy_and_grad(pts)
        # Project gradient onto tangent space
        g_tangent = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        return g_tangent.flatten()
    
    # Try L-BFGS-B
    try:
        res = minimize(
            objective, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000}
        )
        if time.time() < deadline:
            cand = _project_and_normalize(res.x.reshape(-1, 3)).flatten()
            cand_e = objective(cand)
            if cand_e < best_energy:
                best_pts = cand
                best_energy = cand_e
    except Exception:
        pass
    
    # Check time and try a second round with tighter tolerance if time permits
    if time.time() < deadline - 30:
        try:
            x0 = best_pts.copy()
            res = minimize(
                objective, x0, jac=grad, method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 20000}
            )
            if time.time() < deadline:
                cand = _project_and_normalize(res.x.reshape(-1, 3)).flatten()
                cand_e = objective(cand)
                if cand_e < best_energy:
                    best_pts = cand
                    best_energy = cand_e
        except Exception:
            pass
    
    # Final check and return
    result = _project_and_normalize(best_pts.reshape(-1, 3))
    
    # Validate: check for near-duplicate points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(result[i] - result[j])
            if d < 1e-9:
                # Add small perturbation
                result[i] += np.random.normal(0, 1e-8, 3)
                result[i] /= np.linalg.norm(result[i])
    
    # Ensure non-finite values are handled
    if not np.all(np.isfinite(result)):
        # Fall back to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        result = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return result
# EVOLVE-BLOCK-END

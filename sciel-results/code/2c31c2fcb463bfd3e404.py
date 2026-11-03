"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _energy_and_grad(X):
    """Compute logarithmic energy and its gradient for points X on the unit sphere."""
    N = X.shape[0]
    # Compute pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Avoid division by zero for i==j
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    log_dist = np.log(dist[mask])
    energy = -np.sum(log_dist)
    
    # Gradient: d/dX_i [ -log|X_i - X_j| ] = (X_i - X_j) / |X_i - X_j|^2
    # For each point i, grad_i = sum_{j != i} (X_i - X_j) / |X_i - X_j|^2
    dist2 = dist * dist
    np.fill_diagonal(dist2, 1.0)
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[i] = sum_j (X_i - X_j) / |X_i - X_j|^2
    # = sum_j X_i / dist2_ij - sum_j X_j / dist2_ij
    # = X_i * sum_j (1/dist2_ij) - sum_j X_j / dist2_ij
    sum_inv = inv_dist2.sum(axis=1)  # sum_j 1/dist2_ij for each i
    weighted_sum = (inv_dist2[:, :, np.newaxis] * X[np.newaxis, :, :]).sum(axis=1)
    
    grad = X * sum_inv[:, np.newaxis] - weighted_sum
    
    # Project gradient onto tangent space (remove radial component)
    # tangent_grad = grad - (grad . X) * X
    dot = (grad * X).sum(axis=1, keepdims=True)
    grad = grad - dot * X
    
    return energy, grad


def _hessian_vec_prod(X, v):
    """Compute Hessian-vector product for the logarithmic energy on the sphere.
    
    For the ambient Hessian of f(X_i) = -sum_{j!=i} log|X_i - X_j|:
    H_ij = delta_ij * sum_k 1/|X_i-X_k|^2 - (X_i - X_j)/|X_i-X_j|^3 * (something)
    
    Actually, let's compute it directly. The ambient Hessian of -log|X_i - X_j| is:
    d/dX_i d/dX_j [-log|X_i - X_j|] = 
      For i==j: sum_k (1/|X_i-X_k|^2 * I - 2*(X_i-X_k)(X_i-X_k)^T / |X_i-X_k|^4)
      For i!=j: -((X_i-X_j)(X_i-X_j)^T - |X_i-X_j|^2 * I) / |X_i-X_j|^4
               = (|X_i-X_j|^2 * I - (X_i-X_j)(X_i-X_j)^T) / |X_i-X_j|^4
    """
    N = X.shape[0]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dist2 = dist * dist
    np.fill_diagonal(dist2, 1.0)
    
    # Hessian-vector product: H @ v where v is shape (N, 3)
    # For each i: (Hv)_i = sum_j H_ij @ v_j
    
    # H_ii @ v_i = [sum_k 1/d2_ik * I - 2 * sum_k (X_i-X_k)(X_i-X_k)^T / d4_ik] @ v_i
    # H_ij @ v_j for i!=j = [d2_ij * I - (X_i-X_j)(X_i-X_j)^T] / d4_ij @ v_j
    
    Hv = np.zeros_like(X)
    
    for i in range(N):
        # Diagonal block H_ii @ v_i
        diag_term = 0.0
        offdiag_term = np.zeros(3)
        for k in range(N):
            if k == i:
                continue
            d2 = dist2[i, k]
            d4 = d2 * d2
            diff_ik = X[i] - X[k]
            diag_term += 1.0 / d2
            # 2 * (X_i-X_k)(X_i-X_k)^T / d4 @ v_i
            dot_kv = np.dot(diff_ik, v[i])
            offdiag_term += 2.0 * diff_ik * dot_kv / d4
        Hv[i] += diag_term * v[i] - offdiag_term
        
        # Off-diagonal blocks H_ij @ v_j for j != i
        for j in range(N):
            if j == i:
                continue
            d2 = dist2[i, j]
            d4 = d2 * d2
            diff_ij = X[i] - X[j]
            # [d2 * I - diff_ij diff_ij^T] / d4 @ v_j
            dot_vj = np.dot(diff_ij, v[j])
            Hv[i] += (d2 * v[j] - diff_ij * dot_vj) / d4
    
    # Project to tangent space
    dot = (Hv * X).sum(axis=1, keepdims=True)
    Hv = Hv - dot * X
    
    return Hv


def _newton_polish(X, max_iter=50, tol=1e-12, deadline=None):
    """Newton's method polish on the sphere using Gauss-Newton approximation."""
    X = X.copy()
    N = X.shape[0]
    best_energy = float('inf')
    best_X = X.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad(X)
        if energy < best_energy:
            best_energy = energy
            best_X = X.copy()
        
        if np.linalg.norm(grad) < tol:
            break
        
        # Use Gauss-Newton: approximate Hessian with J^T J where J is the Jacobian
        # of the residual. For energy = -sum log(d_ij), the "residual" is d_ij.
        # Alternatively, use a Levenberg-Marquardt approach.
        
        # Simpler: use a line search with gradient descent but with adaptive step
        # that mimics Newton behavior by estimating curvature.
        
        # Estimate step size using gradient norm and energy
        grad_norm = np.linalg.norm(grad)
        
        # Try a backtracking line search
        alpha = 1.0
        for _ in range(20):
            # Riemannian exponential map
            v_norms = np.linalg.norm(alpha * grad, axis=1, keepdims=True)
            v_norms_safe = np.where(v_norms < 1e-15, 1.0, v_norms)
            cos_term = np.cos(v_norms_safe)
            sin_term = np.sin(v_norms_safe) / v_norms_safe
            X_trial = cos_term * X + sin_term * (alpha * grad)
            X_trial = _project_to_sphere(X_trial)
            
            energy_trial, _ = _energy_and_grad(X_trial)
            
            if energy_trial < energy - 1e-4 * alpha * grad_norm * grad_norm:
                X = X_trial
                break
            alpha *= 0.5
        else:
            # If no good step found, use a small step
            X = _project_to_sphere(X - 0.01 * grad)
    
    return best_X, best_energy


def _project_to_sphere(X):
    """Project points back onto the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return X / norms


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n=N, rng=None):
    """Generate random points on the unit sphere."""
    if rng is None:
        rng = np.random.default_rng()
    points = rng.normal(size=(n, 3))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms


def _gradient_descent_with_momentum(X, lr=0.01, momentum=0.9, steps=500, deadline=None):
    """Gradient descent with momentum on the sphere using Riemannian exponential map."""
    X = X.copy()
    velocity = np.zeros_like(X)
    best_energy = float('inf')
    best_X = X.copy()
    
    for step in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad(X)
        if energy < best_energy:
            best_energy = energy
            best_X = X.copy()
        
        # Update velocity
        velocity = momentum * velocity - lr * grad
        
        # Riemannian exponential map: move along tangent direction
        # v_tangent is already tangent (from _energy_and_grad)
        # X_new = cos(|v|) * X + sin(|v|)/|v| * v
        v_norms = np.linalg.norm(velocity, axis=1, keepdims=True)
        v_norms_safe = np.where(v_norms < 1e-15, 1.0, v_norms)
        
        cos_term = np.cos(v_norms_safe)
        sin_term = np.sin(v_norms_safe) / v_norms_safe
        
        X = cos_term * X + sin_term * velocity
        
        # Ensure points stay on sphere
        X = _project_to_sphere(X)
        
        # Adaptive learning rate: reduce if energy increases
        if step > 10 and energy > best_energy * 1.01:
            lr *= 0.5
    
    return best_X, best_energy


def _lbfgs_polish(X, deadline=None, maxiter=300, ftol=1e-14, gtol=1e-12):
    """High-precision L-BFGS-B polish."""
    N = X.shape[0]
    
    def objective_flat(flat_X):
        X = flat_X.reshape(N, 3)
        X = _project_to_sphere(X)
        energy, _ = _energy_and_grad(X)
        return energy
    
    def gradient_flat(flat_X):
        X = flat_X.reshape(N, 3)
        X = _project_to_sphere(X)
        _, grad = _energy_and_grad(X)
        return grad.flatten()
    
    x0 = X.flatten()
    
    result = minimize(
        objective_flat,
        x0,
        jac=gradient_flat,
        method='L-BFGS-B',
        options={
            'maxiter': maxiter,
            'ftol': ftol,
            'gtol': gtol,
            'maxfun': maxiter * 10
        }
    )
    
    X_opt = _project_to_sphere(result.x.reshape(N, 3))
    energy, _ = _energy_and_grad(X_opt)
    
    return X_opt, energy


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin before 150s limit
    
    best_X = None
    best_energy = float('inf')
    
    # Multiple restarts to escape local minima
    rng = np.random.default_rng(42)
    
    # Start with Fibonacci spiral as first candidate
    X = _fibonacci_spiral()
    
    # Phase 1: Gradient descent with momentum
    X, energy = _gradient_descent_with_momentum(X, lr=0.01, momentum=0.9, steps=200, deadline=deadline)
    
    if energy < best_energy:
        best_energy = energy
        best_X = X.copy()
    
    # Try a few random restarts if time permits
    n_restarts = 3
    for r in range(n_restarts):
        if time.time() > deadline - 60.0:
            break
        
        X = _random_sphere(rng=rng)
        
        # Phase 1: Gradient descent
        X, energy = _gradient_descent_with_momentum(X, lr=0.01, momentum=0.9, steps=150, deadline=deadline)
        
        if energy < best_energy:
            best_energy = energy
            best_X = X.copy()
    
    if time.time() > deadline - 30.0:
        # Not enough time for polish, return current best
        if best_X is not None:
            return best_X
        return _fibonacci_spiral()
    
    # Phase 2: L-BFGS-B polish with tight tolerances
    X, energy = _lbfgs_polish(best_X, deadline=deadline, maxiter=300, ftol=1e-14, gtol=1e-12)
    
    if energy < best_energy:
        best_energy = energy
        best_X = X.copy()
    
    # Phase 3: Newton polish for final convergence
    if time.time() < deadline - 10.0:
        X, energy = _newton_polish(best_X, max_iter=30, tol=1e-13, deadline=deadline)
        if energy < best_energy:
            best_energy = energy
            best_X = X.copy()
    
    # Final validation: ensure all points are valid
    if best_X is None:
        best_X = _fibonacci_spiral()
    
    norms = np.linalg.norm(best_X, axis=1)
    if np.any(norms < 1e-9) or not np.all(np.isfinite(best_X)):
        # Fallback to Fibonacci spiral if something went wrong
        return _fibonacci_spiral()
    
    return best_X
# EVOLVE-BLOCK-END

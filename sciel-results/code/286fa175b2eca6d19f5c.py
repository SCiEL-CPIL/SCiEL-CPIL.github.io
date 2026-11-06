"""Logarithmic energy minimisation for N points on the unit sphere."""
import time
import numpy as np
import math
from scipy.optimize import minimize

N = 150
# Deadline guard: leave a safety margin under the 150 s limit
_DEADLINE = time.time() + 135.0


def _energy(pts):
    """Sum_{i<j} log(1 / |x_i - x_j|) using math.fsum."""
    # Use dot-product trick for speed: |xi-xj|^2 = 2 - 2*xi.xj
    G = pts @ pts.T
    d2 = 2.0 - 2.0 * G
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    # Only upper triangle
    iu = np.triu_indices(N, 1)
    terms = -np.log(d[iu])
    return math.fsum(terms)


def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _retract(pts):
    """Project back onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _objective_and_grad(x):
    """
    Objective and gradient for L-BFGS.
    x is a flat vector of shape (N*3,).
    We use a penalty-free approach: we minimize the energy on the sphere
    by using the retraction trick with L-BFGS-B, but since L-BFGS-B doesn't
    handle manifold constraints directly, we use a penalty method or
    instead use the gradient projected onto the tangent space.
    
    Actually, for simplicity and robustness, let's use a penalty approach:
    minimize E(pts) + mu * sum(||xi||^2 - 1)^2
    """
    pts = x.reshape(N, 3)
    # Energy
    G = pts @ pts.T
    d2 = 2.0 - 2.0 * G
    # Avoid division by zero
    d2 = np.maximum(d2, 1e-30)
    d = np.sqrt(d2)
    iu = np.triu_indices(N, 1)
    # Energy = sum -log(d_ij)
    energy = -np.sum(np.log(d[iu]))
    
    # Gradient of energy wrt pts
    # d/dx_i sum_{j>i} -log|xi-xj| + d/dx_i sum_{j<i} -log|xj-xi|
    # = sum_{j != i} (xi - xj) / |xi - xj|^2
    diff = pts[:, None, :] - pts[None, :, :]  # (N, N, 3)
    dist = np.sqrt(d2)  # (N, N)
    np.fill_diagonal(dist, 1.0)
    inv_d2 = 1.0 / (dist * dist)
    np.fill_diagonal(inv_d2, 0.0)
    grad_energy = np.einsum('ij,ijk->ik', inv_d2, diff)
    
    # Penalty term: mu * sum(||xi||^2 - 1)^2
    mu = 100.0
    norms2 = np.sum(pts * pts, axis=1)
    penalty = mu * np.sum((norms2 - 1.0) ** 2)
    # Gradient of penalty: 4*mu*(||xi||^2 - 1)*xi
    grad_penalty = 4.0 * mu * (norms2 - 1.0)[:, None] * pts
    
    total = energy + penalty
    grad = grad_energy + grad_penalty
    return total, grad.flatten()


def _lbfgs(pts, maxiter=2000, gtol=1e-10):
    """Run L-BFGS-B with penalty on the sphere constraint."""
    x0 = pts.flatten()
    res = minimize(_objective_and_grad, x0, method='L-BFGS-B',
                   jac=True, options={'maxiter': maxiter, 'gtol': gtol,
                                      'maxls': 50})
    pts_out = res.x.reshape(N, 3)
    pts_out = _retract(pts_out)
    return pts_out


def _perturb(pts, seed, sigma=0.1):
    rng = np.random.default_rng(seed)
    p = pts + rng.normal(0, sigma, pts.shape)
    p = _retract(p)
    return p


def construct():
    t0 = time.time()
    
    # Start with Fibonacci sphere
    pts0 = _fibonacci_sphere(N)
    
    # First pass: L-BFGS from Fibonacci
    best_pts = pts0.copy()
    best_E = _energy(pts0)
    
    # Run L-BFGS
    p1 = _lbfgs(pts0, maxiter=3000, gtol=1e-12)
    E1 = _energy(p1)
    if E1 < best_E:
        best_pts = p1
        best_E = E1
    
    # Check time
    if time.time() > _DEADLINE - 20:
        best_pts = _retract(best_pts)
        return best_pts
    
    # Try perturbations
    for seed in range(1, 8):
        if time.time() > _DEADLINE - 15:
            break
        p = _perturb(pts0, seed, sigma=0.15)
        p2 = _lbfgs(p, maxiter=2000, gtol=1e-11)
        E2 = _energy(p2)
        if E2 < best_E:
            best_pts = p2
            best_E = E2
    
    # Final polish: more iterations, tighter tolerance
    if time.time() < _DEADLINE - 5:
        best_pts = _lbfgs(best_pts, maxiter=3000, gtol=1e-14)
        best_E = _energy(best_pts)
    
    # Ensure valid output
    best_pts = _retract(best_pts)
    
    # Verify no points too close
    d2 = 2.0 - 2.0 * (best_pts @ best_pts.T)
    np.fill_diagonal(d2, 1.0)
    min_d = np.sqrt(np.min(d2))
    if min_d < 1e-9:
        # Fallback to Fibonacci if degenerate
        best_pts = _fibonacci_sphere(N)
    
    return best_pts

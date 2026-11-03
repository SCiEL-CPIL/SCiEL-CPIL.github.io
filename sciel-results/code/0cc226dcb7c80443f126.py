"""Minimal logarithmic energy of 150 points on the sphere – Riemannian GD + polish."""
import time
import math
import numpy as np
from numpy.linalg import norm

N = 150
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0


def _energy_and_grad(pts):
    """Compute log-energy E = sum_{i<j} log(1/|x_i-x_j|) and its Euclidean gradient.
    Gradient of log(1/d) w.r.t. x_i is (x_i - x_j)/|x_i - x_j|^2 for each pair (i,j).
    We return (energy, grad) where grad has shape (N,3)."""
    n = pts.shape[0]
    # Pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N,N,3)
    dist2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist2, 1.0)  # avoid div by zero
    dist = np.sqrt(dist2)

    # Energy: sum over i<j of log(1/d) = -sum_{i<j} log(d)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    d_upper = dist[iu]
    energy = -np.sum(np.log(d_upper))

    # Gradient: for each i, grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j!=i} diff[i,j] / dist2[i,j]
    inv_dist2 = np.zeros_like(dist2)
    inv_dist2[dist2 > 0] = 1.0 / dist2[dist2 > 0]
    np.fill_diagonal(inv_dist2, 0.0)
    grad = np.einsum('ij,ijk->ik', inv_dist2, diff)

    return energy, grad


def _riemannian_grad(pts, grad):
    """Project Euclidean gradient onto tangent space: grad - (grad·x) x."""
    return grad - np.einsum('ij,ij->i', grad, pts)[:, np.newaxis] * pts


def _exp_map(pts, tangent):
    """Exponential map on the unit sphere: exp_x(v) = x cos|v| + (v/|v|) sin|v|."""
    v_norm = norm(tangent, axis=1, keepdims=True)
    small = v_norm < 1e-10
    # For non-small
    cos_t = np.cos(v_norm)
    sin_t = np.sin(v_norm)
    # Handle small norms: use series
    result = pts * cos_t + tangent * (sin_t / np.where(small, 1.0, v_norm))
    # For small norms, sin_t/v_norm ≈ 1 - v_norm^2/6
    if np.any(small):
        v2 = v_norm[small, 0]**2
        result[small] = pts[small] * (1.0 - v2/2.0) + tangent[small] * (1.0 - v2/6.0)
    # Normalize to ensure unit sphere
    result = result / norm(result, axis=1, keepdims=True)
    return result


def _backtracking_line_search(pts, grad, rgrad, max_iter=30):
    """Backtracking line search for step size in Riemannian GD."""
    e0, _ = _energy_and_grad(pts)
    alpha = 0.1
    c1 = 1e-4
    for _ in range(max_iter):
        new_pts = _exp_map(pts, -alpha * rgrad)
        e_new, _ = _energy_and_grad(new_pts)
        if e_new <= e0 - c1 * alpha * np.sum(-rgrad * rgrad):
            return alpha, new_pts
        alpha *= 0.5
    return alpha, new_pts


def construct():
    t_start = time.time()
    deadline = t_start + TIME_LIMIT - SAFETY_MARGIN

    # Fibonacci spiral initial points
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    best_pts = pts.copy()
    best_energy = _energy_and_grad(best_pts)[0]

    # Riemannian gradient descent with backtracking line search
    alpha = 0.1
    patience = 0
    max_patience = 200
    min_improvement = 1e-12

    while time.time() < deadline:
        e, grad = _energy_and_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                break

        rgrad = _riemannian_grad(pts, grad)
        grad_norm = norm(rgrad)
        if grad_norm < 1e-10:
            break

        alpha, new_pts = _backtracking_line_search(pts, rgrad, rgrad)
        pts = new_pts

    # Final polish: use scipy's minimize on the sphere via parameterization
    # Use a simple high-precision local polish with smaller steps
    t_polish_start = time.time()
    polish_deadline = deadline - 1.0

    # Do a few more iterations with smaller learning rate for polish
    alpha = 0.01
    for _ in range(200):
        if time.time() > polish_deadline:
            break
        e, grad = _energy_and_grad(best_pts)
        rgrad = _riemannian_grad(best_pts, grad)
        grad_norm = norm(rgrad)
        if grad_norm < 1e-12:
            break
        new_pts = _exp_map(best_pts, -alpha * rgrad)
        e_new, _ = _energy_and_grad(new_pts)
        if e_new < e:
            best_pts = new_pts
            best_energy = e_new
        else:
            alpha *= 0.5
            if alpha < 1e-10:
                break

    # Ensure all points are on the unit sphere and non-degenerate
    best_pts = best_pts / norm(best_pts, axis=1, keepdims=True)

    return best_pts

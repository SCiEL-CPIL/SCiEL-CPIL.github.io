"""
Strategy: Gram Matrix Optimization via Projected Gradient Descent.
1. Represent the configuration of N points on the sphere by its Gram matrix G, where G_ij = x_i . x_j.
2. The constraint that points lie on the unit sphere translates to G being Positive Semi-Definite (PSD) with unit diagonal (trace N).
3. The logarithmic energy is a function of the eigenvalues of G (which are the squared chordal distances).
4. We minimize the energy over the manifold of PSD matrices with fixed trace using projected gradient descent.
   The projection onto the PSD cone with fixed trace involves eigen-decomposition and clipping negative eigenvalues, followed by renormalization.
5. After converging to a low-energy Gram matrix, we extract the points via eigendecomposition (taking the top N eigenvectors) and perform a final Riemannian polish on the point configuration to ensure exact unit norm and minimize residual errors.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

def _log_energy_and_grad_sphere(x):
    """
    Computes the logarithmic energy and its Euclidean gradient for points on the sphere.
    x: (N, 3) array of points on the unit sphere.
    Returns: energy, grad (N, 3)
    """
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _riemannian_project(grad, x):
    dot = np.sum(grad * x, axis=1, keepdims=True)
    return grad - dot * x

def _exponential_map(x, v):
    norm_v = np.linalg.norm(v, axis=1, keepdims=True)
    mask = norm_v > 1e-10
    new_x = np.zeros_like(x)
    
    cos_theta = np.cos(norm_v)
    sin_theta_over_norm = np.zeros_like(norm_v)
    sin_theta_over_norm[mask] = np.sin(norm_v[mask]) / norm_v[mask]
    
    new_x = cos_theta * x + sin_theta_over_norm * v
    return new_x

def _project_psd_trace(G, trace_val):
    """
    Project a symmetric matrix G onto the set of PSD matrices with trace equal to trace_val.
    """
    # Symmetrize
    G = 0.5 * (G + G.T)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    
    # Clip negative eigenvalues
    eigenvalues_clipped = np.maximum(eigenvalues, 0.0)
    
    # If all eigenvalues are zero, return identity scaled
    if np.sum(eigenvalues_clipped) < 1e-12:
        return np.eye(N) * (trace_val / N)
    
    # Normalize to target trace
    current_trace = np.sum(eigenvalues_clipped)
    scale = trace_val / current_trace
    eigenvalues_scaled = eigenvalues_clipped * scale
    
    # Reconstruct
    G_new = (eigenvectors * eigenvalues_scaled) @ eigenvectors.T
    return 0.5 * (G_new + G_new.T)

def _gram_energy_and_grad(G):
    """
    Compute energy and gradient with respect to the Gram matrix G.
    Energy E = sum_{i<j} log(1 / sqrt(2 - 2 G_ij)) = -0.5 sum_{i!=j} log(sqrt(2 - 2 G_ij))
    Let d_ij^2 = 2 - 2 G_ij.
    E = -0.5 sum_{i!=j} 0.5 log(d_ij^2) = -0.25 sum_{i!=j} log(d_ij^2)
    
    Gradient dE/dG_ij:
    For i != j:
    dE/dG_ij = -0.25 * (1/d_ij^2) * d(d_ij^2)/dG_ij
    d(d_ij^2)/dG_ij = -2
    So dE/dG_ij = -0.25 * (-2) / d_ij^2 = 0.5 / d_ij^2 = 0.5 / (2 - 2 G_ij) = 0.25 / (1 - G_ij)
    
    For i == j:
    G_ii is fixed to 1 in the constraint, but in the unconstrained gradient,
    dE/dG_ii involves terms where i is involved in pairs. However, since we project
    onto the manifold of fixed diagonal, the gradient components for the diagonal
    are effectively replaced by the projection step. We can compute the full gradient
    and then the projection handles the constraints.
    
    Actually, let's be careful. The energy depends on off-diagonal elements.
    E = sum_{i<j} -log(sqrt(2-2G_ij)) = sum_{i<j} -0.5 log(2-2G_ij)
    
    dE/dG_ij (for i<j) = -0.5 * (1/(2-2G_ij)) * (-2) = 1 / (2-2G_ij) = 0.5 / (1-G_ij)
    
    Since G is symmetric, the gradient matrix H will have H_ij = 0.5 / (1-G_ij) for i!=j.
    What about H_ii?
    The energy does not depend on G_ii directly if we assume G_ii=1.
    However, in the projection step, we allow G_ii to vary slightly before projection?
    No, the standard approach is:
    1. Compute gradient of the objective with respect to G.
    2. Since the objective is defined on the manifold of constant diagonal, the gradient
       in the ambient space is zero on the diagonal? Or we just compute the off-diagonal
       part and let the projection handle the rest?
       
    Let's compute the gradient of the function f(G) = sum_{i<j} -0.5 log(2-2G_ij).
    df/dG_ij = 1/(2-2G_ij) for i!=j.
    df/dG_ii = 0.
    
    So the gradient matrix is:
    H_ij = 1 / (2 - 2 G_ij) for i != j
    H_ii = 0
    
    Then we take a step G_new = G - alpha * H.
    Then project G_new to PSD with trace N and unit diagonal?
    Wait, the trace constraint is sum G_ii = N. If we enforce unit diagonal, trace is automatically N.
    The projection _project_psd_trace enforces trace N, but does it enforce unit diagonal?
    No. It enforces PSD and trace N.
    However, the energy function is only well-defined if G_ii=1.
    If G_ii != 1, the distance formula d_ij^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i.x_j = G_ii + G_jj - 2 G_ij.
    
    So we should use the general distance formula:
    d_ij^2 = G_ii + G_jj - 2 G_ij.
    
    Energy E = sum_{i<j} -0.5 log(G_ii + G_jj - 2 G_ij).
    
    Gradient:
    dE/dG_kk (diagonal):
    Sum over all j != k of -0.5 * (1/d_kj^2) * 1 = -0.5 sum_{j!=k} 1/d_kj^2.
    
    dE/dG_ij (off-diagonal, i!=j):
    Term for pair (i,j): -0.5 * (1/d_ij^2) * (-2) = 1/d_ij^2.
    So H_ij = 1/d_ij^2.
    
    This gradient is for the ambient space.
    We then project the updated G to the manifold of PSD matrices with trace N.
    Does this manifold include matrices with G_ii != 1? Yes.
    But we want points on the unit sphere, so we need G_ii = 1.
    
    If we just project to PSD + Trace N, the resulting G might have G_ii != 1.
    Then we extract points from G. If G_ii != 1, the points won't be on the unit sphere.
    
    Solution:
    After the PSD+Trace projection, we can renormalize the rows of the point matrix?
    Or, we can constrain the diagonal to be 1 in the projection.
    
    A common trick:
    1. Compute gradient H.
    2. Step: G_temp = G - alpha * H.
    3. Project G_temp to PSD.
    4. Normalize the diagonal to 1? This is not a simple projection.
    
    Alternative:
    Use the fact that if G is PSD and we take the square root G^{1/2}, the rows are the points.
    If we enforce G_ii=1, the points are on the unit sphere.
    
    Let's stick to the simpler approach:
    Optimize G in the space of symmetric matrices.
    Objective: E(G) = sum_{i<j} -0.5 log(G_ii + G_jj - 2 G_ij).
    Constraint: G is PSD and G_ii = 1 for all i.
    
    This is a difficult constraint to project onto directly.
    
    Let's try a different path:
    Optimize the points directly but use the Gram matrix insight for initialization or a different metric?
    No, the prompt asks for a structural change to SDP/Gram.
    
    Let's implement the PSD projection with Trace N, and then **renormalize the points** after extraction.
    If G is PSD with trace N, let X = G^{1/2}. The rows of X have squared norms G_ii.
    If we normalize each row to unit length, we get points on the sphere.
    This changes the Gram matrix slightly, but it's a valid point configuration.
    We can then do a final polish on these points.
    
    This hybrid approach:
    1. Optimize G to minimize energy, subject to G being PSD and Trace(G)=N.
       The energy is calculated using the general distance formula.
    2. Extract X = G^{1/2}.
    3. Normalize rows of X to get points on the sphere.
    4. Polish these points using Riemannian Gradient Descent.
    
    This avoids the hard constraint of G_ii=1 during the main optimization loop,
    relying on the trace constraint to keep the "size" of the configuration bounded,
    and the normalization step to enforce the sphere constraint.
    
    Let's implement _gram_energy_and_grad for the general case.
    """
    # d_ij^2 = G_ii + G_jj - 2 G_ij
    # We need to compute this for all pairs.
    
    # Diag of G
    diag_G = np.diag(G)
    
    # d_sq[i,j] = diag_G[i] + diag_G[j] - 2 G[i,j]
    d_sq = diag_G[:, np.newaxis] + diag_G[np.newaxis, :] - 2.0 * G
    
    # Ensure numerical stability (avoid log of negative or zero)
    d_sq = np.maximum(d_sq, 1e-12)
    
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: sum_{i<j} -0.5 log(d_sq[i,j])
    # Sum over i<j is 0.5 * sum_{i!=j}
    log_terms = -0.5 * np.log(d_sq)
    energy = 0.5 * np.sum(log_terms[mask])
    
    # Gradient
    # H_ii = -0.5 * sum_{j!=i} 1/d_sq[i,j]
    # H_ij = 1/d_sq[i,j] for i!=j
    
    inv_d_sq = 1.0 / d_sq
    # Zero out diagonal for off-diag calculation
    inv_d_sq_off = inv_d_sq.copy()
    np.fill_diagonal(inv_d_sq_off, 0.0)
    
    # Diagonal part of gradient
    grad_diag = -0.5 * np.sum(inv_d_sq_off, axis=1)
    
    # Off-diagonal part of gradient
    grad_off = inv_d_sq_off
    
    # Combine
    grad = grad_off
    np.fill_diagonal(grad, grad_diag)
    
    return energy, grad

def _extract_points_from_gram(G):
    """
    Extract points from a PSD Gram matrix.
    1. Eigen decompose G.
    2. Take square root of eigenvalues.
    3. Points are rows of V * sqrt(eigenvalues).
    4. Normalize rows to unit length.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    
    # X = eigenvectors * sqrt_eigenvalues
    # eigenvectors is (N, N), sqrt_eigenvalues is (N,)
    # We want rows to be points.
    # G = V D V^T. X = V D^{1/2}.
    # Row i of X is the i-th point?
    # G_ij = sum_k V_ik D_kk V_jk.
    # x_i . x_j = sum_k (V_ik sqrt(D_kk)) (V_jk sqrt(D_kk)).
    # So yes, rows of V D^{1/2} are the points.
    
    X = eigenvectors * sqrt_eigenvalues[np.newaxis, :]
    
    # Normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    X = X / norms
    
    return X

def _local_optimize_riemannian(x, time_budget=10.0):
    """
    Riemannian Gradient Descent with Armijo line search.
    """
    x_curr = x.copy()
    best_energy, _ = _log_energy_and_grad_sphere(x_curr)
    
    step_size = 0.1
    for i in range(1000):
        if time.time() > time_budget:
            break
            
        energy, grad = _log_energy_and_grad_sphere(x_curr)
        if energy < best_energy:
            best_energy = energy
            
        rgrad = _riemannian_project(grad, x_curr)
        
        # Armijo line search
        alpha = step_size
        for _ in range(20):
            v = alpha * rgrad
            x_new = _exponential_map(x_curr, v)
            e_new, _ = _log_energy_and_grad_sphere(x_new)
            if e_new < energy - 1e-4 * alpha * np.sum(rgrad**2):
                break
            alpha *= 0.5
            
        x_curr = x_new
        step_size *= 0.99
        
    return x_curr

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize G
    # Start with a random PSD matrix with trace N
    # Or start from a good point configuration's Gram matrix
    
    # Let's start from a Fibonacci sphere
    def _fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return x
    
    x_init = _fibonacci_sphere(N)
    G_init = x_init @ x_init.T
    
    # Ensure G_init is symmetric
    G_init = 0.5 * (G_init + G_init.T)
    
    G_curr = G_init.copy()
    
    # Project to ensure PSD and trace N
    G_curr = _project_psd_trace(G_curr, N)
    
    best_energy = float('inf')
    best_x = x_init.copy()
    
    # Projected Gradient Descent on G
    lr = 0.01
    n_iters = 500
    
    for i in range(n_iters):
        if time.time() > deadline - 20:
            break
            
        energy, grad = _gram_energy_and_grad(G_curr)
        
        # Update
        G_new = G_curr - lr * grad
        
        # Project
        G_new = _project_psd_trace(G_new, N)
        
        # Check convergence
        G_curr = G_new
        
        # Extract points and evaluate on sphere
        x_curr = _extract_points_from_gram(G_curr)
        e_sphere, _ = _log_energy_and_grad_sphere(x_curr)
        
        if e_sphere < best_energy:
            best_energy = e_sphere
            best_x = x_curr.copy()
        
        # Adaptive learning rate
        if i % 50 == 0:
            lr *= 0.9

    # Final Polish
    if time.time() < deadline - 5:
        x_polished = _local_optimize_riemannian(best_x, time_budget=15.0)
        e_polished, _ = _log_energy_and_grad_sphere(x_polished)
        if e_polished < best_energy:
            best_energy = e_polished
            best_x = x_polished

    return best_x

# EVOLVE-BLOCK-END

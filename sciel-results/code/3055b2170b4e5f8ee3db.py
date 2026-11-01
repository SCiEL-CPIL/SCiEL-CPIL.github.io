import numpy as np
import time
import math

N = 100

def _energy_from_gram(G):
    """
    Compute energy from a Gram matrix G.
    d_ij^2 = 2 - 2 * G_ij
    Energy = sum_{i<j} log(1 / sqrt(2 - 2 * G_ij))
           = -0.5 * sum_{i<j} log(2 - 2 * G_ij)
    """
    # Ensure numerical stability
    # G_ij should be in [-1, 1]
    # dist_sq = 2 - 2*G_ij
    dist_sq = 2.0 - 2.0 * G
    
    # Mask diagonal (dist_sq is 0 there, log is -inf)
    mask = ~np.eye(N, dtype=bool)
    
    # Clip to avoid log(0) or log(negative) due to numerical errors
    # dist_sq must be > 0. If G_ij is close to 1, dist_sq is close to 0.
    # Minimum distance is handled by the problem constraints, but numerically we clip.
    dist_sq = np.clip(dist_sq, 1e-12, None)
    
    log_dist_sq = np.log(dist_sq)
    
    # Sum over i < j
    # Sum_{i<j} log(d_ij^2) = 0.5 * Sum_{i!=j} log(d_ij^2)
    total_log = 0.5 * np.sum(log_dist_sq[mask])
    
    # Energy = -0.5 * Sum_{i<j} log(d_ij) = -0.5 * 0.5 * Sum_{i<j} log(d_ij^2)
    # Wait.
    # E = sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
    # log(d_ij) = 0.5 log(d_ij^2)
    # E = -0.5 sum_{i<j} log(d_ij^2)
    
    # My formula above: total_log = 0.5 * Sum_{i!=j} log(d_ij^2) = Sum_{i<j} log(d_ij^2)
    # So E = -0.5 * total_log
    
    energy = -0.5 * total_log
    return energy

def _optimize_gram(G0, deadline):
    """
    Optimize the Gram matrix using a projected gradient method.
    Variables: G (N x N)
    Constraints: G is PSD, diag(G) = 1.
    Objective: Minimize E(G).
    
    Gradient of E w.r.t G:
    E = -0.5 sum_{i<j} log(2 - 2 G_ij)
    dE/dG_ij = -0.5 * (1/(2-2G_ij)) * (-2) = 1/(2-2G_ij) for i!=j
    dE/dG_ii = 0 (but we enforce diag=1)
    
    We use a simple iterative scheme:
    1. Compute gradient.
    2. Update G = G - alpha * grad.
    3. Project to PSD: Eigenvalue decomposition, clip negative eigenvalues to 0, reassemble.
    4. Project to unit diagonal: Set diagonal to 1.
    5. Repeat.
    
    This is a heuristic "proximal gradient" method on the Gram matrix space.
    """
    G = G0.copy()
    best_G = G.copy()
    best_E = _energy_from_gram(G)
    
    alpha = 0.01
    min_alpha = 1e-6
    max_iter = 2000
    
    for k in range(max_iter):
        if time.time() > deadline:
            break
            
        # Compute Gradient
        dist_sq = 2.0 - 2.0 * G
        dist_sq = np.clip(dist_sq, 1e-12, None)
        
        grad = 1.0 / dist_sq
        # Set diagonal to 0 because we project diagonal to 1 anyway
        np.fill_diagonal(grad, 0.0)
        
        # Update
        G_new = G - alpha * grad
        
        # Project to PSD
        # Eigenvalue decomposition
        try:
            w, v = np.linalg.eigh(G_new)
            # Clip negative eigenvalues
            w = np.clip(w, 0, None)
            G_psd = v @ np.diag(w) @ v.T
        except np.linalg.LinAlgError:
            # Fallback if SVD fails, just keep old G
            break
            
        # Project to unit diagonal
        # We want diag(G) = 1.
        # Simple projection: set diagonal to 1.
        # More sophisticated: scale rows/cols, but that breaks PSD.
        # Setting diagonal to 1 is a standard heuristic for correlation matrices.
        np.fill_diagonal(G_psd, 1.0)
        
        # Check Energy
        E_new = _energy_from_gram(G_psd)
        
        if E_new < best_E:
            best_E = E_new
            best_G = G_psd.copy()
            G = G_psd
            # Slow down alpha
            alpha = max(alpha * 0.99, min_alpha)
        else:
            # Reduce alpha
            alpha *= 0.5
            if alpha < min_alpha:
                break
                
    return best_G, best_E

def _gram_to_points(G):
    """
    Extract points from a Gram matrix.
    G = X X^T.
    We need X of shape (N, 3).
    Perform eigendecomposition G = V diag(W) V^T.
    Take the top 3 eigenvectors.
    X = V_top3 * sqrt(W_top3)
    """
    w, v = np.linalg.eigh(G)
    # Sort eigenvalues in descending order
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]
    
    # Take top 3
    w_top3 = w[:3]
    v_top3 = v[:, :3]
    
    # Ensure non-negative eigenvalues for sqrt
    w_top3 = np.clip(w_top3, 0, None)
    
    X = v_top3 * np.sqrt(w_top3)
    
    # Normalize rows to ensure unit sphere (handles numerical drift)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X = X / norms
    
    return X

def _fibonacci_sphere(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _points_to_gram(X):
    return X @ X.T

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Generate initial Gram matrices from different point configurations
    starts = []
    
    # 1. Fibonacci Sphere
    X_fib = _fibonacci_sphere(N)
    G_fib = _points_to_gram(X_fib)
    starts.append(G_fib)
    
    # 2. Random Points
    X_rand = np.random.randn(N, 3)
    X_rand /= np.linalg.norm(X_rand, axis=1, keepdims=True)
    G_rand = _points_to_gram(X_rand)
    starts.append(G_rand)
    
    # 3. Fibonacci with rotation
    u, s, vh = np.linalg.svd(np.random.randn(3, 3))
    R = u @ vh
    X_rot = X_fib @ R.T
    G_rot = _points_to_gram(X_rot)
    starts.append(G_rot)
    
    best_G_global = None
    best_E_global = float('inf')
    best_X_global = None
    
    # Allocate time to each start
    # We have ~140 seconds.
    # 3 starts.
    # Give each ~40s, leaving 20s for final polish/extraction.
    
    time_alloc = [40.0, 40.0, 40.0]
    
    for i, G0 in enumerate(starts):
        if time.time() > deadline - 5.0:
            break
            
        t_end_for_this = time.time() + time_alloc[i]
        if t_end_for_this > deadline - 5.0:
            t_end_for_this = deadline - 5.0
            
        G_opt, E_opt = _optimize_gram(G0, t_end_for_this)
        
        if E_opt < best_E_global:
            best_E_global = E_opt
            best_G_global = G_opt.copy()
            # Extract points immediately to check validity/energy in point space
            X_candidate = _gram_to_points(G_opt)
            # Verify energy in point space to be safe
            # (The Gram energy should match, but let's be sure)
            
    if best_G_global is not None:
        best_X_global = _gram_to_points(best_G_global)
        
        # Final Polish: 
        # The Gram optimization might stop at a point where the rank-3 projection 
        # isn't perfectly optimized in point space. 
        # We can do a few steps of point-space gradient descent on the result 
        # to polish it, using the remaining time.
        
        if time.time() < deadline - 2.0:
             # Implement a quick point-space polish
             X = best_X_global.copy()
             # Simple gradient step
             # Reuse logic from previous thought process but simplified
             
             def _energy_and_grad_point(X):
                 diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
                 dist_sq = np.sum(diff * diff, axis=2)
                 np.fill_diagonal(dist_sq, 1.0)
                 mask = ~np.eye(N, dtype=bool)
                 
                 # Energy
                 dist_sq_clipped = np.clip(dist_sq, 1e-12, None)
                 log_dist_sq = np.log(dist_sq_clipped)
                 energy = -0.25 * np.sum(log_dist_sq[mask])
                 
                 # Gradient
                 inv_dist_sq = np.zeros_like(dist_sq)
                 inv_dist_sq[mask] = 1.0 / dist_sq_clipped[mask]
                 grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
                 
                 return energy, grad
             
             E, g = _energy_and_grad_point(X)
             best_E = E
             best_X = X.copy()
             
             step = 0.005
             for k in range(500):
                 if time.time() > deadline - 1.0:
                     break
                 
                 g_tang = g - np.sum(g * X, axis=1, keepdims=True) * X
                 X_new = X - step * g_tang
                 X_new /= np.linalg.norm(X_new, axis=1, keepdims=True)
                 
                 E_new, g_new = _energy_and_grad_point(X_new)
                 
                 if E_new < E:
                     X = X_new
                     E = E_new
                     g = g_new
                     if E < best_E:
                         best_E = E
                         best_X = X.copy()
                     step *= 0.99
                 else:
                     step *= 0.5
                     if step < 1e-8:
                         break
             
             best_X_global = best_X
             
    # Final check
    if best_X_global is None:
        # Fallback to Fibonacci if everything failed
        best_X_global = _fibonacci_sphere(N)
        
    # Ensure valid output
    best_X_global = best_X_global / np.linalg.norm(best_X_global, axis=1, keepdims=True)
    
    return best_X_global

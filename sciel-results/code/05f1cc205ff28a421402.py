import numpy as np
import time
import math

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def fibonacci_spiral():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def random_points(seed):
        rng = np.random.default_rng(seed)
        pts = rng.normal(size=(N, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        return pts / norms

    def energy_and_grad(pts):
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d_sq, 1.0)
        
        # Avoid log(0) and numerical issues
        d_sq = np.maximum(d_sq, 1e-12)
        d = np.sqrt(d_sq)
        
        # Energy: sum_{i<j} -log(d_ij)
        iu = np.triu_indices(N, k=1)
        energy = np.sum(-np.log(d[iu]))
        
        # Gradient: 
        # E = sum_{i<j} -log(d_ij)
        # dE/dx_i = sum_{j != i} d/dx_i (-log(d_ij))
        # d/dx_i log(d_ij) = (1/d_ij) * d/dx_i d_ij = (1/d_ij) * (x_i - x_j)/d_ij = (x_i - x_j)/d_ij^2
        # So dE/dx_i = - sum_{j != i} (x_i - x_j)/d_ij^2
        
        inv_d_sq = np.zeros((N, N))
        inv_d_sq[iu] = 1.0 / d_sq[iu]
        inv_d_sq = inv_d_sq + inv_d_sq.T
        
        # grad[i] = - sum_j diff[i,j] * inv_d_sq[i,j]
        # diff[i,j] = x_i - x_j
        grad = -np.einsum('ijk,ij->ik', diff, inv_d_sq)
        
        # Riemannian Gradient: Project to tangent space
        # P_i = I - x_i x_i^T
        # Rgrad_i = grad_i - (grad_i . x_i) x_i
        grad_dot_x = np.sum(grad * pts, axis=1, keepdims=True)
        rgrad = grad - grad_dot_x * pts
        
        return energy, rgrad

    def riemannian_optimize(pts_init, max_iters=5000, tol=1e-8):
        pts = pts_init.copy()
        E, rgrad = energy_and_grad(pts)
        
        # Backtracking line search
        alpha = 1.0
        c1 = 1e-4
        
        for k in range(max_iters):
            if time.time() > deadline:
                break
                
            # Check convergence
            g_norm = np.linalg.norm(rgrad)
            if g_norm < tol:
                break
                
            # Line search
            while True:
                # Update
                pts_new = pts - alpha * rgrad
                # Project back to sphere
                norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
                pts_new = pts_new / norms
                
                E_new, _ = energy_and_grad(pts_new)
                
                if E_new <= E - c1 * alpha * np.sum(rgrad * (pts_new - pts)):
                    # Sufficient decrease
                    pts = pts_new
                    E = E_new
                    _, rgrad = energy_and_grad(pts)
                    break
                else:
                    alpha *= 0.5
                    if alpha < 1e-12:
                        # Failed line search, break
                        return pts, E
            
        return pts, E

    # Initial configurations
    best_pts = None
    best_energy = float('inf')
    
    # 1. Fibonacci Spiral
    pts_fib = fibonacci_spiral()
    pts_opt, E_opt = riemannian_optimize(pts_fib, max_iters=3000)
    if E_opt < best_energy:
        best_energy = E_opt
        best_pts = pts_opt
        
    # 2. Random Restart 1
    if time.time() < deadline - 10.0:
        pts_rand1 = random_points(42)
        pts_opt2, E_opt2 = riemannian_optimize(pts_rand1, max_iters=3000)
        if E_opt2 < best_energy:
            best_energy = E_opt2
            best_pts = pts_opt2

    # 3. Random Restart 2
    if time.time() < deadline - 10.0:
        pts_rand2 = random_points(123)
        pts_opt3, E_opt3 = riemannian_optimize(pts_rand2, max_iters=3000)
        if E_opt3 < best_energy:
            best_energy = E_opt3
            best_pts = pts_opt3
            
    # 4. Perturbed Fibonacci
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(999)
        pts_pert = pts_fib + rng.normal(scale=0.1, size=(N, 3))
        norms = np.linalg.norm(pts_pert, axis=1, keepdims=True)
        pts_pert = pts_pert / norms
        pts_opt4, E_opt4 = riemannian_optimize(pts_pert, max_iters=3000)
        if E_opt4 < best_energy:
            best_energy = E_opt4
            best_pts = pts_opt4

    # Final polish with more iterations if time permits
    if best_pts is not None and time.time() < deadline - 5.0:
        pts_final, E_final = riemannian_optimize(best_pts, max_iters=10000, tol=1e-10)
        if E_final < best_energy:
            best_energy = E_final
            best_pts = pts_final

    # Ensure valid output
    if best_pts is None:
        best_pts = pts_fib
        
    # Normalize again
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts

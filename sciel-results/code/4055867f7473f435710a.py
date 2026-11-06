"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import math
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def energy(pts):
        """Compute log(1/d) = -log(d) energy for all pairs using math.fsum."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1.0)
        dist = np.maximum(dist, 1e-12)
        e = -np.log(dist)
        iu = np.triu_indices(N, k=1)
        terms = e[iu[0], iu[1]].tolist()
        return math.fsum(terms)

    def fibonacci_sphere():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def random_sphere(n, seed=None):
        rng = np.random.default_rng(seed)
        pts = rng.standard_normal((n, 3))
        return normalize(pts)

    def lbfgs_optimize(pts, max_iter=3000, tol=1e-14):
        """Optimize using L-BFGS-B on the sphere via retraction."""
        pts = pts.copy()
        best_pts = pts.copy()
        best_e = energy(pts)
        
        def obj_and_grad(x):
            p = normalize(x.reshape(N, 3))
            e = energy(p)
            # Gradient
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, 1.0)
            dist = np.maximum(dist, 1e-12)
            inv_d2 = 1.0 / (dist * dist)
            inv_d2 = np.where(np.eye(N, dtype=bool), 0.0, inv_d2)
            grad = np.einsum('ij,ijk->ik', inv_d2, diff)
            # Project to tangent
            dot = np.einsum('ij,ij->i', grad, p)
            grad_t = grad - dot[:, np.newaxis] * p
            return e, grad_t.reshape(-1)
        
        x0 = pts.reshape(-1)
        e0, g0 = obj_and_grad(x0)
        
        try:
            res = minimize(
                lambda x: obj_and_grad(x)[0],
                x0,
                jac=lambda x: obj_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12, 'maxfun': max_iter * 2}
            )
            p_opt = normalize(res.x.reshape(N, 3))
            e_opt = energy(p_opt)
            if e_opt < best_e:
                best_e = e_opt
                best_pts = p_opt
        except Exception:
            pass
        
        return best_pts, best_e

    def gradient_descent_polish(pts, max_iter=5000, lr_init=1e-5):
        """Final polish with small gradient steps."""
        pts = pts.copy()
        lr = lr_init
        best_pts = pts.copy()
        best_e = energy(pts)
        for it in range(max_iter):
            if time.time() > deadline - 3.0:
                break
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, 1.0)
            dist = np.maximum(dist, 1e-12)
            inv_d2 = 1.0 / (dist * dist)
            inv_d2 = np.where(np.eye(N, dtype=bool), 0.0, inv_d2)
            grad = np.einsum('ij,ijk->ik', inv_d2, diff)
            dot = np.einsum('ij,ij->i', grad, pts)
            grad_t = grad - dot[:, np.newaxis] * pts
            
            pts_new = normalize(pts - lr * grad_t)
            e_new = energy(pts_new)
            if e_new < best_e:
                best_e = e_new
                best_pts = pts_new.copy()
                pts = pts_new
                lr = min(lr * 1.02, 1e-3)
            else:
                lr *= 0.7
                if lr < 1e-12:
                    break
        return best_pts, best_e

    # Generate diverse initializations
    inits = []
    
    # Fibonacci sphere
    fib = normalize(fibonacci_sphere())
    inits.append(fib)
    
    # Random initializations with different seeds
    for seed in [42, 123, 999, 7, 2024, 555, 314, 271]:
        inits.append(random_sphere(N, seed=seed))
    
    # Perturbed Fibonacci
    rng = np.random.default_rng(42)
    for noise_scale in [0.01, 0.05, 0.1]:
        perturbed = normalize(fib + noise_scale * rng.standard_normal((N, 3)))
        inits.append(perturbed)
    
    # Optimize each initialization
    best_pts = None
    best_e = float('inf')
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
        e_init = energy(init_pts)
        if e_init < best_e:
            best_e = e_init
            best_pts = init_pts.copy()
        
        # L-BFGS optimization
        pts_opt, e_opt = lbfgs_optimize(init_pts, max_iter=2000, tol=1e-12)
        if e_opt < best_e:
            best_e = e_opt
            best_pts = pts_opt
    
    # Final polish on best result
    if best_pts is not None and time.time() < deadline - 5.0:
        best_pts, best_e = gradient_descent_polish(best_pts, max_iter=10000, lr_init=1e-6)
    
    # Ensure validity
    best_pts = normalize(best_pts)
    
    # Check no points too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    min_dist = dist.min()
    if min_dist < 1e-9:
        # Nudge apart
        rng = np.random.default_rng(0)
        for i in range(N):
            for j in range(i+1, N):
                if dist[i,j] < 1e-9:
                    best_pts[j] = normalize(best_pts[j] + 1e-6 * rng.standard_normal(3))
                    best_pts = normalize(best_pts)
                    # Recompute distances
                    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
                    dist = np.linalg.norm(diff, axis=2)
                    np.fill_diagonal(dist, 1.0)
    
    return best_pts
# EVOLVE-BLOCK-END

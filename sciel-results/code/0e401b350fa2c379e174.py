"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def get_fibonacci_points(n, jitter=0.0):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        if jitter > 0:
            pts += jitter * np.random.randn(n, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

    def log_energy(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        e = -0.25 * np.sum(np.log(dist_sq))
        return e

    def gradient(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        inv_dist_sq = 1.0 / dist_sq
        mask = ~np.eye(N, dtype=bool)
        inv_dist_sq = inv_dist_sq * mask
        grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        dot = np.sum(grad * pts, axis=1, keepdims=True)
        proj_grad = grad - dot * pts
        return proj_grad

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def optimize_l_bfgs(pts, max_time=30.0):
        """Optimize using L-BFGS-B with sphere constraint via normalization."""
        best_pts = pts.copy()
        best_e = log_energy(pts)
        
        # Convert to spherical coordinates for optimization
        # Use azimuthal and polar angles
        x = np.arccos(np.clip(pts[:, 2], -1.0, 1.0))  # polar angle theta
        y = np.arctan2(pts[:, 1], pts[:, 0])  # azimuthal angle phi
        
        def spherical_to_cartesian(theta, phi):
            sin_theta = np.sin(theta)
            return np.column_stack([
                sin_theta * np.cos(phi),
                sin_theta * np.sin(phi),
                np.cos(theta)
            ])
        
        def objective_and_grad(params):
            theta = params[:N]
            phi = params[N:2*N]
            pts = spherical_to_cartesian(theta, phi)
            e = log_energy(pts)
            
            # Compute gradient via chain rule
            # dE/dtheta_i, dE/dphi_i
            grad_pts = gradient(pts)
            # dpts/dtheta = [cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)]
            # dpts/dphi = [-sin(phi), cos(phi), 0]
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            
            dpts_dtheta = np.column_stack([
                cos_theta * cos_phi,
                cos_theta * sin_phi,
                -sin_theta
            ])
            dpts_dphi = np.column_stack([
                -sin_phi,
                cos_phi,
                np.zeros(N)
            ])
            
            grad_theta = np.sum(grad_pts * dpts_dtheta, axis=1)
            grad_phi = np.sum(grad_pts * dpts_dphi, axis=1)
            
            grad = np.concatenate([grad_theta, grad_phi])
            return e, grad
        
        # Use L-BFGS-B
        try:
            result = minimize(
                objective_and_grad,
                np.concatenate([x, y]),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            if result.success or result.fun < best_e:
                new_pts = spherical_to_cartesian(result.x[:N], result.x[N:2*N])
                new_pts = normalize(new_pts)
                e_new = log_energy(new_pts)
                if e_new < best_e:
                    best_e = e_new
                    best_pts = new_pts
        except Exception:
            pass
        
        return best_pts, best_e

    def optimize_gradient_descent(pts, max_time=20.0):
        """Fallback gradient descent with adaptive learning rate."""
        best_pts = pts.copy()
        best_e = log_energy(pts)
        lr = 0.01
        momentum = 0.9
        v = np.zeros_like(pts)
        c1 = 1e-4
        rho = 0.5
        
        while time.time() < start_time + max_time:
            grad = gradient(pts)
            v = momentum * v - lr * grad
            direction = v
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-10:
                break
            
            e_curr = log_energy(pts)
            alpha = 1.0
            max_backtracks = 10
            backtracks = 0
            new_pts = None
            
            while backtracks < max_backtracks:
                candidate = pts + alpha * direction
                candidate = normalize(candidate)
                e_cand = log_energy(candidate)
                slope = np.sum(grad * direction)
                
                if e_cand <= e_curr + c1 * alpha * slope:
                    new_pts = candidate
                    break
                else:
                    alpha *= rho
                    backtracks += 1
            
            if new_pts is None:
                new_pts = normalize(pts + alpha * direction)
                e_new = log_energy(new_pts)
                if e_new < best_e:
                    best_e = e_new
                    best_pts = new_pts.copy()
                lr *= 0.5
                if lr < 1e-6:
                    break
            else:
                e_new = log_energy(new_pts)
                if e_new < best_e:
                    best_e = e_new
                    best_pts = new_pts.copy()
                
                if backtracks == 0:
                    lr *= 1.05
                elif backtracks > 2:
                    lr *= 0.9
                lr = np.clip(lr, 1e-6, 0.1)
                pts = new_pts
        
        return best_pts, best_e

    # Try multiple initial configurations
    best_pts = None
    best_e = float('inf')
    
    # Configuration 1: Fibonacci spiral
    if time.time() < deadline - 10.0:
        pts = get_fibonacci_points(N)
        pts, e1 = optimize_l_bfgs(pts, max_time=40.0)
        if e1 < best_e:
            best_e = e1
            best_pts = pts.copy()
    
    # Configuration 2: Fibonacci with jitter
    if time.time() < deadline - 10.0:
        np.random.seed(42)
        pts = get_fibonacci_points(N, jitter=0.01)
        pts, e2 = optimize_l_bfgs(pts, max_time=30.0)
        if e2 < best_e:
            best_e = e2
            best_pts = pts.copy()
    
    # Configuration 3: Random uniform on sphere
    if time.time() < deadline - 10.0:
        np.random.seed(123)
        pts = np.random.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        pts, e3 = optimize_l_bfgs(pts, max_time=25.0)
        if e3 < best_e:
            best_e = e3
            best_pts = pts.copy()
    
    # Configuration 4: Another random seed
    if time.time() < deadline - 10.0:
        np.random.seed(777)
        pts = np.random.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        pts, e4 = optimize_l_bfgs(pts, max_time=20.0)
        if e4 < best_e:
            best_e = e4
            best_pts = pts.copy()
    
    # Final refinement with gradient descent on the best result
    if best_pts is not None and time.time() < deadline - 5.0:
        best_pts, best_e = optimize_gradient_descent(best_pts, max_time=10.0)
    
    if best_pts is None:
        best_pts = get_fibonacci_points(N)
    
    return normalize(best_pts)

# EVOLVE-BLOCK-END

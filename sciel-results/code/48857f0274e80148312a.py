"""
Strategy: Riemannian Optimization on the Sphere.
- Instead of embedding in R^3 and projecting (which can cause step-size issues 
  and metric distortion), we work directly on the manifold S^2.
- We use the Riemannian Gradient Descent / L-BFGS logic.
- The gradient on the sphere is the projection of the Euclidean gradient.
- The update step is performed via the exponential map (or a retraction like 
  projection-to-sphere, which is simpler and often sufficient for L-BFGS-like 
  line searches). We use a robust Riemannian L-BFGS implementation.
- Initialization: Fibonacci spiral is a strong starting point for log-energy.
- Polishing: We use a more aggressive local search with smaller steps near the 
  end.
- Key difference: The line search is performed along the geodesic (or retraction) 
  path, ensuring the points stay on the sphere without ad-hoc normalization 
  inside the objective function (which breaks differentiability).
"""

import numpy as np
from scipy.optimize import minimize
import time
from math import log, fsum

N = 150
DEADLINE_FRACTION = 0.90  # Safety margin for 120s limit

def _energy_and_grad_manifold(pts):
    """
    Compute logarithmic energy and Riemannian gradient on the sphere.
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    
    pts: (N, 3) array of unit vectors.
    
    The Riemannian gradient grad_M E at p is the projection of the 
    Euclidean gradient onto the tangent space T_p S^2.
    
    Euclidean gradient component for point i:
    grad_E[i] = sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    
    Riemannian gradient:
    grad_M[i] = grad_E[i] - (grad_E[i] . x_i) * x_i
    """
    n = len(pts)
    # Pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists = np.sqrt(dists_sq)
    
    # Avoid division by zero
    mask = dists > 1e-12
    inv_d = np.zeros_like(dists)
    inv_d[mask] = 1.0 / dists[mask]
    
    # Energy
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dists[iu]))
    
    # Gradient accumulation
    # grad_E[i] = sum_j (x_i - x_j) / d_ij^2
    # coeff[i,j] = 1/d_ij^2 if i!=j else 0
    coeff = np.zeros_like(dists_sq)
    mask_sq = dists_sq > 1e-24
    coeff[mask_sq] = 1.0 / dists_sq[mask_sq]
    np.fill_diagonal(coeff, 0.0)
    
    # grad_E[i] = x_i * sum_j coeff[i,j] - sum_j coeff[i,j] * x_j
    sum_coeff = np.sum(coeff, axis=1)
    sum_coeff_pts = coeff @ pts
    
    grad_E = pts * sum_coeff[:, np.newaxis] - sum_coeff_pts
    
    # Project to tangent space
    dot_gx = np.sum(grad_E * pts, axis=1)
    grad_M = grad_E - dot_gx[:, np.newaxis] * pts
    
    return energy, grad_M

def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points in Cartesian coordinates."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * i
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y, z])

def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

class RiemannianLbfgs:
    """
    A simple Riemannian L-BFGS implementation.
    It maintains the history of (s, y) pairs in the tangent space.
    The update is: p_{k+1} = Retract(p_k, alpha * d_k)
    where d_k is the search direction in the tangent space.
    """
    def __init__(self, m=10):
        self.m = m
        self.s_history = []
        self.y_history = []
        self.rho_history = []
        
    def reset(self):
        self.s_history = []
        self.y_history = []
        self.rho_history = []

    def get_search_direction(self, grad):
        """
        Compute the L-BFGS search direction in the tangent space.
        grad: (N, 3) Riemannian gradient.
        """
        if not self.s_history:
            return -grad
        
        # Two-loop recursion for L-BFGS
        # We need to flatten grad to work with 1D arrays for the history
        g_flat = grad.flatten()
        
        # q = g
        q = g_flat.copy()
        
        # Loop 1: backward
        alphas = []
        for i in range(len(self.s_history) - 1, -1, -1):
            s = self.s_history[i].flatten()
            y = self.y_history[i].flatten()
            rho = self.rho_history[i]
            
            alpha = rho * np.dot(s, q)
            alphas.append(alpha)
            q = q - alpha * y
            
        # y0 = g (initial approximation of Hessian inverse is identity)
        # r = H0 * q. Since H0 = I, r = q.
        # But we might want to scale it. Standard L-BFGS uses H0 = I.
        r = q.copy()
        
        # Loop 2: forward
        for i in range(len(self.s_history)):
            s = self.s_history[i].flatten()
            y = self.y_history[i].flatten()
            rho = self.rho_history[i]
            
            beta = rho * np.dot(y, r)
            r = r + (alphas[i] - beta) * s
            
        return r.reshape(grad.shape)
    
    def update(self, s, y):
        """
        Update the history with new step s and gradient change y.
        s: (N, 3) step taken (in tangent space, or retraction difference)
        y: (N, 3) difference in gradients (grad_new - grad_old)
        """
        # Check curvature condition
        sy = np.sum(s * y)
        if sy > 1e-10:
            self.s_history.append(s)
            self.y_history.append(y)
            self.rho_history.append(1.0 / sy)
            
            if len(self.s_history) > self.m:
                self.s_history.pop(0)
                self.y_history.pop(0)
                self.rho_history.pop(0)

def _riemannian_line_search(pts, grad, direction, energy0, fmin, c1=1e-4, max_iter=20):
    """
    Backtracking line search on the sphere.
    We use the retraction: Retract(p, v) = normalize(p + v)
    This is a valid retraction.
    """
    alpha = 1.0
    g_dir = np.sum(grad * direction)
    
    if g_dir >= 0:
        # Direction is not a descent direction? Should not happen if we use -grad
        # But due to numerical errors, maybe.
        return None, 0.0, energy0

    for _ in range(max_iter):
        # Step
        new_pts = pts + alpha * direction
        new_pts = _project_to_sphere(new_pts)
        
        # Energy
        new_energy, new_grad = _energy_and_grad_manifold(new_pts)
        
        # Armijo condition
        if new_energy <= energy0 + c1 * alpha * g_dir:
            return new_pts, alpha, new_energy, new_grad
        
        alpha *= 0.5
        if alpha < 1e-8:
            break
            
    return None, 0.0, energy0, grad

def construct():
    start_time = time.time()
    total_time = 120.0
    deadline = start_time + total_time * DEADLINE_FRACTION
    
    rng = np.random.default_rng(42)
    
    # Initial guess: Fibonacci spiral
    fib_pts = _fibonacci_spiral(N)
    best_pts = fib_pts.copy()
    best_energy, _ = _energy_and_grad_manifold(best_pts)
    
    # Optimization function
    def optimize_from(start_pts, max_iter=1000, ftol=1e-8):
        opt = RiemannianLbfgs(m=10)
        pts = start_pts.copy()
        energy, grad = _energy_and_grad_manifold(pts)
        
        for i in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Check convergence
            grad_norm = np.linalg.norm(grad)
            if grad_norm < ftol:
                break
                
            # Get search direction
            direction = opt.get_search_direction(grad)
            
            # Line search
            new_pts, alpha, new_energy, new_grad = _riemannian_line_search(
                pts, grad, direction, energy, ftol
            )
            
            if new_pts is None:
                # Line search failed, try smaller step or break
                # Simple fallback: gradient descent with small step
                step = -grad * 0.01
                new_pts = _project_to_sphere(pts + step)
                new_energy, new_grad = _energy_and_grad_manifold(new_pts)
                if new_energy >= energy:
                    break
                alpha = 0.01
            
            # Update history
            # s is the step in the tangent space (approximated by new_pts - pts for small steps)
            # For retraction, s = new_pts - pts is a good approximation of the vector in T_p
            s = new_pts - pts
            y = new_grad - grad
            
            opt.update(s, y)
            
            pts = new_pts
            energy = new_energy
            grad = new_grad
            
        return pts, energy
    
    # Multi-restart strategy
    # 1. Start with Fibonacci
    # 2. Try a few random restarts
    # 3. Try perturbed Fibonacci
    
    restarts = []
    # Fibonacci
    restarts.append(fib_pts)
    
    # Random restarts
    for i in range(3):
        rand_pts = rng.standard_normal((N, 3))
        restarts.append(_project_to_sphere(rand_pts))
        
    # Perturbed Fibonacci
    for scale in [0.1, 0.05, 0.01]:
        pert_pts = fib_pts + rng.standard_normal((N, 3)) * scale
        restarts.append(_project_to_sphere(pert_pts))
    
    for init_pts in restarts:
        if time.time() > deadline - 10:
            break
            
        try:
            cand_pts, cand_energy = optimize_from(init_pts)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_pts = cand_pts.copy()
        except Exception:
            continue
            
    # Final Polish: Run one more optimization from the best point with tighter tolerances
    if time.time() < deadline - 2 and best_pts is not None:
        try:
            cand_pts, cand_energy = optimize_from(best_pts, max_iter=2000, ftol=1e-10)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_pts = cand_pts.copy()
        except Exception:
            pass
            
    # Ensure final points are valid
    return _project_to_sphere(best_pts)

if __name__ == "__main__":
    pts = construct()
    # Verification
    energy, _ = _energy_and_grad_manifold(pts)
    print(f"Energy: {energy}")
    print(f"Points shape: {pts.shape}")
    print(f"Norms: {np.linalg.norm(pts, axis=1)[:5]}")

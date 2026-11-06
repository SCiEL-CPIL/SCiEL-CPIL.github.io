import numpy as np
import time
from scipy.spatial import ConvexHull
import math

N = 130

def _project_to_sphere(P):
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    return P / norms

def _riemannian_step(P, grad, step_size):
    # grad is Euclidean gradient
    # Project to tangent space
    # P shape (N, 3), grad shape (N, 3)
    # dot product of grad with P for each point
    dots = np.sum(P * grad, axis=1, keepdims=True)
    tangent_grad = grad - dots * P
    P_new = P - step_size * tangent_grad
    return _project_to_sphere(P_new)

def _compute_circumradii(P):
    """
    Computes the circumradius of each facet of the convex hull.
    Returns an array of circumradii (angular distance).
    """
    try:
        hull = ConvexHull(P)
        facets = hull.simplices
    except Exception:
        # If hull fails (e.g. coplanar points), return a large penalty
        return np.array([np.pi])
    
    radii = []
    for tri in facets:
        p1, p2, p3 = P[tri[0]], P[tri[1]], P[tri[2]]
        # Circumcenter of triangle on sphere
        # Solve for c such that c.p1 = c.p2 = c.p3 and |c|=1
        # c is in plane of p1, p2, p3.
        # c = a*p1 + b*p2 + c*p3
        # System of linear equations:
        # c.p1 = c.p2 => a + b(p2.p1) + c(p3.p1) = 1
        # c.p1 = c.p3 => a(p1.p2) + b + c(p3.p2) = 1 ... wait, simpler:
        
        # Let c = x*p1 + y*p2 + z*p3
        # c.p1 = 1 => x + y(p2.p1) + z(p3.p1) = 1
        # c.p2 = 1 => x(p1.p2) + y + z(p3.p2) = 1
        # c.p3 = 1 => x(p1.p3) + y(p2.p3) + z = 1
        
        p1p2 = np.dot(p1, p2)
        p1p3 = np.dot(p1, p3)
        p2p3 = np.dot(p2, p3)
        
        A = np.array([
            [1.0, p1p2, p1p3],
            [p1p2, 1.0, p2p3],
            [p1p3, p2p3, 1.0]
        ])
        b = np.array([1.0, 1.0, 1.0])
        
        try:
            coeffs = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Degenerate triangle
            radii.append(np.pi)
            continue
            
        c = coeffs[0]*p1 + coeffs[1]*p2 + coeffs[2]*p3
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            radii.append(np.pi)
            continue
        c = c / c_norm
        
        # Angular distance from c to p1
        # cos(r) = c.p1
        cos_r = np.dot(c, p1)
        # Clip for numerical stability
        cos_r = np.clip(cos_r, -1.0, 1.0)
        r = np.arccos(cos_r)
        radii.append(r)
        
    return np.array(radii)

def _covering_radius_surrogate(P, tau=0.1):
    """
    Smooth surrogate for max circumradius.
    """
    radii = _compute_circumradii(P)
    if len(radii) == 0:
        return np.pi, np.zeros((N, 3))
    
    # Log-sum-exp
    max_r = np.max(radii)
    log_sum_exp = max_r + np.log(np.sum(np.exp((radii - max_r) / tau)))
    surrogate = tau * log_sum_exp
    
    # Numerical Gradient
    grad = np.zeros((N, 3))
    eps = 1e-6
    for i in range(N):
        for j in range(3):
            P_plus = P.copy()
            P_plus[i, j] += eps
            P_plus = _project_to_sphere(P_plus)
            f_plus = _covering_radius_surrogate(P_plus, tau)[0]
            
            P_minus = P.copy()
            P_minus[i, j] -= eps
            P_minus = _project_to_sphere(P_minus)
            f_minus = _covering_radius_surrogate(P_minus, tau)[0]
            
            grad[i, j] = (f_plus - f_minus) / (2 * eps)
            
    return surrogate, grad

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # 1. Initialization: Thomson-like repulsion
    # Random start
    np.random.seed(42)
    P = np.random.randn(N, 3)
    P = _project_to_sphere(P)
    
    # Optimize repulsion for uniformity
    # Energy: sum 1/d^2
    def repulsion_energy(P):
        E = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(P[i] - P[j])
                if d < 1e-6:
                    E += 1e10
                else:
                    E += 1.0 / (d * d)
        return E
    
    # Gradient of repulsion energy
    def repulsion_grad(P):
        grad = np.zeros((N, 3))
        for i in range(N):
            for j in range(i+1, N):
                d_vec = P[i] - P[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-6:
                    continue
                # d/dPi (1/d^2) = -2/d^4 * d_vec
                coeff = -2.0 / (d**4)
                grad[i] += coeff * d_vec
                grad[j] -= coeff * d_vec
        return grad
    
    # Riemannian GD for repulsion
    lr = 0.01
    for step in range(200):
        if time.time() > deadline - 60: break
        E = repulsion_energy(P)
        g = repulsion_grad(P)
        P = _riemannian_step(P, g, lr)
        
    # 2. Optimize Covering Radius
    # Use Log-Sum-Exp surrogate
    tau = 0.05
    lr_cov = 0.005
    
    # Reduce tau over time to sharpen
    for step in range(100):
        if time.time() > deadline - 10: break
        
        # Adaptive tau? Start higher, go lower
        current_tau = max(0.01, tau * (1.0 - step/100.0))
        
        try:
            val, g = _covering_radius_surrogate(P, current_tau)
        except Exception:
            break
            
        P = _riemannian_step(P, g, lr_cov)
        
        # Optional: Periodic re-initialization of stuck points?
        # For now, just GD.
        
    # Final polish: Exact max radius
    # We can try a few steps of minimizing the exact max radius using subgradients
    # But the surrogate should be close enough.
    
    return P

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import numpy as np
from math import log, fsum
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from math import log, fsum
import time

N = 150
DEADLINE_FRACTION = 0.85  # safety margin

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e30
            terms.append(-log(d))
    return fsum(terms)

def _energy_and_grad(pts):
    """Compute log energy and gradient on the sphere (tangent-space projected)."""
    n = len(pts)
    # Pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dists = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum_{i<j} -log(d_ij)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dists[iu]))
    
    # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # inv_d2[i,j] = 1/dists[i,j]^2 for i != j
    inv_d2 = np.zeros_like(dists)
    mask = dists > 1e-14
    inv_d2[mask] = 1.0 / (dists[mask] ** 2)
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad_i = sum_j diff[i,j] * inv_d2[i,j]
    grads = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)  # (n, 3)
    
    # Project to tangent space
    for i in range(n):
        g = grads[i]
        p = pts[i]
        grads[i] = g - np.dot(g, p) * p
    
    return energy, grads

def _hessian_matrix(pts):
    """Compute the full Hessian matrix of the log energy on the sphere.
    
    For each pair (i,j), the Hessian contribution to the (i,i) block is:
    H_ii = sum_{j != i} [ (-1/d_ij) * I + (1/d_ij^2 + 1/d_ij) * r_hat_ij r_hat_ij^T ]
    projected to tangent space at x_i.
    
    The (i,j) block for i != j is:
    H_ij = [ (1/d_ij^2 + 1/d_ij) * r_hat_ij r_hat_ij^T ]
    (since f'(d) = -1/d, f''(d) = 1/d^2, and the cross term comes from the product rule)
    
    Wait, let's re-derive carefully.
    E = sum_{i<j} -log(|xi - xj|)
    
    Let r_ij = xi - xj, d_ij = |r_ij|, u_ij = r_ij / d_ij.
    
    Gradient at xi: sum_{j != i} u_ij / d_ij = sum_{j != i} r_ij / d_ij^2
    (This matches the current code)
    
    Hessian:
    d/dt [sum_j u_ij(t) / d_ij(t)] where xi(t) = xi + t*v_i (tangent)
    = sum_j [ (v_i · u_ij) u_ij / d_ij - u_ij (v_i · u_ij) / d_ij ] ... this is getting messy.
    
    Let's use the formula for the Hessian of f(|xi - xj|):
    ∇^2 f(|x|) = f''(|x|) xx^T/|x|^2 + f'(|x|) (I - xx^T/|x|^2)
    
    For f(d) = -log(d):
    f'(d) = -1/d
    f''(d) = 1/d^2
    
    So ∇^2 (-log|xi - xj|) at xi = (1/d^2) u u^T + (-1/d) (I - u u^T)
    = (-1/d) I + (1/d^2 + 1/d) u u^T
    
    This is the contribution to the (i,i) block from pair (i,j).
    
    For the (i,j) block (i != j), we need the cross-derivative:
    d^2/dt1 dt2 [-log|xi(t1) - xj(t2)|]
    = -d/dt2 [ (xi - xj) / |xi - xj|^2 ] · (dxj/dt2)
    = -[ - (xi - xj) / |xi - xj|^3 ] · v_j ... wait.
    
    Let r = xi - xj. ∂/∂xj (r/|r|^2) = ∂/∂xj ((xi - xj)/|xi - xj|^2)
    = -I/|r|^2 + 2r r^T/|r|^4
    = (-1/d^2) I + (2/d^4) r r^T
    
    So the (i,j) block of the Hessian (for i != j) is:
    - [(-1/d^2) I + (2/d^4) r r^T] = (1/d^2) I - (2/d^4) r r^T
    = (1/d^2) (I - 2 u u^T)
    
    But we need to project these to the tangent spaces.
    The Hessian on the sphere is the restriction of the ambient Hessian to the tangent space,
    minus the curvature terms. For the sphere, the second fundamental form is -I (with sign convention),
    so the Hessian on the manifold is:
    Hess_manifold = Proj_i (Hess_ambient) Proj_j - (curvature terms)
    
    Actually, for a function on the sphere, the Hessian in the tangent space is:
    (Hess v)_i = Proj_i (∑_j H_ij v_j)
    where H_ij is the ambient Hessian block, and Proj_i is projection to tangent space at xi.
    But we also need to account for the fact that the points are constrained to the sphere.
    
    A simpler approach: use the fact that the Hessian on the sphere is:
    Hess = Proj (Hess_ambient) Proj
    where Proj is the block-diagonal projection to tangent spaces.
    
    Let's just compute the ambient Hessian and then project.
    """
    n = len(pts)
    # Ambient Hessian: n x n blocks, each 3x3
    H = np.zeros((n * 3, n * 3))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Diagonal block: sum over k != i
                Hii = np.zeros((3, 3))
                for k in range(n):
                    if k == i:
                        continue
                    d = np.linalg.norm(pts[i] - pts[k])
                    if d < 1e-14:
                        continue
                    u = (pts[i] - pts[k]) / d
                    Hii += (-1.0 / d) * np.eye(3) + (1.0 / d**2 + 1.0 / d) * np.outer(u, u)
                H[i*3:(i+1)*3, i*3:(i+1)*3] = Hii
            else:
                # Off-diagonal block (i,j)
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-14:
                    continue
                u = (pts[i] - pts[j]) / d
                Hij = (1.0 / d**2) * (np.eye(3) - 2.0 * np.outer(u, u))
                H[i*3:(i+1)*3, j*3:(j+1)*3] = Hij
                H[j*3:(j+1)*3, i*3:(i+1)*3] = Hij.T
    
    return H

def _tangent_basis(pts):
    """Compute an orthonormal tangent basis for each point on the sphere."""
    n = len(pts)
    bases = []
    for i in range(n):
        p = pts[i]
        # Find a vector not parallel to p
        if abs(p[0]) < 0.9:
            v = np.array([1.0, 0.0, 0.0])
        else:
            v = np.array([0.0, 1.0, 0.0])
        t1 = v - np.dot(v, p) * p
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(p, t1)
        t2 = t2 / np.linalg.norm(t2)
        bases.append(np.column_stack([t1, t2]))
    return np.array(bases)  # shape (n, 3, 2)

def _hessian_tangent(pts):
    """Compute the Hessian of the log energy restricted to the tangent space.
    
    Returns a (2n x 2n) matrix representing the Hessian in the tangent coordinates.
    """
    n = len(pts)
    H_ambient = _hessian_matrix(pts)
    bases = _tangent_basis(pts)  # (n, 3, 2)
    
    # Build the projection matrix: for each point i, the tangent basis is B_i (3x2)
    # The tangent Hessian is: B^T H_ambient B, where B is the block-diagonal matrix of B_i
    H_tan = np.zeros((2 * n, 2 * n))
    for i in range(n):
        for j in range(n):
            # H_tan[i,j] = B_i^T H_ambient[i,j] B_j
            Bi = bases[i]  # 3x2
            Bj = bases[j]  # 3x2
            Hij = H_ambient[i*3:(i+1)*3, j*3:(j+1)*3]  # 3x3
            H_tan[i*2:(i+1)*2, j*2:(j+1)*2] = Bi.T @ Hij @ Bj
    
    return H_tan

def _newton_step(pts, H_tan, grad_tan, max_step=0.1):
    """Take a Newton step in tangent space, with step size limiting."""
    n = len(pts)
    try:
        # Solve H_tan * delta = -grad_tan
        delta = np.linalg.solve(H_tan, -grad_tan)
    except np.linalg.LinAlgError:
        # Fallback to gradient descent if Hessian is singular
        delta = -grad_tan
    
    # Limit step size
    norm_delta = np.linalg.norm(delta)
    if norm_delta > max_step:
        delta = delta * (max_step / norm_delta)
    
    # Convert tangent delta back to ambient space
    bases = _tangent_basis(pts)
    new_pts = np.zeros_like(pts)
    for i in range(n):
        # delta[i] is a 2-vector in tangent coordinates
        ambient_delta = bases[i] @ delta[i*2:(i+1)*2]
        new_pts[i] = pts[i] + ambient_delta
        # Normalize to stay on sphere
        new_pts[i] = new_pts[i] / np.linalg.norm(new_pts[i])
    
    return new_pts

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _spectral_init(n):
    """Initialize using a spectral method: start with a regular grid on the sphere
    and apply a few iterations of a repulsive force to spread points."""
    # Start with a grid of points on the sphere
    n_lat = int(np.ceil(np.sqrt(n)))
    n_lon = int(np.ceil(n / n_lat))
    
    pts = []
    for i in range(n_lat):
        theta = (i + 0.5) * np.pi / n_lat
        for j in range(n_lon):
            if len(pts) >= n:
                break
            phi = (j + 0.5) * 2 * np.pi / n_lon
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            pts.append([x, y, z])
    
    pts = np.array(pts[:n])
    return _normalize(pts)

def _local_optimize_newton(pts, time_budget, maxiter=50):
    """Local optimization using Newton's method on the sphere."""
    best_pts = _normalize(pts.copy())
    best_energy, _ = _energy_and_grad(best_pts)
    
    for it in range(maxiter):
        if time.time() > time_budget:
            break
        
        # Compute energy and gradient
        energy, grad_ambient = _energy_and_grad(best_pts)
        if energy < best_energy:
            best_energy = energy
        
        # Convert gradient to tangent coordinates
        bases = _tangent_basis(best_pts)
        grad_tan = np.zeros(2 * N)
        for i in range(N):
            grad_tan[i*2:(i+1)*2] = bases[i].T @ grad_ambient[i]
        
        # Check convergence
        if np.linalg.norm(grad_tan) < 1e-10:
            break
        
        # Compute Hessian in tangent space
        H_tan = _hessian_tangent(best_pts)
        
        # Take Newton step
        new_pts = _newton_step(best_pts, H_tan, grad_tan)
        
        # Check if the step improved the energy
        new_energy, _ = _energy_and_grad(new_pts)
        if new_energy < energy:
            best_pts = new_pts
        else:
            # If Newton step doesn't improve, try a smaller step
            new_pts_half = _newton_step(best_pts, H_tan, grad_tan, max_step=0.01)
            new_energy_half, _ = _energy_and_grad(new_pts_half)
            if new_energy_half < energy:
                best_pts = new_pts_half
            else:
                # Fallback to gradient descent
                step_size = 0.01
                grad_ambient_scaled = grad_ambient * step_size
                new_pts_grad = best_pts - grad_ambient_scaled
                new_pts_grad = _normalize(new_pts_grad)
                new_energy_grad, _ = _energy_and_grad(new_pts_grad)
                if new_energy_grad < energy:
                    best_pts = new_pts_grad
                else:
                    break
    
    return best_pts, best_energy

def _random_points(n, rng):
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)

def construct():
    start_time = time.time()
    total_time = 150.0
    deadline = start_time + total_time * DEADLINE_FRACTION
    
    rng = np.random.default_rng(12345)
    
    # Start with multiple diverse initializations
    inits = [
        _fibonacci_spiral(N),
        _spectral_init(N),
        _random_points(N, rng),
    ]
    
    best_pts = inits[0]
    best_energy = _log_energy(best_pts)
    
    for init_pts in inits:
        if time.time() >= deadline:
            break
        
        # Local optimization using Newton's method
        pts, energy = _local_optimize_newton(init_pts, deadline)
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Additional restarts with random perturbations
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() >= deadline:
            break
        
        # Perturb the best points
        perturbed = best_pts + rng.standard_normal((N, 3)) * 0.05
        perturbed = _normalize(perturbed)
        
        pts, energy = _local_optimize_newton(perturbed, deadline, maxiter=30)
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 10:
        final_pts, final_energy = _local_optimize_newton(best_pts, deadline, maxiter=100)
        if final_energy < best_energy:
            best_pts = final_pts
            best_energy = final_energy
    
    return best_pts
# EVOLVE-BLOCK-END

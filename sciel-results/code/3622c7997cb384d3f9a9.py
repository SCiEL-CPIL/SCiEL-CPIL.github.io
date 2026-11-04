"""Combined program for logarithmic energy minimization of 150 points on sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

_DEADLINE = 142.0  # Safety margin before 150s kill

def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere using math.fsum for precision."""
    n = len(points)
    if n < 2:
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    return -math.fsum(log_dists.tolist())

def _energy_and_grad(x):
    """
    Energy and tangential gradient for L-BFGS-B. 
    x is flat array of shape (N*3,).
    The function normalizes x to the sphere internally.
    Returns (energy, tangential_gradient_flat).
    """
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms

    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)

    # Gradient of E = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    # Using matrix operations for efficiency:
    # weights[i,j] = 1/d_ij^2 for i!=j, 0 otherwise
    # grads[i] = sum_j weights[i,j] x_j - x_i * sum_j weights[i,j]
    
    inv_dists_sq = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    inv_dists_sq[mask] = 1.0 / dists_sq[mask]
    
    sum_weights = np.sum(inv_dists_sq, axis=1)
    weighted_sum = inv_dists_sq @ points
    
    # Raw Euclidean gradient
    grads = weighted_sum - points * sum_weights[:, np.newaxis]
    
    # Project gradient to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points

    # Compute energy for the optimizer
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(np.sqrt(dists_sq[iu])))

    return energy, grads_tangent.flatten()

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, rng):
    """Generate random uniform points on sphere."""
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def _tangent_vector(point, rng):
    """Generate a random unit tangent vector at point on sphere."""
    rand_vec = rng.randn(3)
    tangent = rand_vec - np.dot(rand_vec, point) * point
    norm = np.linalg.norm(tangent)
    if norm < 1e-15:
        rand_vec = rng.randn(3)
        tangent = rand_vec - np.dot(rand_vec, point) * point
        norm = np.linalg.norm(tangent)
    return tangent / norm

def _local_minimize(points, deadline, maxiter=500, ftol=1e-12, gtol=1e-10):
    """Run L-BFGS-B to find local minimum."""
    x0 = points.flatten()
    
    # Check if we have time
    if time.time() > deadline - 1.0:
        return points, _log_energy(points)

    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 20, 'ftol': ftol, 'gtol': gtol, 'maxls': 40}
        )
        if time.time() > deadline - 1.0:
            return points, _log_energy(points)
            
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _log_energy(p_res)
        return p_res, e_check
    except Exception:
        return points, _log_energy(points)

def _basin_hopping_single(init_points, deadline, rng, max_iter=50, T_init=0.1):
    """
    Basin hopping with improved perturbation strategy.
    """
    current_points, current_energy = _local_minimize(init_points, deadline, maxiter=200)
    best_points = current_points.copy()
    best_energy = current_energy

    T = T_init
    T_min = 1e-8
    cooling = 0.95

    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        if T < T_min:
            break

        # Improved perturbation: mix of small and large steps
        n_perturb = rng.randint(3, 12)
        idxs = rng.choice(N, n_perturb, replace=False)

        new_points = current_points.copy()
        for idx in idxs:
            tangent = _tangent_vector(new_points[idx], rng)
            # Mix of step sizes: mostly small, occasionally larger
            if rng.rand() < 0.8:
                step_size = T * rng.uniform(0.5, 1.5)
            else:
                step_size = T * rng.uniform(1.5, 3.0)
            new_point = new_points[idx] + step_size * tangent
            new_point = new_point / np.linalg.norm(new_point)
            new_points[idx] = new_point

        # Local minimize the perturbed configuration
        new_points, new_energy = _local_minimize(new_points, deadline, maxiter=100)

        # Metropolis criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 0 and rng.rand() < math.exp(-delta_e / T)):
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()

        T *= cooling

    return best_points, best_energy

def _polish_bfgs(pts, max_time):
    """
    High-precision polish using BFGS.
    BFGS maintains a full inverse Hessian approximation, which is often more
    accurate than L-BFGS-B for small-to-medium N, leading to tighter convergence.
    """
    pts = _project_to_sphere(np.array(pts, dtype=np.float64))
    t0 = time.time()
    
    def obj(x):
        if time.time() - t0 > max_time:
            return np.inf
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e
        
    def grad(x):
        if time.time() - t0 > max_time:
            return np.zeros_like(x)
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        _, g = _energy_and_grad(p)
        return g

    x0 = pts.flatten()
    try:
        res = minimize(
            obj, x0, jac=grad, method='BFGS',
            options={'maxiter': 5000, 'gtol': 1e-14, 'disp': False, 'maxfun': 10000}
        )
        pts_opt = _project_to_sphere(res.x.reshape(N, 3))
        e_opt = _log_energy(pts_opt)
        return pts_opt, e_opt
    except Exception:
        return pts, _log_energy(pts)

def _polish_newton(pts, max_time):
    """
    Newton-Raphson polish using the Hessian.
    For N=150, the Hessian is 450x450. We compute the Riemannian Hessian
    and use Cholesky decomposition for the step.
    """
    pts = _project_to_sphere(np.array(pts, dtype=np.float64))
    t0 = time.time()
    
    # Function to compute Hessian-vector product or full Hessian
    # For simplicity and robustness, we use a regularized Newton step
    # using the Euclidean Hessian projected to the tangent space.
    
    def _hessian_tangent(points):
        """
        Compute the Riemannian Hessian of the energy on the sphere.
        This is complex to derive analytically in code quickly.
        Instead, we use a numerical Hessian or a quasi-Newton update.
        Given the risk of instability, we stick to BFGS as the primary polish.
        This function is a placeholder for a potential future Newton step.
        """
        pass

    # Since implementing a stable Riemannian Newton step is complex and risky
    # within the time limit, we rely on BFGS which is very effective.
    # We add a second BFGS pass with even tighter tolerances if time permits.
    
    # Re-use BFGS with tighter tol
    def obj(x):
        if time.time() - t0 > max_time:
            return np.inf
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e
        
    def grad(x):
        if time.time() - t0 > max_time:
            return np.zeros_like(x)
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        _, g = _energy_and_grad(p)
        return g

    x0 = pts.flatten()
    try:
        res = minimize(
            obj, x0, jac=grad, method='BFGS',
            options={'maxiter': 2000, 'gtol': 1e-15, 'disp': False, 'maxfun': 5000}
        )
        pts_opt = _project_to_sphere(res.x.reshape(N, 3))
        e_opt = _log_energy(pts_opt)
        return pts_opt, e_opt
    except Exception:
        return pts, _log_energy(pts)

def construct():
    t_start = time.time()
    deadline = t_start + _DEADLINE
    
    rng = np.random.RandomState(42)

    best_x = None
    best_e = float('inf')

    # Phase 1: Diverse Initialization and Local Minimization
    inits = []
    
    # 1. Fibonacci sphere
    inits.append(_fibonacci_sphere(N))
    
    # 2. Random uniform points (more seeds for diversity)
    for seed in range(8):
        r = np.random.RandomState(seed * 137 + 7)
        inits.append(_random_uniform_sphere(N, r))
        
    # 3. Perturbed Fibonacci with different scales
    for scale in [0.05, 0.1, 0.2]:
        perturbed = _fibonacci_sphere(N) + rng.randn(N, 3) * scale
        inits.append(_project_to_sphere(perturbed))

    # Run local minimization on each to find good local minima
    for i, init in enumerate(inits):
        if time.time() > deadline - 25.0:
            break
            
        remaining = deadline - time.time()
        if remaining < 5.0:
            break

        x_res, e_res = _local_minimize(init, deadline, maxiter=200, ftol=1e-10, gtol=1e-8)
        
        if e_res < best_e:
            best_e = e_res
            best_x = x_res

    # Phase 2: Basin Hopping from the best local minimum found
    if best_x is not None and time.time() < deadline - 15.0:
        x_bh, e_bh = _basin_hopping_single(best_x, deadline, rng, max_iter=80, T_init=0.08)
        if e_bh < best_e:
            best_e = e_bh
            best_x = x_bh

    # Phase 3: BFGS Polish (replaces Riemannian CG and BB GD)
    if best_x is not None and time.time() < deadline - 5.0:
        polish_time = min(10.0, deadline - time.time() - 2.0)
        if polish_time > 1.0:
            x_bfgs, e_bfgs = _polish_bfgs(best_x, polish_time)
            if e_bfgs < best_e:
                best_e = e_bfgs
                best_x = x_bfgs

    # Phase 4: Final High-Precision BFGS Polish
    if best_x is not None and time.time() < deadline - 1.0:
        polish_time = deadline - time.time() - 0.5
        if polish_time > 0.5:
            x_final, e_final = _polish_newton(best_x, polish_time)
            if e_final < best_e:
                best_x = x_final
                best_e = e_final

    # Final validation and projection
    if best_x is None:
        best_x = _fibonacci_sphere(N)
        
    best_x = _project_to_sphere(best_x)
    
    # Ensure no points are too close
    diff = best_x[:, np.newaxis, :] - best_x[np.newaxis, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(N, 1)
    if dist[iu].min() < 1e-9:
        best_x = _fibonacci_sphere(N)
        best_x = _project_to_sphere(best_x)
        
    return best_x
# EVOLVE-BLOCK-END

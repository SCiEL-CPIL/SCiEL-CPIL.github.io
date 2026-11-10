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
DEADLINE_FRACTION = 0.82  # safety margin

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

def _hessian_2vec(pts, vec):
    """Compute Hessian-vector product for the log energy on the sphere.
    
    H_ij (as a bilinear form on tangent vectors) for the log energy:
    The Hessian of -log|xi - xj| at point xi in direction u_i (tangent) is:
    d2/dt2 [-log|xi + t*u_i - xj|] at t=0
    
    For a single pair (i,j) with d = |xi - xj|, r = xi - xj:
    The Hessian quadratic form in direction u (at xi) is:
    (u·r)^2 / d^4 - |u|^2 / d^2  ... wait, let me derive properly.
    
    f(d) = -log(d), d = |r|
    df/dd = -1/d
    d^2f/dd^2 = 1/d^2
    
    Hessian of f(|r|) in direction u:
    = f''(|r|) * (u·r_hat)^2 + f'(|r|) * (div of gradient in direction u)
    
    Actually, for g(x) = f(|x - a|):
    ∇g = f'(|x-a|) * (x-a)/|x-a|
    Hessian g = f''(|x-a|) * (x-a)(x-a)^T/|x-a|^2 + f'(|x-a|) * (I - (x-a)(x-a)^T/|x-a|^2)
    
    So Hessian quadratic form u^T H u = f''(d) * (u·r_hat)^2 + f'(d) * (|u|^2 - (u·r_hat)^2)
    = f'(d) * |u|^2 + (f''(d) - f'(d)) * (u·r_hat)^2
    
    With f(d) = -log(d): f'(d) = -1/d, f''(d) = 1/d^2
    So: u^T H u = (-1/d) * |u|^2 + (1/d^2 + 1/d) * (u·r_hat)^2
    
    For the full energy E = sum_{i<j} -log|xi-xj|:
    The Hessian at point i in direction u_i (tangent to sphere at xi) gets contributions
    from all j != i.
    
    For the full Hessian-vector product, we need:
    (H * v)_i = sum_{j != i} [ (-1/d_ij) * v_i + (1/d_ij^2 + 1/d_ij) * (v_i · r_hat_ij) * r_hat_ij ]
    
    But we also need to account for the fact that moving point i affects the energy
    through all pairs involving i. The gradient we computed is already the sum over j.
    
    Actually, for Hessian-vector product where v is a collection of tangent vectors v_i:
    The second derivative of E in direction v:
    d2E/dt2 = sum_{i<j} d2/dt2 [-log|xi(t) - xj(t)|]
    
    For pair (i,j), with ri = xi - xj, di = |ri|, ri_hat = ri/di:
    If we move xi in direction ui and xj in direction uj (both tangent):
    The relative velocity is (ui - uj).
    d/dt |ri(t)| = (ri_hat) · (ui - uj)
    d2/dt2 |ri(t)| = (ui - uj)·(ui - uj) - ((ri_hat)·(ui - uj))^2  [on the sphere, but for small t]
    
    Actually this is getting complex. Let me use a simpler approach:
    Use a conjugate gradient or L-BFGS with the gradient we have, but do multiple
    diverse restarts with basin-hopping style moves.
    """
    pass  # We'll use L-BFGS with good gradient instead of full Hessian

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

def _random_points(n, rng):
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)

def _local_optimize(pts, time_budget, maxiter=300):
    """Local optimization using L-BFGS-B with gradient on the sphere."""
    best_pts = _normalize(pts.copy())
    best_energy, _ = _energy_and_grad(best_pts)
    
    def objective_and_grad(x):
        p = _normalize(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()
    
    def objective(x):
        e, _ = objective_and_grad(x)
        return e
    
    def grad(x):
        _, g = objective_and_grad(x)
        return g
    
    try:
        result = minimize(
            objective, best_pts.flatten(), method='L-BFGS-B', jac=grad,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 40}
        )
        new_pts = _normalize(result.x.reshape(N, 3))
        new_energy, _ = _energy_and_grad(new_pts)
        if new_energy < best_energy:
            best_pts = new_pts
            best_energy = new_energy
    except Exception:
        pass
    
    return best_pts, best_energy

def _basin_hopping_move(pts, rng, step_size=0.1):
    """Random perturbation for basin hopping: move each point by a random tangent vector."""
    n = len(pts)
    new_pts = np.zeros_like(pts)
    for i in range(n):
        # Generate random tangent vector at pts[i]
        rand_vec = rng.standard_normal(3)
        tangent = rand_vec - np.dot(rand_vec, pts[i]) * pts[i]
        tan_norm = np.linalg.norm(tangent)
        if tan_norm < 1e-15:
            tangent = np.array([1.0, 0.0, 0.0]) - np.dot([1.0, 0.0, 0.0], pts[i]) * pts[i]
            tan_norm = np.linalg.norm(tangent)
        tangent = tangent / tan_norm
        # Move by step_size in random tangent direction
        new_pts[i] = pts[i] + step_size * tangent
    return _normalize(new_pts)

def construct():
    start_time = time.time()
    total_time = 150.0
    deadline = start_time + total_time * DEADLINE_FRACTION
    
    rng = np.random.default_rng(12345)
    
    # Start with Fibonacci spiral and optimize
    best_pts = _fibonacci_spiral(N)
    best_energy, _ = _energy_and_grad(best_pts)
    
    # Initial local optimization
    if time.time() < deadline:
        pts, energy = _local_optimize(best_pts, deadline)
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Basin-hopping style global search
    # Multiple diverse initializations + basin hopping
    n_restarts = 15
    for restart in range(n_restarts):
        if time.time() >= deadline:
            break
        
        # Diverse initialization
        if restart % 4 == 0:
            init_pts = _fibonacci_spiral(N)
        elif restart % 4 == 1:
            init_pts = _random_points(N, rng)
        elif restart % 4 == 2:
            # Spherical cap sampling
            init_pts = np.zeros((N, 3))
            for i in range(N):
                # Sample uniformly on sphere
                z = rng.uniform(-1, 1)
                theta = rng.uniform(0, 2 * np.pi)
                r = np.sqrt(1 - z*z)
                init_pts[i] = [r * np.cos(theta), r * np.sin(theta), z]
        else:
            # Perturbed Fibonacci
            init_pts = _fibonacci_spiral(N)
            init_pts = init_pts + rng.standard_normal((N, 3)) * 0.1
            init_pts = _normalize(init_pts)
        
        # Local optimization from this start
        pts, energy = _local_optimize(init_pts, deadline)
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
        
        # Basin hopping: several random moves + local re-optimization
        n_bh_steps = 5
        bh_step_size = 0.05
        for bh_step in range(n_bh_steps):
            if time.time() >= deadline:
                break
            
            # Random move
            moved_pts = _basin_hopping_move(pts, rng, bh_step_size)
            
            # Local optimization from moved point
            new_pts, new_energy = _local_optimize(moved_pts, deadline, maxiter=100)
            
            if new_energy < energy:
                pts = new_pts
                energy = new_energy
                if energy < best_energy:
                    best_pts = pts
                    best_energy = energy
            else:
                # Accept with some probability (simulated annealing style)
                # But for simplicity, just reject and try next move
                pass
    
    # Final polish: one more careful local optimization
    if time.time() < deadline - 5:
        final_pts, final_energy = _local_optimize(best_pts, deadline, maxiter=500)
        if final_energy < best_energy:
            best_pts = final_pts
            best_energy = final_energy
    
    return best_pts
# EVOLVE-BLOCK-END

"""Optimize 150 points on unit sphere to minimize logarithmic energy."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Compute log energy sum_{i<j} log(1/d) and its gradient.
    
    dE/dx_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)

    mask = dist_sq > 1e-30
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    np.fill_diagonal(inv_dist_sq, 0.0)

    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)

    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = d[d > 1e-15]
    if len(d) == 0:
        energy = 0.0
    else:
        energy = math.fsum(np.log(1.0 / d).tolist())

    return energy, grad


def _energy_only(pts):
    """Compute only the energy, faster when gradient not needed."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = d[d > 1e-15]
    if len(d) == 0:
        return 0.0
    return math.fsum(np.log(1.0 / d).tolist())


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_sphere(n, seed=None):
    """Generate Fibonacci spiral points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
        offset = rng.uniform(0, 1)
    else:
        offset = 0.0
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _random_sphere(n, seed):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)


def _icosahedron_based(n):
    """Generate points based on icosahedron vertices, then subdivide."""
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Add more points by perturbing
    rng = np.random.RandomState(42)
    pts = verts.copy()
    while len(pts) < n:
        idx = rng.randint(0, len(pts))
        new_pt = pts[idx] + rng.randn(3) * 0.3
        new_pt = new_pt / np.linalg.norm(new_pt)
        pts = np.vstack([pts, new_pt])
    
    return _project_to_sphere(pts[:n])


def _lbfgs_optimize(pts, time_limit=5.0):
    """Run L-BFGS-B optimization with time limit."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())

    def objective_flat(x):
        if time.time() - start_time > time_limit:
            return np.inf
        p = _project_to_sphere(x.reshape(-1, 3))
        e = _energy_only(p)
        return e

    def grad_flat(x):
        if time.time() - start_time > time_limit:
            return np.zeros(len(x))
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()

    x0 = pts.flatten()

    result = minimize(
        objective_flat,
        x0,
        jac=grad_flat,
        method='L-BFGS-B',
        options={
            'maxiter': 15000,
            'maxfun': 40000,
            'ftol': 1e-16,
            'gtol': 1e-14,
            'maxls': 200
        }
    )

    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    energy = _energy_only(final_pts)
    return final_pts, energy


def _adam_optimize(pts, time_limit=10.0):
    """Adam optimizer on the sphere with correct gradient."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    lr = 0.05
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0

    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    while time.time() - start_time < time_limit:
        t += 1
        energy, grad = _energy_and_grad(pts)

        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        m_tangent = m_hat - np.sum(m_hat * pts, axis=1, keepdims=True) * pts

        step = lr * m_tangent / (np.sqrt(v_hat) + eps)
        pts = pts - step
        pts = _project_to_sphere(pts)

        if t % 200 == 0 and np.linalg.norm(grad_tangent) < 1e-11:
            break

    return best_pts, best_energy


def _simulated_annealing(pts, time_limit=15.0, seed=42):
    """Simulated annealing to escape local minima."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    current_energy = _energy_only(pts)
    best_pts = pts.copy()
    best_energy = current_energy

    T = 0.05
    T_min = 1e-7
    cooling = 0.998
    max_steps = 20000

    rng = np.random.RandomState(seed)

    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break

        if rng.rand() < 0.05:
            noise_scale = T * 0.5
        else:
            noise_scale = T * 0.1
        perturbation = rng.randn(n, 3) * noise_scale
        new_pts = pts + perturbation
        new_pts = _project_to_sphere(new_pts)

        new_energy = _energy_only(new_pts)

        if new_energy < current_energy or (T > T_min and rng.rand() < math.exp(-(new_energy - current_energy) / T)):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()

        T *= cooling
        if T < T_min:
            break

    return best_pts, best_energy


def _final_polish(pts, time_limit=5.0):
    """High-precision final polish with tight tolerances."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())

    def objective_flat(x):
        if time.time() - start_time > time_limit:
            return np.inf
        p = _project_to_sphere(x.reshape(-1, 3))
        return _energy_only(p)

    def grad_flat(x):
        if time.time() - start_time > time_limit:
            return np.zeros(len(x))
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()

    x0 = pts.flatten()

    for pass_num in range(4):
        if time.time() - start_time > time_limit:
            break
        tol_factor = 10 ** (-pass_num)
        result = minimize(
            objective_flat,
            x0,
            jac=grad_flat,
            method='L-BFGS-B',
            options={
                'maxiter': 20000,
                'maxfun': 50000,
                'ftol': 1e-16 * tol_factor,
                'gtol': 1e-14 * tol_factor,
                'maxls': 300
            }
        )
        x0 = _project_to_sphere(result.x.reshape(-1, 3)).flatten()

    final_pts = _project_to_sphere(x0.reshape(-1, 3))
    energy = _energy_only(final_pts)
    return final_pts, energy


def _velocity_verlet(pts, time_limit=8.0, seed=42):
    """Velocity Verlet dynamics on the sphere."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)
    rng = np.random.RandomState(seed)
    
    # Initialize velocities
    vel = rng.randn(n, 3) * 0.01
    vel = vel - np.sum(vel * pts, axis=1, keepdims=True) * pts  # Project to tangent
    
    dt = 0.01
    damping = 0.999
    
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    for step in range(50000):
        if time.time() - start_time > time_limit:
            break
        
        _, grad = _energy_and_grad(pts)
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Half-step velocity
        vel = vel + 0.5 * dt * grad_tangent
        vel = vel * damping
        
        # Full-step position
        pts_new = pts + dt * vel
        pts_new = _project_to_sphere(pts_new)
        
        # Recompute gradient at new position
        _, grad_new = _energy_and_grad(pts_new)
        grad_new_tangent = grad_new - np.sum(grad_new * pts_new, axis=1, keepdims=True) * pts_new
        
        # Half-step velocity again
        vel = vel + 0.5 * dt * grad_new_tangent
        vel = vel * damping
        vel = vel - np.sum(vel * pts_new, axis=1, keepdims=True) * pts_new
        
        pts = pts_new
        
        energy = _energy_only(pts)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Adaptive dt: reduce if energy increases significantly
        if step % 100 == 0 and energy > best_energy + 0.1:
            dt *= 0.9
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = 145.0

    best_pts = None
    best_energy = np.inf

    # Generate diverse starting points
    starts = []
    starts.append(_fibonacci_sphere(N))
    starts.append(_fibonacci_sphere(N, seed=1))
    starts.append(_fibonacci_sphere(N, seed=7))
    starts.append(_fibonacci_sphere(N, seed=13))
    starts.append(_fibonacci_sphere(N, seed=42))
    starts.append(_icosahedron_based(N))
    for s in range(2, 10):
        starts.append(_random_sphere(N, seed=s * 100))
    starts.append(_random_sphere(N, seed=999))
    starts.append(_random_sphere(N, seed=12345))

    # Phase 1: L-BFGS-B from each diverse start
    for i, start_pts in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 25:
            break
        time_limit = min(6.0, deadline - elapsed - 20)
        if time_limit < 2.0:
            break
        candidate_pts, candidate_energy = _lbfgs_optimize(start_pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_energy = candidate_energy
            best_pts = candidate_pts.copy()

    # Phase 2: Velocity Verlet from best point
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 30:
            time_limit = min(8.0, deadline - elapsed - 25)
            candidate_pts, candidate_energy = _velocity_verlet(best_pts, time_limit=time_limit, seed=42)
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Phase 3: Adam optimization from best point
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 25:
            time_limit = min(8.0, deadline - elapsed - 20)
            candidate_pts, candidate_energy = _adam_optimize(best_pts, time_limit=time_limit)
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Phase 4: Simulated annealing to escape local minima
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 25:
            time_limit = min(15.0, deadline - elapsed - 20)
            candidate_pts, candidate_energy = _simulated_annealing(
                best_pts, time_limit=time_limit, seed=42
            )
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Phase 5: Another SA round with different seed
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 20:
            time_limit = min(12.0, deadline - elapsed - 15)
            candidate_pts, candidate_energy = _simulated_annealing(
                best_pts, time_limit=time_limit, seed=123
            )
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Phase 6: Third SA round
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 15:
            time_limit = min(10.0, deadline - elapsed - 12)
            candidate_pts, candidate_energy = _simulated_annealing(
                best_pts, time_limit=time_limit, seed=777
            )
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Phase 7: Velocity Verlet again after SA
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 15:
            time_limit = min(8.0, deadline - elapsed - 12)
            candidate_pts, candidate_energy = _velocity_verlet(best_pts, time_limit=time_limit, seed=999)
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Phase 8: Final L-BFGS-B polish
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 12:
            time_limit = min(8.0, deadline - elapsed - 8)
            candidate_pts, candidate_energy = _lbfgs_optimize(best_pts, time_limit=time_limit)
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Phase 9: High-precision final polish
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 6:
            time_limit = min(5.0, deadline - elapsed - 3)
            candidate_pts, candidate_energy = _final_polish(best_pts, time_limit=time_limit)
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Phase 10: Ultra-precision final polish
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 2:
            time_limit = min(3.0, deadline - elapsed - 1)
            candidate_pts, candidate_energy = _final_polish(best_pts, time_limit=time_limit)
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Final projection and validation
    best_pts = _project_to_sphere(best_pts)

    # Validate: check for very close points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)

    if min_dist < 1e-9:
        return _fibonacci_sphere(N)

    return best_pts
# EVOLVE-BLOCK-END

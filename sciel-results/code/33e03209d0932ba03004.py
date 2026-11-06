"""Combined program: MODEL-AND-OPTIMIZE with Nesterov and Ultra-Precise Polish."""
import numpy as np
import math
import time

N = 100


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    if np.any(d_upper < 1e-9):
        return 1e18
    if not np.all(np.isfinite(d_upper)):
        return 1e18
    return math.fsum(np.log(1.0 / d_upper).tolist())


def _log_energy_grad(pts):
    """Compute logarithmic energy and gradient on sphere (vectorized)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)

    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    if np.any(d_upper < 1e-12):
        return 1e18, np.zeros_like(pts)

    energy = -np.sum(np.log(d_upper))

    inv_d2 = np.zeros_like(d2)
    mask_non_diag = ~np.eye(n, dtype=bool)
    inv_d2[mask_non_diag] = 1.0 / d2[mask_non_diag]

    grad = -np.sum(inv_d2[:, :, np.newaxis] * diff, axis=1)

    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad -= radial * pts

    return energy, grad


def _fibonacci_sphere(n, offset=0.5):
    """Generate Fibonacci spiral points on unit sphere."""
    golden_ratio = 0.5 * (1 + np.sqrt(5))
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * golden_ratio * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _so3_random_rotation(rng):
    """Generate a random SO(3) rotation matrix."""
    q = rng.randn(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _optimize_lbfgs(pts0, deadline, maxiter=2000, ftol=1e-14, gtol=1e-10):
    """Optimize using L-BFGS-B with sphere projection."""
    from scipy.optimize import minimize

    x0 = pts0.flatten()

    def objective(x):
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms
        e, _ = _log_energy_grad(pts)
        return e

    def grad(x):
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms
        _, g = _log_energy_grad(pts)
        return g.flatten()

    result = minimize(objective, x0, jac=grad, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'maxfun': maxiter * 4,
                               'ftol': ftol, 'gtol': gtol, 'm': 50})

    if result.success or result.fun < objective(x0):
        best_pts = _project_to_sphere(result.x.reshape(-1, 3))
        best_energy = _log_energy(best_pts)
        return best_pts, best_energy
    else:
        return pts0, _log_energy(pts0)


def _optimize_nesterov(pts0, deadline, max_iter=10000):
    """
    Nesterov accelerated gradient descent with adaptive momentum.
    This method is often more effective than standard GD in escaping shallow minima.
    """
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)

    v = np.zeros_like(pts)
    lr = 0.005
    momentum = 0.92
    no_improve_count = 0
    best_lr = lr

    for it in range(max_iter):
        if time.time() > deadline:
            break

        e, grad = _log_energy_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
            no_improve_count = 0
            best_lr = lr
        else:
            no_improve_count += 1

        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-14:
            break

        # Nesterov lookahead
        v_new = momentum * v - lr * grad
        pts_lookahead = pts + v_new
        pts_lookahead = _project_to_sphere(pts_lookahead)

        _, grad_lookahead = _log_energy_grad(pts_lookahead)

        # Update velocity
        v = momentum * v - lr * grad_lookahead
        pts_new = pts + v
        pts_new = _project_to_sphere(pts_new)

        e_new = _log_energy(pts_new)
        if e_new < e:
            pts = pts_new
            # Adaptive learning rate
            if no_improve_count == 0:
                lr = min(lr * 1.02, 0.01)
        else:
            lr *= 0.7
            v *= 0.3
            if no_improve_count > 50:
                lr = best_lr * 0.5
                v = np.zeros_like(pts)
                no_improve_count = 0
            if lr < 1e-15:
                break

    return best_pts, best_energy


def _molecular_dynamics(pts0, deadline, n_steps=20000, dt=0.005, T0=0.5, T1=0.001, seed=None):
    """
    Molecular dynamics with velocity Verlet integration and adaptive cooling.
    Uses physical dynamics with inertia and thermal fluctuations to escape local minima.
    """
    rng = np.random.RandomState(seed)
    pts = pts0.copy()
    vel = np.zeros_like(pts)
    
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Compute initial forces
    _, grad = _log_energy_grad(pts)
    forces = -grad  # Force is negative gradient
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Adaptive temperature (exponential cooling)
        T = T0 * (T1 / T0) ** (step / n_steps)
        
        # Velocity Verlet step 1: update position by half step
        pts_half = pts + vel * dt + 0.5 * forces * dt * dt
        pts_half = _project_to_sphere(pts_half)
        
        # Compute forces at new position
        _, grad_new = _log_energy_grad(pts_half)
        forces_new = -grad_new
        
        # Velocity Verlet step 2: update velocity
        vel_new = 0.5 * (forces + forces_new) * dt
        
        # Add thermal noise (Langevin-like)
        thermal_noise = rng.randn(N, 3) * math.sqrt(2 * T * dt)
        vel_new += thermal_noise
        
        # Damping to remove excess energy
        damping = 0.98
        vel_new *= damping
        
        # Update position
        pts_new = pts_half + vel_new * dt
        pts_new = _project_to_sphere(pts_new)
        
        # Update state
        vel = vel_new
        forces = forces_new
        pts = pts_new
        
        # Track best
        e = _log_energy(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
    
    return best_pts, best_energy


def _ultra_precise_polish(pts0, deadline):
    """
    Final polishing phase using very tight tolerances and small steps.
    Combines L-BFGS with a final Nesterov pass at low learning rate.
    """
    # First, a high-precision L-BFGS
    pts, e = _optimize_lbfgs(pts0, deadline, maxiter=3000, ftol=1e-16, gtol=1e-12)
    
    # Then, a gentle Nesterov pass to squeeze out any remaining error
    if time.time() < deadline - 1:
        pts, e = _optimize_nesterov(pts, deadline, max_iter=5000)
        
    return pts, e


def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_pts = None
    best_energy = np.inf

    # Generate diverse starting configurations
    starts = []
    # Fibonacci spirals with different offsets
    for offset in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        starts.append(_fibonacci_sphere(N, offset=offset))

    # Random starts with various seeds
    for seed in [0, 42, 123, 456, 789, 1000, 2024, 3141, 2718, 1618]:
        starts.append(_random_sphere(N, seed=seed))

    # Rotated Fibonacci for diversity
    rng = np.random.RandomState(42)
    for _ in range(5):
        base = _fibonacci_sphere(N, offset=0.5)
        R = _so3_random_rotation(rng)
        starts.append(base @ R.T)

    # Phase 1: Quick L-BFGS on all starts
    candidates = []
    for i, pts0 in enumerate(starts):
        if time.time() > deadline - 80:
            break
        pts_opt, e_opt = _optimize_lbfgs(pts0, deadline - 80, maxiter=500)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
        candidates.append((e_opt, pts_opt))

    # Sort and keep top candidates
    candidates.sort(key=lambda x: x[0])

    # Phase 2: Nesterov optimization on top candidates
    for e_cand, pts_cand in candidates[:8]:
        if time.time() > deadline - 40:
            break
        pts_nest, e_nest = _optimize_nesterov(pts_cand, deadline - 40, max_iter=5000)
        if e_nest < best_energy:
            best_energy = e_nest
            best_pts = pts_nest

    # Phase 3: Molecular dynamics to escape local minima
    if best_pts is not None and time.time() < deadline - 20:
        md_deadline = deadline - 15
        pts_md, e_md = _molecular_dynamics(best_pts, md_deadline, n_steps=15000, 
                                            dt=0.005, T0=0.3, T1=0.001, seed=42)
        if e_md < best_energy:
            best_energy = e_md
            best_pts = pts_md
        
        # Re-optimize after MD with L-BFGS
        if time.time() < deadline - 10:
            pts_reopt, e_reopt = _optimize_lbfgs(pts_md, deadline - 10, maxiter=2000)
            if e_reopt < best_energy:
                best_energy = e_reopt
                best_pts = pts_reopt

    # Phase 4: Ultra-precise polish
    if best_pts is not None and time.time() < deadline - 5:
        pts_polish, e_polish = _ultra_precise_polish(best_pts, deadline - 1)
        if e_polish < best_energy:
            best_energy = e_polish
            best_pts = pts_polish

    # Validate final result
    if best_pts is None:
        best_pts = _fibonacci_sphere(N, offset=0.5)

    # Ensure all points are on unit sphere and valid
    norms = np.linalg.norm(best_pts, axis=1)
    best_pts = best_pts / norms[:, np.newaxis]

    # Check for near-duplicate points
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    if min_dist < 1e-9:
        # Perturb and re-optimize
        rng = np.random.RandomState(12345)
        best_pts = best_pts + rng.randn(N, 3) * 1e-6
        best_pts = _project_to_sphere(best_pts)
        if time.time() < deadline - 1:
            best_pts, _ = _optimize_lbfgs(best_pts, deadline - 0.5, maxiter=500)

    return best_pts

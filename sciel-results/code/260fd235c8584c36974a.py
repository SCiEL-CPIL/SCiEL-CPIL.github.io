"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize as scipy_minimize


def _energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -math.fsum(np.log(dist[iu]))


def _energy_and_grad(points):
    """Compute energy and gradient on the sphere.
    
    For energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    The gradient projected onto the tangent space of the sphere is:
    grad_tangent_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2 + (x_i . sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2) * x_i
    This ensures the gradient is tangent to the sphere (orthogonal to x_i).
    """
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Raw gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    raw_grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Project onto tangent space: grad_tan = raw_grad - (raw_grad . x_i) * x_i
    dot_with_x = np.sum(raw_grad * points, axis=1, keepdims=True)
    grad = raw_grad - dot_with_x * points
    
    # Compute energy
    iu = np.triu_indices(n, k=1)
    energy = -math.fsum(np.log(dist[iu]))
    
    return energy, grad


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _geodesic_dome(n, seed=None):
    """Generate geodesic dome points by subdividing an icosahedron."""
    rng = np.random.default_rng(seed)
    
    # Icosahedron vertices
    phi_g = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi_g, 0], [1, phi_g, 0], [-1, -phi_g, 0], [1, -phi_g, 0],
        [0, -1, phi_g], [0, 1, phi_g], [0, -1, -phi_g], [0, 1, -phi_g],
        [phi_g, 0, -1], [phi_g, 0, 1], [-phi_g, 0, -1], [-phi_g, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Icosahedron faces
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide faces to get approximately n points
    # Each face has 3 vertices, we can subdivide each edge into k segments
    # Number of points after subdivision: 12 + 30*(k-1) + 20*k*(k-1)/2
    # For k=1: 12, k=2: 42, k=3: 92, k=4: 162
    # We want ~150, so k=4 gives 162 which is close
    
    k = 4  # subdivisions per edge
    all_points = []
    seen = set()
    
    for face in faces:
        a, b, c = verts[face[0]], verts[face[1]], verts[face[2]]
        for i in range(k + 1):
            for j in range(k + 1 - i):
                l = k - i - j
                # Barycentric coordinates
                pt = (i * a + j * b + l * c) / k
                pt = pt / np.linalg.norm(pt)
                # Round to avoid duplicates
                key = tuple(np.round(pt, 10))
                if key not in seen:
                    seen.add(key)
                    all_points.append(pt)
    
    all_points = np.array(all_points)
    
    # If we have more than n points, randomly select n
    if len(all_points) > n:
        idx = rng.choice(len(all_points), size=n, replace=False)
        all_points = all_points[idx]
    # If we have fewer, pad with random points
    elif len(all_points) < n:
        extra = _random_sphere(n - len(all_points), seed=rng.integers(0, 2**32))
        all_points = np.vstack([all_points, extra])
    
    return all_points


def _velocity_verlet_step(points, velocities, dt):
    """One velocity-Verlet step with projection onto sphere."""
    n = len(points)
    
    # Compute forces (negative gradient of energy)
    _, grad = _energy_and_grad(points)
    forces = -grad  # force = -gradient
    
    # Update velocities: v_new = v + 0.5 * dt * F
    velocities = velocities + 0.5 * dt * forces
    
    # Update positions: x_new = x + dt * v_new
    points_new = points + dt * velocities
    
    # Project back to sphere
    points_new = _project_to_sphere(points_new)
    
    # Compute new forces
    _, grad_new = _energy_and_grad(points_new)
    forces_new = -grad_new
    
    # Update velocities again: v_new = v + 0.5 * dt * F_new
    velocities = velocities + 0.5 * dt * forces_new
    
    return points_new, velocities


def _damped_dynamics(points, max_time=100.0):
    """Run damped dynamics to minimize energy using tangent-space gradient descent."""
    n = len(points)
    current_energy = _energy(points)
    
    # Adaptive timestep
    dt = 0.005
    velocities = np.zeros_like(points)
    
    # Velocity-Verlet with damping
    damping = 0.98
    
    max_steps = 10000
    best_points = points.copy()
    best_energy = current_energy
    stagnant = 0
    
    for step in range(max_steps):
        # Check time
        if step % 50 == 0 and time.time() > max_time:
            break
        
        # Compute forces (tangent-space gradient)
        _, grad = _energy_and_grad(points)
        forces = -grad
        
        # Damped velocity update in tangent space
        velocities = damping * velocities + dt * forces
        # Ensure velocities are tangent
        vel_dot_x = np.sum(velocities * points, axis=1, keepdims=True)
        velocities = velocities - vel_dot_x * points
        
        # Update positions
        points_new = points + dt * velocities
        points_new = _project_to_sphere(points_new)
        
        # Compute new energy
        new_energy = _energy(points_new)
        
        if new_energy < current_energy - 1e-12:
            points = points_new
            current_energy = new_energy
            stagnant = 0
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
            # Try to increase dt slightly
            dt = min(dt * 1.02, 0.05)
        else:
            # Reject step, reduce dt
            dt *= 0.8
            velocities *= 0.3
            stagnant += 1
        
        if dt < 1e-8 or stagnant > 200:
            break
    
    return best_points, best_energy


def _lbfgs_refine(points, max_iter=1000):
    """Refine using L-BFGS-B from scipy with proper tangent-space handling."""
    def objective_and_grad(x):
        pts = x.reshape(-1, 3)
        # Project to sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        
        e, g = _energy_and_grad(pts)
        # The gradient g is already in tangent space
        # For L-BFGS-B, we need gradient with respect to x
        # Chain rule: dx/dpts = I - pts*pts^T (projection Jacobian)
        # grad_x = (I - pts*pts^T) @ g = g - (g.pts)*pts
        # But g is already tangent, so g.pts = 0, thus grad_x = g
        # However, we need to account for the normalization
        # d(pts)/d(x) where pts = x/|x|
        # J = (I - pts*pts^T)/|x| ... this is complex
        # Simpler: just use g directly as it's tangent
        return e, g.flatten()
    
    # Flatten points
    x0 = points.flatten()
    
    result = scipy_minimize(
        objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'gtol': 1e-15, 'ftol': 1e-15, 'maxls': 100}
    )
    
    pts = _project_to_sphere(result.x.reshape(-1, 3))
    return pts, _energy(pts)


def _local_perturb_restart(points, rng, strength=0.1):
    """Perturb points locally to escape local minima."""
    n = len(points)
    # Perturb a subset of points
    n_perturb = max(1, n // 10)
    idx = rng.choice(n, size=n_perturb, replace=False)
    
    new_points = points.copy()
    for i in idx:
        # Generate random tangent vector
        rand_vec = rng.standard_normal(3)
        # Make it tangent to sphere at points[i]
        tangent = rand_vec - np.dot(rand_vec, points[i]) * points[i]
        norm = np.linalg.norm(tangent)
        if norm > 1e-10:
            tangent = tangent / norm
        else:
            tangent = rand_vec / np.linalg.norm(rand_vec)
        
        new_points[i] = points[i] + strength * tangent
        new_points[i] = new_points[i] / np.linalg.norm(new_points[i])
    
    return new_points


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin before 150s limit
    
    best_points = None
    best_energy = float('inf')
    
    # Multiple random restarts with local perturbation
    n_restarts = 15
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Generate initial points
        if restart == 0:
            points = _fibonacci_sphere(N)
        elif restart == 1:
            points = _geodesic_dome(N, seed=rng.integers(0, 2**32))
        elif restart == 2 and best_points is not None:
            # Perturb the best solution found so far
            points = _local_perturb_restart(best_points, rng, strength=0.05)
        elif restart == 3 and best_points is not None:
            points = _local_perturb_restart(best_points, rng, strength=0.1)
        elif restart == 4 and best_points is not None:
            points = _local_perturb_restart(best_points, rng, strength=0.2)
        else:
            points = _random_sphere(N, seed=rng.integers(0, 2**32))
        
        # Phase 1: Damped dynamics
        try:
            pts, e = _damped_dynamics(points, max_time=min(25.0, deadline - time.time() - 10.0))
            if e < best_energy:
                best_energy = e
                best_points = pts.copy()
        except Exception:
            continue
        
        # Phase 2: L-BFGS refinement if time permits
        if time.time() < deadline - 10.0 and best_points is not None:
            try:
                pts, e = _lbfgs_refine(best_points, max_iter=300)
                if e < best_energy:
                    best_energy = e
                    best_points = pts.copy()
            except Exception:
                pass
        
        # Phase 3: More dynamics if time permits
        if time.time() < deadline - 5.0 and best_points is not None:
            try:
                pts, e = _damped_dynamics(best_points, max_time=min(10.0, deadline - time.time() - 3.0))
                if e < best_energy:
                    best_energy = e
                    best_points = pts.copy()
            except Exception:
                pass
    
    # Final high-precision L-BFGS polish if time permits
    if time.time() < deadline - 5.0 and best_points is not None:
        try:
            pts, e = _lbfgs_refine(best_points, max_iter=3000)
            if e < best_energy:
                best_energy = e
                best_points = pts.copy()
        except Exception:
            pass
    
    # One more dynamics pass
    if time.time() < deadline - 3.0 and best_points is not None:
        try:
            pts, e = _damped_dynamics(best_points, max_time=min(5.0, deadline - time.time() - 1.0))
            if e < best_energy:
                best_energy = e
                best_points = pts.copy()
        except Exception:
            pass
    
    # Final L-BFGS polish after dynamics
    if time.time() < deadline - 1.0 and best_points is not None:
        try:
            pts, e = _lbfgs_refine(best_points, max_iter=1000)
            if e < best_energy:
                best_energy = e
                best_points = pts.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback to Fibonacci
        best_points = _fibonacci_sphere(N)
    
    # Ensure validity
    best_points = _project_to_sphere(best_points)
    
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                # Perturb slightly
                best_points[j] += rng.normal(0, 1e-6, 3)
                best_points[j] = best_points[j] / np.linalg.norm(best_points[j])
    
    return best_points
# EVOLVE-BLOCK-END

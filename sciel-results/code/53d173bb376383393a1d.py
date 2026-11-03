"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 115.0  # safety margin before 120s kill

# EVOLVE-BLOCK-START

def _log_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/d_ij)"""
    n = len(points)
    # Compute all pairwise distances
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dists, 1.0)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    return math.fsum(np.log(1.0 / dists[iu]))


def _log_energy_fast(points):
    """Faster version using vectorized ops, returns energy and gradient info."""
    n = len(points)
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diffs**2, axis=2)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    iu = np.triu_indices(n, k=1)
    # energy = sum log(1/d) = -sum log(d)
    energy = -np.sum(np.log(dists[iu]))
    return energy


def _project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _fibonacci_sphere(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _geodesic_dome(n_target):
    """
    Generate points using geodesic dome (subdivided icosahedron) as initial configuration.
    We subdivide an icosahedron and project to sphere, then select n_target points
    with uniform spacing.
    """
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Faces of icosahedron
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    
    # Subdivide faces
    def midpoint(v1, v2):
        m = (v1 + v2) / 2.0
        m = m / np.linalg.norm(m)
        return m
    
    def subdivide(verts, faces, subdivisions):
        for _ in range(subdivisions):
            new_faces = []
            mid_cache = {}
            for f in faces:
                a, b, c = f
                ab = midpoint(verts[a], verts[b])
                bc = midpoint(verts[b], verts[c])
                ca = midpoint(verts[c], verts[a])
                # Add midpoints to vertices
                verts = np.vstack([verts, [ab, bc, ca]])
                new_faces.extend([
                    (a, len(verts)-3, len(verts)-2),
                    (b, len(verts)-2, len(verts)-1),
                    (c, len(verts)-1, len(verts)-3),
                    (len(verts)-3, len(verts)-2, len(verts)-1)
                ])
            return verts, new_faces
    
    # Determine subdivisions to get close to n_target points
    # Icosahedron has 12 vertices, 20 faces
    # After k subdivisions: 10*4^k + 2 vertices
    # k=0: 12, k=1: 42, k=2: 162, k=3: 642
    # We want ~150, so k=2 gives 162 which is close
    # Let's try k=2 first
    best_points = None
    for k in [2, 3]:
        v, f = subdivide(verts.copy(), list(faces), k)
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        if len(v) >= n_target:
            # Select n_target points with good spread using farthest-point sampling
            selected = [0]
            # Use a simple greedy approach: pick points that maximize minimum distance to already selected
            # For efficiency, just take every other point or use random selection
            indices = np.random.choice(len(v), size=n_target, replace=False)
            best_points = v[indices]
            break
    
    if best_points is None:
        # Fallback to fibonacci
        best_points = _fibonacci_sphere(n_target)
    
    return best_points


def _gradient_log_energy(points):
    """
    Compute gradient of log energy with respect to point positions.
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    dE/dx_i = sum_{j != i} (x_i - x_j) / d_ij^2
    """
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i+1, n):
            diff = points[i] - points[j]
            d_sq = np.dot(diff, diff)
            if d_sq < 1e-30:
                continue
            d = np.sqrt(d_sq)
            # d/dx_i log(1/d) = d/dx_i (-log d) = -(1/d) * (1/d) * diff = -diff/d^2
            # Wait: log(1/d) = -log(d)
            # d(-log(d))/dx_i where d = |x_i - x_j|
            # = -(1/d) * (1/d) * (x_i - x_j) = -(x_i - x_j)/d^2
            # Actually: d/dx_i log(1/|x_i-x_j|) = d/dx_i (-log|x_i-x_j|)
            # = -(1/|x_i-x_j|) * (1/|x_i-x_j|) * (x_i-x_j)/|x_i-x_j|
            # = -(x_i - x_j) / |x_i-x_j|^2
            term = -diff / d_sq
            grad[i] += term
            grad[j] -= term
    return grad


def _project_gradient(points, grad):
    """Project gradient to tangent space of sphere at each point."""
    n = len(points)
    proj_grad = np.zeros_like(grad)
    for i in range(n):
        # Tangent space: remove radial component
        radial = np.dot(grad[i], points[i]) * points[i]
        proj_grad[i] = grad[i] - radial
    return proj_grad


def _newton_step_sphere(points, max_step=0.1):
    """
    One step of gradient descent on sphere with line search.
    Uses projected gradient and backtracking line search.
    """
    n = len(points)
    energy = _log_energy_fast(points)
    grad = _gradient_log_energy(points)
    proj_grad = _project_gradient(points, grad)
    
    # Scale gradient
    grad_norm = np.linalg.norm(proj_grad)
    if grad_norm < 1e-15:
        return points, energy
    
    # Initial step size
    lr = 0.01 / (1.0 + grad_norm)
    
    for _ in range(50):
        # Move in direction of negative projected gradient
        new_points = points - lr * proj_grad
        new_points = _project_to_sphere(new_points)
        
        new_energy = _log_energy_fast(new_points)
        
        if new_energy < energy:
            return new_points, new_energy
        lr *= 0.5
        if lr < 1e-12:
            break
    
    return points, energy


def _lbfgs_optimize(points, time_limit):
    """
    Use L-BFGS-B with spherical constraint handled via parameterization.
    We use the fact that we can flatten and use scipy's minimize with bounds,
    but better: use the tangent space approach with scipy.
    """
    start_time = time.time()
    
    def objective_and_grad(x):
        """x is flat array of 3N coordinates, projected to sphere."""
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        energy = _log_energy_fast(pts)
        grad = _gradient_log_energy(pts)
        proj_grad = _project_gradient(pts, grad)
        return energy, proj_grad.flatten()
    
    x0 = points.flatten()
    
    result = minimize(
        objective_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
    )
    
    pts = _project_to_sphere(result.x.reshape(-1, 3))
    return pts


def _simulated_annealing(points, time_limit, n_iter=2000):
    """
    Simulated annealing to escape local minima.
    """
    start_time = time.time()
    current = points.copy()
    current_energy = _log_energy_fast(current)
    best = current.copy()
    best_energy = current_energy
    
    T = 0.01  # Initial temperature
    T_min = 1e-8
    cooling = 0.999
    
    for it in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random point and perturb it
        i = np.random.randint(N)
        # Random direction in tangent space
        rand_dir = np.random.randn(3)
        rand_dir = rand_dir - np.dot(rand_dir, current[i]) * current[i]
        rand_norm = np.linalg.norm(rand_dir)
        if rand_norm < 1e-15:
            continue
        rand_dir = rand_dir / rand_norm
        
        # Step size proportional to temperature
        step = T * 0.1
        new_point = current[i] + step * rand_dir
        new_point = new_point / np.linalg.norm(new_point)
        
        new_points = current.copy()
        new_points[i] = new_point
        
        new_energy = _log_energy_fast(new_points)
        delta = new_energy - current_energy
        
        if delta < 0 or np.random.random() < math.exp(-delta / max(T, T_min)):
            current = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy
        
        T *= cooling
    
    return best


def _polish(points, time_limit):
    """
    High-precision polish: alternating between gradient steps and Newton-like refinement.
    """
    start_time = time.time()
    current = points.copy()
    energy = _log_energy_fast(current)
    
    for iteration in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        # Gradient step with adaptive learning rate
        grad = _gradient_log_energy(current)
        proj_grad = _project_gradient(current, grad)
        grad_norm = np.linalg.norm(proj_grad)
        
        if grad_norm < 1e-14:
            break
        
        # Adaptive learning rate
        lr = 0.05 / (1.0 + grad_norm * 10)
        
        # Try step
        new_points = current - lr * proj_grad
        new_points = _project_to_sphere(new_points)
        new_energy = _log_energy_fast(new_points)
        
        if new_energy < energy:
            current = new_points
            energy = new_energy
        else:
            # Backtrack
            lr *= 0.5
            new_points = current - lr * proj_grad
            new_points = _project_to_sphere(new_points)
            new_energy = _log_energy_fast(new_points)
            if new_energy < energy:
                current = new_points
                energy = new_energy
            else:
                # Try smaller step
                for _ in range(10):
                    lr *= 0.5
                    new_points = current - lr * proj_grad
                    new_points = _project_to_sphere(new_points)
                    new_energy = _log_energy_fast(new_points)
                    if new_energy < energy:
                        current = new_points
                        energy = new_energy
                        break
    
    return current


def construct():
    """
    Construct 150 points on the unit sphere minimizing logarithmic energy.
    Strategy:
    1. Generate initial points using geodesic dome and fibonacci spiral
    2. Optimize with L-BFGS-B
    3. Polish with gradient descent
    4. Use simulated annealing to escape local minima
    5. Final polish
    """
    global DEADLINE
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return DEADLINE - elapsed()
    
    # Generate multiple initial configurations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_fibonacci_sphere(N))
    
    # 2. Geodesic dome
    try:
        gd = _geodesic_dome(N)
        inits.append(gd)
    except Exception:
        pass
    
    # 3. Perturbed fibonacci
    rng = np.random.default_rng(42)
    perturbed = _fibonacci_sphere(N) + 0.05 * rng.randn(N, 3)
    perturbed = _project_to_sphere(perturbed)
    inits.append(perturbed)
    
    # 4. Another random perturbation
    perturbed2 = _fibonacci_sphere(N) + 0.1 * rng.randn(N, 3)
    perturbed2 = _project_to_sphere(perturbed2)
    inits.append(perturbed2)
    
    # Optimize each initial configuration
    best_points = inits[0]
    best_energy = _log_energy_fast(inits[0])
    
    for idx, init in enumerate(inits):
        if remaining() < 15:
            break
        
        # L-BFGS optimization
        try:
            pts = _lbfgs_optimize(init, time_limit=max(0, remaining() - 10))
            pts = _project_to_sphere(pts)
            e = _log_energy_fast(pts)
            
            if e < best_energy:
                best_energy = e
                best_points = pts.copy()
        except Exception:
            pass
        
        # Polish
        if remaining() < 10:
            break
        
        try:
            pts = _polish(pts, time_limit=max(0, remaining() - 5))
            pts = _project_to_sphere(pts)
            e = _log_energy_fast(pts)
            
            if e < best_energy:
                best_energy = e
                best_points = pts.copy()
        except Exception:
            pass
    
    # Simulated annealing on the best configuration
    if remaining() > 20:
        try:
            sa_pts = _simulated_annealing(best_points, time_limit=max(0, remaining() - 10), n_iter=1000)
            sa_pts = _project_to_sphere(sa_pts)
            sa_energy = _log_energy_fast(sa_pts)
            
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_points = sa_pts.copy()
        except Exception:
            pass
    
    # Final polish
    if remaining() > 5:
        try:
            final_pts = _polish(best_points, time_limit=max(0, remaining() - 2))
            final_pts = _project_to_sphere(final_pts)
            final_energy = _log_energy_fast(final_pts)
            
            if final_energy < best_energy:
                best_energy = final_energy
                best_points = final_pts.copy()
        except Exception:
            pass
    
    # Ensure valid output
    best_points = _project_to_sphere(best_points)
    
    # Validate: no points closer than 1e-9
    diffs = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        # Add small perturbation to resolve collisions
        rng = np.random.default_rng(123)
        for _ in range(100):
            best_points = best_points + 1e-6 * rng.randn(N, 3)
            best_points = _project_to_sphere(best_points)
            diffs = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            np.fill_diagonal(dists, 1.0)
            iu = np.triu_indices(N, k=1)
            if np.min(dists[iu]) >= 1e-9:
                break
    
    # Check for non-finite values
    if not np.all(np.isfinite(best_points)):
        # Fallback to fibonacci
        best_points = _fibonacci_sphere(N)
    
    return best_points

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere using math.fsum."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    return -math.fsum(log_dists.tolist())


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, seed=42):
    """Generate random uniform points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _compute_forces(points):
    """
    Compute the forces (negative gradient) on the points.
    Force on i due to j is -dE/dx_i.
    E = sum log(1/d) = -sum log(d).
    dE/dx_i = sum_{j!=i} d/dx_i (-log |x_i - x_j|) = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2.
    So Force = -dE/dx_i = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2.
    """
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0) # Avoid division by zero for self-interaction
    
    # Weights w_ij = 1 / d_ij^2
    # Force on i is sum_j w_ij * (x_j - x_i) = sum_j w_ij * x_j - x_i * sum_j w_ij
    weights = 1.0 / dists_sq
    np.fill_diagonal(weights, 0.0)
    
    sum_weights = np.sum(weights, axis=1, keepdims=True)
    weighted_sum = weights @ points
    
    forces = weighted_sum - points * sum_weights
    
    # Project forces to tangent space to ensure motion stays on the sphere
    # v_tangent = v - (v . n) n
    dot_products = np.sum(forces * points, axis=1, keepdims=True)
    forces_tangent = forces - dot_products * points
    
    return forces_tangent

def _velocity_verlet_optimize(points, max_time=10.0, dt=0.005, damping=0.98):
    """
    Optimize points on sphere using Velocity Verlet integration with damping.
    """
    points = _project_to_sphere(points.copy())
    n = len(points)
    velocities = np.zeros_like(points)
    
    # Initial forces
    forces = _compute_forces(points)
    
    # Compute initial energy
    current_energy = _log_energy(points)
    
    start_time = time.time()
    step = 0
    
    # Adaptive timestep logic: reduce dt if energy increases or step size is too large
    # Simple approach: fixed dt with damping, maybe reduce dt if oscillations detected
    # For simplicity and robustness, we use a fixed dt and damping.
    
    while time.time() - start_time < max_time:
        # Velocity Verlet update
        # x(t+dt) = x(t) + v(t)dt + 0.5 a(t) dt^2
        # Note: a(t) is proportional to force. We treat mass=1.
        # We work in Cartesian coordinates but project back to sphere.
        
        new_points = points + velocities * dt + 0.5 * forces * (dt * dt)
        
        # Project to sphere
        new_points = _project_to_sphere(new_points)
        
        # Compute new forces
        new_forces = _compute_forces(new_points)
        
        # Update velocity
        # v(t+dt) = v(t) + 0.5 (a(t) + a(t+dt)) dt
        # Apply damping
        new_velocities = (velocities + 0.5 * (forces + new_forces) * dt) * damping
        
        # Project velocity to tangent space
        dot_products = np.sum(new_velocities * new_points, axis=1, keepdims=True)
        new_velocities = new_velocities - dot_products * new_points
        
        # Update state
        points = new_points
        velocities = new_velocities
        forces = new_forces
        
        # Check energy
        new_energy = _log_energy(points)
        
        # If energy increases significantly, reduce velocity or step back?
        # Damping should handle most of it.
        
        step += 1
        
        # Break if convergence is reached (velocity very small)
        if np.linalg.norm(velocities) < 1e-8:
            break

    return points


def _geodesic_dome_init(n):
    """Generate initial points using a geodesic dome subdivision."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Subdivide faces
    # Icosahedron has 20 faces
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide each face into 4 triangles (frequency 2)
    def subdivide_face(v0, v1, v2, subdiv=2):
        """Subdivide a triangle into subdiv^2 smaller triangles."""
        if subdiv == 1:
            return [(v0, v1, v2)]
        
        # Midpoints
        m01 = (v0 + v1) / 2.0
        m12 = (v1 + v2) / 2.0
        m20 = (v2 + v0) / 2.0
        m01 = m01 / np.linalg.norm(m01)
        m12 = m12 / np.linalg.norm(m12)
        m20 = m20 / np.linalg.norm(m20)
        
        return (subdivide_face(v0, m01, m20, subdiv-1) +
                subdivide_face(m01, v1, m12, subdiv-1) +
                subdivide_face(m20, m12, v2, subdiv-1) +
                subdivide_face(m01, m12, m20, subdiv-1))
    
    # Generate all vertices from subdivided faces
    all_verts = set()
    for f in faces:
        v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
        sub_faces = subdivide_face(v0, v1, v2, subdiv=2)
        for sf in sub_faces:
            for v in sf:
                all_verts.add(tuple(np.round(v, 12)))
    
    # Convert to array
    result = np.array(list(all_verts))
    result = result / np.linalg.norm(result, axis=1, keepdims=True)
    
    # If we have more points than needed, randomly select n
    if len(result) > n:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(result), size=n, replace=False)
        result = result[idx]
    elif len(result) < n:
        # Pad with random points if needed
        extra = _random_uniform_sphere(n - len(result), seed=99)
        result = np.vstack([result, extra])
    
    return result


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Multiple initializations including geodesic dome
    inits = [
        _geodesic_dome_init(N),
        _fibonacci_sphere(N),
        _random_uniform_sphere(N, seed=1),
        _random_uniform_sphere(N, seed=2),
        _random_uniform_sphere(N, seed=3),
        _random_uniform_sphere(N, seed=4),
        _random_uniform_sphere(N, seed=5),
        _random_uniform_sphere(N, seed=6),
    ]
    
    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)
    
    for idx, init in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
        
        try:
            # Use Velocity Verlet for optimization
            # Allocate time based on remaining budget
            time_left = deadline - time.time() - 5.0 # 5s safety margin for final check
            time_per_init = max(1.0, time_left / max(1, len(inits) - idx))
            
            p_res = _velocity_verlet_optimize(init, max_time=time_per_init)
            e_check = _log_energy(p_res)
            
            if e_check < best_energy:
                best_energy = e_check
                best_points = p_res.copy()
        except Exception:
            pass
    
    # Final polish using Velocity Verlet with smaller dt and longer time if possible
    if time.time() < deadline - 2.0:
        time_left = deadline - time.time() - 1.0
        try:
            p_res = _velocity_verlet_optimize(best_points, max_time=time_left, dt=0.001, damping=0.99)
            e_check = _log_energy(p_res)
            if e_check < best_energy:
                best_energy = e_check
                best_points = p_res.copy()
        except Exception:
            pass
    
    best_points = _project_to_sphere(best_points)
    
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)
    
    # Check for points too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    if np.min(dists_sq[iu]) < (1e-9) ** 2:
        return _fibonacci_sphere(N)
    
    return best_points
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START

def _covering_radius(points):
    """
    Compute the covering radius in radians using vectorized circumcenter calculation.
    """
    if len(points) < 3:
        return np.pi / 2
    
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        return np.pi
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi / 2
    
    faces = hull.simplices  # shape (F, 3)
    F = len(faces)
    
    # Extract face vertices: shape (F, 3, 3)
    p0 = points[faces[:, 0]]  # (F, 3)
    p1 = points[faces[:, 1]]
    p2 = points[faces[:, 2]]
    
    # Compute circumcenter for each face using vectorized operations
    A = p1 - p0  # (F, 3)
    B = p2 - p0  # (F, 3)
    
    # Dot products
    A_A = np.einsum('ij,ij->i', A, A)  # (F,)
    A_B = np.einsum('ij,ij->i', A, B)
    B_B = np.einsum('ij,ij->i', B, B)
    
    det = A_A * B_B - A_B * A_B  # (F,)
    
    # u and v for circumcenter: O = p0 + u*A + v*B
    safe_det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    
    u = (0.5 * A_A * B_B - 0.5 * A_B * B_B) / safe_det
    v = (0.5 * A_B * A_A - 0.5 * A_A * B_B) / safe_det
    
    # O = p0 + u*A + v*B, shape (F, 3)
    O = p0 + u[:, np.newaxis] * A + v[:, np.newaxis] * B
    
    # Normalize O to get spherical circumcenter
    norm_O = np.linalg.norm(O, axis=1, keepdims=True)
    safe_norm = np.where(norm_O < 1e-12, 1e-12, norm_O)
    C = O / safe_norm  # (F, 3)
    
    # For degenerate faces, use centroid as fallback
    degenerate = np.abs(det) < 1e-12
    if np.any(degenerate):
        centroids = (p0 + p1 + p2) / 3.0
        cent_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        safe_cent_norm = np.where(cent_norms < 1e-12, 1e-12, cent_norms)
        C[degenerate] = centroids[degenerate] / safe_cent_norm[degenerate]
    
    # Calculate angular distances from C to each vertex
    d0 = np.arccos(np.clip(np.einsum('ij,ij->i', C, p0), -1, 1))
    d1 = np.arccos(np.clip(np.einsum('ij,ij->i', C, p1), -1, 1))
    d2 = np.arccos(np.clip(np.einsum('ij,ij->i', C, p2), -1, 1))
    
    r_face = np.maximum(np.maximum(d0, d1), d2)  # (F,)
    
    return float(np.max(r_face))

def _energy(points):
    """
    Compute the Riesz s=1 energy (sum of pairwise distances) for energy minimization.
    Lower energy corresponds to more uniform distribution.
    """
    # Compute all pairwise distances
    # points shape (N, 3)
    # Use broadcasting to get pairwise dot products
    dots = np.dot(points, points.T)
    # Distance = arccos(dot) for unit vectors
    # But for energy minimization, using Euclidean distance (chordal) is often faster and smoother
    # Let's use chordal distance squared for smoothness, or just chordal distance.
    # The task mentions "Energy minimization (Coulomb/Riesz s=1)". Riesz s=1 is sum of distances.
    # Let's compute sum of chordal distances.
    dists = np.sqrt(np.maximum(1 - dots, 0) * 2)
    # Zero out diagonal
    np.fill_diagonal(dists, 0)
    return np.sum(dists)

def _energy_grad(points):
    """
    Compute gradient of the Riesz s=1 energy.
    E = sum_{i<j} ||p_i - p_j||
    dE/dp_i = sum_{j!=i} (p_i - p_j) / ||p_i - p_j||
    """
    N = len(points)
    grad = np.zeros_like(points)
    
    # Vectorized gradient calculation
    # diff[i,j] = p_i - p_j
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, 3)
    dists = np.linalg.norm(diff, axis=2)  # (N, N)
    
    # Avoid division by zero on diagonal
    mask = dists > 1e-12
    inv_dists = np.zeros_like(dists)
    inv_dists[mask] = 1.0 / dists[mask]
    
    # grad_i = sum_j (p_i - p_j) / dist_ij
    # This is sum over j of diff[i,j] * inv_dists[i,j]
    grad = np.sum(diff * inv_dists[:, :, np.newaxis], axis=1)
    
    return grad

def _optimize_energy(start_points, time_limit, rng):
    """
    Optimize points by minimizing Riesz energy using L-BFGS-B.
    """
    points = start_points.copy()
    
    # Flatten points for optimizer
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        if np.any(norms < 1e-12):
            return 1e10
        pts = pts / norms
        return _energy(pts)
    
    def gradient(x):
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        if np.any(norms < 1e-12):
            return np.zeros_like(x)
        pts = pts / norms
        g = _energy_grad(pts)
        # Project gradient to tangent space
        # grad_tangent = grad - (grad . p) * p
        dots = np.einsum('ij,ij->i', g, pts)
        g_tangent = g - dots[:, np.newaxis] * pts
        return g_tangent.flatten()
    
    start_time = time.time()
    
    try:
        res = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            jac=gradient,
            options={'maxiter': 1000, 'gtol': 1e-6, 'ftol': 1e-6, 'maxfun': 5000}
        )
        if res.success or res.fun < 1e10:
            best_pts = res.x.reshape(N, 3)
            best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
            return best_pts
    except Exception:
        pass
    
    # Fallback to hill climbing if energy optimization fails or times out
    return _hill_climb(start_points, time_limit, rng)

def _hill_climb(start_points, time_limit, rng, initial_step=0.1):
    """
    Local optimization via hill-climbing with adaptive step size.
    """
    points = start_points.copy()
    current_cr = _covering_radius(points)
    best_points = points.copy()
    best_cr = current_cr
    
    start_time = time.time()
    
    step_size = initial_step
    min_step = 1e-5
    max_iter = 500000
    no_improve_count = 0
    no_improve_limit = 200
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        idx = rng.integers(N)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        
        improved = False
        for sign in [1, -1]:
            new_point = points[idx] + sign * step_size * direction
            new_norm = np.linalg.norm(new_point)
            if new_norm < 1e-12:
                continue
            new_point /= new_norm
            
            old_point = points[idx]
            points[idx] = new_point
            
            new_cr = _covering_radius(points)
            
            if new_cr < current_cr - 1e-12:
                current_cr = new_cr
                improved = True
                if new_cr < best_cr:
                    best_cr = new_cr
                    best_points = points.copy()
                step_size = min(step_size * 1.03, 0.3)
                no_improve_count = 0
                break
            else:
                points[idx] = old_point
        
        if not improved:
            no_improve_count += 1
            step_size = max(step_size * 0.95, min_step)
            
            if no_improve_count > no_improve_limit:
                step_size = rng.uniform(0.05, 0.15)
                no_improve_count = 0
        
        if step_size < min_step * 1.01:
            break
    
    return best_points, best_cr

def construct():
    total_deadline = 145.0
    start_time = time.time()
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_cr = np.pi
    
    # 1. Fibonacci spiral start
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Energy minimization on Fibonacci start
    elapsed = time.time() - start_time
    alloc_time = min(30.0, total_deadline - elapsed - 5.0)
    if alloc_time > 2.0:
        energy_points = _optimize_energy(fib_points, alloc_time, rng)
        energy_cr = _covering_radius(energy_points)
        if energy_cr < best_cr:
            best_cr = energy_cr
            best_points = energy_points.copy()
    
    # 3. Hill climbing from best so far
    if best_points is not None:
        elapsed = time.time() - start_time
        alloc_time = min(20.0, total_deadline - elapsed - 5.0)
        if alloc_time > 2.0:
            hc_points, hc_cr = _hill_climb(best_points, alloc_time, rng, initial_step=0.05)
            if hc_cr < best_cr:
                best_cr = hc_cr
                best_points = hc_points.copy()
    
    # 4. Random restarts with energy minimization
    num_restarts = 5
    elapsed = time.time() - start_time
    remaining_time = total_deadline - elapsed - 5.0
    if remaining_time > 5.0:
        time_per_restart = remaining_time / num_restarts
        
        for i_restart in range(num_restarts):
            if time.time() - start_time > total_deadline - 5.0:
                break
            
            random_points = rng.normal(size=(N, 3))
            random_points /= np.linalg.norm(random_points, axis=1, keepdims=True)
            
            alloc_time = min(time_per_restart, total_deadline - (time.time() - start_time) - 5.0)
            if alloc_time < 2.0:
                break
            
            # Try energy minimization first
            energy_points = _optimize_energy(random_points, alloc_time * 0.6, rng)
            energy_cr = _covering_radius(energy_points)
            
            # Then hill climb from there
            if energy_cr < best_cr:
                best_cr = energy_cr
                best_points = energy_points.copy()
            
            # Final hill climb
            remaining_alloc = total_deadline - (time.time() - start_time) - 5.0
            if remaining_alloc > 1.0:
                hc_points, hc_cr = _hill_climb(energy_points, remaining_alloc, rng, initial_step=0.03)
                if hc_cr < best_cr:
                    best_cr = hc_cr
                    best_points = hc_points.copy()
    
    if best_points is not None:
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END

import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # safety margin before 150s kill


def compute_radius(points):
    """Compute the maximum radius r such that circles of radius r centered at points
    are inside the unit circle and do not overlap."""
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    if min_boundary <= 0:
        return 0.0
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_dist / 2.0, min_boundary)
    return r if r > 0 else 0.0


def build_contact_graph(points, r_target, k=6):
    """Build a nearest-neighbor contact graph. Each node connects to its k nearest neighbors."""
    tree = cKDTree(points)
    # Query k+1 nearest neighbors (including self)
    distances, indices = tree.query(points, k=k+1)
    
    # Build edge list (undirected, no self-loops)
    edges = set()
    for i in range(N):
        for j in indices[i, 1:]:  # skip self
            if j > i:
                edges.add((i, j))
            else:
                edges.add((j, i))
    
    return list(edges)


def topology_energy_and_grad(points, edges, r_target):
    """
    Topology-aware energy:
    - Spring forces: penalize deviation from target contact distance (2*r_target)
    - Boundary penalty: penalize circles extending outside unit circle
    
    Returns (energy, gradient)
    """
    E = 0.0
    grad = np.zeros_like(points)
    
    # Spring forces along contact edges
    k_spring = 1.0
    for (i, j) in edges:
        vec = points[i] - points[j]
        d = np.linalg.norm(vec)
        if d < 1e-10:
            continue
        # Target distance is 2*r_target
        target_d = 2.0 * r_target
        # Spring force: pull/push to target distance
        force_mag = k_spring * (d - target_d)
        force_vec = force_mag * vec / d
        
        grad[i] += force_vec
        grad[j] -= force_vec
        
        E += 0.5 * force_mag * (d - target_d)
    
    # Boundary penalty
    norms = np.linalg.norm(points, axis=1)
    boundary_limit = 1.0 - r_target
    for i in range(N):
        if norms[i] > boundary_limit:
            violation = norms[i] - boundary_limit
            # Gradient points inward
            if norms[i] > 1e-10:
                grad[i] += 2.0 * violation * points[i] / norms[i]
            E += violation ** 2
    
    return E, grad


def force_directed_optimize(points, r_target, time_limit=10.0):
    """
    Force-directed layout optimization with adaptive cooling.
    Maintains contact topology and optimizes positions.
    """
    pts = points.copy()
    start_time = time.time()
    
    # Initial temperature for perturbation
    T = 0.02
    T_min = 0.001
    cooling_rate = 0.995
    
    # Adaptive step size
    step_size = 0.01
    min_step = 0.001
    max_step = 0.1
    
    best_pts = pts.copy()
    best_E = topology_energy_and_grad(pts, build_contact_graph(pts, r_target, k=6), r_target)[0]
    
    iteration = 0
    while time.time() - start_time < time_limit:
        # Build current contact graph
        edges = build_contact_graph(pts, r_target, k=6)
        
        # Compute energy and gradient
        E, grad = topology_energy_and_grad(pts, edges, r_target)
        
        # Gradient descent step with adaptive step size
        pts_new = pts - step_size * grad
        
        # Project inside unit circle (hard constraint)
        norms = np.linalg.norm(pts_new, axis=1)
        for i in range(N):
            if norms[i] > 1.0:
                pts_new[i] = pts_new[i] / norms[i] * (1.0 - 0.001)
        
        # Compute new energy
        E_new, _ = topology_energy_and_grad(pts_new, build_contact_graph(pts_new, r_target, k=6), r_target)
        
        # Accept/reject with simulated annealing criterion
        delta_E = E_new - E
        if delta_E < 0 or np.random.random() < np.exp(-delta_E / max(T, T_min)):
            pts = pts_new
            # Adapt step size: increase if improving, decrease if worsening
            if delta_E < 0:
                step_size = min(step_size * 1.1, max_step)
            else:
                step_size = max(step_size * 0.9, min_step)
            
            if E_new < best_E:
                best_E = E_new
                best_pts = pts.copy()
        else:
            # Reject: reduce step size slightly
            step_size = max(step_size * 0.95, min_step)
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
        
        # Periodic topology refresh: occasionally re-randomize some contacts
        if iteration % 100 == 0 and T > T_min * 2:
            # Perturb a random subset of points
            n_perturb = max(1, N // 10)
            indices = np.random.choice(N, n_perturb, replace=False)
            for idx in indices:
                pts[idx] += np.random.normal(0, T, 2)
                norm = np.linalg.norm(pts[idx])
                if norm > 1.0:
                    pts[idx] = pts[idx] / norm * (1.0 - 0.01)
        
        iteration += 1
    
    return best_pts


def initial_rings_optimized():
    """
    Generate initial points using concentric rings matching the known optimal structure:
    1 center + rings of 6, 12, 18, 13 (total 50)
    """
    points = np.zeros((N, 2))
    points[0] = [0, 0]
    count = 1
    
    # Ring radii and counts
    rings = [
        (0.15, 6),   # inner ring
        (0.30, 12),  # second ring
        (0.45, 18),  # third ring
        (0.58, 13),  # fourth ring (remaining)
    ]
    
    for ring_radius, ring_count in rings:
        for k in range(ring_count):
            if count >= N:
                break
            angle = 2 * np.pi * k / ring_count + np.pi / ring_count  # offset for symmetry
            points[count, 0] = ring_radius * np.cos(angle)
            points[count, 1] = ring_radius * np.sin(angle)
            count += 1
    
    return points


def initial_hexagonal():
    """Generate initial points using a hexagonal lattice pattern."""
    spacing = 0.22
    points = []
    
    max_k = 10
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = i * spacing + (j % 2) * spacing * 0.5
            y = j * spacing * np.sqrt(3) / 2
            if x**2 + y**2 < 0.85:
                points.append([x, y])
                
    points = np.array(points)
    
    if len(points) >= N:
        # Select N points closest to origin
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        selected = points[indices[:N]]
    else:
        needed = N - len(points)
        random_pts = np.random.uniform(-0.5, 0.5, size=(needed, 2))
        selected = np.vstack([points, random_pts])
        
    return selected


def initial_random():
    """Generate random initial points inside unit circle."""
    # Uniform in disk
    theta = np.random.uniform(0, 2 * np.pi, N)
    r = np.sqrt(np.random.uniform(0, 1, N)) * 0.9
    points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return points


def construct():
    deadline = time.time() + DEADLINE
    
    best_overall_r = 0.0
    best_overall_points = None
    
    # Multiple initializations
    initializations = [
        initial_rings_optimized(),
        initial_hexagonal(),
        initial_random()
    ]
    
    for init_pts in initializations:
        if time.time() > deadline - 15:
            break
        
        # Estimate initial radius
        r_est = compute_radius(init_pts)
        if r_est <= 0:
            r_est = 0.1
        
        # Force-directed optimization
        try:
            fd_pts = force_directed_optimize(init_pts, r_est, time_limit=20.0)
            r_fd = compute_radius(fd_pts)
            
            if r_fd > best_overall_r:
                best_overall_r = r_fd
                best_overall_points = fd_pts.copy()
        except Exception:
            pass
        
        # If we have time, do a second pass with improved radius estimate
        if best_overall_points is not None and time.time() < deadline - 10:
            try:
                r_improved = best_overall_r * 1.005
                fd_pts2 = force_directed_optimize(best_overall_points, r_improved, time_limit=15.0)
                r_fd2 = compute_radius(fd_pts2)
                
                if r_fd2 > best_overall_r:
                    best_overall_r = r_fd2
                    best_overall_points = fd_pts2.copy()
            except Exception:
                pass
    
    # Final polish: try to increase radius slightly
    if best_overall_points is not None and time.time() < deadline - 5:
        try:
            for _ in range(5):
                if time.time() > deadline - 2:
                    break
                r_try = best_overall_r * 1.002
                polished = force_directed_optimize(best_overall_points, r_try, time_limit=5.0)
                r_new = compute_radius(polished)
                if r_new > best_overall_r:
                    best_overall_r = r_new
                    best_overall_points = polished.copy()
                else:
                    break
        except Exception:
            pass
    
    return best_overall_points if best_overall_points is not None else initial_rings_optimized()


if __name__ == "__main__":
    pts = construct()
    r = compute_radius(pts)
    print(f"Radius: {r}")
    print(f"Shape: {pts.shape}")

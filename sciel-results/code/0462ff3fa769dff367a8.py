"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

def _compute_radius(points):
    """Compute the feasible radius for a set of points."""
    if points is None or len(points) == 0:
        return 0.0
    # Distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distances
    d = pdist(points)
    min_dist = np.min(d)
    min_pair = min_dist / 2.0
    
    return min(min_boundary, min_pair)

def _generate_hexagonal_lattice(n_points, spacing=None):
    """Generate a hexagonal lattice of points."""
    if spacing is None:
        spacing = 2.0 / (np.sqrt(n_points) + 1.0)
    
    k = int(1.0 / spacing) + 3
    rows = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            rows.append((x, y))
    
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    # Keep points inside the unit circle with some margin
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    
    if len(points) >= n_points:
        idx = np.argsort(radius[mask])[:n_points]
        return points[idx]
    return points

def _generate_ring_configuration(n_points):
    """Generate a ring-based configuration."""
    # Use multiple rings with varying counts
    best_points = None
    best_r = 0.0
    
    # Try different ring structures
    for n_rings in range(3, 10):
        # Distribute points across rings
        ring_counts = []
        remaining = n_points
        for i in range(n_rings):
            if i == n_rings - 1:
                ring_counts.append(remaining)
            else:
                # More points in outer rings
                count = int(n_points * (i + 1) / (n_rings * (n_rings + 1) / 2))
                count = max(1, min(count, remaining))
                ring_counts.append(count)
                remaining -= count
        
        if any(rc <= 0 for rc in ring_counts):
            continue
        
        points_list = []
        for i, rc in enumerate(ring_counts):
            r = (i + 1) / (n_rings + 0.5)
            if rc == 1 and i == 0:
                points_list.append(np.array([0.0, 0.0]))
            else:
                angles = np.linspace(0, 2 * np.pi, rc, endpoint=False)
                # Stagger alternating rings
                if i % 2 == 1 and rc > 1:
                    angles += np.pi / rc
                xs = r * np.cos(angles)
                ys = r * np.sin(angles)
                points_list.append(np.column_stack((xs, ys)))
        
        if not points_list:
            continue
        
        points = np.vstack(points_list)
        if len(points) != n_points:
            continue
        
        r = _compute_radius(points)
        if r > best_r:
            best_r = r
            best_points = points
    
    return best_points

def _physics_relax(points, n_iter=2000, dt=0.01, cooling=True):
    """
    Physics-based relaxation using repulsive forces.
    Treats circles as repulsive particles with hard-core constraint.
    """
    points = points.copy()
    n = len(points)
    
    # Parameters
    repulsion_strength = 1.0
    boundary_strength = 10.0
    temperature = 0.01
    
    for it in range(n_iter):
        if time.time() > _deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random small perturbation
                    diff = np.random.randn(2) * 1e-10
                    dist = np.linalg.norm(diff)
                
                # Repulsive force (inverse square law, capped)
                if dist < 2.0:
                    force_mag = repulsion_strength / (dist * dist + 1e-6)
                    force = force_mag * diff / dist
                    forces[i] += force
                    forces[j] -= force
        
        # Boundary force (push back inside if too close to edge)
        for i in range(n):
            dist_to_center = np.linalg.norm(points[i])
            if dist_to_center > 0.95:
                # Push toward center
                force = boundary_strength * (dist_to_center - 0.95) * points[i] / dist_to_center
                forces[i] -= force
        
        # Add random perturbation (Langevin dynamics)
        noise = np.random.randn(n, 2) * temperature
        forces += noise
        
        # Update positions
        points += dt * forces
        
        # Cool down temperature
        if cooling:
            temperature *= 0.999
        
        # Hard constraint: ensure points stay inside unit circle
        dists = np.linalg.norm(points, axis=1)
        mask = dists > 0.99
        if np.any(mask):
            points[mask] *= 0.99 / dists[mask]
    
    return points

def _gradient_ascent(points, n_iter=500, lr=0.001):
    """
    Gradient ascent on a soft-penalty energy function.
    Maximizes the minimum of (pair distances/2, boundary distances).
    """
    points = points.copy()
    n = len(points)
    
    def soft_min(values, beta=50.0):
        """Smooth approximation of the minimum."""
        return -np.log(np.sum(np.exp(-beta * values))) / beta
    
    for it in range(n_iter):
        if time.time() > _deadline:
            break
        
        # Compute current constraints
        d = pdist(points)
        min_pair = np.min(d) / 2.0
        dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
        min_boundary = np.min(dist_boundary)
        
        current_r = min(min_pair, min_boundary)
        
        # Compute gradients (approximate)
        # For simplicity, use finite differences
        eps = 1e-6
        grad = np.zeros_like(points)
        
        # Sample a subset of points for gradient computation (for speed)
        if n > 50:
            sample_idx = np.random.choice(n, size=50, replace=False)
        else:
            sample_idx = np.arange(n)
        
        for i in sample_idx:
            for dim in range(2):
                orig = points[i, dim]
                points[i, dim] = orig + eps
                d_new = pdist(points)
                min_pair_new = np.min(d_new) / 2.0
                dist_boundary_new = 1.0 - np.linalg.norm(points, axis=1)
                min_boundary_new = np.min(dist_boundary_new)
                r_new = min(min_pair_new, min_boundary_new)
                
                points[i, dim] = orig - eps
                d_new2 = pdist(points)
                min_pair_new2 = np.min(d_new2) / 2.0
                dist_boundary_new2 = 1.0 - np.linalg.norm(points, axis=1)
                min_boundary_new2 = np.min(dist_boundary_new2)
                r_new2 = min(min_pair_new2, min_boundary_new2)
                
                points[i, dim] = orig
                grad[i, dim] = (r_new - r_new2) / (2 * eps)
        
        # Update
        points += lr * grad
        
        # Enforce boundary constraint
        dists = np.linalg.norm(points, axis=1)
        mask = dists > 0.99
        if np.any(mask):
            points[mask] *= 0.99 / dists[mask]
        
        # Decay learning rate
        lr *= 0.995
    
    return points

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 140.0  # 10s margin before 150s limit
    
    best_points = None
    best_r = 0.0
    
    # Strategy 1: Hexagonal lattice
    if time.time() < _deadline:
        pts = _generate_hexagonal_lattice(N)
        if pts is not None and len(pts) == N:
            r = _compute_radius(pts)
            if r > best_r:
                best_r = r
                best_points = pts
    
    # Strategy 2: Ring configuration
    if time.time() < _deadline:
        pts = _generate_ring_configuration(N)
        if pts is not None:
            r = _compute_radius(pts)
            if r > best_r:
                best_r = r
                best_points = pts
    
    # Strategy 3: Physics-based relaxation on best candidate
    if best_points is not None and time.time() < _deadline - 20:
        try:
            relaxed = _physics_relax(best_points, n_iter=1000, dt=0.005)
            r_relaxed = _compute_radius(relaxed)
            if r_relaxed > best_r:
                best_r = r_relaxed
                best_points = relaxed
        except Exception:
            pass
    
    # Strategy 4: Gradient ascent on best candidate
    if best_points is not None and time.time() < _deadline - 10:
        try:
            ascended = _gradient_ascent(best_points, n_iter=200, lr=0.002)
            r_ascended = _compute_radius(ascended)
            if r_ascended > best_r:
                best_r = r_ascended
                best_points = ascended
        except Exception:
            pass
    
    # Final physics relaxation if time remains
    if best_points is not None and time.time() < _deadline - 5:
        try:
            final = _physics_relax(best_points, n_iter=500, dt=0.002, cooling=True)
            r_final = _compute_radius(final)
            if r_final > best_r:
                best_r = r_final
                best_points = final
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: simple hexagonal
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        k = int(1.0 / spacing) + 3
        rows = [(((i + 0.5 * (j % 2)) * spacing), j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            idx = np.argsort(radius[radius <= 1.0 - spacing / 2.0])[:N]
            return points[idx]
        return points[:N]
    
    return best_points
# EVOLVE-BLOCK-END

"""Packing 300 equal circles in a unit circle using rigid-body mechanics approach."""
import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START

def _hex_lattice_init(n_points, radius=1.0):
    """Generate initial points on a hexagonal lattice within the unit disk."""
    s = np.sqrt(2.0 * np.pi / (n_points * np.sqrt(3.0) / 2.0))
    s = s * 0.92
    
    k = int(1.0 / s) + 2
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= radius - s / 2.0:
                points.append((x, y))
    
    points = np.array(points)
    if len(points) < n_points:
        while len(points) < n_points:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, radius - s / 2.0)
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            points = np.vstack([points, [[x, y]]])
    
    radii = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(radii, kind='stable')[:n_points]
    return points[idx]


def _compute_score(points):
    """Compute the packing score: min of pairwise distances/2 and distance to boundary."""
    n = len(points)
    if n < 2:
        return 0.0
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary_dist = np.min(1.0 - radii)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)


def _spring_force(points, target_dist, k_spring=100.0, k_boundary=500.0):
    """
    Compute spring forces to maintain target distance between particles.
    Also includes boundary repulsion.
    """
    n = len(points)
    forces = np.zeros_like(points)
    
    # Pairwise spring forces
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                continue
            
            # Spring force: pull together if dist > target, push apart if dist < target
            # F = k * (target - dist) * unit_vector
            if dist < target_dist:
                # Push apart
                f = k_spring * (target_dist - dist)
            else:
                # Weak attraction to maintain structure
                f = k_spring * 0.1 * (target_dist - dist)
            
            unit = diff / dist
            forces[i] += f * unit
            forces[j] -= f * unit
    
    # Boundary forces: push inward if too close to boundary
    for i in range(n):
        r = np.linalg.norm(points[i])
        if r > 0.01:
            # Boundary repulsion: stronger as we approach the wall
            boundary_margin = 1.0 - r
            if boundary_margin < 0.1:
                f = k_boundary * (0.1 - boundary_margin) ** 2
                unit = -points[i] / r
                forces[i] += f * unit
    
    return forces


def _compute_kinetic_energy(points, velocities):
    """Compute kinetic energy of the system."""
    return 0.5 * np.sum(velocities ** 2)


def _compute_potential_energy(points, target_dist, k_spring=100.0, k_boundary=500.0):
    """Compute potential energy of the spring system."""
    n = len(points)
    pe = 0.0
    
    # Pairwise spring potential
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                continue
            displacement = target_dist - dist
            pe += 0.5 * k_spring * displacement ** 2
    
    # Boundary potential
    for i in range(n):
        r = np.linalg.norm(points[i])
        boundary_margin = 1.0 - r
        if boundary_margin < 0.1:
            pe += 0.5 * k_boundary * (0.1 - boundary_margin) ** 2
    
    return pe


def _verlet_step(points, velocities, forces, dt, damping=0.99):
    """One Verlet integration step with damping."""
    n = len(points)
    masses = np.ones(n)
    
    # Update positions
    new_points = points + velocities * dt + 0.5 * forces / masses * dt ** 2
    
    # Compute new forces
    new_forces = _spring_force(new_points, velocities.shape[0] * 0 + 0.053)
    
    # Update velocities with damping
    new_velocities = (velocities + 0.5 * (forces + new_forces) / masses * dt) * damping
    
    return new_points, new_velocities, new_forces


def _optimize_mechanics(points, target_r, max_iters=5000, dt=0.001, damping=0.995):
    """
    Optimize using rigid-body mechanics: particles connected by springs.
    """
    n = len(points)
    target_dist = 2 * target_r
    
    velocities = np.zeros_like(points)
    forces = _spring_force(points, target_dist)
    
    best_points = points.copy()
    best_score = _compute_score(points)
    
    for it in range(max_iters):
        # Verlet step
        points, velocities, forces = _verlet_step(points, velocities, forces, dt, damping)
        
        # Ensure points stay inside unit circle
        radii = np.hypot(points[:, 0], points[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            for i in np.where(mask)[0]:
                scale = 0.999 / radii[i]
                points[i] *= scale
        
        # Compute score every 50 iterations
        if it % 50 == 0:
            score = _compute_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
        
        # Damping increases over time for stability
        damping = 0.995 * (0.999 ** it)
    
    return best_points, best_score


def _gradient_ascent(points, lr=0.01, max_iters=1000):
    """
    Gradient ascent on the score using finite differences.
    """
    best_points = points.copy()
    best_score = _compute_score(points)
    
    for it in range(max_iters):
        # Compute gradient via finite differences
        grad = np.zeros_like(points)
        eps = 1e-6
        
        for i in range(len(points)):
            for d in range(2):
                p_plus = points.copy()
                p_plus[i, d] += eps
                score_plus = _compute_score(p_plus)
                
                p_minus = points.copy()
                p_minus[i, d] -= eps
                score_minus = _compute_score(p_minus)
                
                grad[i, d] = (score_plus - score_minus) / (2 * eps)
        
        # Update
        points = points + lr * grad
        
        # Ensure inside unit circle
        radii = np.hypot(points[:, 0], points[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            for i in np.where(mask)[0]:
                scale = 0.999 / radii[i]
                points[i] *= scale
        
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Adaptive learning rate
        lr *= 0.999
    
    return best_points, best_score


def construct():
    """Construct 300 circle centers in the unit disk maximizing the minimum radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial guess: hexagonal lattice
    np.random.seed(42)
    points = _hex_lattice_init(N)
    
    # Compute initial score
    current_score = _compute_score(points)
    best_points = points.copy()
    best_score = current_score
    
    # Stage 1: Mechanics-based optimization with increasing target radius
    target_r = best_score
    for stage in range(5):
        if time.time() > deadline:
            break
        
        target_r = best_score * (1.0 + 0.02 * (stage + 1))
        
        # Run mechanics optimization
        new_points, new_score = _optimize_mechanics(
            best_points, target_r, 
            max_iters=2000, dt=0.0005, damping=0.998
        )
        
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
    
    # Stage 2: Gradient ascent polish
    if time.time() < deadline:
        new_points, new_score = _gradient_ascent(best_points, lr=0.005, max_iters=200)
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
    
    # Stage 3: Another mechanics pass with tighter target
    if time.time() < deadline:
        target_r = best_score
        new_points, new_score = _optimize_mechanics(
            best_points, target_r,
            max_iters=3000, dt=0.0003, damping=0.999
        )
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
    
    # Stage 4: Final gradient polish
    if time.time() < deadline:
        new_points, new_score = _gradient_ascent(best_points, lr=0.002, max_iters=100)
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
    
    # Ensure all points are strictly inside the unit circle
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    mask = radii < 1.0
    if not np.all(mask):
        for i in range(len(best_points)):
            if radii[i] >= 1.0:
                scale = 0.999 / radii[i] if radii[i] > 0 else 1.0
                best_points[i] *= scale
    
    return best_points
# EVOLVE-BLOCK-END

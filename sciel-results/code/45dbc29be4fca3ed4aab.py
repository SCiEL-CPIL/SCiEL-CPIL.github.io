"""Pack 50 equal circles in a unit circle using a repulsive force-field annealing approach."""
import numpy as np
import time

N = 50


def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    # Distance to boundary
    dists_to_boundary = 1.0 - np.linalg.norm(centers, axis=1)
    min_boundary = np.min(dists_to_boundary)
    
    # Pairwise distances
    min_pair = np.inf
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair:
                min_pair = d
    min_pair = min_pair / 2.0
    
    return min(min_boundary, min_pair)


def _repulsive_forces(centers, power=2.0, cutoff=0.5):
    """
    Compute repulsive forces between particles using a soft repulsive potential.
    Uses Method of Images concept: we add a virtual charge at the center to
    create an effective boundary repulsion.
    """
    n = len(centers)
    forces = np.zeros_like(centers)
    
    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Jitter
                diff = np.array([np.random.randn(), np.random.randn()]) * 1e-6
                dist = np.linalg.norm(diff)
            # Repulsive force: F = k * r^(1-power) * r_hat, with soft cutoff
            if dist < cutoff:
                f_mag = (cutoff - dist) ** power / dist
                forces[i] += f_mag * diff
                forces[j] -= f_mag * diff
    
    # Boundary repulsion: push particles inward if they get close to boundary
    for i in range(n):
        r = np.linalg.norm(centers[i])
        if r > 0.95:
            # Strong repulsion from boundary
            f_mag = (r - 0.95) ** 2 * 5.0
            forces[i] -= f_mag * centers[i] / r
    
    return forces


def _ring_init(n_rings, total_n):
    """Initialize particles in concentric rings."""
    centers = np.zeros((total_n, 2))
    
    # Center particle
    centers[0] = [0.0, 0.0]
    remaining = total_n - 1
    ring_idx = 1
    while remaining > 0:
        # Number of particles in this ring
        if n_rings == 1 or ring_idx == n_rings:
            n_in_ring = remaining
        else:
            # Distribute remaining particles among remaining rings
            n_in_ring = max(1, int(remaining / (n_rings - ring_idx + 1)))
        
        # Radius of this ring
        # Approximate radius based on circumference
        if n_in_ring == 1:
            ring_r = 0.5
        else:
            # Circumference ~ 2*pi*r, spacing ~ 2*r_particle
            # For now, use a heuristic
            ring_r = ring_idx * 0.25
        
        # Place particles on the ring
        for k in range(n_in_ring):
            idx = ring_idx + k
            if idx >= total_n:
                break
            angle = 2 * np.pi * k / n_in_ring + (ring_idx % 2) * np.pi / n_in_ring
            centers[idx] = [ring_r * np.cos(angle), ring_r * np.sin(angle)]
        
        remaining -= n_in_ring
        ring_idx += 1
    
    return centers


def _anneal(centers, max_iter=5000, initial_temp=0.01, cool_rate=0.999):
    """Simulated annealing with repulsive forces."""
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    temp = initial_temp
    for it in range(max_iter):
        # Compute forces
        forces = _repulsive_forces(centers)
        
        # Move particles
        step = temp * 0.1
        new_centers = centers + forces * step + np.random.randn(*centers.shape) * temp * 0.01
        
        # Keep inside boundary
        norms = np.linalg.norm(new_centers, axis=1)
        mask = norms > 0.98
        new_centers[mask] = 0.98 * new_centers[mask] / norms[mask, None]
        
        # Compute new radius
        new_radius = _compute_radius(new_centers)
        
        # Accept or reject
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = new_centers.copy()
        
        # Metropolis criterion: accept worse solutions with probability exp(-delta/T)
        delta = best_radius - new_radius
        if delta < 0 or np.random.rand() < np.exp(-delta / (temp + 1e-10)):
            centers = new_centers
        else:
            centers = centers  # Keep old
        
        # Cool down
        temp *= cool_rate
        
        if temp < 1e-6:
            break
    
    return best_centers, best_radius


def _polish(centers, max_iter=2000):
    """Gradient-based polish to improve the packing."""
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    lr = 0.001
    for it in range(max_iter):
        # Compute gradient of the minimum radius
        # We want to maximize min(min_pairwise/2, min_boundary)
        # Use subgradient method
        forces = np.zeros_like(centers)
        
        # Find the binding constraint
        min_boundary = 1.0 - np.min(np.linalg.norm(centers, axis=1))
        min_pair = np.inf
        bind_i, bind_j = -1, -1
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_pair:
                    min_pair = d
                    bind_i, bind_j = i, j
        min_pair /= 2.0
        
        current_r = min(min_boundary, min_pair)
        
        if min_boundary < min_pair:
            # Boundary is binding: push inward
            for i in range(len(centers)):
                r = np.linalg.norm(centers[i])
                if r > 1e-10:
                    forces[i] = -centers[i] / r
        else:
            # Pairwise is binding: push apart
            if bind_i >= 0:
                diff = centers[bind_i] - centers[bind_j]
                d = np.linalg.norm(diff)
                if d > 1e-10:
                    forces[bind_i] = diff / d
                    forces[bind_j] = -diff / d
        
        # Add small random perturbation to escape local minima
        forces += np.random.randn(*centers.shape) * 0.001
        
        new_centers = centers + lr * forces
        
        # Keep inside boundary
        norms = np.linalg.norm(new_centers, axis=1)
        mask = norms > 0.99
        new_centers[mask] = 0.99 * new_centers[mask] / norms[mask, None]
        
        new_radius = _compute_radius(new_centers)
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = new_centers.copy()
            lr *= 1.01  # Increase learning rate slightly
        else:
            centers = new_centers
            lr *= 0.95
        
        if lr < 1e-7:
            break
    
    return best_centers, best_radius


def construct():
    """
    Construct the packing of 50 equal circles in a unit circle.
    Uses a repulsive force-field annealing approach with Method of Images
    concept for boundary repulsion.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_centers = None
    best_radius = 0.0
    
    # Try multiple initializations
    n_rings_options = [2, 3, 4, 5]
    
    for n_rings in n_rings_options:
        if time.time() > deadline - 30.0:
            break
        
        # Initialize
        centers = _ring_init(n_rings, N)
        
        # Annealing phase
        centers, radius = _anneal(centers, max_iter=3000, initial_temp=0.02, cool_rate=0.9995)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
        
        # Polish phase
        if time.time() > deadline - 15.0:
            break
        centers, radius = _polish(centers, max_iter=1000)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    # Final polish with more iterations if time allows
    if best_centers is not None and time.time() < deadline - 5.0:
        centers, radius = _polish(best_centers, max_iter=2000)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback: simple hexagonal lattice
        centers = np.zeros((N, 2))
        for i in range(N):
            r = 0.8 * np.sqrt((i + 0.5) / N)
            theta = 2 * np.pi * i / N
            centers[i] = [r * np.cos(theta), r * np.sin(theta)]
    
    return best_centers

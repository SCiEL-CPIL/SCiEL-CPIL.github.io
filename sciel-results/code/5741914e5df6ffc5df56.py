"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import random

N = 50

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.linalg.norm(centers, axis=1)
    r_boundary = np.min(dists_boundary)
    if r_boundary < 0:
        return 0.0
    # Pairwise distances
    r_pairs = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < r_pairs:
                r_pairs = d
    return min(r_pairs / 2.0, r_boundary)


def _objective(centers_flat):
    """Objective to maximize (we minimize the negative)."""
    centers = centers_flat.reshape(N, 2)
    return -_compute_radius(centers)


def _hexagonal_init():
    """Create a hexagonal lattice initialization."""
    points = []
    # Try various spacings
    best_points = None
    best_count = 0
    for spacing in np.linspace(0.1, 0.3, 50):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        pts = np.array(rows)
        radius = np.hypot(pts[:, 0], pts[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        pts = pts[mask]
        if len(pts) >= N:
            # Sort by distance from origin
            idx = np.argsort(radius[mask])[:N]
            pts = pts[idx]
            if len(pts) > best_count:
                best_count = len(pts)
                best_points = pts
    if best_points is None:
        # Fallback: random
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.random.uniform(0, 0.9, N)
        best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return best_points


def _ring_init():
    """Create a ring-based initialization."""
    centers = []
    # Center
    centers.append([0.0, 0.0])
    # Rings
    ring_radii = [0.3, 0.55, 0.75, 0.9]
    counts_per_ring = [12, 16, 15, 6]  # Total: 1+12+16+15+6 = 50
    for r, count in zip(ring_radii, counts_per_ring):
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + np.random.uniform(0, 0.1)
        for a in angles:
            centers.append([r * np.cos(a), r * np.sin(a)])
    return np.array(centers[:N])


def _force_directed_relaxation(centers, max_iters=2000, deadline=None):
    """Force-directed relaxation with adaptive time-stepping."""
    centers = centers.copy()
    dt = 0.01
    min_dt = 1e-6
    max_dt = 0.05
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        forces = np.zeros_like(centers)
        
        # Repulsive forces between circles
        for i in range(N):
            for j in range(i + 1, N):
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    d = np.random.uniform(-0.01, 0.01, 2)
                    dist = np.linalg.norm(d)
                # Strong repulsion when close
                if dist < 0.3:
                    force_mag = 10.0 / (dist * dist)
                elif dist < 0.5:
                    force_mag = 1.0 / dist
                else:
                    force_mag = 0.0
                forces[i] += force_mag * d / dist
                forces[j] -= force_mag * d / dist
        
        # Attractive force toward boundary (to maximize radius)
        for i in range(N):
            r = np.linalg.norm(centers[i])
            if r > 1e-10:
                # Gentle pull outward
                forces[i] += 0.5 * centers[i] / r
        
        # Boundary constraint: push inward if too close to boundary
        for i in range(N):
            r = np.linalg.norm(centers[i])
            if r > 0.95:
                forces[i] -= 2.0 * centers[i] / r
        
        # Adaptive time step
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 1.0:
            dt = max_dt * 0.5
        elif max_force < 0.1:
            dt = min(max_dt, dt * 1.1)
        else:
            dt = min(max_dt, max(min_dt, dt))
        
        # Update positions
        new_centers = centers + dt * forces
        
        # Project back inside unit circle
        for i in range(N):
            r = np.linalg.norm(new_centers[i])
            if r > 0.99:
                new_centers[i] = 0.99 * new_centers[i] / r
        
        # Check if valid
        new_radius = _compute_radius(new_centers)
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = new_centers.copy()
            centers = new_centers
        else:
            # Reduce step
            dt *= 0.9
            centers = centers  # Keep old positions
        
    return best_centers


def _gradient_descent(centers, max_iters=5000, deadline=None):
    """Gradient descent on the soft-min objective."""
    centers = centers.copy()
    lr = 0.01
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient of -radius
        grad = np.zeros_like(centers)
        
        # Gradient from pairwise distances
        for i in range(N):
            for j in range(i + 1, N):
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    continue
                # Contribution to radius from this pair
                contrib = dist / 2.0
                # Find the minimum contribution
                # Approximate gradient
                grad[i] += 0.1 * d / dist
                grad[j] -= 0.1 * d / dist
        
        # Gradient from boundary
        for i in range(N):
            r = np.linalg.norm(centers[i])
            if r > 1e-10:
                grad[i] -= 0.1 * centers[i] / r
        
        # Update
        new_centers = centers - lr * grad
        
        # Project back inside
        for i in range(N):
            r = np.linalg.norm(new_centers[i])
            if r > 0.99:
                new_centers[i] = 0.99 * new_centers[i] / r
        
        new_radius = _compute_radius(new_centers)
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = new_centers.copy()
            centers = new_centers
            lr = min(lr * 1.05, 0.05)
        else:
            lr *= 0.9
            if lr < 1e-6:
                break
        
    return best_centers


def construct():
    """Main function to construct the circle packing."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Generate multiple initializations
    inits = []
    
    # Hexagonal
    inits.append(_hexagonal_init())
    
    # Ring-based
    inits.append(_ring_init())
    
    # Random with some structure
    for _ in range(5):
        angles = np.sort(np.random.uniform(0, 2 * np.pi, N))
        radii = np.random.uniform(0.2, 0.95, N)
        centers = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(centers)
    
    # Best result so far
    best_centers = None
    best_radius = 0.0
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Force-directed relaxation
        centers_fd = _force_directed_relaxation(init, max_iters=1000, deadline=deadline)
        r_fd = _compute_radius(centers_fd)
        if r_fd > best_radius:
            best_radius = r_fd
            best_centers = centers_fd.copy()
        
        # Gradient descent from the relaxed state
        centers_gd = _gradient_descent(centers_fd, max_iters=2000, deadline=deadline)
        r_gd = _compute_radius(centers_gd)
        if r_gd > best_radius:
            best_radius = r_gd
            best_centers = centers_gd.copy()
        
        # Also try gradient descent from original init
        if time.time() < deadline:
            centers_gd2 = _gradient_descent(init, max_iters=2000, deadline=deadline)
            r_gd2 = _compute_radius(centers_gd2)
            if r_gd2 > best_radius:
                best_radius = r_gd2
                best_centers = centers_gd2.copy()
    
    # If we still have time, do more refinement
    if best_centers is not None and time.time() < deadline - 5.0:
        # Multiple rounds of force-directed + gradient
        for _ in range(3):
            if time.time() > deadline:
                break
            centers_refined = _force_directed_relaxation(best_centers, max_iters=500, deadline=deadline)
            r_ref = _compute_radius(centers_refined)
            if r_ref > best_radius:
                best_radius = r_ref
                best_centers = centers_refined.copy()
            
            centers_refined2 = _gradient_descent(best_centers, max_iters=1000, deadline=deadline)
            r_ref2 = _compute_radius(centers_refined2)
            if r_ref2 > best_radius:
                best_radius = r_ref2
                best_centers = centers_refined2.copy()
    
    # Final validation
    if best_centers is None:
        best_centers = _hexagonal_init()
    
    # Ensure valid output
    result = np.asarray(best_centers, dtype=np.float64)
    if result.shape != (N, 2) or not np.all(np.isfinite(result)):
        result = _hexagonal_init()
    
    return result
# EVOLVE-BLOCK-END

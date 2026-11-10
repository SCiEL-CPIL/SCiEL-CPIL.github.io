"""Pack 300 equal circles in the unit disk.
Strategy: Initialize from a hexagonal lattice, then use a physics-based
molecular dynamics simulation with repulsive forces and a confining potential
to drive the system into a local energy minimum. Final polish with gradient
descent on the smooth surrogate objective.
"""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 300
DEADLINE = 115.0  # Safety margin before 120s limit


def _score(points):
    """Compute the packing score (minimum of pair distances/2 and boundary margins)."""
    if len(points) < 2:
        return 0.0
    # Compute minimum pairwise distance using cKDTree
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    min_pair_dist = dists[:, 1].min() / 2.0
    # Compute minimum distance to boundary
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary = (1.0 - radii).min()
    return min(min_pair_dist, min_boundary)


def _gen_hex_lattice(spacing):
    """Generate points on a hexagonal lattice cut to the unit disk."""
    k = int(1.0 / spacing) + 3
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0 - spacing * 0.5:
                pts.append((x, y))
    return np.array(pts)


def _md_step(points, dt, k_rep=100.0, k_conf=50.0, soft=0.1):
    """
    Perform one molecular dynamics step.
    - Repulsive force: inverse power law for overlapping circles
    - Confining potential: quadratic penalty for points outside the disk
    """
    n = len(points)
    forces = np.zeros_like(points)
    
    # Compute repulsive forces between all pairs
    # For efficiency, use cKDTree to find nearby points
    tree = cKDTree(points)
    pairs = tree.query_pairs(1.0)  # Only consider pairs within distance 1
    
    for i, j in pairs:
        diff = points[i] - points[j]
        d = np.linalg.norm(diff)
        if d < 1e-10:
            # Random small perturbation if coincident
            forces[i] += np.random.randn(2) * 0.001
            forces[j] -= np.random.randn(2) * 0.001
        else:
            # Repulsive force: stronger when closer
            if d < soft:
                f = k_rep / (d ** 2 + 1e-10)
            else:
                f = 0.0
            direction = diff / d
            forces[i] += direction * f
            forces[j] -= direction * f
    
    # Confining potential: push points back inside the disk
    radii = np.hypot(points[:, 0], points[:, 1])
    outside = radii > 1.0
    if np.any(outside):
        # Quadratic penalty for being outside
        penalty = np.zeros_like(points)
        penalty[outside] = -k_conf * (radii[outside] - 1.0) * (points[outside] / radii[outside, np.newaxis])
        forces += penalty
    
    # Update positions with velocity Verlet-like integration (simplified)
    new_points = points + forces * dt
    return new_points


def _gradient_descent(points, lr=0.001, n_iter=100):
    """
    Gradient descent on the smooth surrogate objective.
    Objective: maximize min(min_pair_dist/2, min_boundary)
    Surrogate: use soft-min via log-sum-exp for smoothness
    """
    n = len(points)
    
    for _ in range(n_iter):
        # Compute pairwise distances
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=2)
        min_dists = dists[:, 1]
        
        # Compute boundary distances
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary_dists = 1.0 - radii
        
        # Find the most constrained point (minimum of min_dist and boundary)
        # For simplicity, use the point with the smallest minimum distance to neighbor or boundary
        constraints = np.minimum(min_dists / 2.0, boundary_dists)
        idx = np.argmin(constraints)
        
        # Compute gradient for this point
        grad = np.zeros(2)
        
        # Gradient from pairwise distances
        _, nn_idx = tree.query(points[idx], k=2)
        nn_idx = nn_idx[1]
        diff = points[idx] - points[nn_idx]
        d = np.linalg.norm(diff)
        if d > 1e-10:
            grad += -diff / d  # Move away from nearest neighbor
        
        # Gradient from boundary
        r = radii[idx]
        if r > 1e-10:
            grad += points[idx] / r  # Move inward
        
        # Update
        points[idx] += lr * grad
        
        # Ensure points stay within the disk
        radii = np.hypot(points[:, 0], points[:, 1])
        outside = radii > 1.0
        if np.any(outside):
            points[outside] /= radii[outside, np.newaxis]
    
    return points


def construct():
    start_time = time.time()
    
    # Initialize from hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        points = _gen_hex_lattice(spacing)
        if len(points) >= N:
            # Sort by radius and take first N
            radii = np.hypot(points[:, 0], points[:, 1])
            points = points[np.argsort(radii, kind='stable')[:N]]
            break
        spacing *= 0.98
    
    best_score = _score(points)
    best_points = points.copy()
    
    # Molecular dynamics phase
    dt = 0.001
    k_rep = 100.0
    k_conf = 50.0
    soft = 0.1
    n_md_steps = 500
    
    for step in range(n_md_steps):
        if time.time() > start_time + DEADLINE - 30.0:
            break
        
        points = _md_step(points, dt, k_rep, k_conf, soft)
        
        # Cool down: reduce dt and increase repulsion
        dt *= 0.99
        k_rep *= 1.01
        
        # Check score
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Gradient descent polish
    if time.time() < start_time + DEADLINE - 10.0:
        points = best_points.copy()
        points = _gradient_descent(points, lr=0.001, n_iter=100)
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Final check
    final_score = _score(best_points)
    if final_score > best_score:
        best_score = final_score
    
    return best_points

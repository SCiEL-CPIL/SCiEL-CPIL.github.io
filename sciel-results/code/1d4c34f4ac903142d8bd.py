"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax
import time

def _get_icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    return verts

def _generate_mackay_icosahedron(n_target, spacing=1.1):
    """
    Generate a Mackay icosahedral cluster with exactly n_target atoms.
    Shell structure:
      - Core: 1 atom at origin
      - Shell 1 (12 atoms): icosahedron vertices at radius r1
      - Shell 2 (42 atoms): icosahedral second shell at radius r2
    For n up to 60, we use core + shell1 + part of shell2.
    """
    # Core atom
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Get icosahedron vertices and normalize to unit sphere
    verts = _get_icosahedron_vertices()
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Shell 1: 12 vertices at radius r1
    r1 = spacing * 1.0
    for v in verts:
        points.append(r1 * v)
    
    # If we need more atoms, add shell 2
    if n_target > 13:
        # Generate shell 2 positions
        # Shell 2 consists of 42 atoms: 30 at face centers and 12 at edge midpoints (approximately)
        # For simplicity, we'll generate positions on the icosahedron surface at radius r2
        r2 = spacing * np.sqrt(5) / 2 * 1.2  # Approximate second shell radius
        
        # Generate 42 points on the icosahedron surface
        # Using a simple Fibonacci sphere for even distribution
        def fibonacci_sphere(n_points, radius):
            points = []
            golden_angle = np.pi * (3 - np.sqrt(5))
            for i in range(n_points):
                y = 1 - (i / (n_points - 1)) * 2  # y goes from 1 to -1
                radius_at_y = np.sqrt(1 - y**2)  # radius at y
                theta = golden_angle * i  # golden angle increment
                x = np.cos(theta) * radius_at_y
                z = np.sin(theta) * radius_at_y
                points.append(radius * np.array([x, y, z]))
            return points
        
        shell2_points = fibonacci_sphere(42, r2)
        
        # Add shell 2 points until we reach n_target
        for p in shell2_points:
            if len(points) >= n_target:
                break
            # Check minimum distance to existing points
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
    
    # If we still don't have enough, add more points
    while len(points) < n_target:
        # Add random points near the cluster
        angle1 = np.random.uniform(0, 2 * np.pi)
        angle2 = np.random.uniform(0, np.pi)
        r = spacing * np.random.uniform(0.8, 1.5)
        p = r * np.array([
            np.sin(angle2) * np.cos(angle1),
            np.sin(angle2) * np.sin(angle1),
            np.cos(angle2)
        ])
        if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
            points.append(p)
    
    return np.array(points[:n_target])

def _generate_defect_icosahedron(n_target, spacing=1.1, defect_type='remove'):
    """
    Generate an icosahedral cluster with defects for diversity.
    """
    # Start with a slightly larger icosahedron and remove atoms
    n_base = min(55, max(n_target + 5, 20))
    base_cluster = _generate_mackay_icosahedron(n_base, spacing)
    
    if len(base_cluster) < n_target:
        return base_cluster
    
    # Remove atoms to reach target size
    # Remove atoms from the outer shell first (higher indices)
    indices_to_remove = list(range(len(base_cluster)))
    np.random.shuffle(indices_to_remove)
    indices_to_remove = indices_to_remove[:len(base_cluster) - n_target]
    
    # Keep atoms not in removal list
    keep_mask = np.ones(len(base_cluster), dtype=bool)
    keep_mask[indices_to_remove] = False
    defective_cluster = base_cluster[keep_mask]
    
    return defective_cluster

def _generate_random_compact(n, spacing=1.1):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(n)
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
            points.append(p)
    return np.array(points)

def construct(instance):
    """
    Best of multiple icosahedral and defect-seeded starting clusters, each relaxed locally.
    Uses time guard to ensure we finish within 300s.
    """
    start_time = time.time()
    deadline = start_time + 280  # 20s safety margin
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best = None
    
    # Generate multiple diverse starting points
    strategies = [
        ('icosahedral', lambda: _generate_mackay_icosahedron(size)),
        ('defect_remove', lambda: _generate_defect_icosahedron(size, defect_type='remove')),
        ('defect_remove2', lambda: _generate_defect_icosahedron(size, defect_type='remove')),
        ('random', lambda: _generate_random_compact(size)),
        ('random2', lambda: _generate_random_compact(size)),
    ]
    
    for name, gen_func in strategies:
        if time.time() > deadline:
            break
        
        try:
            points = gen_func()
            if len(points) != size:
                continue
            
            energy, positions = relax(points, potential, parameter)
            
            if best is None or energy < best[0]:
                best = (energy, positions)
        except Exception:
            continue
    
    # If we still don't have a result, fall back to random
    if best is None:
        points = _generate_random_compact(size)
        energy, positions = relax(points, potential, parameter)
        best = (energy, positions)
    
    return best[1]
# EVOLVE-BLOCK-END

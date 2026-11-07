"""Sutton-Chen transition-metal cluster global minima - multi-template with relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
import time

# Known Doye-Wales cluster sizes and their approximate energies for reference
# We'll use template-based construction with multiple symmetry types

def _fcc_cluster(n):
    """Generate a truncated FCC cluster with n atoms."""
    # FCC lattice points: positions where all coords are integers or all are half-integers
    # with even sum, scaled to nearest-neighbor distance ~1.0
    # For SC potential, equilibrium NN distance is around 0.72-0.75
    a = 0.72  # lattice constant
    positions = []
    max_radius = a * (n / 4.0) ** (1.0 / 3.0) * 1.2
    
    # Generate FCC lattice points
    n_max = int(max_radius / a) + 2
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            for k in range(-n_max, n_max + 1):
                # FCC: integer points with i+j+k even, plus face-center offsets
                r = np.array([i, j, k]) * a
                if np.linalg.norm(r) <= max_radius:
                    if (i + j + k) % 2 == 0:
                        positions.append(r)
                # Face centers
                for dc in [(0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    r2 = np.array([i, j, k]) * a + np.array(dc) * a
                    if np.linalg.norm(r2) <= max_radius:
                        positions.append(r2)
    
    # Remove duplicates and sort by radius
    positions = np.array(positions)
    if len(positions) == 0:
        return np.zeros((n, 3)) * 0.72
    
    # Deduplicate
    unique = []
    for p in positions:
        if not any(np.linalg.norm(p - q) < 0.1 for q in unique):
            unique.append(p)
    positions = np.array(unique)
    
    # Sort by distance from origin and take n closest
    dists = np.linalg.norm(positions, axis=1)
    order = np.argsort(dists)
    positions = positions[order]
    
    if len(positions) >= n:
        positions = positions[:n]
    else:
        # Pad with random points near surface
        while len(positions) < n:
            angle1 = np.random.uniform(0, 2 * np.pi)
            angle2 = np.random.uniform(0, np.pi)
            r = max_radius * 0.95
            p = np.array([r * np.sin(angle2) * np.cos(angle1),
                         r * np.sin(angle2) * np.sin(angle1),
                         r * np.cos(angle2)])
            positions = np.vstack([positions, p])
    
    return positions[:n]


def _hcp_cluster(n):
    """Generate a truncated HCP cluster with n atoms."""
    a = 0.72  # nearest-neighbor distance
    c = a * np.sqrt(8.0 / 3.0)  # c/a ratio for ideal HCP
    
    positions = []
    max_layers = int((n / 2.0) ** 0.5) + 3
    
    for layer in range(-max_layers, max_layers + 1):
        z = layer * c / 2.0
        # Each layer is a hexagonal close-packed 2D arrangement
        for i in range(-max_layers, max_layers + 1):
            for j in range(-max_layers, max_layers + 1):
                # Hexagonal coordinates
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0) / 2.0 * j
                r = np.array([x, y, z])
                if np.linalg.norm(r) <= a * (n / 2.0) ** (1.0 / 3.0) * 1.3:
                    positions.append(r)
    
    positions = np.array(positions)
    if len(positions) == 0:
        return np.zeros((n, 3)) * 0.72
    
    # Deduplicate
    unique = []
    for p in positions:
        if not any(np.linalg.norm(p - q) < 0.1 for q in unique):
            unique.append(p)
    positions = np.array(unique)
    
    dists = np.linalg.norm(positions, axis=1)
    order = np.argsort(dists)
    positions = positions[order]
    
    if len(positions) >= n:
        positions = positions[:n]
    else:
        while len(positions) < n:
            angle1 = np.random.uniform(0, 2 * np.pi)
            angle2 = np.random.uniform(0, np.pi)
            r = a * (n / 2.0) ** (1.0 / 3.0)
            p = np.array([r * np.sin(angle2) * np.cos(angle1),
                         r * np.sin(angle2) * np.sin(angle1),
                         r * np.cos(angle2)])
            positions = np.vstack([positions, p])
    
    return positions[:n]


def _icosahedral_cluster(n):
    """Generate an icosahedral Mackay-type cluster."""
    # Icosahedral vertices
    t = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
    ])
    # Actually, proper icosahedron has 12 vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts[0])
    
    # Mackay icosahedra: 13, 55, 147, 309, ...
    # For general n, use concentric icosahedral shells
    shell_sizes = [1, 12, 42, 92, 162, 252]
    cumulative = [0]
    for s in shell_sizes:
        cumulative.append(cumulative[-1] + s)
    
    positions = [np.zeros(3)]  # center atom
    
    # Generate shell points
    for shell_idx in range(1, len(shell_sizes) + 1):
        shell_n = shell_sizes[shell_idx - 1]
        shell_r = shell_idx * 0.72
        
        # Generate points on icosahedral shell
        # Use the 12 vertices and their rotations
        for v in verts:
            for _ in range(shell_n // 12 + 1):
                if len(positions) >= n:
                    break
                # Add point at radius shell_r in direction v with small perturbation
                p = v * shell_r
                if not any(np.linalg.norm(p - q) < 0.3 for q in positions):
                    positions.append(p)
        if len(positions) >= n:
            break
    
    positions = np.array(positions)
    if len(positions) >= n:
        return positions[:n]
    else:
        # Pad
        while len(positions) < n:
            angle1 = np.random.uniform(0, 2 * np.pi)
            angle2 = np.random.uniform(0, np.pi)
            r = len(positions) ** (1.0 / 3.0) * 0.72
            p = np.array([r * np.sin(angle2) * np.cos(angle1),
                         r * np.sin(angle2) * np.sin(angle1),
                         r * np.cos(angle2)])
            positions = np.vstack([positions, p])
        return positions[:n]


def _spherical_shells(n):
    """Generate a spherical shell cluster."""
    positions = [np.zeros(3)]  # center
    shells = [1, 12, 42, 92, 162]
    shell_radii = [0, 0.72, 1.44, 2.16, 2.88]
    
    for shell_idx in range(1, len(shells) + 1):
        target = shells[shell_idx - 1]
        r = shell_radii[shell_idx - 1]
        
        # Use Fibonacci sphere for uniform distribution
        golden = (1 + np.sqrt(5)) / 2
        for i in range(target):
            if len(positions) >= n:
                break
            z = 1 - 2 * (i + 0.5) / target
            radius_xy = np.sqrt(max(0, 1 - z * z))
            theta = 2 * np.pi * i / golden
            p = np.array([r * radius_xy * np.cos(theta),
                         r * radius_xy * np.sin(theta),
                         r * z])
            if not any(np.linalg.norm(p - q) < 0.35 for q in positions):
                positions.append(p)
    
    positions = np.array(positions)
    if len(positions) >= n:
        return positions[:n]
    else:
        while len(positions) < n:
            angle1 = np.random.uniform(0, 2 * np.pi)
            angle2 = np.random.uniform(0, np.pi)
            r = n ** (1.0 / 3.0) * 0.72
            p = np.array([r * np.sin(angle2) * np.cos(angle1),
                         r * np.sin(angle2) * np.sin(angle1),
                         r * np.cos(angle2)])
            positions = np.vstack([positions, p])
        return positions[:n]


def _random_compact(n, seed=42):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = 0.72 * n ** (1.0 / 3.0) * 0.9
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 50:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.65 for q in points):
            points.append(p)
        attempts += 1
    
    # If we couldn't place enough, add more densely
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            points.append(p)
    
    return np.array(points[:n])


def construct(instance):
    """Multi-template construction with local relaxation."""
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    
    start_time = time.time()
    deadline = start_time + 280  # Leave 50s margin for safety
    
    # Generate multiple templates
    templates = []
    
    # FCC template
    try:
        templates.append(_fcc_cluster(n))
    except Exception:
        pass
    
    # HCP template
    try:
        templates.append(_hcp_cluster(n))
    except Exception:
        pass
    
    # Icosahedral template
    try:
        templates.append(_icosahedral_cluster(n))
    except Exception:
        pass
    
    # Spherical shells
    try:
        templates.append(_spherical_shells(n))
    except Exception:
        pass
    
    # Random compact with different seeds
    for seed in [42, 123, 777, 2024]:
        try:
            templates.append(_random_compact(n, seed=seed))
        except Exception:
            pass
    
    # Relax each template and keep the best
    best_energy = np.inf
    best_positions = None
    
    for i, template in enumerate(templates):
        if time.time() > deadline:
            break
        
        try:
            energy, positions = relax(template, 'sutton_chen', kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    # If no valid result, try one more random attempt
    if best_positions is None:
        try:
            template = _random_compact(n, seed=999)
            energy, positions = relax(template, 'sutton_chen', kind)
            best_positions = positions
            best_energy = energy
        except Exception:
            # Last resort: simple linear arrangement
            best_positions = np.zeros((n, 3))
            for i in range(n):
                best_positions[i] = [i * 0.72, 0, 0]
    
    # Verify minimum distance constraint
    if best_positions is not None and len(best_positions) == n:
        # Check for any pair closer than 0.35
        from scipy.spatial.distance import pdist
        min_dist = pdist(best_positions)
        if np.min(min_dist) < 0.35:
            # Perturb to fix
            rng = np.random.default_rng(42)
            for _ in range(10):
                perturbed = best_positions + rng.normal(0, 0.05, best_positions.shape)
                min_d = pdist(perturbed)
                if np.min(min_d) >= 0.35:
                    best_positions = perturbed
                    break
    
    return best_positions
# EVOLVE-BLOCK-END

"""Rewritten program: systematic multi-structure initialization + shake-relax refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _time_left(deadline, margin=2.0):
    return (deadline - margin) - time.time() > 0


def _is_valid(positions, min_dist=0.35):
    if not np.all(np.isfinite(positions)):
        return False
    n = positions.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i] - positions[j]) < min_dist:
                return False
    return True


def _make_fcc(n, spacing=1.0):
    """Generate n points from an FCC lattice, centered, sorted by radius."""
    pts = []
    max_coord = int(np.ceil(n ** (1.0 / 3.0) * 1.5)) + 3
    fcc_pts = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC primitive vectors
                fcc_pts.append(np.array([i, j, k]) * spacing)
                fcc_pts.append(np.array([i, j, k]) * spacing + np.array([0, 0.5, 0.5]) * spacing)
                fcc_pts.append(np.array([i, j, k]) * spacing + np.array([0.5, 0, 0.5]) * spacing)
                fcc_pts.append(np.array([i, j, k]) * spacing + np.array([0.5, 0.5, 0]) * spacing)
    fcc_pts = np.array(fcc_pts)
    # Center on centroid
    fcc_pts -= np.mean(fcc_pts, axis=0)
    # Sort by distance from origin
    dists = np.linalg.norm(fcc_pts, axis=1)
    idx = np.argsort(dists)
    pts = fcc_pts[idx[:n]]
    return pts


def _make_bcc(n, spacing=1.0):
    """Generate n points from a BCC lattice, centered, sorted by radius."""
    pts = []
    max_coord = int(np.ceil(n ** (1.0 / 3.0) * 1.5)) + 3
    bcc_pts = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                bcc_pts.append(np.array([i, j, k]) * spacing)
                bcc_pts.append(np.array([i, j, k]) * spacing + np.array([0.5, 0.5, 0.5]) * spacing)
    bcc_pts = np.array(bcc_pts)
    bcc_pts -= np.mean(bcc_pts, axis=0)
    dists = np.linalg.norm(bcc_pts, axis=1)
    idx = np.argsort(dists)
    pts = bcc_pts[idx[:n]]
    return pts


def _make_hcp(n, spacing=1.0):
    """Generate n points from an HCP-like structure."""
    pts = []
    # HCP: close-packed layers with ABAB stacking
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    max_layer = int(np.ceil(np.sqrt(n / 2.0))) + 2
    for layer in range(-max_layer, max_layer + 1):
        z = layer * c
        # AB stacking
        if layer % 2 == 0:
            offset = np.array([0, 0, 0])
        else:
            offset = np.array([a / np.sqrt(3.0), a / 2.0, 0])
        for i in range(-max_layer, max_layer + 1):
            for j in range(-max_layer, max_layer + 1):
                p = np.array([i * a + (j % 2) * a / 2.0, j * a * np.sqrt(3.0) / 2.0, z]) + offset
                pts.append(p)
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    pts = pts[idx[:n]]
    return pts


def _make_mackay(n):
    """Generate points from a Mackay icosahedron (nested shells)."""
    pts = []
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2.0
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    ico_vertices /= np.linalg.norm(ico_vertices[0])
    
    # Shell radii for Mackay icosahedron
    shell_radii = [1.0, 1.0 * np.sqrt(1 + 2 * phi), 1.0 * np.sqrt(1 + 4 * phi)]
    shell_counts = [12, 42, 92]
    
    target_n = n
    for r, count in zip(shell_radii, shell_counts):
        if target_n <= 0:
            break
        take = min(count, target_n)
        for i in range(take):
            pts.append(ico_vertices[i] * r)
        target_n -= take
    
    # If we still need more, add random points on a larger shell
    if target_n > 0:
        rng = np.random.default_rng(42)
        for _ in range(target_n):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            pts.append(v * shell_radii[-1] * 1.2)
    
    pts = np.array(pts[:n])
    return pts


def _make_spherical_shells(n):
    """Generate points on concentric spherical shells."""
    pts = []
    shell_radii = [0.72, 1.44, 2.16, 2.88]
    # Distribute points among shells
    shell_counts = []
    remaining = n
    for i, r in enumerate(shell_radii):
        if i == len(shell_radii) - 1:
            shell_counts.append(remaining)
        else:
            count = min(int(remaining * 0.4), int(4 * np.pi * r * r / 0.2))
            shell_counts.append(count)
            remaining -= count
    
    rng = np.random.default_rng(123)
    for r, count in zip(shell_radii, shell_counts):
        for _ in range(count):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            pts.append(v * r)
    
    pts = np.array(pts[:n])
    return pts


def _shake_refine(positions, potential, parameter, n_shakes=10, shake_amp=0.1):
    """Apply shake perturbations and relax to find better local minima."""
    best_energy, best_pos = sutton_chen_energy_gradient(positions, parameter)
    rng = np.random.default_rng(hash(positions.tobytes()) % 2**32)
    for _ in range(n_shakes):
        perturbed = positions + rng.normal(0, shake_amp, positions.shape)
        energy, new_pos = relax(perturbed, potential, parameter)
        if energy < best_energy:
            best_energy, best_pos = energy, new_pos
    return best_energy, best_pos


def construct(instance):
    """Systematic multi-structure initialization + shake-relax refinement."""
    start_time = time.time()
    deadline = start_time + 280.0  # 50s safety margin
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Generate diverse structural seeds
    seeds = []
    
    # FCC
    seeds.append(_make_fcc(size, spacing=0.72))
    
    # BCC
    seeds.append(_make_bcc(size, spacing=0.72))
    
    # HCP
    seeds.append(_make_hcp(size, spacing=0.72))
    
    # Mackay icosahedron
    seeds.append(_make_mackay(size))
    
    # Spherical shells
    seeds.append(_make_spherical_shells(size))
    
    # Random compact
    rng = np.random.default_rng(size)
    radius = 0.72 * size ** (1.0 / 3.0) * 0.75
    points = []
    while len(points) < size:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.55 for q in points):
            points.append(p)
    seeds.append(np.array(points))
    
    # Relax each seed and keep the best
    best_energy = np.inf
    best_positions = None
    
    for seed in seeds:
        if not _time_left(deadline):
            break
        if not _is_valid(seed):
            continue
        try:
            energy, positions = relax(seed, potential, parameter)
            if energy < best_energy and _is_valid(positions):
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    if best_positions is None:
        # Fallback: return a simple valid configuration
        best_positions = np.zeros((size, 3))
        for i in range(size):
            best_positions[i] = [i * 0.72, (i % 3) * 0.72, (i // 3) * 0.72]
        return best_positions
    
    # Shake-refine the best result
    if _time_left(deadline):
        try:
            refined_energy, refined_pos = _shake_refine(best_positions, potential, parameter, n_shakes=5, shake_amp=0.08)
            if refined_energy < best_energy and _is_valid(refined_pos):
                best_energy = refined_energy
                best_positions = refined_pos
        except Exception:
            pass
    
    # Final high-precision polish if time permits
    if _time_left(deadline, margin=5.0):
        try:
            polished_energy, polished_pos = relax(best_positions, potential, parameter)
            if polished_energy < best_energy and _is_valid(polished_pos):
                best_energy = polished_energy
                best_positions = polished_pos
        except Exception:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END

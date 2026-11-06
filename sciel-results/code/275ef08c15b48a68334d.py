"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 285.0  # Safety margin before 300s limit


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _macky_icosahedron(n, scale=1.0):
    """Generate Mackay icosahedral cluster with exactly n atoms.
    
    Mackay icosahedra have layers: 1, 12, 42, 120, 220, ...
    For n not matching a complete Mackay number, we fill shells
    optimally for Dzugutov potential (polytetrahedral order).
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of regular icosahedron (outer shell)
    verts_outer = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms_outer = np.linalg.norm(verts_outer, axis=1, keepdims=True)
    verts_outer = verts_outer / norms_outer  # unit sphere
    
    # 42 vertices of second shell (Mackay M2)
    # These are midpoints of edges of the icosahedron plus face centers
    # Actually, Mackay M2 = 42 atoms: 30 edge midpoints + 12 face centers
    # Let's construct them properly
    
    # Edge midpoints of icosahedron
    edges = []
    # Find all edges by checking distances
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts_outer[i] - verts_outer[j])
            if d < 0.1:  # very close
                continue
            # Regular icosahedron edge length is 2 (for unit vertices)
            if abs(d - 2.0) < 0.1 or abs(d - np.sqrt(2+np.sqrt(5))) < 0.01:
                edges.append((i, j))
    
    # Actually, let's use a different approach - generate points on concentric shells
    # Shell 0: center (1 atom)
    # Shell 1: icosahedron vertices (12 atoms)  
    # Shell 2: 42 atoms (edge midpoints + face centers)
    # Shell 3: 120 atoms
    
    points = []
    
    # Center atom
    points.append(np.array([0.0, 0.0, 0.0]))
    
    if n > 1:
        # Shell 1: icosahedron vertices
        shell1 = verts_outer * scale
        points.extend(shell1)
    
    if n > 13:
        # Shell 2: Generate 42 points
        # 30 edge midpoints + 12 face centers
        # Edge midpoints
        edge_mids = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(verts_outer[i] - verts_outer[j])
                # For unit icosahedron, edge length ~ 1.0515
                if 0.9 < d < 1.2:
                    mid = (verts_outer[i] + verts_outer[j]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    edge_mids.append(mid)
        
        # Face centers - find triangular faces
        face_centers = []
        for i in range(12):
            for j in range(i+1, 12):
                for k in range(j+1, 12):
                    d1 = np.linalg.norm(verts_outer[i] - verts_outer[j])
                    d2 = np.linalg.norm(verts_outer[j] - verts_outer[k])
                    d3 = np.linalg.norm(verts_outer[k] - verts_outer[i])
                    # All three distances should be similar (edge length)
                    if 0.9 < d1 < 1.2 and 0.9 < d2 < 1.2 and 0.9 < d3 < 1.2:
                        center = (verts_outer[i] + verts_outer[j] + verts_outer[k]) / 3.0
                        center = center / np.linalg.norm(center)
                        face_centers.append(center)
        
        shell2 = np.array(edge_mids + face_centers) * (scale * 1.8)
        points.extend(shell2)
    
    if n > 55:
        # Shell 3: Generate 120 points on a larger sphere
        # Use golden spiral distribution
        rng = np.random.default_rng(42)
        shell3 = []
        for i in range(120):
            # Fibonacci sphere
            y = 1 - 2 * (i + 0.5) / 120
            r = np.sqrt(1 - y**2)
            theta = 2 * np.pi * i / 120
            p = np.array([r * np.cos(theta), y, r * np.sin(theta)])
            shell3.append(p)
        shell3 = np.array(shell3) * (scale * 2.5)
        points.extend(shell3)
    
    all_points = np.array(points)
    
    if len(all_points) >= n:
        return all_points[:n].copy()
    
    # If we need more, add points in gaps
    rng = np.random.default_rng(n)
    while len(all_points) < n:
        # Add point near existing points but displaced
        idx = rng.integers(0, len(all_points))
        p = all_points[idx] + rng.normal(0, 0.1, 3)
        all_points = np.vstack([all_points, p])
    
    return all_points[:n].copy()


def _fcc_cluster(n, scale=1.0):
    """Generate FCC-like cluster."""
    rng = np.random.default_rng(777)
    points = []
    
    # Generate FCC lattice points within a sphere
    a = scale * 0.5  # lattice constant
    max_coord = int(scale * 2)
    
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            for z in range(-max_coord, max_coord + 1):
                if len(points) >= n:
                    break
                # FCC positions
                pos = np.array([x, y, z]) * a
                if np.linalg.norm(pos) <= scale * 1.2:
                    points.append(pos)
                # FCC face centers
                for fc in [
                    [x+0.5, y+0.5, z], [x+0.5, y, z+0.5], [x, y+0.5, z+0.5],
                    [x+0.5, y-0.5, z], [x+0.5, y, z-0.5], [x, y-0.5, z-0.5]
                ]:
                    if len(points) >= n:
                        break
                    pos = np.array(fc) * a
                    if np.linalg.norm(pos) <= scale * 1.2:
                        points.append(pos)
    
    # Fill remaining
    while len(points) < n:
        p = rng.uniform(-scale, scale, 3)
        if np.linalg.norm(p) <= scale:
            points.append(p)
    
    return np.array(points[:n])


def _random_compact(n, seed, spacing=1.05, radius_factor=0.85):
    """Generate a random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * radius_factor
    points = []
    min_dist = 0.7 * spacing
    attempts = 0
    max_attempts = n * 1000
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms with strict constraint, relax it
    if len(points) < n:
        min_dist = 0.5
        attempts = 0
        while len(points) < n and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= min_dist for q in points):
                    points.append(p)
            attempts += 1
    
    # Last resort: just add random points
    while len(points) < n:
        p = rng.uniform(-radius * 0.5, radius * 0.5, 3)
        points.append(p)
    
    return np.array(points[:n])


def _perturb_and_optimize(positions, potential, parameter, n_perturbations=3, step_size=0.15):
    """Try multiple perturbations and keep the best."""
    best_e = np.inf
    best_pos = positions.copy()
    rng = np.random.default_rng(hash(tuple(positions[0])) % 2**32)
    
    for _ in range(n_perturbations):
        perturbed = positions + rng.normal(0, step_size, positions.shape)
        try:
            e, pos = relax(perturbed, potential, parameter)
            if e < best_e:
                best_e = e
                best_pos = pos
        except:
            pass
    
    return best_e, best_pos


def construct(instance):
    """Construct low-energy Dzugutov cluster using structured initializations and relaxation."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    def _try_relax(points):
        nonlocal best_energy, best_positions
        try:
            energy, positions = relax(points, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
                return True
        except:
            pass
        return False
    
    # Base scale for this cluster size
    base_scale = size ** (1.0 / 3.0) * 0.38
    
    # Strategy 1: Mackay icosahedron
    ico_points = _macky_icosahedron(size, scale=base_scale)
    _try_relax(ico_points)
    
    if _time_remaining() < 50:
        return best_positions
    
    # Strategy 2: Perturbed Mackay icosahedron
    e, pos = _perturb_and_optimize(ico_points, potential, parameter, n_perturbations=3, step_size=0.12)
    if e < best_energy:
        best_energy = e
        best_positions = pos
    
    if _time_remaining() < 50:
        return best_positions
    
    # Strategy 3: Different scale icosahedron
    ico_points2 = _macky_icosahedron(size, scale=base_scale * 0.9)
    _try_relax(ico_points2)
    
    if _time_remaining() < 50:
        return best_positions
    
    # Strategy 4: FCC-like structure
    fcc_points = _fcc_cluster(size, scale=base_scale * 1.1)
    _try_relax(fcc_points)
    
    if _time_remaining() < 50:
        return best_positions
    
    # Strategy 5: Random compact
    rc_points = _random_compact(size, seed=size, spacing=1.05, radius_factor=0.85)
    _try_relax(rc_points)
    
    if _time_remaining() < 50:
        return best_positions
    
    # Strategy 6: Perturbed random compact
    e, pos = _perturb_and_optimize(rc_points, potential, parameter, n_perturbations=3, step_size=0.1)
    if e < best_energy:
        best_energy = e
        best_positions = pos
    
    if _time_remaining() < 50:
        return best_positions
    
    # Strategy 7: Second random compact with different seed
    rc_points2 = _random_compact(size, seed=size + 123, spacing=1.0, radius_factor=0.9)
    _try_relax(rc_points2)
    
    if _time_remaining() < 50:
        return best_positions
    
    # Strategy 8: Icosahedron with larger scale
    ico_points3 = _macky_icosahedron(size, scale=base_scale * 1.1)
    _try_relax(ico_points3)
    
    if _time_remaining() < 50:
        return best_positions
    
    # Strategy 9: Third random compact
    rc_points3 = _random_compact(size, seed=size + 246, spacing=1.1, radius_factor=0.8)
    _try_relax(rc_points3)
    
    if _time_remaining() < 50:
        return best_positions
    
    # Strategy 10: Perturbed FCC
    e, pos = _perturb_and_optimize(fcc_points, potential, parameter, n_perturbations=2, step_size=0.15)
    if e < best_energy:
        best_energy = e
        best_positions = pos
    
    if _time_remaining() < 50:
        return best_positions
    
    # Strategy 11: Fourth random compact with tighter packing
    rc_points4 = _random_compact(size, seed=size + 369, spacing=0.95, radius_factor=0.95)
    _try_relax(rc_points4)
    
    if _time_remaining() < 50:
        return best_positions
    
    # Strategy 12: Icosahedron with smaller scale
    ico_points4 = _macky_icosahedron(size, scale=base_scale * 0.8)
    _try_relax(ico_points4)
    
    return best_positions
# EVOLVE-BLOCK-END

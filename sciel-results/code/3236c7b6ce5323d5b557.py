"""
Morse cluster global minima: structured template + multi-restart local relaxation.
Uses BCC, FCC, icosahedral, and decahedral shells as physically-motivated seeds,
then relaxes each with the provided local optimizer. Time-budgeted.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _bcc_points(n, rng):
    """Generate n points on a BCC lattice shell structure."""
    points = [np.array([0.0, 0.0, 0.0])]
    # BCC layers: (0,0,0) and (0.5,0.5,0.5) offsets
    offsets = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.5]),
        np.array([0.5, -0.5, 0.5]),
        np.array([-0.5, 0.5, 0.5]),
        np.array([-0.5, -0.5, 0.5]),
        np.array([0.5, 0.5, -0.5]),
        np.array([0.5, -0.5, -0.5]),
        np.array([-0.5, 0.5, -0.5]),
        np.array([-0.5, -0.5, -0.5]),
    ]
    # Generate BCC lattice points in a cube, sort by radius
    max_shell = int(np.ceil(n ** (1.0/3.0))) + 2
    candidates = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                base = np.array([float(i), float(j), float(k)])
                for off in offsets:
                    p = base + off
                    r = np.linalg.norm(p)
                    candidates.append((r, p))
    candidates.sort(key=lambda x: x[0])
    for r, p in candidates:
        if len(points) >= n:
            break
        # Check minimum distance
        if all(np.linalg.norm(p - q) >= 0.55 for q in points):
            points.append(p)
    # If we still need more, add random points on a sphere shell
    while len(points) < n:
        p = rng.normal(0, 0.8, 3)
        p = p / np.linalg.norm(p) * (n ** (1.0/3.0) * 0.7)
        if all(np.linalg.norm(p - q) >= 0.55 for q in points):
            points.append(p)
    return np.array(points[:n])


def _fcc_points(n, rng):
    """Generate n points on an FCC lattice structure."""
    points = [np.array([0.0, 0.0, 0.0])]
    # FCC lattice vectors
    max_shell = int(np.ceil(n ** (1.0/3.0))) + 2
    candidates = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                # FCC: (i,j,k) where i+j+k even
                if (i + j + k) % 2 == 0:
                    p = np.array([float(i), float(j), float(k)]) * 0.7071
                    r = np.linalg.norm(p)
                    candidates.append((r, p))
    candidates.sort(key=lambda x: x[0])
    for r, p in candidates:
        if len(points) >= n:
            break
        if all(np.linalg.norm(p - q) >= 0.55 for q in points):
            points.append(p)
    while len(points) < n:
        p = rng.normal(0, 0.8, 3)
        p = p / np.linalg.norm(p) * (n ** (1.0/3.0) * 0.7)
        if all(np.linalg.norm(p - q) >= 0.55 for q in points):
            points.append(p)
    return np.array(points[:n])


def _icosahedral_points(n, rng):
    """Generate points in icosahedral shells (Mackay icosahedron)."""
    # Icosahedral shell radii and counts: 1, 12, 42, 92, ...
    # Shell radii: 0, 1, sqrt(3), sqrt(7), ...
    # For n up to 80, we need shells 1, 12, 42
    points = [np.array([0.0, 0.0, 0.0])]
    
    # 12 vertices of icosahedron
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    # Shell 1: 12 vertices at radius ~1
    shell1_radius = 1.0
    for v in ico_verts:
        if len(points) >= n:
            break
        p = v * shell1_radius
        if all(np.linalg.norm(p - q) >= 0.55 for q in points):
            points.append(p)
    
    # Shell 2: 42 points (edges and face centers of icosahedron)
    if len(points) < n:
        shell2_radius = np.sqrt(3.0) * 0.85
        # Edge midpoints
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                if d < 1.2:  # adjacent vertices
                    edges.append((i, j))
        for i, j in edges:
            if len(points) >= n:
                break
            p = (ico_verts[i] + ico_verts[j]) / 2.0
            p = p / np.linalg.norm(p) * shell2_radius
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        # Face centers
        if len(points) < n:
            faces = []
            for i in range(12):
                for j in range(12):
                    for k in range(12):
                        if i < j < k:
                            d1 = np.linalg.norm(ico_verts[i] - ico_verts[j])
                            d2 = np.linalg.norm(ico_verts[j] - ico_verts[k])
                            d3 = np.linalg.norm(ico_verts[k] - ico_verts[i])
                            if d1 < 1.2 and d2 < 1.2 and d3 < 1.2:
                                faces.append((i, j, k))
            for i, j, k in faces:
                if len(points) >= n:
                    break
                p = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3.0
                p = p / np.linalg.norm(p) * shell2_radius
                if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                    points.append(p)
    
    while len(points) < n:
        p = rng.normal(0, 0.8, 3)
        p = p / np.linalg.norm(p) * (n ** (1.0/3.0) * 0.7)
        if all(np.linalg.norm(p - q) >= 0.55 for q in points):
            points.append(p)
    return np.array(points[:n])


def _decahedral_points(n, rng):
    """Generate points in decahedral (twinned icosahedra) structure."""
    points = [np.array([0.0, 0.0, 0.0])]
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    # Use 5-fold twinned icosahedra
    # First shell: 12 vertices
    shell1_radius = 1.0
    for v in ico_verts:
        if len(points) >= n:
            break
        p = v * shell1_radius
        if all(np.linalg.norm(p - q) >= 0.55 for q in points):
            points.append(p)
    
    # Second shell: 42 points
    if len(points) < n:
        shell2_radius = 1.7
        # Generate points on a sphere using golden spiral
        count = min(42, n - len(points))
        golden_angle = np.pi * (3 - np.sqrt(5))
        for idx in range(count):
            if len(points) >= n:
                break
            y = 1.0 - 2.0 * (idx + 0.5) / count
            r = np.sqrt(1.0 - y*y)
            theta = golden_angle * idx
            p = np.array([np.cos(theta) * r, y, np.sin(theta) * r]) * shell2_radius
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
    
    while len(points) < n:
        p = rng.normal(0, 0.8, 3)
        p = p / np.linalg.norm(p) * (n ** (1.0/3.0) * 0.7)
        if all(np.linalg.norm(p - q) >= 0.55 for q in points):
            points.append(p)
    return np.array(points[:n])


def _random_packed_points(n, rng, spacing=1.0):
    """Random packing with minimum distance constraint."""
    points = []
    radius = spacing * n ** (1.0/3.0) * 0.8
    max_attempts = 50000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.55 for q in points):
            points.append(p)
    # Fill remaining with random sphere points
    while len(points) < n:
        p = rng.normal(0, 1.0, 3)
        p = p / np.linalg.norm(p) * (n ** (1.0/3.0) * 0.75)
        points.append(p)
    return np.array(points[:n])


def _distorted_template(template, rng, distortion=0.15):
    """Add small random distortion to a template."""
    distorted = template + rng.normal(0, distortion, template.shape)
    return distorted


def construct(instance):
    """Find low-energy Morse cluster using structured templates + local relaxation."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    rng = np.random.default_rng(size * 7 + 13)
    
    # Generate diverse templates
    templates = []
    
    # 1. BCC
    templates.append(_bcc_points(size, rng))
    # 2. FCC
    templates.append(_fcc_points(size, rng))
    # 3. Icosahedral
    templates.append(_icosahedral_points(size, rng))
    # 4. Decahedral
    templates.append(_decahedral_points(size, rng))
    
    # 5-8. Distorted versions of each
    for t in templates[:4]:
        templates.append(_distorted_template(t, rng, distortion=0.1))
        templates.append(_distorted_template(t, rng, distortion=0.2))
    
    # 9-10. Random packings
    templates.append(_random_packed_points(size, rng))
    templates.append(_random_packed_points(size, rng))
    
    # Scale templates to appropriate size
    # The equilibrium distance is 1.0, so templates should be at that scale
    # BCC/FCC are already at ~1.0 spacing
    # Icosahedral/decahedral need scaling based on n
    
    best_energy = np.inf
    best_positions = None
    
    for idx, template in enumerate(templates):
        if time.time() > deadline - 30.0:
            break
        
        # Scale template so that typical nearest-neighbor distance ~ 1.0
        # Compute median nearest-neighbor distance
        from scipy.spatial import cKDTree
        try:
            tree = cKDTree(template)
            dists, _ = tree.query(template, k=2)
            nn_dists = dists[:, 1]
            median_nn = np.median(nn_dists)
            if median_nn > 0:
                scale = 1.0 / median_nn
                template = template * scale
        except Exception:
            pass
        
        # Ensure minimum distance constraint
        # Relax will handle this, but let's try to fix obvious violations
        try:
            energy, positions = relax(template.copy(), potential, parameter)
        except Exception:
            energy = np.inf
            positions = template
        
        if energy < best_energy:
            best_energy = energy
            best_positions = positions
    
    # If we have time, do additional random restarts from the best position
    if best_positions is not None and time.time() < deadline - 60.0:
        for _ in range(5):
            if time.time() > deadline - 30.0:
                break
            perturbed = best_positions + rng.normal(0, 0.3, best_positions.shape)
            try:
                energy, positions = relax(perturbed, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
    
    # Final check: ensure minimum distance
    if best_positions is not None:
        # Verify and fix any violations
        for _ in range(3):
            from scipy.spatial import cKDTree
            try:
                tree = cKDTree(best_positions)
                dists, idxs = tree.query(best_positions, k=2)
                if np.min(dists[:, 1]) >= 0.5:
                    break
                # Nudge violating pairs apart
                for i in range(len(best_positions)):
                    d, j = dists[i, 1], idxs[i, 1]
                    if d < 0.5:
                        direction = best_positions[i] - best_positions[j]
                        direction = direction / (np.linalg.norm(direction) + 1e-10)
                        best_positions[i] += direction * (0.5 - d) * 0.5
                        best_positions[j] -= direction * (0.5 - d) * 0.5
            except Exception:
                break
    
    if best_positions is None:
        # Fallback: simple random
        best_positions = _random_packed_points(size, rng)
    
    return best_positions
# EVOLVE-BLOCK-END

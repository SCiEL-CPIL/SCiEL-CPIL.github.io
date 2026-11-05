"""Dzugutov cluster global minima search using polytetrahedral/icosahedral initial guesses."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax


# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16.0


def _icosahedron_vertices(radius=1.0):
    """Generate 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts


def _get_icosahedron_topology():
    """Get faces and edges of unit icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts_unit = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts_unit, axis=1)
    verts_unit = verts_unit / norms[:, np.newaxis]
    
    # Edge length of unit icosahedron
    edge_len = 2.0 / np.sqrt(phi * phi + 1)
    
    # Find all edges
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            dist = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if abs(dist - edge_len) < 1e-6:
                edges.append((i, j))
    
    # Find all faces (triangles)
    edge_set = set(tuple(sorted(e)) for e in edges)
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                if (tuple(sorted((i, j))) in edge_set and
                    tuple(sorted((j, k))) in edge_set and
                    tuple(sorted((i, k))) in edge_set):
                    faces.append((i, j, k))
    
    return verts_unit, faces, edges


def _build_mackay_icosahedron(n, rng):
    """Build a Mackay icosahedral cluster with n atoms."""
    # Optimal bond lengths tuned for Dzugutov potential
    r1 = 1.05   # first shell radius
    r2 = 1.85   # second shell radius
    
    points = [np.zeros(3)]  # center
    
    # First shell: icosahedron vertices
    ico = _icosahedron_vertices(r1)
    points.extend(list(ico))
    
    if n <= len(points):
        if n < len(points):
            idx = rng.choice(len(points), n, replace=False)
            points = [points[i] for i in idx]
        return np.array(points)
    
    # Second shell candidates
    verts_unit, faces, edges = _get_icosahedron_topology()
    
    # Face centers (20 positions)
    face_centers = []
    for face in faces:
        fc = np.mean(verts_unit[list(face)], axis=0)
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
    
    # Edge midpoints (30 positions)
    edge_mids = []
    for i, j in edges:
        em = (verts_unit[i] + verts_unit[j]) / 2
        em = em / np.linalg.norm(em)
        edge_mids.append(em)
    
    # Build second shell: prioritize face centers, then edge midpoints
    second_shell = []
    for fc in face_centers:
        second_shell.append(fc * r2)
    for em in edge_mids:
        second_shell.append(em * r2)
    # Add vertex positions at intermediate radius for flexibility
    for v in verts_unit:
        second_shell.append(v * (r1 + r2) / 2)
    
    second_shell = np.array(second_shell)
    
    needed = n - len(points)
    if needed <= 0:
        return np.array(points[:n])
    
    # Select from second shell candidates
    n_select = min(needed, len(second_shell))
    idx = rng.permutation(len(second_shell))
    for i in range(n_select):
        points.append(second_shell[idx[i]])
    
    # If we still need more atoms, add random points in the shell region
    while len(points) < n:
        p = rng.uniform(-r2, r2, 3)
        if np.linalg.norm(p) <= r2:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.55:
                    valid = False
                    break
            if valid:
                points.append(p)
            else:
                points.append(p + 0.15 * rng.standard_normal(3))
        else:
            points.append(p + 0.1 * rng.standard_normal(3))
    
    return np.array(points[:n])


def _build_random_compact(n, rng):
    """Build a random compact cluster with polytetrahedral bias."""
    spacing = 1.15
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    points = []
    attempts = 0
    while len(points) < n and attempts < 15000:
        # Bias toward icosahedral-like positions
        if rng.random() < 0.3 and len(points) >= 13:
            # Place near existing atoms with tetrahedral offset
            base_idx = rng.randint(0, len(points))
            base = points[base_idx]
            # Random tetrahedral direction
            theta = rng.uniform(0, 2 * np.pi)
            phi_angle = np.arccos(2 * rng.random() - 1)
            offset_dir = np.array([
                np.sin(phi_angle) * np.cos(theta),
                np.sin(phi_angle) * np.sin(theta),
                np.cos(phi_angle)
            ])
            p = base + spacing * offset_dir
        else:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) > radius:
                continue
        
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.55:
                valid = False
                break
        if valid:
            points.append(p)
        attempts += 1
    
    # Fallback: add remaining atoms with small offsets
    while len(points) < n:
        if points:
            base = points[len(points) % len(points)]
            p = base + 0.35 * rng.standard_normal(3)
        else:
            p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _build_tetrahedral_cluster(n, rng):
    """Build a cluster based on tetrahedral packing (polytetrahedral model)."""
    r = 1.05
    points = []
    
    # Central tetrahedron
    tet = np.array([
        [1, 1, 1], [1, -1, -1],
        [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    norms = np.linalg.norm(tet, axis=1)
    tet = tet / norms[:, np.newaxis] * r
    points.extend(list(tet))
    
    # Add octahedral positions
    octa = np.array([
        [r * 1.3, 0, 0], [-r * 1.3, 0, 0],
        [0, r * 1.3, 0], [0, -r * 1.3, 0],
        [0, 0, r * 1.3], [0, 0, -r * 1.3]
    ], dtype=float)
    for p in octa:
        if len(points) >= n:
            break
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.55:
                valid = False
                break
        if valid:
            points.append(p)
    
    # Add more atoms in shells with tetrahedral bias
    shell_radius = r * 1.6
    while len(points) < n:
        if rng.random() < 0.4 and len(points) > 4:
            base_idx = rng.randint(0, len(points))
            base = points[base_idx]
            theta = rng.uniform(0, 2 * np.pi)
            phi_angle = np.arccos(2 * rng.random() - 1)
            offset_dir = np.array([
                np.sin(phi_angle) * np.cos(theta),
                np.sin(phi_angle) * np.sin(theta),
                np.cos(phi_angle)
            ])
            p = base + r * 0.9 * offset_dir
        else:
            p = rng.uniform(-shell_radius, shell_radius, 3)
            if np.linalg.norm(p) > shell_radius:
                p = p * shell_radius / np.linalg.norm(p)
        
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.55:
                valid = False
                break
        if valid:
            points.append(p)
        else:
            points.append(p + 0.2 * rng.standard_normal(3))
    
    return np.array(points[:n])


def construct(instance):
    """Find low-energy Dzugutov clusters using polytetrahedral initial guesses."""
    start_time = time.time()
    time_limit = 300.0
    safety_margin = 20.0
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(size * 7 + 13)
    
    best_energy = float('inf')
    best_positions = None
    
    def check_time():
        return (time.time() - start_time) < (time_limit - safety_margin)
    
    def try_relax(pos, label=""):
        nonlocal best_energy, best_positions
        try:
            energy, positions = relax(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            pass
    
    # Strategy 1: Mackay icosahedral structures with varied perturbations
    n_ico_tries = min(8, max(3, int(25 * (300.0 - safety_margin) / 300.0)))
    for i in range(n_ico_tries):
        if not check_time():
            break
        pos = _build_mackay_icosahedron(size, rng)
        scale = 0.03 + 0.04 * i
        pos = pos + scale * rng.standard_normal((size, 3))
        try_relax(pos, f"ico_{i}")
    
    # Strategy 2: Random compact structures with polytetrahedral bias
    n_rand_tries = min(6, max(2, int(20 * (300.0 - safety_margin) / 300.0)))
    for i in range(n_rand_tries):
        if not check_time():
            break
        pos = _build_random_compact(size, rng)
        pos = pos + 0.04 * rng.standard_normal((size, 3))
        try_relax(pos, f"rand_{i}")
    
    # Strategy 3: Tetrahedral structures
    n_tet_tries = min(4, max(2, int(15 * (300.0 - safety_margin) / 300.0)))
    for i in range(n_tet_tries):
        if not check_time():
            break
        pos = _build_tetrahedral_cluster(size, rng)
        pos = pos + 0.04 * rng.standard_normal((size, 3))
        try_relax(pos, f"tet_{i}")
    
    # Strategy 4: Local optimization around best solution with varied perturbations
    if best_positions is not None:
        n_local_tries = min(10, max(4, int(30 * (300.0 - safety_margin) / 300.0)))
        for i in range(n_local_tries):
            if not check_time():
                break
            # Vary perturbation scale: small to medium
            scale = 0.08 + 0.06 * i
            pos = best_positions + scale * rng.standard_normal((size, 3))
            try_relax(pos, f"local_{i}")
    
    # Strategy 5: Additional diverse restarts if time permits
    n_extra = min(8, max(3, int(25 * (300.0 - safety_margin) / 300.0)))
    for i in range(n_extra):
        if not check_time():
            break
        # Cycle through different strategies
        if i % 3 == 0:
            pos = _build_mackay_icosahedron(size, rng)
            scale = 0.12 + 0.04 * (i // 3)
        elif i % 3 == 1:
            pos = _build_random_compact(size, rng)
            scale = 0.08 + 0.04 * (i // 3)
        else:
            pos = _build_tetrahedral_cluster(size, rng)
            scale = 0.08 + 0.04 * (i // 3)
        
        pos = pos + scale * rng.standard_normal((size, 3))
        try_relax(pos, f"extra_{i}")
    
    # Strategy 6: Aggressive local search around best with larger perturbations
    if best_positions is not None and check_time():
        n_aggressive = min(6, max(2, int(20 * (300.0 - safety_margin) / 300.0)))
        for i in range(n_aggressive):
            if not check_time():
                break
            scale = 0.25 + 0.1 * i
            pos = best_positions + scale * rng.standard_normal((size, 3))
            try_relax(pos, f"agg_{i}")
    
    # Strategy 7: If time remains, do a final round of fine-tuning around best
    if best_positions is not None and check_time():
        n_fine = min(4, max(1, int(10 * (300.0 - safety_margin) / 300.0)))
        for i in range(n_fine):
            if not check_time():
                break
            scale = 0.02 + 0.01 * i
            pos = best_positions + scale * rng.standard_normal((size, 3))
            try_relax(pos, f"fine_{i}")
    
    if best_positions is None:
        # Fallback: return a simple structure
        best_positions = _build_random_compact(size, rng)
    
    return best_positions
# EVOLVE-BLOCK-END

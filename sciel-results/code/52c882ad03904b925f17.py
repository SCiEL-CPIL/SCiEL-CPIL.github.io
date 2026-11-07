"""Structure prediction: hierarchical growth from known small-cluster minima with surface-site addition."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from scipy.optimize import minimize

def _icosahedron_vertices(r=1.0):
    """Return 12 vertices of a regular icosahedron at radius r."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts[0]) * r
    return verts

def _icosahedron_edges(verts):
    """Return edge midpoints of icosahedron (30 edges)."""
    n = len(verts)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < np.linalg.norm(verts[0]) * 1.2:
                mid = (verts[i] + verts[j]) / 2
                edges.append(mid)
    return np.array(edges)

def _icosahedron_faces(verts, edges):
    """Return face centers of icosahedron (20 faces)."""
    # Find triangles: three vertices that form an edge-connected triangle
    faces = []
    n = len(verts)
    # Build adjacency
    adj = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < np.linalg.norm(verts[0]) * 1.2:
                adj.add((i, j))
    # Find all triangles
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in adj:
                continue
            for k in range(j + 1, n):
                if (i, k) in adj and (j, k) in adj:
                    center = (verts[i] + verts[j] + verts[k]) / 3
                    faces.append(center)
    return np.array(faces)

def _make_icosahedron_core(n_core, r=0.7):
    """Create a 13-atom icosahedron core (center + 12 vertices)."""
    center = np.array([0.0, 0.0, 0.0])
    verts = _icosahedron_vertices(r)
    pts = [center] + [v for v in verts]
    return np.array(pts[:n_core])

def _add_surface_atoms(core, n_add, kind, rng):
    """Add n_add atoms to core at high-symmetry surface sites."""
    n_core = len(core)
    if n_add <= 0:
        return core
    
    # Identify surface sites: edge midpoints, face centers, vertex outward extensions
    verts = core[1:] if n_core >= 13 else core
    r_core = np.linalg.norm(verts[0]) if len(verts) > 0 else 0.7
    
    # Generate candidate surface positions
    candidates = []
    
    # 1. Edge midpoints (30 sites)
    if n_core >= 13:
        edges = _icosahedron_edges(verts)
        for e in edges:
            candidates.append(e)
    
    # 2. Face centers (20 sites)
    if n_core >= 13:
        faces = _icosahedron_faces(verts, None)
        for f in faces:
            candidates.append(f)
    
    # 3. Outward extensions from vertices
    for v in verts:
        outward = v / np.linalg.norm(v) * (r_core + 0.3)
        candidates.append(outward)
    
    # 4. Random positions on a shell just outside the core
    n_random = max(20, n_add * 3)
    for i in range(n_random):
        # Fibonacci sphere
        z = 1 - 2 * (i + 0.5) / n_random
        r = np.sqrt(1 - z * z)
        theta = 2 * np.pi * i / n_random
        p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * (r_core + 0.25 + rng.uniform(0, 0.2))
        candidates.append(p)
    
    candidates = np.array(candidates)
    
    # Select n_add candidates that maximize minimum distance from core
    selected = []
    for _ in range(n_add):
        best_idx = -1
        best_score = -np.inf
        for idx, cand in enumerate(candidates):
            if any(np.linalg.norm(cand - s) < 0.3 for s in selected):
                continue
            # Score: minimum distance to existing atoms
            d_to_core = np.min([np.linalg.norm(cand - c) for c in core]) if len(core) > 0 else np.inf
            d_to_selected = np.min([np.linalg.norm(cand - s) for s in selected]) if len(selected) > 0 else np.inf
            score = min(d_to_core, d_to_selected)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx >= 0:
            selected.append(candidates[best_idx])
            candidates = np.delete(candidates, best_idx, axis=0)
        else:
            # Add a random position
            p = rng.uniform(0, r_core + 0.5, 3)
            p = p / (np.linalg.norm(p) + 1e-10) * (r_core + 0.3)
            selected.append(p)
    
    return np.vstack([core, np.array(selected)])

def _bulk_fcc_positions(n, spacing=0.75):
    """Generate n positions in an FCC-like arrangement."""
    a = spacing
    max_coord = int(np.ceil((n ** (1/3)) * a * 0.8)) + 2
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i * a + dx * a, j * a + dy * a, k * a + dz * a])
                    if np.linalg.norm(p) <= max_coord * a * 0.9:
                        points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]]

def _random_compact(n, rng, min_dist=0.36):
    """Generate n random compact positions."""
    radius = 0.75 * n ** (1.0 / 3.0) * 1.2
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    if len(points) == n:
        return np.array(points)
    return None

def _check_validity(positions, min_dist=0.35):
    """Check if positions are valid (finite, min distance)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i] - positions[j]) < min_dist:
                return False
    return True

def _relax_with_variants(positions, potential, kind, n):
    """Try multiple relaxation strategies and return best result."""
    best_energy = np.inf
    best_positions = None
    
    # Strategy 1: Standard L-BFGS-B via relax()
    try:
        energy, pos = relax(positions, potential, kind)
        if np.isfinite(energy) and _check_validity(pos):
            best_energy = energy
            best_positions = pos
    except:
        pass
    
    # Strategy 2: Pre-relaxation with small steps
    try:
        pos2 = positions.copy()
        for step in range(50):
            e, g = sutton_chen_energy_gradient(pos2, kind)
            if not np.isfinite(e):
                break
            # Small step
            lr = 0.01 / (1 + step * 0.1)
            pos2 = pos2 - lr * g
            if not _check_validity(pos2):
                break
        energy, pos = relax(pos2, potential, kind)
        if np.isfinite(energy) and _check_validity(pos) and energy < best_energy:
            best_energy = energy
            best_positions = pos
    except:
        pass
    
    # Strategy 3: Perturb and relax
    rng_local = np.random.default_rng(hash(tuple(positions[0].round(4))) % 2**31)
    try:
        perturbed = positions + rng_local.normal(0, 0.05, positions.shape)
        if _check_validity(perturbed):
            energy, pos = relax(perturbed, potential, kind)
            if np.isfinite(energy) and _check_validity(pos) and energy < best_energy:
                best_energy = energy
                best_positions = pos
    except:
        pass
    
    return best_energy, best_positions

def construct(instance):
    """Hierarchical structure prediction: embed small-cluster cores, grow at surface sites."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    rng = np.random.default_rng(size * 7 + 123)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial configurations using hierarchical growth
    configs = []
    
    # Config 1: Icosahedral core + surface growth
    if size >= 13:
        core_size = min(13, size)
        core = _make_icosahedron_core(core_size, r=0.7)
        n_add = size - core_size
        config1 = _add_surface_atoms(core, n_add, kind, rng)
        if config1 is not None and len(config1) == size and _check_validity(config1):
            configs.append(config1)
    
    # Config 2: Smaller core (13) + more surface atoms
    if size > 13:
        core = _make_icosahedron_core(13, r=0.65)
        n_add = size - 13
        config2 = _add_surface_atoms(core, n_add, kind, rng)
        if config2 is not None and len(config2) == size and _check_validity(config2):
            configs.append(config2)
    
    # Config 3: FCC-like bulk
    try:
        fcc = _bulk_fcc_positions(size, spacing=0.72)
        if len(fcc) == size and _check_validity(fcc):
            configs.append(fcc)
    except:
        pass
    
    # Config 4: Random compact
    rc = _random_compact(size, rng, min_dist=0.38)
    if rc is not None:
        configs.append(rc)
    
    # Config 5: Random compact with different parameters
    rc2 = _random_compact(size, rng, min_dist=0.36)
    if rc2 is not None:
        configs.append(rc2)
    
    # Config 6: Spherical shell + icosahedral core
    if size >= 13:
        core_n = min(13, size)
        shell_n = size - core_n
        core = _make_icosahedron_core(core_n, r=0.6)
        shell_r = 0.6 * (size ** (1/3.0)) * 0.95
        shell_pts = []
        for i in range(shell_n):
            z = 1 - 2 * (i + 0.5) / shell_n
            r = np.sqrt(1 - z * z)
            theta = 2 * np.pi * i / shell_n
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * shell_r
            shell_pts.append(p)
        config6 = np.vstack([core, np.array(shell_pts)])
        if len(config6) == size and _check_validity(config6):
            configs.append(config6)
    
    # Config 7: Two-layer icosahedral (for larger clusters)
    if size >= 25:
        # Inner icosahedron (13) + outer shell
        inner = _make_icosahedron_core(13, r=0.5)
        outer_n = size - 13
        outer_r = 0.5 * (size ** (1/3.0)) * 0.85
        outer_pts = []
        for i in range(outer_n):
            z = 1 - 2 * (i + 0.5) / outer_n
            r = np.sqrt(1 - z * z)
            theta = 2 * np.pi * i / outer_n
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * outer_r
            outer_pts.append(p)
        config7 = np.vstack([inner, np.array(outer_pts)])
        if len(config7) == size and _check_validity(config7):
            configs.append(config7)
    
    # Relax each configuration with multiple strategies
    for config in configs:
        if config is None or len(config) != size:
            continue
        if not _check_validity(config):
            continue
        
        energy, positions = _relax_with_variants(config, potential, kind, size)
        if positions is not None and energy < best_energy:
            best_energy = energy
            best_positions = positions
    
    # If no valid result, fall back to simple random
    if best_positions is None:
        rc3 = _random_compact(size, rng, min_dist=0.4)
        if rc3 is not None:
            try:
                energy, positions = relax(rc3, potential, kind)
                if np.all(np.isfinite(positions)):
                    best_positions = positions
            except:
                pass
    
    # Final fallback
    if best_positions is None:
        positions = np.zeros((size, 3))
        for i in range(size):
            positions[i] = [i * 0.5, 0, 0]
        best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END

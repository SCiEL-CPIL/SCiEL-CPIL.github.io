"""Sutton-Chen cluster global minima search using core-embedding structure prediction and basin hopping."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _icosahedron_vertices(r=1.0):
    """Return 12 vertices of a regular icosahedron scaled to radius r."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    return verts * r


def _icosahedron_edges(verts):
    """Return list of edge index pairs for the icosahedron (30 edges)."""
    n = len(verts)
    edges = []
    dists = np.linalg.norm(verts[:, None] - verts[None, :], axis=2)
    # The 12 vertices of a regular icosahedron: each vertex connects to 5 others
    # The shortest distance is the edge length
    min_d = dists.min()
    for i in range(n):
        for j in range(i + 1, n):
            if abs(dists[i, j] - min_d) < 1e-9:
                edges.append((i, j))
    return edges


def _icosahedron_faces(verts, edges):
    """Return list of face (triangle) index tuples for the icosahedron (20 faces)."""
    # Build adjacency
    adj = {i: set() for i in range(len(verts))}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    # Faces are triples where all three pairs are edges
    faces = []
    for i in range(len(verts)):
        for j in sorted(adj[i]):
            for k in sorted(adj[j]):
                if k != i and (i, j, k) not in faces and (j, k, i) not in faces and (k, i, j) not in faces:
                    if k in adj[i]:
                        faces.append((i, j, k))
    return faces


def _face_centers(verts, faces):
    """Return centroid of each triangular face."""
    centers = []
    for f in faces:
        c = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        centers.append(c)
    return np.array(centers)


def _edge_midpoints(verts, edges):
    """Return midpoint of each edge."""
    mids = []
    for i, j in edges:
        m = (verts[i] + verts[j]) / 2.0
        mids.append(m)
    return np.array(mids)


def _build_mackay_icosahedron(n, a=0.72):
    """
    Build a Mackay icosahedral cluster by adding shells.
    Shell 0: 1 atom (center)
    Shell 1: 12 atoms (icosahedron vertices)
    Shell 2: 42 atoms (adds to make 55 total)
    For n up to 55, use shells 0-2. For larger n, add more.
    """
    a = 0.72
    points = [np.zeros(3)]  # center
    
    # Shell 1: icosahedron vertices
    r1 = a
    v1 = _icosahedron_vertices(r1)
    for v in v1:
        if len(points) < n:
            points.append(v.copy())
    
    # Shell 2: 42 atoms - positions at face centers and edge midpoints of the icosahedron
    # Face centers: 20 faces -> 20 points
    # Edge midpoints: 30 edges -> 30 points
    # But we need 42 total for shell 2 (to reach 55)
    # Actually Mackay 55 = 1 + 12 + 42
    # The 42 come from: 20 face centers + 22 more? Let me use a different approach.
    # For shell 2, place atoms at face centers (20) and some edge midpoints
    if len(points) < n:
        r2 = a * 1.48  # approximate radius for shell 2
        # Use face centers scaled
        v1_unit = _icosahedron_vertices(1.0)
        edges = _icosahedron_edges(v1_unit)
        faces = _icosahedron_faces(v1_unit, edges)
        fc = _face_centers(v1_unit, faces)
        # Scale face centers to shell 2 radius
        fc_norm = fc / np.linalg.norm(fc, axis=1, keepdims=True)
        fc_scaled = fc_norm * r2
        for p in fc_scaled:
            if len(points) < n:
                points.append(p)
        # Add edge midpoints if needed
        if len(points) < n:
            em = _edge_midpoints(v1_unit, edges)
            em_norm = em / np.linalg.norm(em, axis=1, keepdims=True)
            em_scaled = em_norm * r2
            for p in em_scaled:
                if len(points) < n:
                    points.append(p)
    
    # If still need more, add a third shell
    if len(points) < n:
        r3 = a * 2.0
        # Add points on a sphere
        theta = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        for t in theta:
            if len(points) < n:
                points.append([r3 * np.cos(t), r3 * np.sin(t), 0])
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _build_fcc_sphere(n, a=0.72):
    """Build an fcc spherical cluster by taking the n closest points to center."""
    basis = [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
    ]
    side = int(np.ceil((n / 4) ** (1.0 / 3.0))) + 2
    candidates = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i * a + b[0] * a, j * a + b[1] * a, k * a + b[2] * a])
                    candidates.append(p)
    candidates = np.array(candidates)
    center = candidates.mean(axis=0)
    dists = np.linalg.norm(candidates - center, axis=1)
    order = np.argsort(dists)
    points = candidates[order[:n]]
    points -= points.mean(axis=0)
    return points


def _build_core_plus_growth(core_n, n, a=0.72, seed=42):
    """
    Build a cluster by starting from a known core structure and adding atoms
    at high-symmetry surface sites.
    """
    rng = np.random.default_rng(seed)
    
    # Choose core based on size
    if core_n is None:
        # Pick a core size that's a known magic number
        magic_numbers = [1, 13, 55, 147]
        core_n = max(mn for mn in magic_numbers if mn <= n)
        if core_n > n:
            core_n = 13 if n >= 13 else 1
    
    # Build the core
    if core_n == 1:
        core = np.zeros((1, 3))
    elif core_n == 13:
        core = _build_mackay_icosahedron(13, a)
    elif core_n == 55:
        core = _build_mackay_icosahedron(55, a)
    else:
        core = _build_fcc_sphere(core_n, a)
    
    if len(core) >= n:
        return core[:n]
    
    # Add remaining atoms at surface sites
    remaining = n - len(core)
    added = []
    
    # Compute surface sites: face centers, edge midpoints, vertex directions
    # Use the outermost shell of the core
    core_center = core.mean(axis=0)
    core = core - core_center
    
    # Find outermost atoms
    dists_from_center = np.linalg.norm(core, axis=1)
    outer_radius = np.percentile(dists_from_center, 80)
    outer_mask = dists_from_center >= outer_radius * 0.8
    
    # Generate candidate surface sites
    candidates = []
    
    # 1. Extend along vertex directions
    outer_atoms = core[outer_mask]
    for atom in outer_atoms:
        if len(added) >= remaining:
            break
        direction = atom / (np.linalg.norm(atom) + 1e-10)
        # Place new atom slightly outside
        new_pos = atom + direction * a * 0.5
        candidates.append(new_pos)
    
    # 2. Face-like sites between pairs of outer atoms
    outer_idx = np.where(outer_mask)[0]
    for i in range(len(outer_idx)):
        for j in range(i + 1, len(outer_idx)):
            if len(added) >= remaining:
                break
            p1 = core[outer_idx[i]]
            p2 = core[outer_idx[j]]
            mid = (p1 + p2) / 2.0
            d = np.linalg.norm(p2 - p1)
            if 0.5 * a < d < 1.5 * a:
                # Push outward from center
                mid_dir = mid / (np.linalg.norm(mid) + 1e-10)
                new_pos = mid + mid_dir * a * 0.3
                candidates.append(new_pos)
    
    # 3. Random perturbations of surface
    for _ in range(remaining * 5):
        if len(added) >= remaining:
            break
        atom = core[np.random.randint(len(core))]
        noise = rng.normal(0, 0.15, 3)
        new_pos = atom + noise
        candidates.append(new_pos)
    
    # Add candidates, checking minimum distance
    all_points = [p for p in core]
    for cand in candidates:
        if len(all_points) >= n:
            break
        # Check minimum distance
        min_d = min(np.linalg.norm(cand - p) for p in all_points)
        if min_d > 0.35:
            all_points.append(cand)
    
    # If still short, add random points
    if len(all_points) < n:
        radius = a * n ** (1.0 / 3.0) * 0.8
        for _ in range(1000):
            if len(all_points) >= n:
                break
            p = rng.uniform(-radius, radius, 3)
            min_d = min(np.linalg.norm(p - q) for q in all_points)
            if min_d > 0.35:
                all_points.append(p)
    
    all_points = np.array(all_points[:n])
    all_points -= all_points.mean(axis=0)
    return all_points


def _random_compact(n, spacing=0.72, seed=42):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    for _ in range(10000):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - np.array(q)) >= 0.5 * spacing for q in points):
                points.append(p)
    if len(points) < n:
        points = [list(rng.uniform(-radius, radius, 3)) for _ in range(n)]
    return np.array(points[:n])


def _basin_hopping(positions, kind, n, max_steps=30, step_size=0.5, deadline=None):
    """
    Perform basin hopping: perturb, relax, accept if better.
    Returns best energy and positions found.
    """
    best_energy, best_pos = relax(positions, 'sutton_chen', kind)
    current_energy, current_pos = best_energy, best_pos
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb
        rng = np.random.default_rng(step * 1000 + int(n))
        noise = rng.normal(0, step_size, (n, 3))
        perturbed = current_pos + noise
        
        # Relax
        try:
            energy, pos = relax(perturbed, 'sutton_chen', kind)
        except:
            continue
        
        if np.isfinite(energy) and energy < best_energy:
            best_energy = energy
            best_pos = pos
            current_pos = pos
            current_energy = energy
        else:
            # Keep current
            current_pos = best_pos
            current_energy = best_energy
    
    return best_energy, best_pos


def construct(instance):
    """Structure prediction via core-embedding + basin hopping."""
    size = instance["n"]
    kind = instance["kind"]
    n_exp = instance["n_exp"]
    m_exp = instance["m_exp"]
    c = instance["c"]
    
    start_time = time.time()
    deadline = start_time + 285  # Safety margin before 330s kill
    
    # Generate diverse initial configurations using core-embedding strategy
    initial_configs = []
    
    # 1. Mackay icosahedron
    try:
        initial_configs.append(_build_mackay_icosahedron(size))
    except:
        pass
    
    # 2. FCC sphere
    try:
        initial_configs.append(_build_fcc_sphere(size))
    except:
        pass
    
    # 3. Core-embedding with different cores and seeds
    for core_n in [13, 55, None]:
        for seed in [42, 123, 456]:
            try:
                initial_configs.append(_build_core_plus_growth(core_n, size, seed=seed))
            except:
                pass
    
    # 4. Random compact
    for seed in [42, 123, 456, 789]:
        try:
            initial_configs.append(_random_compact(size, seed=seed))
        except:
            pass
    
    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Relax all initial configs
    for i, config in enumerate(initial_configs):
        if time.time() > deadline:
            break
        try:
            energy, positions = relax(config, 'sutton_chen', kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
    
    if best_positions is None:
        rng = np.random.default_rng(0)
        best_positions = rng.uniform(-1, 1, (size, 3))
        return best_positions
    
    # Phase 2: Basin hopping from the best configuration
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        # Estimate how many BH steps we can afford
        # Each step: perturb + relax. Relax takes ~2-5s for large n
        max_bh_steps = min(40, int(remaining_time / 5))
        try:
            bh_energy, bh_pos = _basin_hopping(
                best_positions, kind, size,
                max_steps=max_bh_steps,
                step_size=0.5,
                deadline=deadline
            )
            if np.isfinite(bh_energy) and bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos
        except:
            pass
    
    # Phase 3: If time remains, try basin hopping from second-best config
    if time.time() < deadline - 20:
        # Find second best from initial relaxations
        # (we didn't track them, so just perturb best and try again)
        rng = np.random.default_rng(999)
        noise = rng.normal(0, 0.7, (size, 3))
        perturbed = best_positions + noise
        try:
            energy, positions = relax(perturbed, 'sutton_chen', kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END

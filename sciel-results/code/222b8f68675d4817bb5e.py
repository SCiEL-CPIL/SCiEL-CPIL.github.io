"""Dzugutov cluster global minima: icosahedral symmetry manifold + L-BFGS-B + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 295.0  # Safety margin before 330s limit


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _icosahedron_vertices():
    """Return 12 unit icosahedron vertices."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms


def _icosahedron_edges(verts):
    """Return list of edge pairs for unit icosahedron."""
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - 2.0) < 0.01:
                edges.append((i, j))
    return edges


def _icosahedron_face_centers(verts):
    """Return 20 face center unit vectors for icosahedron."""
    edges = _icosahedron_edges(verts)
    # Build face list from edges
    # Each edge belongs to exactly 2 faces
    # Use known face structure
    faces = [
        [0, 1, 8], [0, 8, 4], [0, 4, 2], [0, 2, 6], [0, 6, 1],
        [1, 6, 9], [1, 9, 5], [1, 5, 3], [1, 3, 8],
        [2, 4, 10], [2, 10, 7], [2, 7, 6],
        [3, 5, 11], [3, 11, 10], [3, 10, 4], [3, 4, 8],
        [5, 9, 11], [5, 11, 7], [5, 7, 3],
        [7, 11, 10], [7, 10, 2], [7, 2, 6], [7, 6, 9], [7, 9, 5]
    ]
    # Verify these are valid faces (all edges in face exist)
    edge_set = set()
    for e in edges:
        edge_set.add((min(e), max(e)))
    
    valid_faces = []
    for f in faces:
        valid = True
        for k in range(3):
            i, j = f[k], f[(k+1) % 3]
            if (min(i,j), max(i,j)) not in edge_set:
                valid = False
                break
        if valid:
            valid_faces.append(f)
    
    # If we don't have 20 faces, generate them properly
    if len(valid_faces) != 20:
        # Use edge adjacency to find faces
        # Each vertex connects to 5 others
        adj = {i: [] for i in range(12)}
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        # Find triangular faces by checking triples
        valid_faces = []
        for i in range(12):
            for j in adj[i]:
                if j > i:
                    for k in adj[j]:
                        if k > j and k in adj[i]:
                            face = [i, j, k]
                            # Check not already added
                            if face not in valid_faces:
                                valid_faces.append(face)
    
    face_centers = []
    for f in valid_faces[:20]:
        fc = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
    return np.array(face_centers[:20])


def _build_icosahedral_manifold(n, scale=1.0, twist=0.0, r1=0.55, r2=0.85, r3=1.1):
    """Build cluster positions from icosahedral manifold parameters.
    
    Parameters:
    - n: number of atoms
    - scale: overall scale factor
    - twist: twist angle between shells (radians)
    - r1, r2, r3: shell radii (relative to scale)
    
    Returns array of shape (n, 3).
    """
    verts = _icosahedron_vertices()
    edges = _icosahedron_edges(verts)
    edge_mids = np.array([(verts[i] + verts[j]) / 2.0 for i, j in edges])
    edge_mids = edge_mids / np.linalg.norm(edge_mids, axis=1, keepdims=True)
    face_centers = _icosahedron_face_centers(verts)
    
    # Apply twist to outer shells
    def rotate_z(vecs, angle):
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return vecs @ R.T
    
    points = []
    
    # Shell 0: center
    if n >= 1:
        points.append(np.array([0.0, 0.0, 0.0]))
    
    # Shell 1: 12 vertices at radius r1
    if n >= 13:
        shell1 = rotate_z(verts, 0.0) * (scale * r1)
        points.extend(shell1.tolist())
    
    # Shell 2: 30 edge midpoints at radius r2
    if n >= 43:
        shell2 = rotate_z(edge_mids, twist) * (scale * r2)
        points.extend(shell2.tolist())
    
    # Shell 3: 20 face centers at radius r3
    if n >= 63:
        shell3 = rotate_z(face_centers, 2 * twist) * (scale * r3)
        points.extend(shell3.tolist())
    
    # Shell 4: Another 12 vertices at larger radius
    if n >= 93:
        shell4 = rotate_z(verts, 3 * twist) * (scale * 1.4)
        points.extend(shell4.tolist())
    
    points = np.array(points)
    
    # Trim or pad to exactly n atoms
    if len(points) > n:
        points = points[:n]
    elif len(points) < n:
        # Add remaining atoms on a sphere
        rng = np.random.default_rng(42)
        while len(points) < n:
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(-1, 1)
            r = np.sqrt(1 - z**2)
            radius = scale * (r2 + 0.1 * rng.random())
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * radius
            points = np.vstack([points, p])
    
    return points[:n]


def _energy_and_gradient(positions, potential, parameter):
    """Compute energy and gradient using the provided functions."""
    energy, grad = dzugutov_energy_gradient(positions)
    return energy, grad


def _lbfgs_relax(positions, potential, parameter, maxiter=200):
    """Relax positions using L-BFGS-B with gradient."""
    def objective(x):
        pos = x.reshape(-1, 3)
        e, g = _energy_and_gradient(pos, potential, parameter)
        return e, g.reshape(-1)
    
    x0 = positions.reshape(-1)
    result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                      options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
    return result.fun, result.x.reshape(-1, 3)


def _manifold_optimize(n, potential, parameter, scale, maxiter=50):
    """Optimize over the icosahedral manifold parameters using L-BFGS-B.
    
    Parameters: [scale, twist, r1, r2, r3]
    """
    def manifold_energy(params):
        s, tw, r1, r2, r3 = params
        # Ensure valid radii
        r1 = max(0.3, min(1.0, r1))
        r2 = max(0.6, min(1.5, r2))
        r3 = max(0.9, min(2.0, r3))
        tw = max(-0.5, min(0.5, tw))
        s = max(0.5, min(2.0, s))
        
        pos = _build_icosahedral_manifold(n, scale=s, twist=tw, r1=r1, r2=r2, r3=r3)
        
        # Relax on full Cartesian space first
        e, pos = _lbfgs_relax(pos, potential, parameter, maxiter=30)
        return e
    
    # Initial parameters
    x0 = np.array([scale, 0.0, 0.55, 0.85, 1.1])
    
    result = minimize(manifold_energy, x0, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'ftol': 1e-6})
    
    best_params = result.x
    best_energy = result.fun
    best_pos = _build_icosahedral_manifold(n, scale=best_params[0], twist=best_params[1],
                                            r1=best_params[2], r2=best_params[3], r3=best_params[4])
    best_energy, best_pos = _lbfgs_relax(best_pos, potential, parameter, maxiter=50)
    
    return best_energy, best_pos


def _kick_and_relax(positions, potential, parameter, kick_sigma, seed):
    """Perturb positions and relax."""
    rng = np.random.default_rng(seed)
    perturbed = positions + rng.normal(0, kick_sigma, positions.shape)
    energy, relaxed = relax(perturbed, potential, parameter)
    return energy, relaxed


def _basin_hopping(positions, potential, parameter, n_kicks=8, kick_scales=None, seed=0):
    """Basin hopping with varying kick scales."""
    if kick_scales is None:
        kick_scales = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.15, 0.10]
    
    best_energy, best_pos = positions.copy(), positions.copy()
    
    # Get initial energy
    e0, _ = relax(positions, potential, parameter)
    best_energy = e0
    
    for i in range(min(n_kicks, len(kick_scales))):
        if _time_remaining() < 10:
            break
        energy, new_pos = _kick_and_relax(best_pos, potential, parameter, kick_scales[i], seed + i)
        if energy < best_energy:
            best_energy = energy
            best_pos = new_pos
    
    return best_energy, best_pos


def construct(instance):
    """Construct low-energy Dzugutov cluster using icosahedral manifold optimization + basin hopping."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Base scale estimate
    base_scale = size ** (1.0 / 3.0) * 0.45
    
    # Strategy 1: Icosahedral manifold optimization (primary)
    if _time_remaining() > 120:
        e, pos = _manifold_optimize(size, potential, parameter, base_scale, maxiter=30)
        if e < best_energy:
            best_energy = e
            best_positions = pos.copy()
    
    if best_positions is None or _time_remaining() < 100:
        # Fallback: simple icosahedral seed
        ico_points = _build_icosahedral_manifold(size, scale=base_scale)
        energy, positions = relax(ico_points, potential, parameter)
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
    
    if _time_remaining() < 80:
        return best_positions
    
    # Strategy 2: Basin hopping around best
    bh_energy, bh_pos = _basin_hopping(
        best_positions, potential, parameter,
        n_kicks=10, kick_scales=[0.05, 0.08, 0.12, 0.15, 0.20, 0.25, 0.15, 0.10, 0.08, 0.05],
        seed=size
    )
    if bh_energy < best_energy:
        best_energy = bh_energy
        best_positions = bh_pos.copy()
    
    if _time_remaining() < 60:
        return best_positions
    
    # Strategy 3: Try different manifold parameters
    for scale_factor, twist in [(0.9, 0.1), (1.1, -0.1), (0.95, 0.2), (1.05, -0.2)]:
        if _time_remaining() < 45:
            break
        ico_points = _build_icosahedral_manifold(size, scale=base_scale * scale_factor, twist=twist)
        energy, positions = relax(ico_points, potential, parameter)
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
    
    if _time_remaining() < 40:
        return best_positions
    
    # Strategy 4: Basin hopping with larger kicks
    bh_energy2, bh_pos2 = _basin_hopping(
        best_positions, potential, parameter,
        n_kicks=8, kick_scales=[0.20, 0.30, 0.25, 0.35, 0.20, 0.15, 0.10, 0.05],
        seed=size + 100
    )
    if bh_energy2 < best_energy:
        best_energy = bh_energy2
        best_positions = bh_pos2.copy()
    
    if _time_remaining() < 30:
        return best_positions
    
    # Strategy 5: Random compact initialization
    rng = np.random.default_rng(size)
    radius = 1.1 * size ** (1.0 / 3.0) * 0.8
    points = []
    min_dist = 0.6
    attempts = 0
    max_attempts = size * 200
    while len(points) < size and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    if len(points) < size:
        min_dist = 0.5
        attempts = 0
        while len(points) < size and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= min_dist for q in points):
                    points.append(p)
            attempts += 1
    
    while len(points) < size:
        p = rng.uniform(-radius * 0.5, radius * 0.5, 3)
        points.append(p)
    
    rc_points = np.array(points[:size])
    energy, positions = relax(rc_points, potential, parameter)
    if energy < best_energy:
        best_energy = energy
        best_positions = positions.copy()
    
    if _time_remaining() < 20:
        return best_positions
    
    # Strategy 6: Final basin hopping
    bh_energy3, bh_pos3 = _basin_hopping(
        best_positions, potential, parameter,
        n_kicks=8, kick_scales=[0.05, 0.08, 0.12, 0.15, 0.10, 0.08, 0.05, 0.03],
        seed=size + 200
    )
    if bh_energy3 < best_energy:
        best_energy = bh_energy3
        best_positions = bh_pos3.copy()
    
    return best_positions
# EVOLVE-BLOCK-END

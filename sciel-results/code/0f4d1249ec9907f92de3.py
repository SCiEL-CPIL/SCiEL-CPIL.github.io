"""Program for Dzugutov cluster global minima using systematic polytetrahedral shell construction."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


def _dzugutov_pair(r, A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16):
    """Compute pair potential at distance r."""
    if r < 1e-10:
        return 1e10
    if r < a:
        return A * (r**(-m) - B) * np.exp(c / (r - a))
    elif r < b:
        return B * np.exp(d / (r - b))
    else:
        return 0.0


def _find_optimal_distance():
    """Find the distance that minimizes the pair potential."""
    def V(r):
        return _dzugutov_pair(r)
    res = minimize_scalar(V, bounds=(0.8, 1.85), method='bounded')
    return res.x


def _icosahedron_unit():
    """Return 12 unit-icosahedron vertices (edge length 1)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    edge = np.linalg.norm(verts[0] - verts[1])
    return verts / edge


def _dodecahedron_unit():
    """Return 20 vertices of a regular dodecahedron (edge length 1)."""
    phi = (1 + np.sqrt(5)) / 2
    iphi = 1.0 / phi
    verts = []
    # 8 vertices: (±1, ±1, ±1)
    for sx in [1, -1]:
        for sy in [1, -1]:
            for sz in [1, -1]:
                verts.append([sx, sy, sz])
    # 12 vertices: (0, ±iphi, ±phi), (±iphi, ±phi, 0), (±phi, 0, ±iphi)
    for s in [1, -1]:
        for t in [1, -1]:
            verts.append([0, s*iphi, t*phi])
            verts.append([s*iphi, t*phi, 0])
            verts.append([t*phi, 0, s*iphi])
    verts = np.array(verts, dtype=float)
    # Find edge length and normalize
    # Nearest neighbor distance
    min_d = np.inf
    for i in range(len(verts)):
        for j in range(i+1, len(verts)):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < min_d:
                min_d = d
    return verts / min_d


def _build_shell_positions(n, bond_length, seed=42):
    """
    Build a cluster using icosahedral core + successive shells.
    Shell 0: center (1 atom)
    Shell 1: icosahedron (12 atoms) at distance r1
    Shell 2: 30 atoms at edge-midpoint positions scaled
    Shell 3: 60+ atoms at face-center and other positions
    """
    rng = np.random.default_rng(seed)
    positions = [np.zeros(3)]
    
    # Shell 1: icosahedron
    icosa = _icosahedron_unit()
    r1 = bond_length
    for v in icosa:
        positions.append(v * r1)
    
    if n <= len(positions):
        return np.array(positions[:n])
    
    # Shell 2: positions at edge midpoints of icosahedron, scaled outward
    # The icosahedron has 30 edges. Edge midpoint direction, then scale to shell 2 radius.
    # Edge midpoint of icosahedron vertices i,j: (v_i + v_j)/2, normalize, then place at r2
    icosa_full = _icosahedron_unit()
    edge_midpoints = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(icosa_full[i] - icosa_full[j])
            if abs(d - 1.0) < 0.01:  # it's an edge
                mid_dir = (icosa_full[i] + icosa_full[j]) / 2
                mid_dir = mid_dir / np.linalg.norm(mid_dir)
                edge_midpoints.append(mid_dir)
    
    # Shell 2 radius: for icosahedral packing, r2 ≈ r1 * sqrt(2) or so
    r2 = bond_length * 1.618  # golden ratio
    for em in edge_midpoints:
        if len(positions) >= n:
            break
        p = em * r2
        # Check distance to existing
        existing = np.array(positions)
        dists = np.linalg.norm(existing - p, axis=1)
        if len(dists) > 0 and dists.min() < 0.5:
            continue
        positions.append(p)
    
    if len(positions) >= n:
        return np.array(positions[:n])
    
    # Shell 3: face centers of icosahedron, scaled
    # Icosahedron has 20 faces
    face_centers = []
    # Get face normals from icosahedron
    # Each face is a triangle of 3 mutually-connected vertices
    # For simplicity, use the 20 directions of a dodecahedron (dual of icosahedron)
    dodeca = _dodecahedron_unit()
    r3 = bond_length * 2.236  # sqrt(5)
    for v in dodeca:
        if len(positions) >= n:
            break
        p = v * r3
        existing = np.array(positions)
        dists = np.linalg.norm(existing - p, axis=1)
        if len(dists) > 0 and dists.min() < 0.5:
            continue
        positions.append(p)
    
    if len(positions) >= n:
        return np.array(positions[:n])
    
    # Shell 4 and beyond: add atoms at favorable positions
    # Use a combination of icosahedral and dodecahedral directions at increasing radii
    all_directions = []
    for v in icosa_full:
        all_directions.append(v / np.linalg.norm(v))
    for v in dodeca:
        all_directions.append(v / np.linalg.norm(v))
    # Also add some random directions
    for _ in range(50):
        v = rng.normal(size=3)
        all_directions.append(v / np.linalg.norm(v))
    
    shell_radii = [bond_length * 2.8, bond_length * 3.2, bond_length * 3.6, bond_length * 4.0]
    for r_shell in shell_radii:
        if len(positions) >= n:
            break
        for d in all_directions:
            if len(positions) >= n:
                break
            p = d * r_shell
            existing = np.array(positions)
            dists = np.linalg.norm(existing - p, axis=1)
            if len(dists) > 0 and dists.min() < 0.5:
                continue
            # Prefer positions close to existing atoms
            if len(dists) > 0:
                close = dists < bond_length * 1.4
                if close.sum() >= 2:
                    positions.append(p)
    
    # Fill remaining with random placement
    while len(positions) < n:
        p = rng.uniform(-bond_length * 5, bond_length * 5, 3)
        existing = np.array(positions)
        if len(existing) > 0:
            dists = np.linalg.norm(existing - p, axis=1)
            if dists.min() < 0.5:
                continue
        positions.append(p)
    
    return np.array(positions[:n])


def _build_compact_random(n, bond_length, seed=0):
    """Build a compact random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = bond_length * (n ** (1.0/3.0)) * 0.7
    positions = []
    attempts = 0
    max_attempts = 10000
    
    while len(positions) < n and attempts < max_attempts:
        attempts += 1
        # Sample from ball
        p = rng.normal(size=3)
        p = p / np.linalg.norm(p) * rng.uniform(0, radius)
        if len(positions) > 0:
            existing = np.array(positions)
            dists = np.linalg.norm(existing - p, axis=1)
            if dists.min() < 0.5:
                continue
        positions.append(p)
    
    # If not enough, add more with larger radius
    while len(positions) < n:
        p = rng.uniform(-bond_length * 6, bond_length * 6, 3)
        existing = np.array(positions)
        if len(existing) > 0:
            dists = np.linalg.norm(existing - p, axis=1)
            if dists.min() < 0.5:
                continue
        positions.append(p)
    
    return np.array(positions[:n])


def _build_bcc_fragment(n, bond_length, seed=0):
    """Build a BCC-like fragment."""
    rng = np.random.default_rng(seed)
    a_lat = bond_length * 2 / np.sqrt(3)
    positions = []
    for ix in range(-3, 4):
        for iy in range(-3, 4):
            for iz in range(-3, 4):
                positions.append([ix*a_lat, iy*a_lat, iz*a_lat])
                positions.append([(ix+0.5)*a_lat, (iy+0.5)*a_lat, (iz+0.5)*a_lat])
    positions = np.array(positions)
    # Keep only those within radius
    radius = bond_length * (n ** (1.0/3.0)) * 0.8
    mask = np.linalg.norm(positions, axis=1) <= radius
    positions = positions[mask]
    if len(positions) >= n:
        idx = rng.permutation(len(positions))[:n]
        return positions[idx]
    return positions


def _build_fcc_fragment(n, bond_length, seed=0):
    """Build an FCC-like fragment."""
    rng = np.random.default_rng(seed)
    a_lat = bond_length * 2 / np.sqrt(2)
    positions = []
    for ix in range(-3, 4):
        for iy in range(-3, 4):
            for iz in range(-3, 4):
                positions.append([ix*a_lat, iy*a_lat, iz*a_lat])
                positions.append([(ix+0.5)*a_lat, (iy+0.5)*a_lat, iz*a_lat])
                positions.append([(ix+0.5)*a_lat, iy*a_lat, (iz+0.5)*a_lat])
                positions.append([ix*a_lat, (iy+0.5)*a_lat, (iz+0.5)*a_lat])
    positions = np.array(positions)
    radius = bond_length * (n ** (1.0/3.0)) * 0.8
    mask = np.linalg.norm(positions, axis=1) <= radius
    positions = positions[mask]
    if len(positions) >= n:
        idx = rng.permutation(len(positions))[:n]
        return positions[idx]
    return positions


def construct(instance):
    """Construct low-energy Dzugutov cluster using multi-start polytetrahedral construction."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    opt_dist = _find_optimal_distance()
    
    # Generate diverse candidates
    candidates = []
    
    # 1. Systematic shell construction (primary approach)
    for seed in range(5):
        try:
            pos = _build_shell_positions(size, opt_dist, seed=seed)
            candidates.append(pos)
        except Exception:
            pass
    
    # 2. Compact random
    for seed in range(5):
        try:
            pos = _build_compact_random(size, opt_dist, seed=100+seed)
            candidates.append(pos)
        except Exception:
            pass
    
    # 3. BCC fragment
    try:
        pos = _build_bcc_fragment(size, opt_dist, seed=200)
        candidates.append(pos)
    except Exception:
        pass
    
    # 4. FCC fragment
    try:
        pos = _build_fcc_fragment(size, opt_dist, seed=300)
        candidates.append(pos)
    except Exception:
        pass
    
    # Relax all candidates and pick the best
    best_energy = np.inf
    best_positions = None
    
    for pos in candidates:
        try:
            # Ensure minimum distance constraint
            n = len(pos)
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < 0.5:
                        mid = (pos[i] + pos[j]) / 2
                        diff = pos[i] - pos[j]
                        if np.linalg.norm(diff) < 1e-10:
                            diff = np.array([1.0, 0, 0])
                        else:
                            diff = diff / np.linalg.norm(diff)
                        pos[i] = mid + diff * 0.26
                        pos[j] = mid - diff * 0.26
            
            energy, relaxed = relax(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except Exception:
            continue
    
    if best_positions is None:
        # Fallback
        rng = np.random.default_rng(0)
        positions = rng.uniform(-3, 3, (size, 3))
        energy, best_positions = relax(positions, potential, parameter)
    
    return best_positions
# EVOLVE-BLOCK-END

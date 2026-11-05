"""Program for Dzugutov cluster global minima using polytetrahedral construction."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


def _dzugutov_energy(positions, A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16):
    """Compute total Dzugutov potential energy."""
    n = len(positions)
    E = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-10:
                continue
            if r < a:
                V = A * (r**(-m) - B) * np.exp(c / (r - a))
            elif r < b:
                V = B * np.exp(d / (r - b))
            else:
                V = 0.0
            E += V
    return E


def _find_optimal_distance(A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16):
    """Find the distance that minimizes the pair potential."""
    def V(r):
        if r < a:
            return A * (r**(-m) - B) * np.exp(c / (r - a))
        elif r < b:
            return B * np.exp(d / (r - b))
        else:
            return 0.0
    # Search for minimum in reasonable range
    rs = np.linspace(0.7, 1.85, 1000)
    Vs = [V(r) for r in rs]
    idx = np.argmin(Vs)
    # Refine
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(V, bounds=(rs[max(0,idx-5)], rs[min(len(rs)-1, idx+5)]), method='bounded')
    return res.x


def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    # Normalize to unit edge length
    # Edge length of this icosahedron: distance between (0,1,phi) and (1,phi,0)
    edge = np.linalg.norm(verts[0] - verts[1])
    verts = verts / edge  # now edge length = 1
    return verts


def _make_icosahedral_cluster(n, bond_length):
    """Construct an icosahedral cluster with n atoms."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    if n == 13:
        # 1 center + 12 icosahedron vertices
        verts = _icosahedron_vertices() * bond_length
        return np.vstack([np.array([[0.0, 0.0, 0.0]]), verts])
    
    # For larger clusters, use nested icosahedra or add atoms at favorable positions
    # Start with 13-atom icosahedron
    verts = _icosahedron_vertices() * bond_length
    positions = [np.array([0.0, 0.0, 0.0])]
    positions.extend(verts.tolist())
    
    # For n > 13, add atoms at second shell positions
    # Second shell of icosahedron: 30 positions (edge midpoints scaled) or other polytetrahedral sites
    if n > 13:
        # Generate candidate positions for additional atoms
        # Use positions at various distances and angles
        rng = np.random.default_rng(42)
        existing = np.array(positions)
        
        # Try to add atoms at positions that are roughly bond_length from existing atoms
        # Use a grid of candidate positions on shells
        for shell_r in [bond_length * 1.5, bond_length * 1.8, bond_length * 2.0, bond_length * 2.2]:
            if len(positions) >= n:
                break
            # Generate random directions
            for _ in range(200):
                if len(positions) >= n:
                    break
                # Random direction
                v = rng.normal(size=3)
                v = v / np.linalg.norm(v)
                p = v * shell_r
                # Check minimum distance to existing
                dists = np.linalg.norm(existing - p, axis=1)
                if len(dists) > 0 and dists.min() < 0.5:
                    continue
                # Check that it's close to some existing atoms (for polytetrahedral character)
                if len(dists) > 0:
                    close = dists < bond_length * 1.3
                    if close.sum() >= 3:  # at least 3 neighbors
                        positions.append(p)
                        existing = np.vstack([existing, p])
    
    if len(positions) > n:
        positions = positions[:n]
    
    # If we still don't have enough, add more random atoms
    while len(positions) < n:
        rng = np.random.default_rng(123 + len(positions))
        for attempt in range(1000):
            p = rng.uniform(-bond_length * 3, bond_length * 3, 3)
            existing = np.array(positions)
            dists = np.linalg.norm(existing - p, axis=1)
            if len(dists) > 0 and dists.min() < 0.5:
                continue
            positions.append(p)
            break
        else:
            # Just add at a random far position
            p = rng.uniform(-bond_length * 5, bond_length * 5, 3)
            positions.append(p)
    
    return np.array(positions[:n])


def _make_polytetrahedral_cluster(n, bond_length):
    """Construct a polytetrahedral cluster using tetrahedral building blocks."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    if n == 4:
        # Regular tetrahedron
        verts = np.array([
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
        ]) / np.sqrt(3)
        return verts * bond_length * np.sqrt(3) / 2  # adjust for edge length
    
    # Build by adding tetrahedra
    positions = [np.array([0.0, 0.0, 0.0])]
    rng = np.random.default_rng(99)
    
    while len(positions) < n:
        # Pick an existing atom and add a new atom at tetrahedral distance
        base = positions[rng.integers(len(positions))]
        # Random direction
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v)
        p = base + v * bond_length
        
        existing = np.array(positions)
        dists = np.linalg.norm(existing - p, axis=1)
        if len(dists) > 0 and dists.min() < 0.5:
            continue
        positions.append(p)
    
    return np.array(positions[:n])


def construct(instance):
    """Construct low-energy Dzugutov cluster using polytetrahedral construction + relaxation."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Find optimal bond length
    opt_dist = _find_optimal_distance()
    
    # Generate multiple candidate structures
    candidates = []
    
    # 1. Icosahedral-based
    try:
        icosa = _make_icosahedral_cluster(size, opt_dist)
        candidates.append(("icosa", icosa))
    except Exception:
        pass
    
    # 2. Polytetrahedral
    try:
        poly = _make_polytetrahedral_cluster(size, opt_dist)
        candidates.append(("poly", poly))
    except Exception:
        pass
    
    # 3. Random compact with better initialization
    rng = np.random.default_rng(size)
    radius = opt_dist * size ** (1.0 / 3.0) * 0.8
    for trial in range(6):
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.5 * opt_dist for q in points):
                    points.append(p)
        candidates.append((f"rand{trial}", np.array(points)))
    
    # 4. BCC/FCC-like lattice fragments (as additional candidates)
    for lattice_type in ["bcc", "fcc"]:
        if lattice_type == "bcc":
            # BCC unit cell: 2 atoms
            a_lat = opt_dist * 2 / np.sqrt(3)
            positions = []
            for ix in range(-2, 3):
                for iy in range(-2, 3):
                    for iz in range(-2, 3):
                        positions.append([ix*a_lat, iy*a_lat, iz*a_lat])
                        positions.append([(ix+0.5)*a_lat, (iy+0.5)*a_lat, (iz+0.5)*a_lat])
            positions = np.array(positions)
            # Keep only those within radius
            mask = np.linalg.norm(positions, axis=1) <= radius * 1.2
            positions = positions[mask]
            if len(positions) >= size:
                # Shuffle and take first size
                idx = rng.permutation(len(positions))[:size]
                candidates.append((f"bcc{trial}", positions[idx]))
        else:
            a_lat = opt_dist * 2 / np.sqrt(2)
            positions = []
            for ix in range(-2, 3):
                for iy in range(-2, 3):
                    for iz in range(-2, 3):
                        positions.append([ix*a_lat, iy*a_lat, iz*a_lat])
                        positions.append([(ix+0.5)*a_lat, (iy+0.5)*a_lat, (iz)*a_lat])
                        positions.append([(ix+0.5)*a_lat, (iy)*a_lat, (iz+0.5)*a_lat])
                        positions.append([(ix)*a_lat, (iy+0.5)*a_lat, (iz+0.5)*a_lat])
            positions = np.array(positions)
            mask = np.linalg.norm(positions, axis=1) <= radius * 1.2
            positions = positions[mask]
            if len(positions) >= size:
                idx = rng.permutation(len(positions))[:size]
                candidates.append((f"fcc{trial}", positions[idx]))
    
    # Relax all candidates and pick the best
    best_energy = np.inf
    best_positions = None
    
    for name, pos in candidates:
        try:
            # Ensure minimum distance constraint
            # Check and fix if needed
            n = len(pos)
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < 0.5:
                        # Push apart
                        mid = (pos[i] + pos[j]) / 2
                        diff = pos[i] - pos[j]
                        if np.linalg.norm(diff) < 1e-10:
                            diff = np.array([1.0, 0, 0])
                        diff = diff / np.linalg.norm(diff)
                        pos[i] = mid + diff * 0.25
                        pos[j] = mid - diff * 0.25
            
            energy, relaxed = relax(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except Exception:
            continue
    
    if best_positions is None:
        # Fallback: simple random
        rng = np.random.default_rng(0)
        positions = rng.uniform(-3, 3, (size, 3))
        energy, best_positions = relax(positions, potential, parameter)
    
    return best_positions
# EVOLVE-BLOCK-END

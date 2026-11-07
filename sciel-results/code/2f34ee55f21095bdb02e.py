"""Sutton-Chen transition-metal cluster global minima search.
Strategy: Physics-informed templates (FCC, HCP, icosahedral, random) + Basin Hopping.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 295.0  # Safety margin before 300s limit
_START_TIME = None


def _time_left():
    if _START_TIME is None:
        return _DEADLINE_SEC
    return _DEADLINE_SEC - (time.time() - _START_TIME)


def _fcc_fragment(n, spacing=1.0):
    """Generate n atoms from an FCC lattice fragment."""
    # FCC lattice vectors
    a = spacing
    # Generate FCC points in a cube
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    points.append(p)
    
    points = np.array(points)
    # Center the cluster
    points -= points.mean(axis=0)
    
    # Select n closest to center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _hcp_fragment(n, spacing=1.0):
    """Generate n atoms from an HCP lattice fragment."""
    a = spacing
    c = a * np.sqrt(2.0 / 3.0)  # Ideal HCP c/a ratio
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                # HCP basis in hexagonal coordinates
                # Layer A: (0,0,0), (1,0,0), (0,1,0) in hex basis
                # Convert hex to Cartesian
                for xi, yi in [(0,0), (1,0), (0,1)]:
                    x = a * (xi + 0.5 * yi)
                    y = a * np.sqrt(3)/2 * yi
                    z = c * k
                    points.append([x, y, z])
                # Layer B: shifted
                for xi, yi in [(0,0), (1,0), (0,1)]:
                    x = a * (xi + 0.5 * yi) + a/np.sqrt(3)
                    y = a * np.sqrt(3)/2 * yi + a/2
                    z = c * (k + 0.5)
                    points.append([x, y, z])
    
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _icosahedron_based(n, spacing=1.0):
    """Generate n atoms based on icosahedral symmetry."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ]) * spacing
    verts /= np.linalg.norm(verts[0]) * spacing  # Normalize to unit radius
    verts *= spacing
    
    points = list(verts)
    rng = np.random.default_rng(42)
    
    # Add more atoms in shells
    shell_radii = [spacing * 0.5, spacing * 1.0, spacing * 1.5, spacing * 2.0, spacing * 2.5]
    while len(points) < n:
        # Try adding at icosahedral positions in next shell
        shell_r = shell_radii[min(len(points) // 12, len(shell_radii)-1)]
        for v in verts:
            if len(points) >= n:
                break
            p = v * shell_r
            # Check minimum distance
            if all(np.linalg.norm(p - q) >= 0.4 * spacing for q in points):
                points.append(p)
        # If still not enough, add random points
        if len(points) < n:
            p = rng.uniform(-shell_r, shell_r, 3)
            if np.linalg.norm(p) <= shell_r and all(np.linalg.norm(p - q) >= 0.35 * spacing for q in points):
                points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _random_compact(n, spacing=0.72, seed=0):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms, fill remaining with small random offsets
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3) * 0.1
        points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _basin_hopping(positions, kind, n_exp, m_exp, c, rng, max_iterations=50):
    """Simple basin hopping: perturb, relax, keep best."""
    best_energy, best_positions = sutton_chen_energy_gradient(positions, kind)
    current_energy, current_positions = best_energy, best_positions.copy()
    
    for it in range(max_iterations):
        if _time_left() < 10.0:
            break
        
        # Perturb
        perturbation_scale = 0.1 + 0.3 * (it / max_iterations)
        perturbed = current_positions + rng.normal(0, perturbation_scale, current_positions.shape)
        
        # Relax
        energy, relaxed = relax(perturbed, 'sutton_chen', kind)
        
        if energy < best_energy:
            best_energy, best_positions = energy, relaxed
        current_positions = relaxed
    
    return best_energy, best_positions


def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    size = instance["n"]
    kind = instance.get("kind", "12-6")
    n_exp = instance.get("n_exp", 12)
    m_exp = instance.get("m_exp", 6)
    c = instance.get("c", 144.41)
    
    rng = np.random.default_rng(size)
    
    # Generate diverse starting structures
    starting_structures = []
    
    # FCC fragment
    try:
        fcc = _fcc_fragment(size, spacing=0.72)
        starting_structures.append(fcc)
    except:
        pass
    
    # HCP fragment
    try:
        hcp = _hcp_fragment(size, spacing=0.72)
        starting_structures.append(hcp)
    except:
        pass
    
    # Icosahedron-based
    try:
        ico = _icosahedron_based(size, spacing=0.72)
        starting_structures.append(ico)
    except:
        pass
    
    # Multiple random compact clusters
    for seed in [size, size+1, size+2, size+3]:
        try:
            rand = _random_compact(size, spacing=0.72, seed=seed)
            starting_structures.append(rand)
        except:
            pass
    
    # Relax each starting structure and keep the best
    best_energy = np.inf
    best_positions = None
    
    for i, struct in enumerate(starting_structures):
        if _time_left() < 15.0:
            break
        
        try:
            energy, positions = relax(struct, 'sutton_chen', kind)
            if energy < best_energy:
                best_energy, best_positions = energy, positions
        except:
            continue
    
    # Basin hopping from the best structure
    if best_positions is not None and _time_left() > 20.0:
        try:
            best_energy, best_positions = _basin_hopping(
                best_positions, kind, n_exp, m_exp, c, rng, max_iterations=30
            )
        except:
            pass
    
    # Additional basin hopping from other structures if time permits
    for i, struct in enumerate(starting_structures):
        if _time_left() < 15.0:
            break
        if i == 0 and best_positions is not None:
            continue  # Already did basin hopping from best
        
        try:
            energy, positions = relax(struct, 'sutton_chen', kind)
            if energy < best_energy:
                best_energy, best_positions = energy, positions
            elif _time_left() > 25.0:
                # Basin hopping from this structure
                e2, p2 = _basin_hopping(positions, kind, n_exp, m_exp, c, rng, max_iterations=20)
                if e2 < best_energy:
                    best_energy, best_positions = e2, p2
        except:
            continue
    
    return best_positions
# EVOLVE-BLOCK-END

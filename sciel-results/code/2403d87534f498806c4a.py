"""Structure-motif guided basin hopping for Sutton-Chen cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE = 320.0  # safety margin before 330s kill
_START_TIME = time.time()


def _time_left():
    """Return remaining time in seconds."""
    return _DEADLINE - (time.time() - _START_TIME)


def _fcc_cluster(n, scale=1.0):
    """Generate an FCC-like cluster with n atoms."""
    pts = []
    a = scale
    # Generate FCC lattice points in a sphere
    max_idx = int(np.ceil(n ** (1/3))) + 2
    fcc_offsets = np.array([[0, 0, 0],
                            [0.5, 0.5, 0],
                            [0.5, 0, 0.5],
                            [0, 0.5, 0.5]]) * a
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for off in fcc_offsets:
                    p = np.array([i, j, k]) * a + off
                    if np.linalg.norm(p) <= a * max_idx * 1.1:
                        pts.append(p)
    
    pts = np.array(pts)
    if len(pts) == 0:
        return np.zeros((n, 3))
    
    # Select n closest points to center
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _hcp_cluster(n, scale=1.0):
    """Generate an HCP-like cluster with n atoms."""
    pts = []
    a = scale
    # HCP: two interpenetrating hexagonal close-packed layers
    max_idx = int(np.ceil(n ** (1/3))) + 2
    # Hexagonal basis
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            x = a * (i + 0.5 * (j % 2))
            y = a * np.sqrt(3) / 2 * j
            # Two layers for HCP
            for layer in [0, a / np.sqrt(2)]:
                p = np.array([x, y, layer])
                if np.linalg.norm(p) <= a * max_idx * 1.1:
                    pts.append(p)
    
    pts = np.array(pts)
    if len(pts) == 0:
        return np.zeros((n, 3))
    
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _icosa_cluster(n, scale=1.0):
    """Generate an icosahedral-like cluster with n atoms."""
    pts = []
    a = scale
    # Icosahedron vertices (normalized to unit sphere)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_verts /= np.linalg.norm(ico_verts[0]) * 2  # normalize
    
    # Shell 1: 12 vertices
    shell1 = ico_verts * a
    pts.extend([p.copy() for p in shell1])
    
    # Shell 2: 30 vertices (edges of icosahedron)
    if n > 13:
        # Add midpoints of edges
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                if d < 0.5:  # connected vertices
                    mid = (ico_verts[i] + ico_verts[j]) / 2
                    mid /= np.linalg.norm(mid)
                    edges.append(mid * a * 1.6)
        pts.extend([p.copy() for p in edges])
    
    # Shell 3: center
    pts.append(np.zeros(3))
    
    # Pad with random points if needed
    while len(pts) < n:
        p = a * 1.8 * np.random.randn(3)
        if np.linalg.norm(p) <= a * 2.5:
            pts.append(p)
    
    pts = np.array(pts[:n])
    return pts


def _random_compact(n, scale=0.72, seed=None):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = scale * n ** (1.0 / 3.0) * 0.8
    pts = []
    attempts = 0
    while len(pts) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.4 * scale for q in pts):
                pts.append(p)
        attempts += 1
    
    pts = np.array(pts)
    if len(pts) < n:
        # Fill remaining with random points
        while len(pts) < n:
            p = rng.uniform(-radius, radius, 3)
            pts = np.vstack([pts, p])
    return pts[:n]


def _perturb(positions, sigma):
    """Add Gaussian perturbation."""
    return positions + sigma * np.random.randn(*positions.shape)


def _local_relax(positions, potential, kind, max_iter=1000):
    """Run local minimization and return (energy, positions)."""
    try:
        energy, pos = relax(positions, potential, kind)
        return energy, pos
    except Exception:
        # Fallback: simple gradient descent
        energy, grad = sutton_chen_energy_gradient(positions, kind)
        lr = 0.01
        for _ in range(max_iter):
            positions = positions - lr * grad
            energy, grad = sutton_chen_energy_gradient(positions, kind)
            if np.linalg.norm(grad) < 1e-8:
                break
        return energy, positions


def construct(instance):
    """Structure-motif guided basin hopping for global minimum search."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("rho", instance.get("kind"))
    
    # Time budget management
    t0 = time.time()
    
    # Initial structures: multiple motifs
    initial_structures = []
    
    # 1. FCC-based
    fcc = _fcc_cluster(size, scale=0.72)
    initial_structures.append(fcc)
    
    # 2. HCP-based
    hcp = _hcp_cluster(size, scale=0.72)
    initial_structures.append(hcp)
    
    # 3. Icosahedral
    icosa = _icosa_cluster(size, scale=0.72)
    initial_structures.append(icosa)
    
    # 4-7. Random compact structures
    for i in range(4):
        rand = _random_compact(size, scale=0.72, seed=size + i * 1000)
        initial_structures.append(rand)
    
    # Local relaxation of all initial structures
    best_energy = np.inf
    best_positions = None
    
    for i, init in enumerate(initial_structures):
        if time.time() - t0 > _DEADLINE - 30:
            break
        energy, pos = _local_relax(init, potential, kind)
        if np.isfinite(energy) and energy < best_energy:
            best_energy = energy
            best_positions = pos.copy()
    
    if best_positions is None:
        # Fallback: return the first relaxed structure
        energy, best_positions = _local_relax(initial_structures[0], potential, kind)
        best_energy = energy
    
    # Basin hopping: perturb and re-relax
    sigma = 0.15
    n_bh = 0
    
    while time.time() - t0 < _DEADLINE - 20:
        # Generate perturbation
        perturbed = _perturb(best_positions, sigma)
        
        # Local relaxation
        energy, pos = _local_relax(perturbed, potential, kind)
        
        if np.isfinite(energy):
            # Metropolis acceptance
            if energy < best_energy:
                best_energy = energy
                best_positions = pos.copy()
                # Reduce perturbation size on success
                sigma *= 0.95
            else:
                # Accept with probability exp(-(E - E_best)/kT)
                kT = 0.1
                accept_prob = np.exp(-(energy - best_energy) / kT)
                if np.random.random() < accept_prob:
                    best_positions = pos.copy()
        
        n_bh += 1
        
        # Gradually increase perturbation if stuck
        if n_bh % 10 == 0:
            sigma = min(0.3, sigma * 1.05)
    
    # Final validation
    if best_positions is None or not np.all(np.isfinite(best_positions)):
        # Return a valid fallback
        best_positions = _fcc_cluster(size, scale=0.72)
    
    # Ensure no two atoms are closer than 0.35
    min_dist = np.inf
    for i in range(len(best_positions)):
        for j in range(i + 1, len(best_positions)):
            d = np.linalg.norm(best_positions[i] - best_positions[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 0.35:
        # Scale up slightly to enforce minimum distance
        scale_factor = 0.35 / min_dist * 1.01
        best_positions = best_positions * scale_factor
    
    return best_positions
# EVOLVE-BLOCK-END

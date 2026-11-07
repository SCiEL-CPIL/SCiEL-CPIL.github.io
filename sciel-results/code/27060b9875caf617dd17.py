"""Structure-motif + basin hopping for Sutton-Chen cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 315.0  # safety margin before 330s kill


def _time_left():
    """Return remaining time in seconds."""
    return _DEADLINE - (time.time() - _START_TIME)


def _fcc_cluster(n, scale=0.72):
    """Generate an FCC-like cluster with n atoms."""
    pts = []
    # Generate FCC lattice points in a cube
    a = scale
    max_idx = 10
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC basis: (0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0)
                for dx, dy, dz in [(0, 0, 0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)]:
                    p = np.array([i * a + dx, j * a + dy, k * a + dz])
                    pts.append(p)
    pts = np.array(pts)
    # Keep only the n closest to origin
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    pts = pts[idx]
    # Center
    pts -= pts.mean(axis=0)
    return pts


def _hcp_cluster(n, scale=0.72):
    """Generate an HCP-like cluster with n atoms."""
    pts = []
    a = scale
    c = a * np.sqrt(8.0 / 3.0)
    max_idx = 8
    for layer in range(-max_idx, max_idx + 1):
        z = layer * c
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                if layer % 2 == 0:
                    x = i * a + 0.5 * a * (j % 2)
                    y = j * a * np.sqrt(3.0) / 2.0
                else:
                    x = i * a
                    y = j * a * np.sqrt(3.0) / 2.0 + a * np.sqrt(3.0) / 6.0
                p = np.array([x, y, z])
                pts.append(p)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    pts = pts[idx]
    pts -= pts.mean(axis=0)
    return pts


def _icosa_cluster(n, scale=0.72):
    """Generate an icosahedral-like cluster with n atoms."""
    pts = []
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0])
    
    # Concentric shells
    shell_radii = [0.0, 1.0, np.sqrt(2), np.sqrt(3), 2.0]
    shell_counts = [1, 12, 20, 30, 60]
    
    # Build shells
    for r, count in zip(shell_radii, shell_counts):
        if len(pts) >= n:
            break
        if r == 0:
            pts.append(np.zeros(3))
        else:
            # Use icosahedron vertices scaled to radius r
            for v in ico_verts:
                if len(pts) >= n:
                    break
                pts.append(r * v)
            # Add octahedron vertices if needed
            if len(pts) < n:
                oct_verts = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
                for v in oct_verts:
                    if len(pts) >= n:
                        break
                    pts.append(r * v)
    
    pts = np.array(pts[:n]) * scale
    pts -= pts.mean(axis=0)
    return pts


def _random_compact(n, seed=42):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = 0.72 * n ** (1.0 / 3.0) * 0.8
    pts = []
    min_dist = 0.55
    attempts = 0
    while len(pts) < n and attempts < n * 1000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in pts):
                pts.append(p)
        attempts += 1
    # If we couldn't place enough, just add random points
    while len(pts) < n:
        pts.append(rng.uniform(-radius, radius, 3))
    return np.array(pts)


def _perturb(positions, sigma, rng):
    """Perturb positions by adding Gaussian noise."""
    return positions + rng.normal(0, sigma, positions.shape)


def construct(instance):
    """Structure-motif + basin hopping for Sutton-Chen cluster global minima."""
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    
    # Generate initial structures from different motifs
    initial_structures = []
    
    # FCC
    try:
        initial_structures.append(_fcc_cluster(n))
    except Exception:
        pass
    
    # HCP
    try:
        initial_structures.append(_hcp_cluster(n))
    except Exception:
        pass
    
    # Icosahedral
    try:
        initial_structures.append(_icosa_cluster(n))
    except Exception:
        pass
    
    # Random compact with different seeds
    for seed in [42, 123, 456, 789]:
        try:
            initial_structures.append(_random_compact(n, seed))
        except Exception:
            pass
    
    # Relax each initial structure
    best_energy = np.inf
    best_positions = None
    
    for i, pos in enumerate(initial_structures):
        if _time_left() < 5:
            break
        try:
            energy, positions = relax(pos, 'sutton_chen', kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
    
    # Basin hopping on the best structure
    if best_positions is not None and _time_left() > 10:
        rng = np.random.default_rng(42)
        current_energy = best_energy
        current_positions = best_positions.copy()
        
        # Adaptive step size
        sigma = 0.3
        min_sigma = 0.01
        max_sigma = 0.5
        
        while _time_left() > 5:
            # Perturb
            new_positions = _perturb(current_positions, sigma, rng)
            
            # Relax
            try:
                energy, positions = relax(new_positions, 'sutton_chen', kind)
            except Exception:
                continue
            
            if not np.isfinite(energy):
                continue
            
            # Accept if better or with Metropolis criterion
            if energy < current_energy or (rng.random() < np.exp(-100 * (energy - current_energy) / max(1e-10, abs(current_energy)))):
                current_energy = energy
                current_positions = positions
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
                # Decrease sigma slightly on acceptance
                sigma = max(min_sigma, sigma * 0.95)
            else:
                # Increase sigma on rejection
                sigma = min(max_sigma, sigma * 1.1)
        
        # Final relaxation of the best
        try:
            energy, positions = relax(best_positions, 'sutton_chen', kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
    
    # Ensure no two atoms are too close
    if best_positions is not None:
        # Check minimum distance
        from scipy.spatial.distance import pdist
        min_d = pdist(best_positions).min()
        if min_d < 0.35:
            # Try to fix by relaxing again from a slightly perturbed state
            rng = np.random.default_rng(999)
            for attempt in range(5):
                perturbed = best_positions + rng.normal(0, 0.1, best_positions.shape)
                try:
                    energy, positions = relax(perturbed, 'sutton_chen', kind)
                    if np.isfinite(energy):
                        min_d2 = pdist(positions).min()
                        if min_d2 >= 0.35:
                            best_positions = positions
                            break
                except Exception:
                    continue
    
    return best_positions
# EVOLVE-BLOCK-END

"""Structure-motif + adaptive basin hopping for Sutton-Chen cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill


def _time_left():
    """Return remaining time in seconds."""
    return _DEADLINE - (time.time() - _START_TIME)


def _fcc_cluster(n, a=0.72):
    """Generate an FCC-like cluster with n atoms."""
    pts = []
    max_idx = 12
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC lattice points
                for di, dj, dk in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = a * np.array([i + di, j + dj, k + dk])
                    pts.append(p)
    pts = np.array(pts)
    if len(pts) < n:
        return pts
    # Sort by distance from center and take n closest
    center = pts.mean(axis=0)
    dists = np.linalg.norm(pts - center, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _hcp_cluster(n, a=0.72):
    """Generate an HCP-like cluster with n atoms."""
    pts = []
    max_idx = 10
    c = a * np.sqrt(2.0 / 3.0)
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # HCP: ABAB stacking
                if k % 2 == 0:
                    p = a * np.array([i, j, k * c / a])
                else:
                    p = a * np.array([i + 0.5, j + 0.5, k * c / a])
                pts.append(p)
    pts = np.array(pts)
    if len(pts) < n:
        return pts
    center = pts.mean(axis=0)
    dists = np.linalg.norm(pts - center, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _icosa_cluster(n, a=0.72):
    """Generate an icosahedral-like cluster with n atoms."""
    pts = []
    # Icosahedral shell coordinates
    phi = (1 + np.sqrt(5)) / 2
    # First shell: 12 vertices of icosahedron
    shell1 = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    for s in shell1:
        pts.append(a * np.array(s))
    # Second shell: 30 atoms
    shell2 = []
    for i in range(6):
        for j in range(5):
            angle = 2 * np.pi * j / 5
            r = a * 1.6
            pts.append(r * np.array([np.cos(angle), np.sin(angle), 0.5]))
    # Third shell: 20 atoms
    for i in range(20):
        angle = 2 * np.pi * i / 20
        r = a * 2.2
        pts.append(r * np.array([np.cos(angle), np.sin(angle), 0.3]))
    pts = np.array(pts)
    if len(pts) < n:
        # Add more atoms in a spherical shell
        rng = np.random.default_rng(42)
        while len(pts) < n:
            theta = rng.uniform(0, 2 * np.pi)
            phi_angle = rng.uniform(-np.pi / 2, np.pi / 2)
            r = a * 2.5
            p = r * np.array([np.cos(theta) * np.cos(phi_angle),
                              np.sin(theta) * np.cos(phi_angle),
                              np.sin(phi_angle)])
            pts = np.vstack([pts, p])
    center = pts.mean(axis=0)
    dists = np.linalg.norm(pts - center, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _random_compact(n, spacing=0.72, seed=42):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.7 * spacing for q in points):
                points.append(p)
        attempts += 1
    if len(points) < n:
        # Fill remaining with random points
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)


def _basin_hopping_step(positions, kind, n_exp, m_exp, c, rng):
    """Perform one basin hopping step: perturb, then relax."""
    n = len(positions)
    # Perturbation magnitude
    scale = 0.3 + 0.2 * rng.random()
    perturbed = positions + scale * rng.standard_normal((n, 3))
    # Relax
    energy, relaxed = relax(perturbed, 'sutton_chen', kind)
    return energy, relaxed


def construct(instance):
    """Find global minimum of Sutton-Chen cluster using structure motifs + basin hopping."""
    n = instance["n"]
    kind = instance["kind"]
    n_exp = instance["n_exp"]
    m_exp = instance["m_exp"]
    c = instance["c"]
    
    # Generate initial structures from different motifs
    initial_structures = []
    
    # FCC-like
    try:
        fcc = _fcc_cluster(n)
        if len(fcc) >= n:
            fcc = fcc[:n]
            initial_structures.append(fcc)
    except:
        pass
    
    # HCP-like
    try:
        hcp = _hcp_cluster(n)
        if len(hcp) >= n:
            hcp = hcp[:n]
            initial_structures.append(hcp)
    except:
        pass
    
    # Icosahedral-like
    try:
        ico = _icosa_cluster(n)
        if len(ico) >= n:
            ico = ico[:n]
            initial_structures.append(ico)
    except:
        pass
    
    # Random compact
    for seed in [42, 123, 456]:
        try:
            rand = _random_compact(n, seed=seed)
            if len(rand) == n:
                initial_structures.append(rand)
        except:
            pass
    
    if not initial_structures:
        # Fallback: simple random
        rng = np.random.default_rng(n)
        initial_structures.append(rng.uniform(-1, 1, (n, 3)))
    
    # Relax each initial structure
    best_energy = np.inf
    best_positions = None
    
    for idx, init in enumerate(initial_structures):
        if _time_left() < 5.0:
            break
        
        try:
            energy, positions = relax(init, 'sutton_chen', kind)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except:
            continue
    
    # Basin hopping from best structure
    if best_positions is not None and _time_left() > 10.0:
        rng = np.random.default_rng(int(time.time() * 1000) % 10000)
        current_energy = best_energy
        current_positions = best_positions.copy()
        
        max_iterations = min(50, int(_time_left() / 2.0))
        for it in range(max_iterations):
            if _time_left() < 5.0:
                break
            
            # Perturb
            scale = 0.2 + 0.3 * rng.random()
            perturbed = current_positions + scale * rng.standard_normal((n, 3))
            
            # Relax
            try:
                energy, relaxed = relax(perturbed, 'sutton_chen', kind)
            except:
                continue
            
            # Accept if better
            if energy < current_energy:
                current_energy = energy
                current_positions = relaxed.copy()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed.copy()
    
    if best_positions is None:
        # Last resort: return first initial structure
        best_positions = initial_structures[0]
    
    # Validate: check no atoms closer than 0.35
    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(best_positions[i] - best_positions[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 0.35:
        # Spread out atoms
        center = best_positions.mean(axis=0)
        best_positions = best_positions - center
        best_positions *= 1.5
    
    return best_positions
# EVOLVE-BLOCK-END

import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

def _fcc_cluster(n, spacing=1.0):
    """Generate n atoms from an FCC lattice fragment centered at origin."""
    # FCC lattice constant a such that nearest neighbor distance is 1.0
    # NN distance = a / sqrt(2) => a = sqrt(2)
    a = np.sqrt(2.0)
    max_r = int(np.ceil(n ** (1.0 / 3.0) * 1.5)) + 2
    points = []
    for ix in range(-max_r, max_r + 1):
        for iy in range(-max_r, max_r + 1):
            for iz in range(-max_r, max_r + 1):
                # FCC condition: i+j+k is even
                if (ix + iy + iz) % 2 == 0:
                    p = np.array([ix, iy, iz]) * a
                    points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]]

def _bcc_cluster(n, spacing=1.0):
    """Generate n atoms from a BCC lattice fragment centered at origin."""
    # BCC lattice constant a such that nearest neighbor distance is 1.0
    # NN distance = a * sqrt(3) / 2 => a = 2 / sqrt(3)
    a = 2.0 / np.sqrt(3.0)
    max_r = int(np.ceil(n ** (1.0 / 3.0) * 1.5)) + 2
    points = []
    for ix in range(-max_r, max_r + 1):
        for iy in range(-max_r, max_r + 1):
            for iz in range(-max_r, max_r + 1):
                # Corner
                p1 = np.array([ix, iy, iz]) * a
                points.append(p1)
                # Body center
                p2 = np.array([ix + 0.5, iy + 0.5, iz + 0.5]) * a
                points.append(p2)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]]

def _random_compact(n, spacing=1.0):
    """Generate n atoms randomly in a sphere, ensuring minimum distance."""
    rng = np.random.default_rng()
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    # Start with center
    points.append(np.zeros(3))
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check min distance 0.5 * spacing (equilibrium is 1.0, so 0.5 is repulsive core)
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    # If we didn't get enough, fill with random points (might overlap, relax will fix)
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)

def _jittered_fcc(n, spacing=1.0, jitter=0.05):
    """FCC with random jitter to break symmetry."""
    pts = _fcc_cluster(n, spacing)
    rng = np.random.default_rng()
    jitter_vec = rng.normal(0, jitter, size=(n, 3))
    return pts + jitter_vec

def _jittered_bcc(n, spacing=1.0, jitter=0.05):
    """BCC with random jitter to break symmetry."""
    pts = _bcc_cluster(n, spacing)
    rng = np.random.default_rng()
    jitter_vec = rng.normal(0, jitter, size=(n, 3))
    return pts + jitter_vec

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    time_limit = 290.0 # Safety margin for 300s limit
    
    best_energy = float('inf')
    best_positions = None
    
    # Select generators based on rho
    # Long range (low rho) -> more spherical/compact
    # Short range (high rho) -> more local packing (FCC/BCC)
    if parameter < 5:
        generators = [_random_compact, _fcc_cluster, _bcc_cluster, _jittered_fcc, _jittered_bcc]
    else:
        generators = [_fcc_cluster, _bcc_cluster, _jittered_fcc, _jittered_bcc, _random_compact]
    
    current_pos = None
    kick_temp = 0.1
    
    while time.time() - start_time < time_limit:
        # Pick a generator
        gen = np.random.choice(generators)
        try:
            positions = gen(size, 1.0)
        except Exception:
            continue
        
        # Relax
        try:
            energy, relaxed_pos = relax(positions, potential, parameter)
        except Exception:
            continue
            
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed_pos
            current_pos = relaxed_pos.copy()
            
        # Basin Hopping
        if current_pos is not None:
            # Occasionally kick the best found so far
            if np.random.rand() < 0.5:
                kick = np.random.normal(0, kick_temp, size=(size, 3))
                kicked_pos = current_pos + kick
                try:
                    energy2, relaxed_pos2 = relax(kicked_pos, potential, parameter)
                    if energy2 < best_energy:
                        best_energy = energy2
                        best_positions = relaxed_pos2
                        current_pos = relaxed_pos2.copy()
                except Exception:
                    pass
                
                # Decay kick temperature
                kick_temp *= 0.95
                if kick_temp < 0.01:
                    kick_temp = 0.05
    
    # Fallback
    if best_positions is None:
        best_positions = _random_compact(size, 1.0)
        try:
            _, best_positions = relax(best_positions, potential, parameter)
        except:
            pass

    return best_positions

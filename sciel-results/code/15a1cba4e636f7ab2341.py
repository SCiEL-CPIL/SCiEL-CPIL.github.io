"""
Strategy: Multi-start global optimization for Sutton-Chen transition-metal clusters.
Uses FCC, HCP, BCC, and icosahedral templates + adaptive basin hopping + polish.
"""

# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 295.0
_START_TIME = None

def _time_left():
    return _DEADLINE_SEC - (time.time() - _START_TIME)

def _fcc_fragment(n, spacing=1.0, rng=None):
    """Generate n atoms from an FCC lattice, centered."""
    if rng is None:
        rng = np.random.default_rng()
    
    a = spacing * np.sqrt(2)
    L = n ** (1.0/3.0) * spacing * 1.5
    n_side = int(np.ceil(L / a)) + 2
    
    points = []
    for i in range(-n_side, n_side + 1):
        for j in range(-n_side, n_side + 1):
            for k in range(-n_side, n_side + 1):
                base = np.array([i, j, k]) * a
                for offset in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = base + np.array(offset) * a
                    points.append(p)
    
    points = np.array(points)
    R = 0.5 * n ** (1.0/3.0) * spacing * 1.2
    mask = np.linalg.norm(points, axis=1) <= R
    points = points[mask]
    
    if len(points) < n:
        R *= 1.5
        mask = np.linalg.norm(points, axis=1) <= R
        points = points[mask]
    
    if len(points) < n:
        # Pad with random points
        while len(points) < n:
            p = rng.uniform(-R, R, 3)
            points = np.vstack([points, [p]])
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points

def _hcp_fragment(n, spacing=1.0, rng=None):
    """Generate n atoms from an HCP lattice, centered."""
    if rng is None:
        rng = np.random.default_rng()
    
    a = spacing
    c = a * np.sqrt(8.0/3.0)
    points = []
    n_layers = int(np.ceil(np.sqrt(n))) + 2
    
    for layer in range(-n_layers, n_layers + 1):
        z = layer * c
        if layer % 2 == 0:
            shift = np.array([0, 0])
        else:
            shift = np.array([a/2, a*np.sqrt(3)/6])
        
        n_side = int(np.ceil(np.sqrt(n))) + 2
        for i in range(-n_side, n_side + 1):
            for j in range(-n_side, n_side + 1):
                x = i * a + j * a * 0.5 + shift[0]
                y = j * a * np.sqrt(3) / 2.0 + shift[1]
                points.append([x, y, z])
                
    points = np.array(points)
    R = 0.5 * n ** (1.0/3.0) * spacing * 1.2
    mask = np.linalg.norm(points, axis=1) <= R
    points = points[mask]
    
    if len(points) < n:
        R *= 1.5
        mask = np.linalg.norm(points, axis=1) <= R
        points = points[mask]
    
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-R, R, 3)
            points = np.vstack([points, [p]])
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points

def _bcc_fragment(n, spacing=1.0, rng=None):
    """Generate n atoms from a BCC lattice, centered."""
    if rng is None:
        rng = np.random.default_rng()
    
    # BCC: nearest neighbor distance = a*sqrt(3)/2
    a = spacing * 2.0 / np.sqrt(3)
    L = n ** (1.0/3.0) * spacing * 1.5
    n_side = int(np.ceil(L / a)) + 2
    
    points = []
    for i in range(-n_side, n_side + 1):
        for j in range(-n_side, n_side + 1):
            for k in range(-n_side, n_side + 1):
                base = np.array([i, j, k]) * a
                points.append(base)
                points.append(base + np.array([0.5, 0.5, 0.5]) * a)
    
    points = np.array(points)
    R = 0.5 * n ** (1.0/3.0) * spacing * 1.2
    mask = np.linalg.norm(points, axis=1) <= R
    points = points[mask]
    
    if len(points) < n:
        R *= 1.5
        mask = np.linalg.norm(points, axis=1) <= R
        points = points[mask]
    
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-R, R, 3)
            points = np.vstack([points, [p]])
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points

def _icosahedral_shell(n, spacing=1.0, rng=None):
    """Generate an icosahedral cluster with shells."""
    if rng is None:
        rng = np.random.default_rng()
    
    # icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
        [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0]) * spacing
    
    points = [np.zeros(3)]  # center
    # First shell: 12 atoms
    points.extend(verts.tolist())
    
    # Second shell: 30 atoms (edges of icosahedron)
    # Approximate by scaling
    for scale in [1.6, 2.2]:
        for v in verts:
            points.append(v * scale)
    
    # If we still need more, add more shells
    while len(points) < n:
        scale = 1.0 + 0.3 * len(points)
        for v in verts:
            points.append(v * scale)
            if len(points) >= n:
                break
    
    points = np.array(points[:n])
    points = points - np.mean(points, axis=0)
    return points

def _random_compact(n, spacing=1.0, rng=None):
    """Generate a random compact cluster."""
    if rng is None:
        rng = np.random.default_rng()
        
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    max_attempts = n * 100
    attempts = 0
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points.append(p)
            attempts += 1
            
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    points = np.array(points)
    points = points - np.mean(points, axis=0)
    return points

def _perturb(positions, rng, scale=0.5):
    """Perturb positions with Gaussian noise."""
    noise = rng.normal(0, scale, positions.shape)
    return positions + noise

def _polish(positions, potential, parameter, rng, max_time=5.0):
    """High-precision polish using multiple relaxations with small perturbations."""
    best_energy, best_pos = np.inf, positions.copy()
    start = time.time()
    
    for scale in [0.1, 0.05, 0.02]:
        if time.time() - start > max_time:
            break
        for _ in range(3):
            if time.time() - start > max_time:
                break
            new_pos = _perturb(best_pos, rng, scale)
            try:
                energy, pos = relax(new_pos, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos
            except:
                pass
    
    # Final relax
    try:
        energy, pos = relax(best_pos, potential, parameter)
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
    except:
        pass
    
    return best_energy, best_pos

def construct(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    spacing = 1.0
    
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate templates
    templates = []
    for gen_func in [_fcc_fragment, _hcp_fragment, _bcc_fragment, _icosahedral_shell, _random_compact]:
        try:
            t = gen_func(size, spacing, rng)
            if len(t) == size:
                templates.append(t)
        except:
            pass
    
    # Relax each template
    for t in templates:
        if _time_left() < 5.0:
            break
        try:
            energy, positions = relax(t, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            pass
    
    # Basin Hopping with adaptive scale
    max_iterations = 300
    for i in range(max_iterations):
        if _time_left() < 10.0:
            break
            
        # Adaptive perturbation scale
        scale = 0.5 * (1.0 - i / max_iterations) + 0.1
        
        # Choose base
        if rng.random() < 0.6 and best_positions is not None:
            base = best_positions
        else:
            base = templates[rng.integers(len(templates))]
            
        new_positions = _perturb(base, rng, scale)
        
        try:
            energy, positions = relax(new_positions, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            pass
    
    # Final polish
    if best_positions is not None and _time_left() > 10.0:
        try:
            best_energy, best_positions = _polish(best_positions, potential, parameter, rng, max_time=15.0)
        except:
            pass
    
    if best_positions is None:
        best_positions = templates[0]
        
    return best_positions
# EVOLVE-BLOCK-END

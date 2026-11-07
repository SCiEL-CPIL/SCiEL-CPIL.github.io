"""Rewritten program for Sutton-Chen cluster global minima using structural motifs."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fcc_positions(n, rng):
    """Generate n positions from an FCC lattice fragment."""
    a = 1.0  # lattice constant
    # FCC basis vectors
    basis = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
    ]) * a
    # Generate FCC lattice points
    positions = []
    # Determine extent needed
    L = int(np.ceil(n ** (1.0/3.0))) + 2
    for ix in range(-L, L+1):
        for iy in range(-L, L+1):
            for iz in range(-L, L+1):
                for b in basis:
                    p = np.array([ix, iy, iz]) * a + b
                    positions.append(p)
                if len(positions) >= n:
                    break
            if len(positions) >= n:
                break
        if len(positions) >= n:
            break
    positions = np.array(positions[:n])
    # Center
    positions -= positions.mean(axis=0)
    return positions


def _hcp_positions(n, rng):
    """Generate n positions from an HCP lattice fragment."""
    a = 1.0
    c = 1.633 * a  # ideal HCP c/a ratio
    # HCP basis
    basis = np.array([
        [0, 0, 0],
        [a, 0, 0],
        [a/2, a*np.sqrt(3)/2, 0],
        [a/2, a*np.sqrt(3)/2, c/2],  # second layer
        [0, 0, c/2],
    ])
    positions = []
    L = int(np.ceil(n ** (1.0/3.0))) + 2
    for ix in range(-L, L+1):
        for iy in range(-L, L+1):
            for iz in range(-L, L+1):
                for b in basis:
                    p = np.array([ix, iy, iz]) * a + b
                    positions.append(p)
                if len(positions) >= n:
                    break
            if len(positions) >= n:
                break
        if len(positions) >= n:
            break
    positions = np.array(positions[:n])
    positions -= positions.mean(axis=0)
    return positions


def _bcc_positions(n, rng):
    """Generate n positions from a BCC lattice fragment."""
    a = 1.0
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0.5],
    ]) * a
    positions = []
    L = int(np.ceil(n ** (1.0/3.0))) + 2
    for ix in range(-L, L+1):
        for iy in range(-L, L+1):
            for iz in range(-L, L+1):
                for b in basis:
                    p = np.array([ix, iy, iz]) * a + b
                    positions.append(p)
                if len(positions) >= n:
                    break
            if len(positions) >= n:
                break
        if len(positions) >= n:
            break
    positions = np.array(positions[:n])
    positions -= positions.mean(axis=0)
    return positions


def _icosahedron(n, rng):
    """Generate icosahedral shell positions."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    verts /= np.linalg.norm(verts[0])
    # For larger n, add inner shells
    positions = [np.zeros(3)]
    for v in verts:
        positions.append(v * 0.72)
    if n > 13:
        # Add second shell (cuboctahedron-like)
        for v in verts:
            for scale in [1.0, 1.2]:
                if len(positions) < n:
                    positions.append(v * scale * 0.72)
    # Add random compact points if needed
    while len(positions) < n:
        p = rng.normal(0, 0.5, 3)
        if all(np.linalg.norm(p - q) >= 0.35 for q in positions):
            positions.append(p)
    positions = np.array(positions[:n])
    positions -= positions.mean(axis=0)
    return positions


def _cuboctahedron(n, rng):
    """Generate cuboctahedral motif."""
    phi = (1 + np.sqrt(5)) / 2
    # Cuboctahedron vertices (12 vertices)
    verts = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
    ])
    verts /= np.linalg.norm(verts[0])
    positions = [np.zeros(3)]
    for v in verts:
        positions.append(v * 0.7)
    while len(positions) < n:
        p = rng.normal(0, 0.6, 3)
        if all(np.linalg.norm(p - q) >= 0.35 for q in positions):
            positions.append(p)
    positions = np.array(positions[:n])
    positions -= positions.mean(axis=0)
    return positions


def _random_compact(n, rng, spacing=0.72):
    """Generate random compact cluster."""
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    max_attempts = n * 200
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    # If we couldn't place all, add with smaller min distance
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            points.append(p)
    positions = np.array(points[:n])
    positions -= positions.mean(axis=0)
    return positions


def _perturb(positions, rng, scale=0.1):
    """Add small random perturbation."""
    return positions + rng.normal(0, scale, positions.shape)


def construct(instance):
    """Construct near-optimal Sutton-Chen cluster using structural motifs + relaxation."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin: 300s limit, leave 20s
    
    rng = np.random.default_rng(size * 1000 + hash(kind) % 10000)
    
    # Generate diverse starting configurations
    starters = []
    
    # Crystal structure fragments
    for _ in range(2):
        starters.append(_fcc_positions(size, rng))
    for _ in range(2):
        starters.append(_hcp_positions(size, rng))
    for _ in range(1):
        starters.append(_bcc_positions(size, rng))
    
    # Icosahedral and cuboctahedral motifs
    starters.append(_icosahedron(size, rng))
    starters.append(_cuboctahedron(size, rng))
    
    # Random compact
    for _ in range(3):
        starters.append(_random_compact(size, rng))
    
    # Perturbed versions of best motifs
    for base in [_fcc_positions(size, rng), _hcp_positions(size, rng)]:
        for scale in [0.05, 0.15, 0.3]:
            starters.append(_perturb(base, rng, scale))
    
    best_energy = np.inf
    best_positions = None
    
    for i, start in enumerate(starters):
        if time.time() > deadline:
            break
        
        # Validate starting positions
        if not np.all(np.isfinite(start)):
            continue
        
        # Check minimum distance
        min_dist = np.inf
        for j in range(size):
            for k in range(j+1, size):
                d = np.linalg.norm(start[j] - start[k])
                if d < min_dist:
                    min_dist = d
        if min_dist < 0.35:
            # Try to fix by relaxing
            pass
        
        try:
            energy, positions = relax(start, potential, kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    # If no valid result, try a final random start
    if best_positions is None:
        start = _random_compact(size, rng)
        try:
            energy, positions = relax(start, potential, kind)
            if np.isfinite(energy):
                best_energy = energy
                best_positions = positions
        except Exception:
            best_positions = start
    
    # Final validation
    if best_positions is None:
        best_positions = _random_compact(size, rng)
    
    # Ensure validity: check finite and min distance
    if not np.all(np.isfinite(best_positions)):
        best_positions = _random_compact(size, rng)
    
    # Check and fix min distance if needed
    for j in range(size):
        for k in range(j+1, size):
            d = np.linalg.norm(best_positions[j] - best_positions[k])
            if d < 0.35:
                # Nudge apart
                mid = (best_positions[j] + best_positions[k]) / 2
                diff = best_positions[j] - best_positions[k]
                if np.linalg.norm(diff) < 1e-10:
                    diff = np.array([0.35, 0, 0])
                diff /= np.linalg.norm(diff)
                best_positions[j] = mid + diff * 0.18
                best_positions[k] = mid - diff * 0.18
    
    return best_positions
# EVOLVE-BLOCK-END

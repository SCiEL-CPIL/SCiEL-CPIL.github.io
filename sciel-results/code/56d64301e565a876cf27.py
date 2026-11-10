"""Dzugutov cluster global minima: parameterized polytetrahedral model + optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 318.0  # safety margin before 330s kill

def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)

def _build_icosahedral_points():
    """Generate the 12 vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1],
        [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norm = np.linalg.norm(verts[0])
    return verts / norm

def _generate_polytetrahedral_seed(n, rng, scale=1.0):
    """
    Generate a polytetrahedral seed: place icosahedra sharing vertices.
    For n atoms, we build a cluster of icosahedra with shared vertices.
    """
    ico_verts = _build_icosahedral_points() * scale
    
    # Start with one icosahedron (13 atoms: center + 12 vertices)
    if n <= 13:
        if n == 1:
            return np.array([[0.0, 0.0, 0.0]])
        if n == 2:
            return np.array([[0.0, 0.0, 0.0], [scale, 0.0, 0.0]])
        # Build partial icosahedron
        points = [np.zeros(3)]  # center
        for i in range(min(n - 1, 12)):
            points.append(ico_verts[i])
        return np.array(points)
    
    # For larger n, build a cluster of icosahedra
    # Strategy: start with one icosahedron, then add more by sharing faces/edges
    # Simple approach: place centers at icosahedral positions from the first icosahedron
    
    points = [np.zeros(3)]  # center of first icosahedron
    for i in range(12):
        points.append(ico_verts[i].copy())
    
    # Now add more icosahedra by placing their centers at existing vertex positions
    # and adding their vertices (excluding the shared center)
    n_icos_needed = (n - 13) // 12 + 1  # approximately
    centers = [np.zeros(3)]
    
    # Add more icosahedra centers at vertex positions
    for i in range(1, min(n_icos_needed, 12)):
        center_pos = ico_verts[i].copy()
        centers.append(center_pos)
        # Add vertices of this icosahedron (relative to center_pos)
        for j in range(12):
            new_point = center_pos + ico_verts[j]
            # Check if this point is already close to an existing point
            is_dup = False
            for p in points:
                if np.linalg.norm(p - new_point) < 0.1:
                    is_dup = True
                    break
            if not is_dup:
                points.append(new_point)
    
    # If we still need more atoms, add random points near the surface
    while len(points) < n:
        # Pick a random existing point and add a new one nearby
        idx = rng.integers(len(points))
        base = points[idx]
        offset = rng.normal(0, 0.5, 3)
        new_point = base + offset
        is_dup = False
        for p in points:
            if np.linalg.norm(p - new_point) < 0.5:
                is_dup = True
                break
        if not is_dup:
            points.append(new_point)
    
    return np.array(points[:n])

def _enforce_min_distance(positions, min_dist=0.5, max_iters=20):
    """Enforce minimum distance between atoms."""
    n = len(positions)
    for _ in range(max_iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    diff = positions[i] - positions[j]
                    if d < 1e-10:
                        diff = np.array([1.0, 0.0, 0.0])
                    else:
                        diff /= d
                    push = (min_dist - d) / 2 + 0.01
                    positions[i] += diff * push
                    positions[j] -= diff * push
                    moved = True
        if not moved:
            break
    return positions

def _generate_compact_random(n, rng, scale=1.0):
    """Generate a compact random seed."""
    # Place atoms in a sphere of radius proportional to n^(1/3)
    r_max = scale * n ** (1/3) * 0.5
    positions = rng.uniform(-r_max, r_max, (n, 3))
    return _enforce_min_distance(positions)

def _generate_bcc_seed(n, rng, scale=1.0):
    """Generate a BCC-like seed."""
    # BCC lattice points
    a = scale * 2.0  # lattice constant
    positions = []
    # Generate lattice points in a cube
    n_side = int(np.ceil(n ** (1/3)))
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                # Corner atoms
                pos = np.array([i * a, j * a, k * a])
                positions.append(pos)
                # Body center atom
                if len(positions) < n:
                    pos_center = pos + np.array([a/2, a/2, a/2])
                    positions.append(pos_center)
                if len(positions) >= n:
                    break
            if len(positions) >= n:
                break
        if len(positions) >= n:
            break
    
    positions = np.array(positions[:n])
    return _enforce_min_distance(positions)

def _basin_hopping(positions, potential, rng, max_iters=50, step_scale=0.4, time_check=None):
    """
    Basin hopping algorithm.
    """
    best_positions = positions.copy()
    best_energy, _ = relax(positions, potential)
    current_positions = positions.copy()
    current_energy = best_energy
    
    for it in range(max_iters):
        if time_check is not None and time_check() < 20:
            break
            
        # Random perturbation
        step = step_scale * rng.normal(0, 1, current_positions.shape)
        new_positions = current_positions + step
        
        # Relax
        energy, relaxed = relax(new_positions, potential)
        
        # Metropolis acceptance
        if energy < current_energy:
            current_positions = relaxed.copy()
            current_energy = energy
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed.copy()
        else:
            delta_e = energy - current_energy
            if rng.random() < np.exp(-delta_e / 1.0):
                current_positions = relaxed.copy()
                current_energy = energy
                
    return best_positions, best_energy

def construct(instance):
    """
    Find low-energy Dzugutov clusters using polytetrahedral seeding + basin hopping.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    n = instance["n"]
    potential = instance["potential"]
    rng = np.random.default_rng(n * 7919 + 42)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate multiple diverse seeds
    seeds = []
    
    # Seed 1: Polytetrahedral
    seeds.append(_generate_polytetrahedral_seed(n, rng, scale=1.0))
    
    # Seed 2: Compact random
    seeds.append(_generate_compact_random(n, rng, scale=1.0))
    
    # Seed 3: BCC-like
    seeds.append(_generate_bcc_seed(n, rng, scale=1.0))
    
    # Seed 4: Another polytetrahedral with different scale
    seeds.append(_generate_polytetrahedral_seed(n, rng, scale=1.1))
    
    # Seed 5: Another compact random
    seeds.append(_generate_compact_random(n, rng, scale=1.1))
    
    # Relax all seeds and keep the best
    for seed in seeds:
        seed = _enforce_min_distance(seed)
        energy, positions = relax(seed, potential)
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
    
    # Basin hopping from the best seed
    time_remaining = _get_time_remaining()
    
    if best_positions is not None and time_remaining > 30:
        n_iters = min(150, int((time_remaining - 30) / 0.5))
        best_positions, best_energy = _basin_hopping(
            best_positions, potential, rng,
            max_iters=n_iters, step_scale=0.4,
            time_check=_get_time_remaining
        )
    
    # Final relaxation
    if best_positions is not None:
        final_energy, final_positions = relax(best_positions, potential)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_positions
    
    if best_positions is None:
        best_positions = np.array([[i * 1.1, 0.0, 0.0] for i in range(n)])
        
    return best_positions
# EVOLVE-BLOCK-END

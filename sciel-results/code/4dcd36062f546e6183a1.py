"""Dzugutov cluster global minima search using defect-seeded icosahedral initialization."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time

# Strict deadline: 320s wall-clock limit, use 310s as working limit
WORKING_LIMIT = 310.0
START_TIME = time.time()

def _remaining_time():
    return WORKING_LIMIT - (time.time() - START_TIME)

def _get_icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    return verts

def _normalize_and_scale(verts, target_radius):
    """Normalize vertices to unit sphere then scale to target_radius."""
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts * target_radius

def _generate_mackay_55(spacing=1.05):
    """
    Generate the 55-atom Mackay icosahedron:
      Shell 0: 1 atom at origin
      Shell 1: 12 atoms (icosahedron vertices) at radius spacing
      Shell 2: 42 atoms at radius ~1.7*spacing
    """
    ico_verts = _get_icosahedron_vertices()
    
    # Shell 1: 12 vertices of icosahedron
    shell1 = _normalize_and_scale(ico_verts, spacing)
    
    # Shell 2: 42 atoms
    # The second shell of a Mackay icosahedron consists of:
    # - 30 atoms at edge midpoints (scaled)
    # - 12 atoms at face centers (scaled)
    # 
    # For simplicity, we generate 42 points on a sphere of radius r2
    # using icosahedral symmetry. We'll use the 12 icosahedron vertices
    # and apply the icosahedral rotation group to generate the full shell.
    #
    # Actually, let's use a simpler approach: generate points at the
    # positions corresponding to the 42 vertices of the second shell.
    # The radius of shell 2 is approximately sqrt(5)*spacing/2 * 2 = sqrt(5)*spacing
    # But let's use a more standard value: r2 = spacing * sqrt(5.0) / 2 * 2
    # 
    # For a Mackay icosahedron with nearest-neighbor distance d:
    # Shell 1 radius: d
    # Shell 2 radius: d * sqrt(5) / 2 * 2 = d * sqrt(5)
    # Wait, let me think more carefully.
    #
    # In a Mackay icosahedron, the distance from center to shell 1 is d.
    # The distance from center to shell 2 is d * sqrt(5) / 2 * 2? 
    # Actually, the standard Mackay icosahedron has:
    # - 1 atom at center
    # - 12 atoms at vertices of icosahedron (radius R1)
    # - 42 atoms at second shell (radius R2)
    # where R2/R1 = sqrt(5)/2 * 2 = sqrt(5) ≈ 2.236? No, that's too large.
    #
    # Let me use a different approach: generate 42 points on a sphere
    # using a Fibonacci sphere and select those that are well-distributed.
    
    r2 = spacing * 1.7  # approximate second shell radius
    
    # Generate 42 points on sphere of radius r2 using Fibonacci sphere
    n_pts = 42
    golden_angle = np.pi * (3 - np.sqrt(5))
    shell2 = np.zeros((n_pts, 3))
    for i in range(n_pts):
        z = 1 - 2 * i / (n_pts - 1)
        r = np.sqrt(1 - z*z)
        theta = golden_angle * i
        shell2[i] = np.array([r * np.cos(theta), r * np.sin(theta), z]) * r2
    
    # Build the 55-atom cluster
    points = [np.array([0.0, 0.0, 0.0])]
    points.extend([p.copy() for p in shell1])
    points.extend([p.copy() for p in shell2])
    
    return np.array(points)

def _defect_seed(n_target, spacing=1.05, rng=None):
    """
    Generate a defect-seeded structure by starting from the 55-atom Mackay
    icosahedron and removing atoms to get n_target atoms, or adding atoms
    for n_target > 55.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Start with the 55-atom Mackay icosahedron
    mackay55 = _generate_mackay_55(spacing)
    
    if n_target <= 55:
        # Remove atoms to get n_target
        # Remove from the outer shell first (indices 13-54), then from inner shell (1-12)
        n_remove = 55 - n_target
        # Create a list of indices to remove, prioritizing outer shell
        indices_to_remove = list(range(13, 55))  # outer shell indices
        if n_remove > len(indices_to_remove):
            # Also remove from inner shell
            indices_to_remove.extend(list(range(1, 13)))
        
        # Randomly select which to remove
        rng.shuffle(indices_to_remove)
        remove_set = set(indices_to_remove[:n_remove])
        
        # Keep atoms not in remove_set
        keep_indices = [i for i in range(55) if i not in remove_set]
        return mackay55[keep_indices]
    else:
        # Add atoms for n_target > 55
        n_add = n_target - 55
        # Add atoms on a third shell
        r3 = spacing * 2.5
        n_pts = n_add
        golden_angle = np.pi * (3 - np.sqrt(5))
        shell3 = np.zeros((n_pts, 3))
        for i in range(n_pts):
            z = 1 - 2 * i / max(1, n_pts - 1)
            r = np.sqrt(1 - z*z)
            theta = golden_angle * i
            shell3[i] = np.array([r * np.cos(theta), r * np.sin(theta), z]) * r3
        
        return np.vstack([mackay55, shell3])

def _ensure_min_distance(positions, min_dist=0.5, max_iter=50):
    """Ensure all pairs are at least min_dist apart."""
    positions = positions.copy()
    n = len(positions)
    
    for _ in range(max_iter):
        if _remaining_time() < 5.0:
            break
        min_d = np.inf
        min_pair = (0, 1)
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_d:
                    min_d = d
                    min_pair = (i, j)
        
        if min_d >= min_dist:
            break
        
        i, j = min_pair
        direction = (positions[i] - positions[j]) / (min_d + 1e-10)
        push = (min_dist - min_d) / 2 + 0.01
        positions[i] += direction * push
        positions[j] -= direction * push
    
    return positions

def _basin_hopping(positions, potential, parameter, max_iterations=20, rng=None):
    """
    Basin hopping: local relaxation + random perturbation + accept/reject.
    Returns best energy and positions found.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    current_energy, current_pos = relax(positions, potential, parameter)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    amplitude = 0.3
    beta = 1.0 / 0.2
    
    for it in range(max_iterations):
        if _remaining_time() < 10.0:
            break
        
        # Perturb
        perturbed = current_pos + rng.normal(0, amplitude, positions.shape)
        
        # Relax
        new_energy, new_pos = relax(perturbed, potential, parameter)
        
        # Accept/reject (Metropolis)
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-beta * delta_e):
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        
        # Slowly increase amplitude
        amplitude *= 1.05
    
    return best_energy, best_pos

def construct(instance):
    """
    Find low-energy Dzugutov clusters using defect-seeded icosahedral initialization.
    """
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    global START_TIME
    START_TIME = time.time()
    
    rng = np.random.default_rng(n * 7 + 13)
    
    best_energy = np.inf
    best_positions = None
    
    # Use fewer restarts but with more iterations per restart
    n_starts = 4
    
    for start_idx in range(n_starts):
        if _remaining_time() < 30.0:
            break
        
        # Generate a defect-seeded starting configuration
        spacing = 1.0 + rng.uniform(-0.05, 0.05)
        positions = _defect_seed(n, spacing=spacing, rng=rng)
        
        # Add small random perturbation to break symmetry
        positions = positions + rng.normal(0, 0.05, positions.shape)
        
        # Ensure minimum distance constraint
        positions = _ensure_min_distance(positions, min_dist=0.5, max_iter=30)
        
        # Basin hopping from this start
        energy, pos = _basin_hopping(positions, potential, parameter, 
                                      max_iterations=25, rng=rng)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = pos.copy()
    
    # Final relaxation of best structure
    if best_positions is not None:
        final_energy, final_pos = relax(best_positions, potential, parameter)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_pos
    
    return best_positions
# EVOLVE-BLOCK-END

"""Dzugutov cluster global minima search using icosahedral initialization and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time
import copy

# Deadline guard: 300s total, return with 10s safety margin
TOTAL_TIME_LIMIT = 300.0
SAFETY_MARGIN = 10.0
START_TIME = time.time()

def _remaining_time():
    return TOTAL_TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)

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

def _generate_mackay_shells(n_target, spacing=1.05):
    """
    Generate atoms in Mackay icosahedral shells:
      Shell 0: 1 atom at origin
      Shell 1: 12 atoms (icosahedron vertices)
      Shell 2: 42 atoms (second icosahedral shell)
    Total: 1 + 12 + 42 = 55 atoms
    For n > 55, add a partial third shell or distribute remaining atoms.
    For n < 55, use a subset of the shells.
    """
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = _get_icosahedron_vertices()
    
    # Shell 1: 12 vertices of icosahedron
    shell1 = _normalize_and_scale(ico_verts, spacing)
    
    # Shell 2: 42 atoms - these are the midpoints of edges, face centers, etc.
    # For a Mackay icosahedron, the second shell has 42 atoms.
    # We'll generate them by taking all combinations that give icosahedral symmetry.
    # The 42 atoms in shell 2 can be generated as follows:
    # - 30 edge midpoints (scaled)
    # - 12 face centers (scaled)
    
    # Actually, let's use a simpler approach: generate points on a sphere with icosahedral symmetry
    # For shell 2, the radius is approximately 2*spacing (in Mackay icosahedron)
    
    # Generate shell 2 points using icosahedral symmetry operations
    # The icosahedral group has 60 rotations. We can generate fundamental points and apply symmetries.
    
    # Simpler approach: use known Mackay icosahedron coordinates
    # Shell 2 radius is sqrt(5)*spacing/2 for the 42-atom shell? Let me compute properly.
    
    # For a Mackay icosahedron with shell spacing d:
    # Shell 1 radius: d
    # Shell 2 radius: d * sqrt(5) / 2 * ... 
    
    # Let me use a different approach: generate points on a Fibonacci sphere and keep those
    # that are close to icosahedral symmetry positions.
    
    # Actually, the simplest robust approach: generate a dense set of candidate points on shell 2
    # and select those that maintain minimum distance constraints.
    
    # For now, let's use a simpler strategy: generate points on a sphere of radius 2*spacing
    # using icosahedral symmetry.
    
    # The icosahedral group generators (rotation matrices):
    # We'll use the 12 icosahedron vertices and generate shell 2 by placing atoms at
    # positions that are icosahedrally symmetric.
    
    # Shell 2: 42 atoms
    # These are located at:
    # - 30 positions: midpoints of the 30 edges of the icosahedron, scaled
    # - 12 positions: centers of the 20 triangular faces... wait, icosahedron has 20 faces, not 12.
    
    # Let me reconsider. A Mackay icosahedron with 55 atoms:
    # 1 + 12 + 42 = 55
    # The 42 atoms in the second shell are arranged with icosahedral symmetry.
    
    # I'll generate them by taking the icosahedron vertices and applying symmetry operations
    # to a fundamental set of points.
    
    # For simplicity and robustness, let me use a different strategy:
    # Generate all candidate positions on a sphere of radius r2, then select a subset
    # that maximizes minimum pairwise distance.
    
    r2 = spacing * np.sqrt(5.0)  # approximate second shell radius
    
    # Generate many candidate points on sphere of radius r2
    n_candidates = 200
    # Use Fibonacci sphere for even distribution
    n = n_candidates
    golden_angle = np.pi * (3 - np.sqrt(5))
    candidates = np.zeros((n, 3))
    for i in range(n):
        z = 1 - 2 * i / (n - 1)
        r = np.sqrt(1 - z*z)
        theta = golden_angle * i
        candidates[i] = np.array([r * np.cos(theta), r * np.sin(theta), z]) * r2
    
    # Select 42 points from candidates that maximize minimum distance
    # Greedy farthest-point sampling
    selected = [candidates[0]]
    while len(selected) < 42:
        best_idx = -1
        best_min_dist = -1
        for j in range(n):
            if any(np.allclose(candidates[j], s) for s in selected):
                continue
            min_d = min(np.linalg.norm(candidates[j] - s) for s in selected)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = j
        if best_idx >= 0:
            selected.append(candidates[best_idx])
        else:
            break
    
    shell2 = np.array(selected)
    
    # Build the cluster
    points = [np.array([0.0, 0.0, 0.0])]  # core
    
    if n_target >= 13:
        points.extend([p.copy() for p in shell1])
    
    if n_target > 13:
        remaining = n_target - 1 - 12
        if remaining <= 42:
            points.extend([p.copy() for p in shell2[:remaining]])
        else:
            # Need more than 55 atoms: add a third shell
            points.extend([p.copy() for p in shell2])
            still_needed = n_target - 55
            if still_needed > 0:
                # Generate a third shell
                r3 = spacing * 3.0
                n_cand3 = 300
                candidates3 = np.zeros((n_cand3, 3))
                for i in range(n_cand3):
                    z = 1 - 2 * i / (n_cand3 - 1)
                    r = np.sqrt(1 - z*z)
                    theta = golden_angle * i
                    candidates3[i] = np.array([r * np.cos(theta), r * np.sin(theta), z]) * r3
                
                selected3 = [candidates3[0]]
                while len(selected3) < still_needed:
                    best_idx = -1
                    best_min_dist = -1
                    for j in range(n_cand3):
                        if any(np.allclose(candidates3[j], s) for s in selected3):
                            continue
                        min_d = min(np.linalg.norm(candidates3[j] - s) for s in selected3)
                        if min_d > best_min_dist:
                            best_min_dist = min_d
                            best_idx = j
                    if best_idx >= 0:
                        selected3.append(candidates3[best_idx])
                    else:
                        break
                points.extend([p.copy() for p in selected3])
    
    return np.array(points[:n_target])

def _perturb(positions, amplitude, rng):
    """Randomly perturb positions."""
    perturbed = positions + rng.normal(0, amplitude, positions.shape)
    return perturbed

def _basin_hopping(positions, potential, parameter, max_iterations=50, rng=None):
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
    beta = 1.0 / 0.2  # inverse temperature
    
    for it in range(max_iterations):
        if _remaining_time() < 5.0:
            break
        
        # Perturb
        perturbed = _perturb(current_pos, amplitude, rng)
        
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
        amplitude *= 1.02
    
    return best_energy, best_pos

def _local_relax(positions, potential, parameter):
    """Relax a configuration to a local minimum."""
    energy, relaxed = relax(positions, potential, parameter)
    return energy, relaxed

def _compute_energy(positions, potential='dzugutov'):
    """Compute Dzugutov energy."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 0.5:
                return np.inf
            if r < a:
                energy += A * (r**(-m) - B) * np.exp(c / (r - a))
            elif r < b:
                energy += B * np.exp(d / (r - b))
    return energy

def construct(instance):
    """
    Find low-energy Dzugutov clusters using icosahedral initialization and basin hopping.
    """
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    global START_TIME
    START_TIME = time.time()
    
    rng = np.random.default_rng(n * 7 + 13)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate multiple icosahedral starting structures
    n_starts = 8
    
    for start_idx in range(n_starts):
        if _remaining_time() < 20.0:
            break
        
        # Generate an icosahedral starting configuration with some randomness
        if start_idx == 0:
            # Perfect Mackay icosahedron
            positions = _generate_mackay_shells(n, spacing=1.05)
        else:
            # Perturbed icosahedral structure
            base_spacing = 1.0 + rng.uniform(-0.1, 0.1)
            positions = _generate_mackay_shells(n, spacing=base_spacing)
            # Add small random perturbation
            positions = positions + rng.normal(0, 0.1, positions.shape)
        
        # Ensure minimum distance constraint
        # If any pair is too close, push them apart
        for _ in range(10):
            min_dist = np.inf
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < min_dist:
                        min_dist = d
            if min_dist >= 0.5:
                break
            # Push the closest pair apart
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < 0.5:
                        direction = (positions[i] - positions[j]) / (d + 1e-10)
                        positions[i] += direction * 0.1
                        positions[j] -= direction * 0.1
        
        # Basin hopping from this start
        energy, pos = _basin_hopping(positions, potential, parameter, 
                                      max_iterations=30, rng=rng)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = pos.copy()
        
        # Also try a few more basin hopping iterations if time permits
        if _remaining_time() > 30.0:
            energy2, pos2 = _basin_hopping(pos, potential, parameter,
                                            max_iterations=20, rng=rng)
            if energy2 < best_energy:
                best_energy = energy2
                best_positions = pos2.copy()
    
    # Final relaxation of best structure
    if best_positions is not None:
        final_energy, final_pos = relax(best_positions, potential, parameter)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_pos
    
    return best_positions
# EVOLVE-BLOCK-END

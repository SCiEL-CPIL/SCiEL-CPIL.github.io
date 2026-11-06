"""Dzugutov cluster global minima search with icosahedral seeding and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


def _get_icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    return verts


def _get_icosahedral_shell2_positions():
    """
    Generate 42 positions for the second shell of a Mackay icosahedron.
    These are the face centers and edge midpoints of the icosahedron, scaled appropriately.
    For a Mackay icosahedron, shell 2 has 42 atoms.
    """
    phi = (1 + np.sqrt(5)) / 2
    # The 20 face centers of the icosahedron
    # Each face is a triangle. The face centers are at 1/3 of the way from vertex to opposite face.
    # Actually, for Mackay icosahedron, shell 2 positions are more complex.
    # Let's use a simpler approach: generate points on a sphere with icosahedral symmetry.
    
    # Alternative: use the known structure. For 55-atom Mackay icosahedron:
    # 1 + 12 + 42 = 55
    # Shell 2 (42 atoms) consists of:
    # - 20 face centers
    # - 12 edge midpoints (but that's only 32)
    # Actually the second shell has 42 positions which are more complex.
    
    # Let me use a different approach: generate a Fibonacci sphere and project to icosahedral symmetry.
    # Or better: use the fact that shell 2 atoms are at specific positions.
    
    # For now, let's generate a dense set of points and select those with icosahedral symmetry.
    # Actually, let's just use a simpler initialization strategy.
    
    # Generate 42 points on a sphere with icosahedral-like distribution
    # Use golden spiral for even distribution
    n = 42
    points = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        z = 1 - 2 * i / (n - 1)
        r = np.sqrt(1 - z**2)
        theta = golden_angle * i
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z])
    return np.array(points)


def _generate_mackay_icosahedron(n_target, spacing=1.1):
    """
    Generate a Mackay icosahedral cluster with approximately n_target atoms.
    Shell structure:
      - Core: 1 atom at origin
      - Shell 1 (12 atoms): icosahedron vertices at radius r1
      - Shell 2 (42 atoms): icosahedral second shell at radius r2
    """
    # Core atom
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 icosahedron vertices
    ico_verts = _get_icosahedron_vertices()
    # Normalize to unit sphere
    norms = np.linalg.norm(ico_verts, axis=1)
    ico_verts_unit = ico_verts / norms[:, np.newaxis]
    
    r1 = spacing
    for v in ico_verts_unit:
        points.append(v * r1)
    
    # Shell 2: 42 atoms
    if n_target > 13:
        shell2 = _get_icosahedral_shell2_positions()
        r2 = spacing * 2.0  # Approximate radius for second shell
        n_shell2_needed = min(42, n_target - 13)
        for i in range(n_shell2_needed):
            points.append(shell2[i] * r2)
    
    # If we need more atoms, add more shell 2 positions or a third shell
    while len(points) < n_target:
        # Add atoms at random positions on a larger shell
        r3 = spacing * 2.5
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        x = r3 * np.sin(phi) * np.cos(theta)
        y = r3 * np.sin(phi) * np.sin(theta)
        z = r3 * np.cos(phi)
        points.append(np.array([x, y, z]))
    
    return np.array(points[:n_target])


def _generate_defect_icosahedron(n_target, base_n=55, spacing=1.1, rng=None):
    """
    Generate a cluster by starting from a 55-atom Mackay icosahedron
    and removing or displacing atoms to reach n_target.
    """
    if rng is None:
        rng = np.random.default_rng(n_target)
    
    # Start with 55-atom Mackay icosahedron
    base_cluster = _generate_mackay_icosahedron(55, spacing)
    
    if n_target >= 55:
        # Need to add atoms
        result = base_cluster.copy()
        n_to_add = n_target - 55
        for _ in range(n_to_add):
            # Add atom near a random existing atom with some displacement
            idx = rng.integers(0, len(result))
            new_pos = result[idx] + rng.normal(0, spacing * 0.3, 3)
            result = np.vstack([result, new_pos.reshape(1, 3)])
        return result
    
    # Need to remove atoms (n_target < 55)
    n_to_remove = 55 - n_target
    # Remove atoms from the outer shell first (indices 13-54)
    # Prioritize removing from shell 2 (outer)
    remove_indices = list(range(13, 55))
    rng.shuffle(remove_indices)
    remove_set = set(remove_indices[:n_to_remove])
    
    result = []
    for i, p in enumerate(base_cluster):
        if i not in remove_set:
            result.append(p)
    
    return np.array(result)


def _generate_random_compact(n, spacing=1.1, rng=None):
    """Generate a random compact cluster."""
    if rng is None:
        rng = np.random.default_rng(n)
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
            points.append(p)
    return np.array(points)


def _generate_bcc_like(n, spacing=1.1, rng=None):
    """Generate a BCC-like cluster."""
    if rng is None:
        rng = np.random.default_rng(n)
    
    # Generate BCC lattice points
    points = []
    # Add lattice points in a cube
    n_side = int(np.ceil(n ** (1.0/3.0))) + 1
    for i in range(-n_side, n_side+1):
        for j in range(-n_side, n_side+1):
            for k in range(-n_side, n_side+1):
                # Corner atoms
                points.append(np.array([i, j, k]) * spacing * 0.5)
                # Body center
                points.append((np.array([i, j, k]) + 0.5) * spacing * 0.5)
    
    points = np.array(points)
    # Filter to keep only points within a sphere
    norms = np.linalg.norm(points, axis=1)
    max_r = spacing * n ** (1.0/3.0) * 0.8
    mask = norms <= max_r
    points = points[mask]
    
    # If we have too many, randomly select
    if len(points) > n:
        idx = rng.choice(len(points), size=n, replace=False)
        points = points[idx]
    elif len(points) < n:
        # Add random points
        while len(points) < n:
            p = rng.uniform(-max_r, max_r, 3)
            if np.linalg.norm(p) <= max_r:
                points = np.vstack([points, p.reshape(1, 3)])
    
    return points[:n]


def _local_minimize(positions, potential, parameter, max_iter=500):
    """Perform local minimization using the relax function."""
    energy, pos = relax(positions, potential, parameter)
    return energy, pos


def _basin_hopping(positions, potential, parameter, n_trials=10, step_size=0.5, rng=None):
    """
    Simple basin hopping: perturb, minimize, accept if lower energy.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    current_energy, current_pos = _local_minimize(positions, potential, parameter)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for _ in range(n_trials):
        # Perturb
        perturbed = current_pos + rng.normal(0, step_size, current_pos.shape)
        
        # Minimize
        trial_energy, trial_pos = _local_minimize(perturbed, potential, parameter)
        
        # Accept if lower
        if trial_energy < current_energy:
            current_energy = trial_energy
            current_pos = trial_pos
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = trial_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """
    Find low-energy Dzugutov clusters using icosahedral seeding and basin hopping.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 280  # 50s safety margin
    
    # Generate diverse initial configurations
    rng = np.random.default_rng(size)
    
    candidates = []
    
    # 1. Defect-seeded icosahedral structures (main strategy)
    for attempt in range(3):
        if time.time() > deadline:
            break
        pos = _generate_defect_icosahedron(size, base_n=55, spacing=1.1, rng=rng)
        candidates.append(pos)
    
    # 2. Direct Mackay icosahedron (if size close to 55)
    if abs(size - 55) <= 5:
        pos = _generate_mackay_icosahedron(size, spacing=1.1)
        candidates.append(pos)
    
    # 3. BCC-like structures
    for attempt in range(2):
        if time.time() > deadline:
            break
        pos = _generate_bcc_like(size, spacing=1.1, rng=rng)
        candidates.append(pos)
    
    # 4. Random compact structures
    for attempt in range(2):
        if time.time() > deadline:
            break
        pos = _generate_random_compact(size, spacing=1.1, rng=rng)
        candidates.append(pos)
    
    # Relax each candidate and keep the best
    best_energy = np.inf
    best_pos = None
    
    for i, pos in enumerate(candidates):
        if time.time() > deadline:
            break
        
        try:
            energy, relaxed_pos = _local_minimize(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos.copy()
        except Exception:
            continue
    
    # If we have time, do basin hopping on the best candidate
    if best_pos is not None and time.time() < deadline - 30:
        try:
            bh_energy, bh_pos = _basin_hopping(
                best_pos, potential, parameter,
                n_trials=5, step_size=0.3, rng=rng
            )
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos
        except Exception:
            pass
    
    # If still have time, try basin hopping on other good candidates
    if best_pos is not None and time.time() < deadline - 20:
        # Try a few more perturbations
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            try:
                perturbed = best_pos + rng.normal(0, 0.2, best_pos.shape)
                energy, pos = _local_minimize(perturbed, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos
            except Exception:
                continue
    
    if best_pos is None:
        # Fallback: return a simple random configuration
        best_pos = _generate_random_compact(size, spacing=1.1, rng=rng)
    
    return best_pos
# EVOLVE-BLOCK-END

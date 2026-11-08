"""Sutton-Chen transition-metal cluster global minima - systematic growth construction."""
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
import time


def _fcc_positions(n):
    """Generate positions for a truncated FCC cluster with n atoms."""
    a = 1.0
    max_coord = int(np.ceil((3 * n / (16 * np.pi)) ** (1/3) * a)) + 3
    
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for (di, dj, dk) in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    x = (i + di) * a
                    y = (j + dj) * a
                    z = (k + dk) * a
                    points.append([x, y, z])
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    
    return points


def _hcp_positions(n):
    """Generate positions for a truncated HCP cluster with n atoms."""
    a = 1.0
    # HCP lattice vectors
    a1 = np.array([1, 0, 0])
    a2 = np.array([0.5, np.sqrt(3)/2, 0])
    a3 = np.array([0, 0, np.sqrt(2/3)])
    
    points = []
    max_coord = int(np.ceil((3 * n / (16 * np.pi)) ** (1/3))) + 3
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # A and B sublattice
                for (di, dj, dk) in [(0, 0, 0), (1/3, 2/3, 1/2)]:
                    x = (i + di) * a1[0] + (j + dj) * a2[0] + (k + dk) * a3[0]
                    y = (i + di) * a1[1] + (j + dj) * a2[1] + (k + dk) * a3[1]
                    z = (i + di) * a1[2] + (j + dj) * a2[2] + (k + dk) * a3[2]
                    points.append([x, y, z])
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    
    return points


def _mackay_positions(n):
    """Generate positions for a Mackay icosahedron-based cluster with proper shell structure."""
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (unit radius)
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    positions = [np.zeros(3)]
    
    # Shell 1: 12 atoms
    shell1_radius = 1.0
    for v in ico_verts:
        positions.append(v * shell1_radius)
    
    if n <= 13:
        return np.array(positions[:n])
    
    # Shell 2: 42 atoms
    # These are at positions corresponding to the icosahedral symmetry
    # The 42 positions can be generated as all permutations and sign changes of (1, phi, 0) normalized
    shell2_candidates = []
    for v in ico_verts:
        # Generate points by rotating around each vertex
        for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
            # Create rotation matrix around v
            v_norm = v / np.linalg.norm(v)
            K = np.array([
                [0, -v_norm[2], v_norm[1]],
                [v_norm[2], 0, -v_norm[0]],
                [-v_norm[1], v_norm[0], 0]
            ])
            K2 = K @ K
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K2
            new_v = R @ v
            shell2_candidates.append(new_v)
    
    shell2_candidates = np.array(shell2_candidates)
    shell2_candidates /= np.linalg.norm(shell2_candidates, axis=1, keepdims=True)
    
    # Remove duplicates (keep only unique directions)
    unique_shell2 = []
    for v in shell2_candidates:
        is_dup = False
        for u in unique_shell2:
            if np.dot(v, u) > 0.999:
                is_dup = True
                break
        if not is_dup:
            unique_shell2.append(v)
    
    # Shell 2 radius
    shell2_radius = np.sqrt(1 + 1/phi**2)
    
    for v in unique_shell2:
        if len(positions) >= n:
            break
        positions.append(v * shell2_radius)
    
    if len(positions) >= n:
        return np.array(positions[:n])
    
    # Shell 3: 92 atoms if needed
    if n > 55:
        shell3_radius = np.sqrt(1 + 4/phi**2)
        # Generate more points on shell 3
        n_needed = n - len(positions)
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n_needed * 5):
            z = 1 - 2 * (i + 0.5) / (n_needed * 5)
            r = np.sqrt(1 - z**2)
            theta = golden_angle * i
            v = np.array([r * np.cos(theta), r * np.sin(theta), z])
            # Check if this is close to any existing position
            min_dist = np.min(np.linalg.norm(np.array(positions) - v * shell3_radius, axis=1))
            if min_dist > 0.35:
                positions.append(v * shell3_radius)
                if len(positions) >= n:
                    break
    
    return np.array(positions[:n])


def _generate_interstitial_candidates(positions, n_candidates=50):
    """Generate candidate interstitial sites around existing positions."""
    candidates = []
    
    # For each pair of close atoms, find interstitial positions
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < 2.0:  # Only consider nearby pairs
                # Midpoint
                mid = (positions[i] + positions[j]) / 2
                candidates.append(mid)
                
                # Points perpendicular to the bond
                bond_vec = positions[j] - positions[i]
                bond_unit = bond_vec / d
                # Find perpendicular vectors
                perp1 = np.cross(bond_unit, np.array([1, 0, 0]))
                if np.linalg.norm(perp1) < 1e-6:
                    perp1 = np.cross(bond_unit, np.array([0, 1, 0]))
                perp1 /= np.linalg.norm(perp1)
                perp2 = np.cross(bond_unit, perp1)
                
                for offset in [0.3, 0.5]:
                    candidates.append(mid + perp1 * offset)
                    candidates.append(mid - perp1 * offset)
                    candidates.append(mid + perp2 * offset)
                    candidates.append(mid - perp2 * offset)
    
    # Also add points on a sphere around the cluster
    centroid = np.mean(positions, axis=0)
    max_dist = np.max(np.linalg.norm(positions - centroid, axis=1))
    for r in [max_dist * 0.8, max_dist * 1.0, max_dist * 1.2]:
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n_candidates // 3):
            z = 1 - 2 * (i + 0.5) / (n_candidates // 3)
            rr = np.sqrt(1 - z**2)
            theta = golden_angle * i
            candidates.append(centroid + np.array([rr * np.cos(theta), rr * np.sin(theta), z]) * r)
    
    candidates = np.array(candidates)
    
    # Filter: must be at least 0.35 from all existing atoms
    valid = []
    for cand in candidates:
        dists = np.linalg.norm(positions - cand, axis=1)
        if np.min(dists) >= 0.35:
            valid.append(cand)
    
    return np.array(valid) if valid else np.array([[0, 0, 0]])


def _growth_from_seed(seed_positions, target_n, kind, max_time=30):
    """Grow from a seed cluster to target_n atoms using systematic interstitial addition."""
    positions = seed_positions.copy()
    current_n = len(positions)
    
    # Relax the seed
    try:
        energy, positions = relax(positions, 'sutton_chen', kind)
    except:
        pass
    
    start_time = time.time()
    
    while current_n < target_n:
        if time.time() - start_time > max_time:
            break
        
        # Generate interstitial candidates
        candidates = _generate_interstitial_candidates(positions, n_candidates=40)
        
        if len(candidates) == 0:
            break
        
        # Evaluate candidates and pick the best
        best_energy = np.inf
        best_cand = None
        
        # Limit candidates for speed
        max_eval = min(len(candidates), 25)
        if len(candidates) > max_eval:
            indices = np.random.choice(len(candidates), max_eval, replace=False)
            candidates = candidates[indices]
        
        for cand in candidates:
            test_positions = np.vstack([positions, cand.reshape(1, 3)])
            try:
                e, _ = sutton_chen_energy_gradient(test_positions, kind)
                if e < best_energy and np.isfinite(e):
                    best_energy = e
                    best_cand = cand
            except:
                continue
        
        if best_cand is None:
            # Fallback: add at a random position
            best_cand = np.array([1.5, 0.5, 0.3])
        
        positions = np.vstack([positions, best_cand.reshape(1, 3)])
        current_n += 1
        
        # Relax every few atoms
        if current_n % 3 == 0 or current_n == target_n:
            try:
                energy, positions = relax(positions, 'sutton_chen', kind)
            except:
                pass
    
    return positions


def _multi_start_relax(instance, time_budget=45):
    """Multi-start local optimization with diverse initial configurations."""
    n = instance["n"]
    kind = instance["kind"]
    
    start_time = time.time()
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting configurations
    configs = []
    
    # 1. FCC-based
    try:
        fcc = _fcc_positions(n)
        configs.append(('fcc', fcc))
    except:
        pass
    
    # 2. HCP-based
    try:
        hcp = _hcp_positions(n)
        configs.append(('hcp', hcp))
    except:
        pass
    
    # 3. Mackay icosahedron-based
    try:
        mackay = _mackay_positions(n)
        configs.append(('mackay', mackay))
    except:
        pass
    
    # 4. Random compact clusters
    rng = np.random.default_rng(42)
    for i in range(4):
        radius = 0.72 * n ** (1.0 / 3.0) * 0.8
        points = []
        attempts = 0
        while len(points) < n and attempts < 2000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.7 for q in points):
                    points.append(p)
            attempts += 1
        if len(points) == n:
            configs.append(('random', np.array(points)))
    
    # Relax each configuration
    for name, config in configs:
        if time.time() - start_time > time_budget - 10:
            break
        try:
            energy, positions = relax(config, 'sutton_chen', kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except:
            continue
    
    # If we have time, try growth from the best configuration
    if time.time() - start_time < time_budget - 5 and best_positions is not None:
        try:
            grown = _growth_from_seed(best_positions[:min(len(best_positions), max(13, n-5))], 
                                       n, kind, max_time=10)
            if len(grown) == n:
                energy, grown = relax(grown, 'sutton_chen', kind)
                if np.isfinite(energy) and energy < best_energy:
                    best_energy = energy
                    best_positions = grown.copy()
        except:
            pass
    
    if best_positions is None:
        best_positions = _fcc_positions(n)
    
    return best_positions


def construct(instance):
    """Construct a low-energy Sutton-Chen cluster configuration."""
    n = instance["n"]
    kind = instance["kind"]
    
    # Use multi-start approach with diverse initial configurations
    positions = _multi_start_relax(instance, time_budget=45)
    
    # Final verification
    if not np.all(np.isfinite(positions)):
        positions = _fcc_positions(n)
    
    # Check minimum distance
    min_dist = np.inf
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 0.35:
        positions = _fcc_positions(n)
    
    return positions

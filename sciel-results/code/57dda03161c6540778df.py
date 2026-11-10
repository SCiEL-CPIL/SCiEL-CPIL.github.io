"""Dzugutov cluster global minima: hierarchical assembly + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = None
_DEADLINE = 320.0  # safety margin before 330s kill


def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)


def _generate_icosahedron_shell(n_shell, radius):
    """Generate vertices of an icosahedron scaled to given radius.
    Returns 12 points."""
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit length
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * radius
    return verts


def _generate_mackay_icosahedron(n, rng):
    """Generate a Mackay icosahedral cluster with approximately n atoms.
    Shells: 1 (center), 12 (icosahedron), 42 (next shell), 100, 202..."""
    shells = [1, 12, 42, 100, 202]
    # Determine which shells to use
    atoms = []
    remaining = n
    for i, shell_size in enumerate(shells):
        if remaining <= 0:
            break
        if i == 0:
            atoms.append(np.zeros(3))
            remaining -= 1
        elif i == 1:
            # 12 vertices of icosahedron at radius ~1.0
            radius = 1.0
            verts = _generate_icosahedron_shell(12, radius)
            atoms.extend(verts.tolist())
            remaining -= 12
        elif i == 2:
            # 42 atoms: 30 on edges + 12 on faces of icosahedron
            # Simplified: place at radius ~1.6 with icosahedral symmetry
            radius = 1.6
            # Use a denser icosahedral pattern
            phi = (1 + np.sqrt(5)) / 2
            # Generate points on sphere with icosahedral symmetry
            base_verts = _generate_icosahedron_shell(12, 1.0)
            # Add points at midpoints of edges and face centers
            # For simplicity, use a Fibonacci sphere scaled
            n_pts = min(42, remaining)
            # Use golden spiral for roughly uniform distribution
            golden_angle = np.pi * (3 - np.sqrt(5))
            for j in range(n_pts):
                z = 1 - 2 * (j + 0.5) / n_pts
                r = np.sqrt(1 - z*z)
                theta = golden_angle * j
                pt = np.array([r * np.cos(theta), r * np.sin(theta), z]) * radius
                atoms.append(pt)
            remaining -= n_pts
        elif i == 3:
            radius = 2.2
            n_pts = min(100, remaining)
            golden_angle = np.pi * (3 - np.sqrt(5))
            for j in range(n_pts):
                z = 1 - 2 * (j + 0.5) / n_pts
                r = np.sqrt(1 - z*z)
                theta = golden_angle * j
                pt = np.array([r * np.cos(theta), r * np.sin(theta), z]) * radius
                atoms.append(pt)
            remaining -= n_pts
        elif i == 4:
            radius = 2.8
            n_pts = min(202, remaining)
            golden_angle = np.pi * (3 - np.sqrt(5))
            for j in range(n_pts):
                z = 1 - 2 * (j + 0.5) / n_pts
                r = np.sqrt(1 - z*z)
                theta = golden_angle * j
                pt = np.array([r * np.cos(theta), r * np.sin(theta), z]) * radius
                atoms.append(pt)
            remaining -= n_pts
    
    # If we still need more atoms, add random ones on a larger shell
    if remaining > 0:
        radius = 3.0
        for _ in range(remaining):
            # Random point on sphere
            u = rng.uniform(0, 1)
            v = rng.uniform(0, 1)
            theta = 2 * np.pi * u
            phi_angle = np.arccos(2 * v - 1)
            pt = np.array([
                np.sin(phi_angle) * np.cos(theta),
                np.sin(phi_angle) * np.sin(theta),
                np.cos(phi_angle)
            ]) * radius
            atoms.append(pt)
    
    return np.array(atoms[:n])


def _generate_fcc_cluster(n, rng):
    """Generate an FCC-like cluster with n atoms."""
    atoms = [np.zeros(3)]
    # FCC lattice points
    a = 1.0  # lattice parameter
    # Generate FCC points in a cubic region
    max_coord = 5
    fcc_points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) * a
                for frac in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    pt = np.array([i + frac[0], j + frac[1], k + frac[2]]) * a
                    fcc_points.append(pt)
    fcc_points = np.array(fcc_points)
    # Sort by distance from origin
    dists = np.linalg.norm(fcc_points, axis=1)
    order = np.argsort(dists)
    selected = fcc_points[order[:n]]
    # Center
    selected = selected - np.mean(selected, axis=0)
    return selected


def _generate_random_compact(n, rng):
    """Generate a random compact cluster."""
    spacing = 1.1
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points.append(p)
        attempts += 1
    # If not enough, add more randomly
    while len(points) < n:
        p = rng.uniform(-radius * 1.2, radius * 1.2, 3)
        points.append(p)
    return np.array(points[:n])


def _basin_hopping(positions, potential, parameter, max_iter=20, rng=None):
    """Simple basin hopping: perturb, relax, keep if better."""
    if rng is None:
        rng = np.random.default_rng()
    energy, pos = relax(positions.copy(), potential, parameter)
    for _ in range(max_iter):
        # Perturb
        step = 0.3 * rng.normal(size=pos.shape)
        new_pos = pos + step
        # Check minimum distance
        from scipy.spatial.distance import pdist
        if pdist(new_pos) < 0.5:
            continue
        new_energy, new_pos = relax(new_pos, potential, parameter)
        if new_energy < energy:
            energy = new_energy
            pos = new_pos
    return energy, pos


def construct(instance):
    """Find low-energy Dzugutov cluster using diverse structural heuristics + basin hopping."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    time_remaining = _get_time_remaining()
    if time_remaining <= 0:
        # Fallback: return a simple valid structure
        return _generate_random_compact(size, np.random.default_rng(size))
    
    # Time budget allocation
    # We have up to ~320s total across all 6 instances
    # Each instance gets roughly 320/6 ≈ 53s
    # But we should be adaptive
    
    rng = np.random.default_rng(size * 1000 + 42)
    
    # Generate diverse initial structures
    candidates = []
    
    # 1. Mackay icosahedron
    try:
        ico = _generate_mackay_icosahedron(size, rng)
        if len(ico) == size:
            candidates.append(ico)
    except:
        pass
    
    # 2. FCC cluster
    try:
        fcc = _generate_fcc_cluster(size, rng)
        if len(fcc) == size:
            candidates.append(fcc)
    except:
        pass
    
    # 3-6. Random compact clusters with different seeds
    for i in range(4):
        rng_local = np.random.default_rng(size * 10000 + i * 7919 + 42)
        rand = _generate_random_compact(size, rng_local)
        candidates.append(rand)
    
    # 7. Icosahedron with random perturbations
    try:
        ico_pert = _generate_mackay_icosahedron(size, rng)
        ico_pert = ico_pert + 0.2 * rng.normal(size=ico_pert.shape)
        # Ensure minimum distance
        from scipy.spatial.distance import pdist
        if pdist(ico_pert) >= 0.5:
            candidates.append(ico_pert)
    except:
        pass
    
    # Relax all candidates in parallel (up to 4 threads)
    best_energy = np.inf
    best_positions = None
    
    def _relax_candidate(pos):
        try:
            e, p = relax(pos, potential, parameter)
            return e, p
        except:
            return np.inf, pos
    
    max_workers = min(4, len(candidates))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_relax_candidate, c) for c in candidates]
        for future in concurrent.futures.as_completed(futures):
            try:
                e, p = future.result()
                if e < best_energy:
                    best_energy = e
                    best_positions = p
            except:
                pass
    
    # If time remains, do basin hopping on the best candidate
    time_remaining = _get_time_remaining()
    if best_positions is not None and time_remaining > 5:
        # Do a few basin hopping iterations
        bh_energy, bh_pos = _basin_hopping(
            best_positions, potential, parameter,
            max_iter=min(15, int(time_remaining // 3)),
            rng=np.random.default_rng(size + 123)
        )
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_pos
    
    # Final check: ensure minimum distance
    if best_positions is not None:
        from scipy.spatial.distance import pdist
        if pdist(best_positions) < 0.5:
            # Fallback to a valid structure
            best_positions = _generate_random_compact(size, np.random.default_rng(size + 999))
            best_positions, _ = relax(best_positions, potential, parameter)
    
    return best_positions
# EVOLVE-BLOCK-END

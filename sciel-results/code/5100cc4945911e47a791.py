"""Sutton-Chen transition-metal cluster global minima search.
Uses targeted high-symmetry seeds (Mackay icosahedra, FCC patches) and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _icosahedron_positions(n):
    """Generate positions on icosahedral shells for transition metal clusters."""
    phi = (1 + np.sqrt(5)) / 2
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico_vertices /= np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    
    def fibonacci_sphere(n_points, radius=1.0):
        points = np.zeros((n_points, 3))
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n_points):
            z = 1 - 2 * i / (n_points - 1) if n_points > 1 else 0
            r = np.sqrt(1 - z**2)
            theta = golden_angle * i
            points[i] = [np.cos(theta) * r, np.sin(theta) * r, z] * radius
        return points
    
    shell_sizes = [1, 12, 42, 92, 162, 252, 364]
    shell_radii = [0.0, 1.0, 1.9, 2.7, 3.4, 4.0, 4.6]
    
    positions = []
    remaining = n
    
    for shell_idx, (size, radius) in enumerate(zip(shell_sizes, shell_radii)):
        if remaining <= 0:
            break
        take = min(remaining, size)
        if shell_idx == 0:
            positions.append(np.zeros(3))
        elif shell_idx == 1:
            ico_shell = ico_vertices[:take] * radius
            positions.extend(ico_shell.tolist())
        else:
            if take == 1:
                positions.append([0, 0, radius])
            else:
                fib_pts = fibonacci_sphere(take, radius)
                positions.extend(fib_pts.tolist())
        remaining -= take
    
    if remaining > 0:
        next_radius = shell_radii[-1] + 0.8
        fib_pts = fibonacci_sphere(remaining, next_radius)
        positions.extend(fib_pts.tolist())
    
    positions = np.array(positions[:n])
    positions -= np.mean(positions, axis=0)
    return positions


def _fcc_positions(n):
    """Generate n close-packed positions on an FCC lattice, then center."""
    a = 1.0
    max_k = int(np.ceil((2 * n) ** (1.0 / 3.0)))
    pts = []
    for i in range(-max_k, max_k + 1):
        for j in range(-max_k, max_k + 1):
            for k in range(-max_k, max_k + 1):
                for f in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + f[0], j + f[1], k + f[2]]) * a
                    pts.append(p)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    positions = pts[idx]
    positions -= np.mean(positions, axis=0)
    return positions


def _bcc_positions(n):
    """Generate n positions on a BCC lattice."""
    a = 1.0
    max_k = int(np.ceil((2 * n) ** (1.0 / 3.0)))
    pts = []
    for i in range(-max_k, max_k + 1):
        for j in range(-max_k, max_k + 1):
            for k in range(-max_k, max_k + 1):
                for f in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                    p = np.array([i + f[0], j + f[1], k + f[2]]) * a
                    pts.append(p)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    positions = pts[idx]
    positions -= np.mean(positions, axis=0)
    return positions


def _random_compact(n, rng, spacing=0.72):
    """Random compact cluster with minimum distance constraint."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _ensure_valid(positions, min_dist=0.35):
    """Check that all positions are finite and no two are closer than min_dist."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i] - positions[j]) < min_dist:
                return False
    return True


def _basin_hopping(positions, potential, kind, rng, max_iter=20, perturb_scale=0.3,
                   deadline=None):
    """Basin hopping: relax, perturb, relax. Returns best energy and positions found."""
    best_energy, best_pos = relax(positions, potential, kind)
    current_energy = best_energy
    current_pos = best_pos.copy()
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        noise = rng.normal(0, perturb_scale, current_pos.shape)
        perturbed = current_pos + noise
        
        if not _ensure_valid(perturbed):
            continue
        
        try:
            energy, new_pos = relax(perturbed, potential, kind)
        except Exception:
            continue
        
        if np.isfinite(energy):
            if energy < current_energy:
                current_energy = energy
                current_pos = new_pos.copy()
            elif rng.random() < np.exp(-(energy - current_energy) / (0.1 + 0.01 * i)):
                current_energy = energy
                current_pos = new_pos.copy()
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
    
    return best_energy, best_pos


def _construct_mackay_icosahedron(n):
    """
    Construct a Mackay icosahedron structure for specific n values.
    n=13: 1+12
    n=55: 1+12+42
    n=147: 1+12+42+92
    """
    if n == 13:
        return _icosahedron_positions(13)
    elif n == 55:
        return _icosahedron_positions(55)
    elif n == 147:
        return _icosahedron_positions(147)
    else:
        # Fallback to generic icosahedral packing
        return _icosahedron_positions(n)


def _construct_fcc_patch(n):
    """
    Construct a compact FCC-like patch.
    For n=38, 59, 75, 79, 80, we use FCC lattice clipping.
    """
    return _fcc_positions(n)


def construct(instance):
    """
    Targeted search using high-symmetry seeds.
    """
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    start_time = time.time()
    deadline = start_time + 315.0
    
    best_energy = np.inf
    best_positions = None
    
    seeds = []
    
    # 1. High-priority: Specific known global minima structures
    # n=55 is a known Mackay icosahedron global minimum for many potentials.
    if size == 55:
        try:
            pos = _construct_mackay_icosahedron(55)
            if _ensure_valid(pos):
                seeds.append(pos)
        except Exception:
            pass
            
    # n=13 (not in list but good practice)
    if size == 13:
        try:
            pos = _construct_mackay_icosahedron(13)
            if _ensure_valid(pos):
                seeds.append(pos)
        except Exception:
            pass
            
    # 2. High-priority: FCC patches for larger clusters
    # Doye & Wales often find FCC-like or BCC-like structures for larger n.
    # We prioritize FCC for 38, 59, 75, 79, 80.
    try:
        pos_fcc = _construct_fcc_patch(size)
        if _ensure_valid(pos_fcc):
            seeds.append(pos_fcc)
    except Exception:
        pass
        
    # 3. Medium-priority: Icosahedral packing (general)
    try:
        pos_ico = _icosahedron_positions(size)
        if _ensure_valid(pos_ico):
            seeds.append(pos_ico)
    except Exception:
        pass
        
    # 4. Medium-priority: BCC packing
    try:
        pos_bcc = _bcc_positions(size)
        if _ensure_valid(pos_bcc):
            seeds.append(pos_bcc)
    except Exception:
        pass
        
    # 5. Low-priority: Random compact
    for i in range(3):
        rng_seed = np.random.default_rng(1000 + i * 777)
        pos_rand = _random_compact(size, rng_seed)
        if _ensure_valid(pos_rand):
            seeds.append(pos_rand)
            
    # Run basin hopping on each seed
    # We allocate more iterations to the first few seeds (assumed better)
    for idx, seed_pos in enumerate(seeds):
        if time.time() > deadline:
            break
        
        remaining_time = deadline - time.time()
        n_seeds_left = len(seeds) - idx
        time_per_seed = remaining_time / max(n_seeds_left, 1)
        
        # Heuristic: give more time to earlier seeds
        if idx == 0:
            max_iter = 40
            perturb_scale = 0.2
        elif idx == 1:
            max_iter = 30
            perturb_scale = 0.25
        elif idx == 2:
            max_iter = 25
            perturb_scale = 0.3
        else:
            max_iter = 15
            perturb_scale = 0.35
            
        # Cap iterations based on time
        max_iter = min(max_iter, max(3, int(time_per_seed / 1.5)))
        
        rng = np.random.default_rng(42 + size + idx * 1000)
        
        try:
            energy, positions = _basin_hopping(
                seed_pos, potential, kind, rng,
                max_iter=max_iter,
                perturb_scale=perturb_scale,
                deadline=deadline
            )
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
    
    # Final polish
    if best_positions is not None and time.time() < deadline - 10.0:
        rng = np.random.default_rng(999 + size)
        try:
            energy, positions = _basin_hopping(
                best_positions, potential, kind, rng,
                max_iter=15,
                perturb_scale=0.1,
                deadline=deadline
            )
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
    
    # Fallback
    if best_positions is None:
        best_positions = _fcc_positions(size)
    
    if not _ensure_valid(best_positions):
        best_positions = _fcc_positions(size)
    
    return best_positions
# EVOLVE-BLOCK-END

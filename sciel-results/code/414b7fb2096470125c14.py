"""Dzugutov cluster global minima: icosahedral subunit graph + GA + hinge relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_TOTAL_BUDGET = 300.0
_PER_INSTANCE_BUDGET = _TOTAL_BUDGET / 6.0  # ~50s each
_THREAD_LOCK = threading.Lock()


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining():
    return max(1.0, _TOTAL_BUDGET - _elapsed())


def _dzugutov_energy(pos):
    """Compute Dzugutov energy vectorized."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    E = 0.0
    mask_a = dist < a
    mask_b = (dist < b) & ~mask_a
    
    if np.any(mask_a):
        r = dist[mask_a]
        E += np.sum(A * (r**(-m) - B) * np.exp(c / (r - a)))
    if np.any(mask_b):
        r = dist[mask_b]
        E += np.sum(B * np.exp(d / (r - b)))
    return E


def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron centered at origin, unit radius."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    return verts


def _dodecahedron_vertices():
    """Return 20 vertices of a regular dodecahedron centered at origin, unit radius."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    verts = []
    # Cube vertices
    for sx in [1, -1]:
        for sy in [1, -1]:
            for sz in [1, -1]:
                verts.append([sx, sy, sz])
    # Rectangle vertices (3 orientations)
    for sx in [1, -1]:
        for sy in [1, -1]:
            verts.append([0, sx*inv_phi, sy*phi])
            verts.append([sx*inv_phi, sy*phi, 0])
            verts.append([sy*phi, 0, sx*inv_phi])
    verts = np.array(verts, dtype=float)
    verts /= np.linalg.norm(verts[0])
    return verts


def _cuboctahedron_vertices():
    """Return 12 vertices of a cuboctahedron centered at origin, unit radius."""
    verts = []
    for sx in [1, -1]:
        for sy in [1, -1]:
            verts.append([sx, sy, 0])
            verts.append([sx, 0, sy])
            verts.append([0, sx, sy])
    verts = np.array(verts, dtype=float)
    verts /= np.linalg.norm(verts[0])
    return verts


def _make_polyhedral_cluster(n, poly_type, rng, scale=1.1):
    """Build a cluster from concentric polyhedral shells."""
    if poly_type == 'icosahedron':
        shell_sizes = [1, 12, 30, 55, 91]  # Mackay icosahedral shells
        shell_polys = [None, 'icosahedron', 'icosahedron_edge', 'icosahedron_face', 'icosahedron']
    elif poly_type == 'dodecahedron':
        shell_sizes = [1, 20, 60]
        shell_polys = [None, 'dodecahedron', 'dodecahedron_edge']
    else:
        shell_sizes = [1, 12, 30]
        shell_polys = [None, 'cuboctahedron', 'cuboctahedron_edge']
    
    points = [np.zeros(3)]
    current_size = 1
    shell_idx = 0
    
    while len(points) < n and shell_idx < len(shell_sizes) - 1:
        shell_idx += 1
        target_size = shell_sizes[shell_idx]
        num_to_add = min(target_size - current_size, n - len(points))
        
        if shell_polys[shell_idx] == 'icosahedron':
            verts = _icosahedron_vertices()
            radius = scale * (1 + 0.55 * (shell_idx - 1))
            for v in verts:
                if len(points) >= n:
                    break
                points.append(radius * v)
        elif shell_polys[shell_idx] == 'icosahedron_edge':
            # Approximate edge-midpoint positions: use icosahedron vertices at slightly larger radius
            verts = _icosahedron_vertices()
            radius = scale * (1 + 0.55 * (shell_idx - 1)) * 1.15
            # Add rotated versions for more positions
            ico = _icosahedron_vertices()
            for i in range(12):
                if len(points) >= n:
                    break
                v = ico[i]
                # Rotate by golden angle around z
                angle = i * np.pi * (3 - np.sqrt(5))
                c, s = np.cos(angle), np.sin(angle)
                v_rot = np.array([v[0]*c - v[1]*s, v[0]*s + v[1]*c, v[2]])
                v_rot /= np.linalg.norm(v_rot)
                points.append(radius * v_rot)
        elif shell_polys[shell_idx] == 'dodecahedron':
            verts = _dodecahedron_vertices()
            radius = scale * (1 + 0.55 * (shell_idx - 1))
            for v in verts:
                if len(points) >= n:
                    break
                points.append(radius * v)
        elif shell_polys[shell_idx] == 'dodecahedron_edge':
            verts = _dodecahedron_vertices()
            radius = scale * (1 + 0.55 * (shell_idx - 1)) * 1.1
            for i in range(20):
                if len(points) >= n:
                    break
                v = verts[i % 20]
                angle = i * 0.3
                c, s = np.cos(angle), np.sin(angle)
                v_rot = np.array([v[0]*c - v[1]*s, v[0]*s + v[1]*c, v[2]])
                v_rot /= np.linalg.norm(v_rot)
                points.append(radius * v_rot)
        elif shell_polys[shell_idx] == 'cuboctahedron':
            verts = _cuboctahedron_vertices()
            radius = scale * (1 + 0.55 * (shell_idx - 1))
            for v in verts:
                if len(points) >= n:
                    break
                points.append(radius * v)
        elif shell_polys[shell_idx] == 'cuboctahedron_edge':
            verts = _cuboctahedron_vertices()
            radius = scale * (1 + 0.55 * (shell_idx - 1)) * 1.1
            for i in range(30):
                if len(points) >= n:
                    break
                v = verts[i % 12]
                angle = i * 0.2
                c, s = np.cos(angle), np.sin(angle)
                v_rot = np.array([v[0]*c - v[1]*s, v[0]*s + v[1]*c, v[2]])
                v_rot /= np.linalg.norm(v_rot)
                points.append(radius * v_rot)
        else:
            # Fallback: random compact
            r_max = scale * (len(points) ** (1/3)) * 0.8
            while len(points) < n:
                p = rng.uniform(-r_max, r_max, 3)
                if np.linalg.norm(p) <= r_max:
                    valid = True
                    for q in points:
                        if np.linalg.norm(p - q) < 0.55 * scale:
                            valid = False
                            break
                    if valid:
                        points.append(p)
                    else:
                        base = points[rng.integers(len(points))]
                        p = base + rng.normal(0, 0.3, 3)
                        points.append(p)
                else:
                    continue
        
        current_size = len(points)
    
    # Fill remaining
    while len(points) < n:
        r_max = scale * (n ** (1/3)) * 0.8
        p = rng.uniform(-r_max, r_max, 3)
        if np.linalg.norm(p) <= r_max:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.55 * scale:
                    valid = False
                    break
            if valid:
                points.append(p)
            else:
                base = points[rng.integers(len(points))]
                p = base + rng.normal(0, 0.3, 3)
                points.append(p)
        else:
            base = points[rng.integers(len(points))]
            p = base + rng.normal(0, 0.3, 3)
            points.append(p)
    
    return np.array(points[:n])


def _make_bipolyhedral_seed(n, rng, poly_type='icosahedron'):
    """Create a seed from two interpenetrating polyhedral subunits with a hinge angle."""
    half = n // 2
    other = n - half
    
    # First subunit centered at origin
    sub1 = _make_polyhedral_cluster(half, poly_type, rng, scale=1.0)
    
    # Second subunit: offset and rotated
    offset = 1.1 * np.array([1.0, 0.0, 0.0])
    angle = rng.uniform(0, 2*np.pi)
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])
    sub2 = _make_polyhedral_cluster(other, poly_type, rng, scale=0.95)
    sub2 = sub2 @ R.T + offset
    
    combined = np.vstack([sub1, sub2])
    
    # Apply small random perturbation
    combined += rng.normal(0, 0.05, combined.shape)
    combined -= combined.mean(axis=0)
    
    return combined


def _make_triple_polyhedral_seed(n, rng):
    """Create a seed from three interpenetrating icosahedral subunits."""
    third = n // 3
    remainders = [n - 2*third, third, third]
    
    subunits = []
    for i, size in enumerate(remainders):
        sub = _make_polyhedral_cluster(size, 'icosahedron', rng, scale=1.0)
        # Offset each subunit
        angle = i * 2 * np.pi / 3
        offset = 0.9 * np.array([np.cos(angle), np.sin(angle), 0])
        sub += offset
        subunits.append(sub)
    
    combined = np.vstack(subunits)
    combined += rng.normal(0, 0.05, combined.shape)
    combined -= combined.mean(axis=0)
    
    return combined


def _make_random_compact_seed(n, rng, spacing=1.1):
    """Random compact seed with minimum distance constraint."""
    radius = spacing * n ** (1.0/3.0) * 0.85
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 300:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.55 * spacing:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        if points:
            base = points[rng.integers(len(points))]
            p = base + rng.normal(0, 0.4, 3)
        else:
            p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _relax_and_kick(pos, potential, parameter, rng, max_kicks=4, kick_scale=0.4):
    """Relax, then do kick-and-relax cycles to escape local minima."""
    best_energy = np.inf
    best_pos = pos.copy()
    
    for kick_iter in range(max_kicks + 1):
        try:
            energy, relaxed = relax(pos, potential, parameter)
        except Exception:
            break
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
        
        if kick_iter == max_kicks:
            break
        
        # Adaptive kick scale: larger when stuck
        scale = kick_scale * (1.0 + 0.3 * kick_iter)
        kick = rng.normal(0, scale, relaxed.shape)
        pos = relaxed + kick
    
    return best_energy, best_pos


def _basin_hopping(pos, energy, potential, parameter, rng, max_iters=20, kick_scale=0.5):
    """Basin hopping: kick, relax, accept if lower or Metropolis."""
    current_pos = pos.copy()
    current_energy = energy
    best_energy = energy
    best_pos = pos.copy()
    
    for bh_iter in range(max_iters):
        if _remaining() < 3.0:
            break
        
        kick = rng.normal(0, kick_scale, current_pos.shape)
        kicked = current_pos + kick
        
        try:
            e, relaxed = relax(kicked, potential, parameter)
        except Exception:
            continue
        
        # Metropolis acceptance
        if e < current_energy:
            current_energy = e
            current_pos = relaxed.copy()
        else:
            delta = e - current_energy
            # Accept with probability exp(-delta / T), T ~ 0.5
            if rng.random() < np.exp(-delta / 0.5):
                current_energy = e
                current_pos = relaxed.copy()
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Dzugutov cluster using polyhedral subunit seeds + GA-inspired search."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(n * 1000 + 42)
    
    # Generate diverse seeds from different polyhedral families
    seeds = []
    
    # 1. Standard icosahedral Mackay cluster
    seeds.append(_make_polyhedral_cluster(n, 'icosahedron', rng, scale=1.1))
    
    # 2. Slightly compressed icosahedral
    seeds.append(_make_polyhedral_cluster(n, 'icosahedron', rng, scale=1.02))
    
    # 3. Slightly expanded icosahedral
    seeds.append(_make_polyhedral_cluster(n, 'icosahedron', rng, scale=1.18))
    
    # 4. Dodecahedral cluster
    seeds.append(_make_polyhedral_cluster(n, 'dodecahedron', rng, scale=1.1))
    
    # 5. Cuboctahedral cluster
    seeds.append(_make_polyhedral_cluster(n, 'cuboctahedron', rng, scale=1.1))
    
    # 6. Bipolyhedral (two icosahedral subunits)
    seeds.append(_make_bipolyhedral_seed(n, rng, 'icosahedron'))
    
    # 7. Bipolyhedral (two dodecahedral subunits)
    seeds.append(_make_bipolyhedral_seed(n, rng, 'dodecahedron'))
    
    # 8. Triple polyhedral (three icosahedral subunits)
    seeds.append(_make_triple_polyhedral_seed(n, rng))
    
    # 9. Random compact
    seeds.append(_make_random_compact_seed(n, rng))
    
    # 10. Another random compact
    rng2 = np.random.default_rng(n * 777 + 13)
    seeds.append(_make_random_compact_seed(n, rng2))
    
    # 11. Deformed icosahedral
    ico_seed = _make_polyhedral_cluster(n, 'icosahedron', rng, scale=1.1)
    ico_seed += rng.normal(0, 0.12, ico_seed.shape)
    ico_seed -= ico_seed.mean(axis=0)
    seeds.append(ico_seed)
    
    # 12. Deformed dodecahedral
    dod_seed = _make_polyhedral_cluster(n, 'dodecahedron', rng, scale=1.1)
    dod_seed += rng.normal(0, 0.15, dod_seed.shape)
    dod_seed -= dod_seed.mean(axis=0)
    seeds.append(dod_seed)
    
    # Check time budget
    elapsed = _elapsed()
    remaining = _TOTAL_BUDGET - elapsed
    
    # Determine how many workers we can afford
    n_workers = min(4, len(seeds))
    
    best_energy = np.inf
    best_pos = seeds[0].copy()
    
    # Parallel relaxation with kick-and-relax
    def _process_seed(seed_idx):
        nonlocal best_energy, best_pos
        seed = seeds[seed_idx].copy()
        
        local_best_energy = np.inf
        local_best_pos = seed.copy()
        
        # Multiple kick-and-relax rounds with different kick scales
        kick_scales = [0.3, 0.5, 0.8, 1.2]
        for ks in kick_scales:
            if _remaining() < 5.0:
                break
            try:
                e, p = _relax_and_kick(seed, potential, parameter,
                                       np.random.default_rng(seed_idx * 1000 + int(ks*100)),
                                       max_kicks=3, kick_scale=ks)
            except Exception:
                continue
            
            if e < local_best_energy:
                local_best_energy = e
                local_best_pos = p
        
        with _THREAD_LOCK:
            if local_best_energy < best_energy:
                best_energy = local_best_energy
                best_pos = local_best_pos.copy()
        
        return local_best_energy, local_best_pos
    
    # Use ThreadPoolExecutor for parallelism
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i in range(n_workers):
            if _remaining() < 5.0:
                break
            futures.append(executor.submit(_process_seed, i))
        
        for fut in as_completed(futures):
            try:
                fut.result(timeout=40)
            except Exception:
                pass
    
    # If we have time, do basin hopping on the best result
    if _remaining() > 10 and best_energy < np.inf:
        rng_bh = np.random.default_rng(n * 31337)
        current_pos = best_pos.copy()
        current_energy = best_energy
        
        # Basin hopping with adaptive kick scale
        for bh_iter in range(25):
            if _remaining() < 5.0:
                break
            
            # Adaptive kick: larger when not improving
            kick_scale = 0.4 + 0.1 * min(bh_iter, 10)
            kick = rng_bh.normal(0, kick_scale, current_pos.shape)
            kicked = current_pos + kick
            
            try:
                e, relaxed = relax(kicked, potential, parameter)
            except Exception:
                continue
            
            # Metropolis acceptance
            if e < current_energy:
                current_energy = e
                current_pos = relaxed.copy()
                with _THREAD_LOCK:
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_pos = current_pos.copy()
            else:
                delta = e - current_energy
                if rng_bh.random() < np.exp(-delta / 0.5):
                    current_energy = e
                    current_pos = relaxed.copy()
    
    # If we still have time, try a few more diverse seeds
    if _remaining() > 15:
        extra_seeds = []
        # More deformed variants
        for deform in [0.2, 0.3, 0.4]:
            if _remaining() < 5.0:
                break
            ico_seed = _make_polyhedral_cluster(n, 'icosahedron', rng, scale=1.1)
            ico_seed += rng.normal(0, deform, ico_seed.shape)
            ico_seed -= ico_seed.mean(axis=0)
            extra_seeds.append(ico_seed)
        
        # Random compact with different radii
        for radius_scale in [0.8, 1.0, 1.2]:
            if _remaining() < 5.0:
                break
            extra_seeds.append(_make_random_compact_seed(n, rng, spacing=1.1 * radius_scale))
        
        for seed in extra_seeds:
            if _remaining() < 5.0:
                break
            try:
                e, p = _relax_and_kick(seed, potential, parameter, rng, max_kicks=2, kick_scale=0.5)
            except Exception:
                continue
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
    
    # Final validation: ensure minimum distance constraint
    pos = best_pos.copy()
    for _ in range(100):
        min_dist = np.inf
        min_pair = (0, 1)
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    min_dist = d
                    min_pair = (i, j)
        if min_dist >= 0.5:
            break
        i, j = min_pair
        direction = pos[i] - pos[j]
        if np.linalg.norm(direction) < 1e-10:
            direction = rng.normal(0, 1, 3)
        direction /= np.linalg.norm(direction)
        pos[i] += 0.01 * direction
        pos[j] -= 0.01 * direction
    
    return pos
# EVOLVE-BLOCK-END

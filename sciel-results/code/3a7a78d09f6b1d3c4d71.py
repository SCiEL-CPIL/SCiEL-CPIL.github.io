"""Sutton-Chen transition-metal cluster global minima: model-and-optimize approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Deadline guard
_START_TIME = None
_TIME_LIMIT = 305.0  # Leave 25s safety margin before 330s kill

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _make_fcc(n, spacing=0.72):
    """Generate n atoms in an FCC-like arrangement."""
    a = spacing
    r_max = spacing * n ** (1.0/3.0) * 0.8
    max_idx = int(r_max / (a * 0.5)) + 1
    candidates = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * a * 0.5
                    if np.linalg.norm(p) <= r_max:
                        candidates.append(p)
                else:
                    p = (np.array([i, j, k]) + np.array([1, 1, 1])) * a * 0.25
                    if np.linalg.norm(p) <= r_max:
                        candidates.append(p)
    if len(candidates) == 0:
        return None
    candidates = np.array(candidates)
    d2 = np.sum(candidates**2, axis=1)
    order = np.argsort(d2)
    selected = candidates[order[:n]]
    return selected

def _make_bcc(n, spacing=0.72):
    """Generate n atoms in a BCC-like arrangement."""
    a = spacing
    r_max = spacing * n ** (1.0/3.0) * 0.8
    max_idx = int(r_max / (a * 0.5)) + 1
    candidates = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p_corner = np.array([i, j, k]) * a
                if np.linalg.norm(p_corner) <= r_max:
                    candidates.append(p_corner)
                p_body = (np.array([i, j, k]) + np.array([0.5, 0.5, 0.5])) * a
                if np.linalg.norm(p_body) <= r_max:
                    candidates.append(p_body)
    if len(candidates) == 0:
        return None
    candidates = np.array(candidates)
    d2 = np.sum(candidates**2, axis=1)
    order = np.argsort(d2)
    selected = candidates[order[:n]]
    return selected

def _make_icosahedral_shells(n, spacing=0.72):
    """Generate n atoms in proper Mackay icosahedral shells."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
        [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    points = [np.zeros(3)]
    r1 = spacing * 0.6
    points.extend(verts * r1)
    
    # Find edges
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.linalg.norm(verts[i] - verts[j]) < 1.0:
                edges.append((i, j))
    
    r_edge = r1 * 1.15
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid /= np.linalg.norm(mid)
        points.append(mid * r_edge)
    
    if n <= len(points):
        return np.array(points[:n])
    
    # Add more shells using Fibonacci sphere
    remaining = n - len(points)
    r3 = spacing * 1.2
    golden = (1 + np.sqrt(5)) / 2
    for i in range(remaining):
        y = 1 - (i + 0.5) / remaining * 2
        r = np.sqrt(1 - y*y)
        theta = golden * i * 2 * np.pi
        p = np.array([r * np.cos(theta), y, r * np.sin(theta)]) * r3
        points.append(p)
        
    return np.array(points[:n])

def _make_spherical_shells(n, spacing=0.72, seed=None):
    """Generate n atoms on concentric spherical shells."""
    rng = np.random.default_rng(seed)
    points = [np.zeros(3)]
    remaining = n - 1
    
    if n == 1:
        return np.array(points)
        
    r1 = spacing * 0.7
    r2 = spacing * 1.3
    r3 = spacing * 1.9
    
    weights = [r1**2, r2**2, r3**2]
    total_weight = sum(weights)
    counts = [int(remaining * w / total_weight) for w in weights]
    diff = remaining - sum(counts)
    counts[-1] += diff
    
    all_points = points
    
    for r, count in zip([r1, r2, r3], counts):
        if count <= 0:
            continue
        pts = rng.normal(0, 1, (count, 3))
        norms = np.linalg.norm(pts, axis=1)
        pts /= norms[:, np.newaxis]
        pts *= r
        all_points.extend(pts)
        
    return np.array(all_points[:n])

def _make_random_compact(n, spacing=0.72, seed=None, min_dist=0.5):
    """Random compact cluster with minimum distance constraint (vectorized)."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    max_attempts = 20000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0:
                points.append(p)
            else:
                arr = np.array(points)
                dists = np.linalg.norm(arr - p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(p)
        attempts += 1
    if len(points) < n:
        points = []
        attempts = 0
        while len(points) < n and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if len(points) == 0:
                    points.append(p)
                else:
                    arr = np.array(points)
                    dists = np.linalg.norm(arr - p, axis=1)
                    if np.min(dists) >= 0.36:
                        points.append(p)
            attempts += 1
        if len(points) < n:
            while len(points) < n:
                p = rng.uniform(-radius, radius, 3)
                points.append(p)
    return np.array(points)

def _make_deformed(n, base_positions, spacing=0.72, seed=None):
    """Create a deformed version of base positions."""
    rng = np.random.default_rng(seed)
    positions = base_positions.copy()
    noise = rng.normal(0, 0.05 * spacing, positions.shape)
    positions += noise
    n_pts = len(positions)
    for _ in range(3):
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        mask = dists < 0.36
        if not np.any(mask):
            break
        for i in range(n_pts):
            close = np.where(mask[i])[0]
            for j in close:
                if j <= i:
                    continue
                d = dists[i, j]
                if d > 0:
                    direction = (positions[i] - positions[j]) / d
                    push = (0.36 - d) / 2 + 0.01
                    positions[i] += direction * push
                    positions[j] -= direction * push
    return positions

def _make_hcp(n, spacing=0.72):
    """Generate n atoms in an HCP-like arrangement."""
    a = spacing
    c = spacing * 1.633
    r_max = spacing * n ** (1.0/3.0) * 0.8
    max_idx = int(r_max / (a * 0.5)) + 1
    candidates = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # AB stacking
                if k % 2 == 0:
                    p = np.array([i, j, k]) * np.array([a, a * np.sqrt(3)/2, c/2])
                    p[0] += (j % 2) * a / 2
                    if np.linalg.norm(p) <= r_max:
                        candidates.append(p)
                else:
                    p = np.array([i, j, k]) * np.array([a, a * np.sqrt(3)/2, c/2])
                    p[0] += (j % 2) * a / 2 + a / 2
                    p[1] += a * np.sqrt(3) / 6
                    if np.linalg.norm(p) <= r_max:
                        candidates.append(p)
    if len(candidates) == 0:
        return None
    candidates = np.array(candidates)
    d2 = np.sum(candidates**2, axis=1)
    order = np.argsort(d2)
    selected = candidates[order[:n]]
    return selected

def construct(instance):
    """
    Construct a low-energy Sutton-Chen cluster using model-and-optimize approach.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    potential = instance.get("potential", "sutton_chen")
    
    spacing = 0.72
    
    best_energy = np.inf
    best_positions = None
    second_best_energy = np.inf
    second_best_positions = None
    
    def _try_relax(positions, label=""):
        nonlocal best_energy, best_positions, second_best_energy, second_best_positions
        if _time_left() < 2.0:
            return
        try:
            energy, relaxed = relax(positions, potential, kind)
            if np.isfinite(energy):
                dists = cdist(relaxed, relaxed)
                np.fill_diagonal(dists, np.inf)
                min_d = np.min(dists)
                if min_d >= 0.35:
                    if energy < best_energy:
                        second_best_energy = best_energy
                        second_best_positions = best_positions
                        best_energy = energy
                        best_positions = relaxed.copy()
                    elif energy < second_best_energy:
                        second_best_energy = energy
                        second_best_positions = relaxed.copy()
        except Exception:
            pass
    
    start_structures = []
    
    # 1. FCC-like
    try:
        fcc = _make_fcc(n, spacing)
        if fcc is not None and len(fcc) == n:
            start_structures.append(fcc)
    except Exception:
        pass
    
    # 2. BCC-like
    try:
        bcc = _make_bcc(n, spacing)
        if bcc is not None and len(bcc) == n:
            start_structures.append(bcc)
    except Exception:
        pass
    
    # 3. HCP-like
    try:
        hcp = _make_hcp(n, spacing)
        if hcp is not None and len(hcp) == n:
            start_structures.append(hcp)
    except Exception:
        pass
    
    # 4. Icosahedral
    try:
        ico = _make_icosahedral_shells(n, spacing)
        if ico is not None and len(ico) == n:
            start_structures.append(ico)
    except Exception:
        pass
        
    # 5. Spherical Shells
    for seed in [n+5, n+6]:
        try:
            sh = _make_spherical_shells(n, spacing, seed=seed)
            if sh is not None and len(sh) == n:
                start_structures.append(sh)
        except Exception:
            pass
    
    # 6-9. Random compact with different seeds
    for seed in [n, n+1, n+2, n+3]:
        try:
            rnd = _make_random_compact(n, spacing, seed=seed, min_dist=0.5)
            if len(rnd) == n:
                start_structures.append(rnd)
        except Exception:
            pass
    
    # 10-11. Random with tighter packing
    for seed in [n+10, n+11]:
        try:
            rnd = _make_random_compact(n, spacing, seed=seed, min_dist=0.38)
            if len(rnd) == n:
                start_structures.append(rnd)
        except Exception:
            pass
    
    # 12-13. Deformed FCC variants
    if _make_fcc(n, spacing) is not None:
        for seed in [n+20, n+21]:
            try:
                fcc_base = _make_fcc(n, spacing)
                deformed = _make_deformed(n, fcc_base, spacing, seed=seed)
                start_structures.append(deformed)
            except Exception:
                pass
    
    # Relax all starting structures
    for i, structure in enumerate(start_structures):
        if _time_left() < 3.0:
            break
        _try_relax(structure, f"struct_{i}")
    
    # Basin hopping
    if best_positions is not None and _time_left() > 10.0:
        for k in range(10):
            if _time_left() < 4.0:
                break
            seed = n + 100 + k
            scale = 0.02 + 0.005 * k
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, scale * spacing, best_positions.shape)
            perturbed = best_positions + noise
            _try_relax(perturbed, f"bh_small_{k}")
        
        if _time_left() > 8.0:
            for k in range(8):
                if _time_left() < 4.0:
                    break
                seed = n + 200 + k
                scale = 0.08 + 0.01 * k
                rng = np.random.default_rng(seed)
                noise = rng.normal(0, scale * spacing, best_positions.shape)
                perturbed = best_positions + noise
                _try_relax(perturbed, f"bh_med_{k}")
                
                if second_best_positions is not None and _time_left() > 5.0:
                    noise2 = rng.normal(0, scale * spacing, second_best_positions.shape)
                    perturbed2 = second_best_positions + noise2
                    _try_relax(perturbed2, f"bh_med2_{k}")
        
        if _time_left() > 10.0:
            for k in range(6):
                if _time_left() < 4.0:
                    break
                seed = n + 300 + k
                scale = 0.2 + 0.03 * k
                rng = np.random.default_rng(seed)
                noise = rng.normal(0, scale * spacing, best_positions.shape)
                perturbed = best_positions + noise
                _try_relax(perturbed, f"bh_large_{k}")
    
    # More diverse random starts
    if _time_left() > 15.0:
        for k in range(5):
            if _time_left() < 4.0:
                break
            seed = n + 400 + k
            try:
                md = 0.4 + 0.05 * (k % 3)
                rnd = _make_random_compact(n, spacing, seed=seed, min_dist=md)
                if len(rnd) == n:
                    _try_relax(rnd, f"extra_{k}")
            except Exception:
                pass
    
    # Final fallback
    if best_positions is None or not np.all(np.isfinite(best_positions)):
        positions = np.zeros((n, 3))
        for i in range(n):
            theta = i * 2.399
            r = spacing * 0.3 * np.sqrt(i)
            positions[i] = [r * np.cos(theta), r * np.sin(theta), 0.2 * i * 0.05]
        try:
            energy, relaxed = relax(positions, potential, kind)
            if np.isfinite(energy):
                best_positions = relaxed
        except Exception:
            best_positions = positions
    
    # Final polish
    if best_positions is not None and np.all(np.isfinite(best_positions)) and _time_left() > 5.0:
        try:
            energy, relaxed = relax(best_positions, potential, kind)
            if np.isfinite(energy):
                dists = cdist(relaxed, relaxed)
                np.fill_diagonal(dists, np.inf)
                if np.min(dists) >= 0.35:
                    best_positions = relaxed
        except Exception:
            pass
    
    # Ensure validity
    if not np.all(np.isfinite(best_positions)):
        positions = np.zeros((n, 3))
        side = int(np.ceil(n ** (1/3)))
        for i in range(n):
            ix = i % side
            iy = (i // side) % side
            iz = i // (side * side)
            positions[i] = [ix * 0.5, iy * 0.5, iz * 0.5]
        best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END

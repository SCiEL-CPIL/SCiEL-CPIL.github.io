"""Sutton-Chen transition-metal cluster global minima: basin-hopping with Mackay seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = None
_TIME_LIMIT = 305.0

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _min_dist(positions):
    d = cdist(positions, positions)
    np.fill_diagonal(d, np.inf)
    return float(np.min(d))

# --- Proper Mackay icosahedral shells ---
def _ico_shells():
    """Return unit vectors for Mackay icosahedral shells."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 0.8:
                mid = (verts[i] + verts[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                edges.append(mid)
    edges_u = []
    for e in edges:
        if not any(np.allclose(e, u, atol=1e-6) for u in edges_u):
            edges_u.append(e)
    
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[k] - verts[i])
                if d1 < 0.8 and d2 < 0.8 and d3 < 0.8:
                    fc = (verts[i] + verts[j] + verts[k]) / 3.0
                    fc = fc / np.linalg.norm(fc)
                    faces.append(fc)
    faces_u = []
    for f in faces:
        if not any(np.allclose(f, u, atol=1e-6) for u in faces_u):
            faces_u.append(f)
    
    return verts, edges_u, faces_u

def _make_mackay(n, spacing=0.72):
    """Generate n atoms in proper Mackay icosahedral shells."""
    verts, edges, faces = _ico_shells()
    
    r1 = spacing * 0.6
    r2 = r1 * np.sqrt(2.0)
    r3 = r1 * np.sqrt(3.0)
    r4 = r1 * 2.0
    
    points = [np.zeros(3)]
    
    for v in verts:
        points.append(v * r1)
    
    for e in edges:
        points.append(e * r2)
    
    for f in faces:
        points.append(f * r3)
    
    if n <= len(points):
        return np.array(points[:n])
    
    remaining = n - len(points)
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(remaining):
        y = 1.0 - (i + 0.5) / remaining * 2.0
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = 2.0 * np.pi * golden * i
        p = np.array([r * np.cos(theta), y, r * np.sin(theta)])
        points.append(p * r4)
    
    return np.array(points[:n])

def _make_fcc(n, spacing=0.72):
    a = spacing
    r_max = spacing * n ** (1.0/3.0) * 0.85
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
    if not candidates:
        return None
    candidates = np.array(candidates)
    d2 = np.sum(candidates**2, axis=1)
    order = np.argsort(d2)
    return candidates[order[:n]]

def _make_bcc(n, spacing=0.72):
    a = spacing
    r_max = spacing * n ** (1.0/3.0) * 0.85
    max_idx = int(r_max / a) + 1
    candidates = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p = np.array([i, j, k]) * a
                if np.linalg.norm(p) <= r_max:
                    candidates.append(p)
                p2 = (np.array([i, j, k]) + np.array([0.5, 0.5, 0.5])) * a
                if np.linalg.norm(p2) <= r_max:
                    candidates.append(p2)
    if not candidates:
        return None
    candidates = np.array(candidates)
    d2 = np.sum(candidates**2, axis=1)
    order = np.argsort(d2)
    return candidates[order[:n]]

def _make_random_compact(n, spacing=0.72, seed=None, min_dist=0.5):
    """Random compact generation using vectorized distance checks."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.85
    points = []
    max_attempts = 3000
    
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        # Generate a batch of candidates
        batch_size = min(100, n - len(points))
        batch = rng.uniform(-radius, radius, (batch_size, 3))
        batch = batch[np.linalg.norm(batch, axis=1) <= radius]
        
        for p in batch:
            if len(points) >= n:
                break
            if not points:
                points.append(p)
            else:
                arr = np.array(points)
                dists = np.linalg.norm(arr - p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(p)
    
    # Fallback with relaxed min_dist
    if len(points) < n:
        for _ in range(max_attempts):
            if len(points) >= n:
                break
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) > radius:
                continue
            if not points:
                points.append(p)
            else:
                arr = np.array(points)
                dists = np.linalg.norm(arr - p, axis=1)
                if np.min(dists) >= 0.36:
                    points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])

def _make_deformed(base_positions, spacing=0.72, seed=None, scale=0.05):
    rng = np.random.default_rng(seed)
    positions = base_positions.copy()
    noise = rng.normal(0, scale * spacing, positions.shape)
    positions += noise
    # Resolve collisions
    for _ in range(3):
        d = cdist(positions, positions)
        np.fill_diagonal(d, np.inf)
        mask = d < 0.36
        if not np.any(mask):
            break
        for i in range(len(positions)):
            close = np.where(mask[i])[0]
            for j in close:
                if j <= i:
                    continue
                dist = d[i, j]
                if dist > 0:
                    direction = (positions[i] - positions[j]) / dist
                    push = (0.36 - dist) / 2.0 + 0.01
                    positions[i] += direction * push
                    positions[j] -= direction * push
    return positions

def _basin_hopping(start_pos, potential, kind, spacing, max_iter, t0, t1, rng,
                   best_energy, best_positions, kick_scale=0.15):
    """Simulated-annealing basin hopping. Returns (best_e, best_pos)."""
    cur_e, cur_pos = relax(start_pos, potential, kind)
    if not np.isfinite(cur_e):
        return best_energy, best_positions
    if cur_e < best_energy and _min_dist(cur_pos) >= 0.35:
        best_energy = cur_e
        best_positions = cur_pos.copy()
    
    for it in range(max_iter):
        if _time_left() < 3.0:
            break
        T = t0 * (t1 / t0) ** (it / max(1, max_iter - 1))
        # Kick
        noise = rng.normal(0, kick_scale * spacing, cur_pos.shape)
        kicked = cur_pos + noise
        # Relax
        try:
            new_e, new_pos = relax(kicked, potential, kind)
        except Exception:
            continue
        if not np.isfinite(new_e):
            continue
        if new_e < best_energy and _min_dist(new_pos) >= 0.35:
            best_energy = new_e
            best_positions = new_pos.copy()
        # Metropolis accept
        if new_e < cur_e:
            cur_e, cur_pos = new_e, new_pos
        else:
            delta = new_e - cur_e
            if rng.uniform() < np.exp(-delta / max(T, 1e-10)):
                cur_e, cur_pos = new_e, new_pos
    return best_energy, best_positions

def _multi_start_basin_hopping(seeds, potential, kind, spacing, rng, 
                                best_energy, best_positions, time_budget=60.0):
    """Run basin hopping from multiple seeds in parallel (sequential but diverse)."""
    if not seeds:
        return best_energy, best_positions
    
    # Allocate time budget among seeds
    n_seeds = len(seeds)
    time_per_seed = time_budget / n_seeds
    
    for idx, (label, start_pos) in enumerate(seeds):
        if _time_left() < 5.0:
            break
        
        # Run basin hopping from this seed
        max_iter = 80
        t0 = 0.5
        t1 = 0.01
        
        best_energy, best_positions = _basin_hopping(
            start_pos, potential, kind, spacing,
            max_iter, t0, t1, rng, best_energy, best_positions
        )
    
    return best_energy, best_positions

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    potential = instance.get("potential", "sutton_chen")
    spacing = 0.72
    
    best_energy = np.inf
    best_positions = None
    rng = np.random.default_rng(n + 42)
    
    def _try_relax(positions):
        nonlocal best_energy, best_positions
        if _time_left() < 2.0:
            return
        try:
            energy, relaxed = relax(positions, potential, kind)
            if np.isfinite(energy) and energy < best_energy:
                if _min_dist(relaxed) >= 0.35:
                    best_energy = energy
                    best_positions = relaxed.copy()
        except Exception:
            pass
    
    # Generate diverse seeds
    seeds = []
    
    # Primary: proper Mackay icosahedral
    try:
        mackay = _make_mackay(n, spacing)
        if mackay is not None and len(mackay) == n:
            seeds.append(("mackay", mackay))
    except Exception:
        pass
    
    # FCC
    try:
        fcc = _make_fcc(n, spacing)
        if fcc is not None and len(fcc) == n:
            seeds.append(("fcc", fcc))
    except Exception:
        pass
    
    # BCC
    try:
        bcc = _make_bcc(n, spacing)
        if bcc is not None and len(bcc) == n:
            seeds.append(("bcc", bcc))
    except Exception:
        pass
    
    # Random compact seeds
    for seed_val in [n, n + 1, n + 2, n + 3]:
        try:
            rnd = _make_random_compact(n, spacing, seed=seed_val, min_dist=0.5)
            if len(rnd) == n:
                seeds.append((f"rand_{seed_val}", rnd))
        except Exception:
            pass
    
    # Deformed Mackay variants
    if seeds:
        for k in range(3):
            try:
                deformed = _make_deformed(seeds[0][1], spacing, seed=n + 10 + k, scale=0.04)
                seeds.append((f"deform_mackay_{k}", deformed))
            except Exception:
                pass
    
    # Relax all seeds first to get initial best
    for label, pos in seeds:
        if _time_left() < 4.0:
            break
        _try_relax(pos)
    
    # Now run multi-start basin hopping with the remaining time
    if best_positions is not None and _time_left() > 20.0:
        # Select top seeds for basin hopping: keep the ones that gave best relaxations
        # Sort seeds by their relaxation energy (if we tracked it) or just use all
        time_budget = min(_time_left() - 10.0, 180.0)
        best_energy, best_positions = _multi_start_basin_hopping(
            seeds, potential, kind, spacing, rng,
            best_energy, best_positions, time_budget
        )
    
    # Final polish
    if best_positions is not None and np.all(np.isfinite(best_positions)) and _time_left() > 5.0:
        try:
            energy, relaxed = relax(best_positions, potential, kind)
            if np.isfinite(energy) and _min_dist(relaxed) >= 0.35:
                best_positions = relaxed
        except Exception:
            pass
    
    # Fallback
    if best_positions is None or not np.all(np.isfinite(best_positions)):
        positions = np.zeros((n, 3))
        for i in range(n):
            theta = i * 2.399
            r = spacing * 0.3 * np.sqrt(i)
            positions[i] = [r * np.cos(theta), r * np.sin(theta), 0.1 * i * 0.05]
        try:
            energy, relaxed = relax(positions, potential, kind)
            if np.isfinite(energy):
                best_positions = relaxed
        except Exception:
            best_positions = positions
    
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

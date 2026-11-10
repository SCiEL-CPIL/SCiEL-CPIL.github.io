"""Portfolio bootstrap: Simulated Annealing with local relaxation refinement."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Global time management
_GLOBAL_T0 = time.time()
_GLOBAL_DEADLINE = 325.0  # Safety margin below 330s kill

def _global_time_left():
    return _GLOBAL_DEADLINE - (time.time() - _GLOBAL_T0)

def _get_potential_params(kind):
    """Return (n_exp, m_exp, c) for the given kind."""
    params = {
        '12-6': (12, 6, 144.41),
        '9-6': (9, 6, 39.432),
        '10-8': (10, 8, 34.408)
    }
    return params.get(kind, (12, 6, 144.41))

def _icosahedron_verts():
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], float)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / n

def _cuboctahedron_verts():
    # 12 vertices: permutations of (±1, ±1, 0)
    v = []
    for x in (-1, 1):
        for y in (-1, 1):
            v.append([x, y, 0.0])
            v.append([x, 0.0, y])
            v.append([0.0, x, y])
    v = np.array(v, float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def _fcc_lattice(n):
    """Pack n atoms in an FCC-like arrangement, scaled by radius."""
    R = 2.0
    pts = []
    s = int(R) + 1
    for x in range(-s, s + 1):
        for y in range(-s, s + 1):
            for z in range(-s, s + 1):
                for shift in [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]:
                    p = np.array([x + shift[0], y + shift[1], z + shift[2]])
                    if np.linalg.norm(p) <= R:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) == 0:
        return pts
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    pts = pts[order]
    if len(pts) > n:
        pts = pts[:n]
    if len(pts) >= 2:
        nn = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        nn = nn[nn > 1e-6]
        if len(nn):
            scale = 0.72 / np.median(nn)
            pts = pts * scale
    return pts

def _random_cluster(size, rng, spacing=0.72, min_dist=0.85):
    radius = spacing * size ** (1.0 / 3.0) * 0.9
    pts = []
    attempts = 0
    while len(pts) < size and attempts < size * size * 40:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
        attempts += 1
    while len(pts) < size:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts[:size])

def _sanitize(positions, min_dist=0.36):
    """Ensure positions are finite and have minimum separation."""
    if not np.all(np.isfinite(positions)):
        return None
    if len(positions) < 2:
        return positions
    d = np.sqrt(((positions[:, None, :] - positions[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    if d.min() < min_dist:
        scale = min_dist / d.min() + 0.05
        positions = positions * scale
    return positions

def _simulated_annealing(positions, potential, kind, rng, time_limit, T0=5.0, alpha=0.999, relax_interval=50):
    """
    Run Simulated Annealing with periodic local relaxation.
    Returns (best_energy, best_positions).
    """
    n = len(positions)
    current_pos = np.asarray(positions, float).copy()
    current_e, _ = sutton_chen_energy_gradient(current_pos, kind)
    best_pos = current_pos.copy()
    best_e = current_e
    
    T = T0
    step_count = 0
    t_start = time.time()
    
    # Determine perturbation scale based on cluster size
    # Larger clusters need larger steps to explore, but also more atoms to move
    base_sigma = 0.15
    
    while (time.time() - t_start) < time_limit:
        # Check global deadline
        if _global_time_left() < 5.0:
            break
            
        # Perturbation: move 1 or 2 random atoms
        n_move = 1 if rng.random() < 0.8 else 2
        idx = rng.choice(n, n_move, replace=False)
        current_pos[idx] += rng.normal(0, base_sigma, (n_move, 3))
        
        # Sanitize to avoid overlaps
        current_pos = _sanitize(current_pos)
        if current_pos is None:
            current_pos = best_pos + rng.normal(0, 0.1, best_pos.shape)
            current_e, _ = sutton_chen_energy_gradient(current_pos, kind)
            continue
            
        # Compute new energy
        new_e, _ = sutton_chen_energy_gradient(current_pos, kind)
        delta_e = new_e - current_e
        
        # Metropolis acceptance
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current_e = new_e
            if current_e < best_e:
                best_e = current_e
                best_pos = current_pos.copy()
        else:
            # Revert if rejected
            current_pos[idx] -= rng.normal(0, base_sigma, (n_move, 3)) # Approximate revert not exact, better to keep track
            # Actually, simpler to just not update current_pos if rejected? 
            # But we already modified it. Let's just re-compute from previous state? 
            # To be safe and simple, we store prev_pos.
            
        # Correction: Store previous state to revert properly
        # This is slightly inefficient but robust.
        
        # Better approach: 
        # 1. Save prev
        # 2. Perturb
        # 3. Calc E
        # 4. Accept/Reject
        # 5. If reject, restore prev
        
        # Let's refactor the loop logic for correctness
        
        # -- Refactored Loop --
        prev_pos = current_pos.copy()
        n_move = 1 if rng.random() < 0.8 else 2
        idx = rng.choice(n, n_move, replace=False)
        current_pos[idx] += rng.normal(0, base_sigma, (n_move, 3))
        
        current_pos = _sanitize(current_pos)
        if current_pos is None:
            current_pos = prev_pos
            continue
            
        new_e, _ = sutton_chen_energy_gradient(current_pos, kind)
        delta_e = new_e - current_e
        
        if delta_e < 0 or (T > 1e-6 and rng.random() < math.exp(-delta_e / T)):
            current_e = new_e
            if current_e < best_e:
                best_e = current_e
                best_pos = current_pos.copy()
        else:
            current_pos = prev_pos
        
        step_count += 1
        
        # Periodic local relaxation
        if step_count % relax_interval == 0:
            try:
                relaxed_e, relaxed_pos = relax(current_pos, potential, kind)
                if np.all(np.isfinite(relaxed_pos)):
                    current_pos = relaxed_pos
                    current_e = relaxed_e
                    if current_e < best_e:
                        best_e = current_e
                        best_pos = current_pos.copy()
            except Exception:
                pass
            
        # Cool down
        T *= alpha
        if T < 1e-4:
            T = 1e-4 # Floor temperature to allow some exploration
            
    return best_e, best_pos

def construct(instance):
    global _T0
    _T0 = time.time()
    size = int(instance["n"])
    potential = instance["potential"]
    kind = instance.get("kind")
    
    # Check global time budget
    if _global_time_left() < 10.0:
        # Emergency: return a quick random pack
        rng = np.random.default_rng(hash((potential, kind, size)) & 0xFFFFFFFF)
        return _random_cluster(size, rng, 0.72, 0.85)
    
    rng = np.random.default_rng(hash((potential, kind, size)) & 0xFFFFFFFF)
    
    # Generate diverse initial configurations
    seeds = []
    
    # 1. Icosahedron-based
    ico = _icosahedron_verts()
    ico_scaled = ico * 0.60
    if size <= 12:
        seeds.append(ico_scaled[:size])
    else:
        # Add shells
        pts = ico_scaled.copy()
        shell_radius = 1.2
        while len(pts) < size:
            # Generate random points on a shell
            new_pts = rng.normal(0, 1, (size - len(pts) + 10, 3))
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts *= shell_radius
            pts = np.vstack([pts, new_pts])
            shell_radius += 0.8
        seeds.append(pts[:size])
        
    # 2. FCC Lattice
    fcc = _fcc_lattice(size)
    if len(fcc) == size:
        seeds.append(fcc)
        
    # 3. Random Pack
    rand1 = _random_cluster(size, rng, 0.72, 0.85)
    seeds.append(rand1)
    
    # 4. Another Random Pack
    rand2 = _random_cluster(size, rng, 0.72, 0.85)
    seeds.append(rand2)
    
    best_e = np.inf
    best_pos = None
    
    # Allocate time for SA runs
    # We have ~50s per instance. Let's do 3 parallel SA runs sequentially, each getting ~15s.
    # Or 1 long run? Multiple runs from different seeds is better for global search.
    n_sa_runs = 3
    time_per_run = min(15.0, (_global_time_left() - 5.0) / n_sa_runs)
    
    for i in range(n_sa_runs):
        if _global_time_left() < 5.0:
            break
            
        seed = seeds[i % len(seeds)]
        # Add slight noise to break symmetry
        seed = seed + rng.normal(0, 0.05, seed.shape)
        seed = _sanitize(seed)
        
        try:
            e, pos = _simulated_annealing(seed, potential, kind, rng, time_limit=time_per_run)
            if np.all(np.isfinite(pos)) and e < best_e:
                best_e = e
                best_pos = pos
        except Exception:
            continue
            
    # If no SA run succeeded, fallback to simple relax of first seed
    if best_pos is None:
        try:
            best_e, best_pos = relax(seeds[0], potential, kind)
        except Exception:
            best_pos = seeds[0]
            best_e = sutton_chen_energy_gradient(best_pos, kind)[0]
            
    # Final Polish
    if _global_time_left() > 2.0:
        try:
            best_e, best_pos = relax(best_pos, potential, kind)
            if not np.all(np.isfinite(best_pos)):
                best_pos = _sanitize(best_pos)
        except Exception:
            pass
            
    return best_pos
# EVOLVE-BLOCK-END

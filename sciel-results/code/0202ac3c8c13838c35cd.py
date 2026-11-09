"""Combined: structure-based seeds + coarse-to-fine SC relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_KIND_PARAMS = {
    '12-6': (12, 6, 144.41),
    '9-6': (9, 6, 39.432),
    '10-8': (10, 8, 34.408),
}

_SPACING = {
    '12-6': 0.72,
    '9-6': 0.72,
    '10-8': 0.72,
}

_SOFT_PARAMS = {
    '12-6': {'n': 6.0, 'm': 3.0, 'c_scale': 0.5},
    '9-6': {'n': 4.5, 'm': 3.0, 'c_scale': 0.5},
    '10-8': {'n': 5.0, 'm': 4.0, 'c_scale': 0.5},
}


def _validate_positions(pos):
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.35


def _fcc_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    basis = np.array([[0, 0, 0],
                      [0, 0.5, 0.5],
                      [0.5, 0, 0.5],
                      [0.5, 0.5, 0]], dtype=float)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k], dtype=float) + b
                    sites.append(p)
    sites = np.array(sites) * spacing
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _hcp_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    a = spacing
    c = spacing * np.sqrt(8.0 / 3.0)
    v1 = np.array([a, 0, 0])
    v2 = np.array([-a / 2, a * np.sqrt(3) / 2, 0])
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                base = i * v1 + j * v2
                z = k * c
                sites.append(base + np.array([0, 0, z]))
                sites.append(base + np.array([a / 2, a * np.sqrt(3) / 6, z + c / 2]))
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _bcc_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    a = spacing
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                sites.append(np.array([i, j, k], dtype=float) * a)
                sites.append((np.array([i, j, k], dtype=float) + 0.5) * a)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _icosahedral_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)

    points = [np.zeros(3)]
    if n <= 1:
        return np.array(points[:n])

    r1 = spacing
    for v in ico_verts:
        points.append(v * r1)
    if len(points) >= n:
        return np.array(points[:n])

    r2 = spacing * np.sqrt(3.0)
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if 1.0 < d < 2.0:
                mid = (ico_verts[i] + ico_verts[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                points.append(mid * r2)
    if len(points) >= n:
        return np.array(points[:n])

    r3 = spacing * np.sqrt(5.0)
    remaining = n - len(points)
    if remaining > 0:
        for k in range(remaining):
            t = (k + 0.5) / remaining
            inc = np.pi * (3 - np.sqrt(5))
            z = 1 - 2 * t
            r = np.sqrt(1 - z * z)
            theta = inc * k
            p = np.array([r * np.cos(theta), r * np.sin(theta), z])
            points.append(p * r3)

    return np.array(points[:n])


def _random_compact(n, spacing=0.72, seed=42, min_dist_frac=0.7):
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    min_d = min_dist_frac * spacing
    points = []
    max_attempts = n * 500
    attempt = 0
    while len(points) < n and attempt < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_d:
                    ok = False
                    break
            if ok:
                points.append(p)
        attempt += 1
    while len(points) < n:
        if points:
            base = points[len(points) % len(points)]
        else:
            base = np.zeros(3)
        p = base + rng.normal(0, spacing * 0.3, 3)
        points.append(p)
    return np.array(points[:n])


def _try_relax(pos, potential, kind):
    try:
        if not _validate_positions(pos):
            return np.inf, None
        energy, relaxed = relax(pos, potential, kind)
        if np.isfinite(energy) and _validate_positions(relaxed):
            return energy, relaxed
    except Exception:
        pass
    return np.inf, None


def _basin_hopping(pos, potential, kind, time_limit=10.0, max_steps=50, noise_scale=0.1):
    """
    Perform basin hopping starting from pos.
    Returns (best_energy, best_positions).
    """
    start_time = time.time()
    n = len(pos)
    current_pos = pos.copy()
    current_energy, _ = _try_relax(current_pos, potential, kind)
    
    if not np.isfinite(current_energy):
        return np.inf, None

    best_energy = current_energy
    best_positions = current_pos.copy()
    
    rng = np.random.default_rng(hash(tuple(pos.flatten())) % 2**32)
    
    # Adaptive noise scaling
    # Start with small noise, increase if stuck, decrease if improving
    current_noise = noise_scale
    
    for step in range(max_steps):
        if time.time() > start_time + time_limit:
            break
        
        # Perturb current position
        noise = rng.normal(0, current_noise, size=pos.shape)
        perturbed_pos = current_pos + noise
        
        # Relax perturbed position
        new_energy, new_pos = _try_relax(perturbed_pos, potential, kind)
        
        if not np.isfinite(new_energy):
            # If relaxation failed or produced invalid pos, try smaller noise or skip
            current_noise *= 0.9
            continue
            
        # Metropolis accept/reject or just keep best? 
        # For global minima search, "keep best" (greedy hill climbing on basin minima) 
        # is often sufficient if we have many restarts. 
        # However, standard Basin Hopping uses Metropolis to escape local minima.
        # Given the time budget, we will use a simple "if better, accept" strategy 
        # combined with occasional "random jumps" to escape.
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_positions = new_pos.copy()
            
        # Simple strategy: Accept if better, or with some probability if worse (Metropolis)
        # But since we don't have a temperature parameter easily tuned, 
        # we'll just move to the new position if it's better, 
        # or if we are stuck for a few steps, take a larger jump.
        
        # Let's implement a simple version:
        # If new_energy < current_energy, accept.
        # Else, accept with probability exp(-(new_energy - current_energy) / T)
        # But T is tricky. 
        # Alternative: Just keep the best found so far, and use the current position 
        # as the base for the next perturbation only if it improved, 
        # otherwise perturb the best position.
        
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            current_noise = max(0.01, current_noise * 0.9) # Reduce noise if improving
        else:
            # Stay at current_pos for next step, but increase noise slightly
            current_noise = min(0.5, current_noise * 1.1)
            
            # To avoid getting stuck in a local minimum of the basin minima,
            # we can occasionally perturb the BEST position found so far.
            if step % 10 == 0:
                base_pos = best_positions + rng.normal(0, current_noise, size=pos.shape)
                e2, p2 = _try_relax(base_pos, potential, kind)
                if np.isfinite(e2) and e2 < current_energy:
                    current_energy = e2
                    current_pos = p2
                elif np.isfinite(e2) and e2 < best_energy:
                    best_energy = e2
                    best_positions = p2

    return best_energy, best_positions


def construct(instance):
    start_time = time.time()
    deadline = start_time + 290.0

    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    spacing = _SPACING.get(kind, 0.72)

    # Generate diverse structural seeds
    candidates = []
    candidates.append(_fcc_cluster(n, spacing))
    candidates.append(_fcc_cluster(n, spacing * 0.9))
    candidates.append(_fcc_cluster(n, spacing * 1.1))
    candidates.append(_hcp_cluster(n, spacing))
    candidates.append(_hcp_cluster(n, spacing * 0.9))
    candidates.append(_hcp_cluster(n, spacing * 1.1))
    candidates.append(_bcc_cluster(n, spacing))
    candidates.append(_bcc_cluster(n, spacing * 1.05))
    candidates.append(_icosahedral_cluster(n, spacing))
    candidates.append(_icosahedral_cluster(n, spacing * 1.1))
    candidates.append(_icosahedral_cluster(n, spacing * 0.9))

    # Random seeds with varied parameters
    for seed in range(1, 20):
        if time.time() > deadline - 30:
            break
        for md_frac in [0.65, 0.75, 0.85]:
            if time.time() > deadline - 30:
                break
            pos = _random_compact(n, spacing, seed=seed * 100 + int(md_frac * 100), min_dist_frac=md_frac)
            if _validate_positions(pos):
                candidates.append(pos)

    best_energy = np.inf
    best_positions = None

    # Allocate time per candidate
    # We have ~260s for search.
    # Let's say we process candidates sequentially.
    # Each basin hopping run gets a fixed time slice, or we iterate until deadline.
    
    # Strategy: 
    # 1. Relax each seed directly to get a baseline.
    # 2. Run Basin Hopping on the best few seeds.
    
    # First pass: Quick relaxation of all seeds to find good starting points for BH
    relaxed_seeds = []
    for pos in candidates:
        if time.time() > deadline - 20:
            break
        e, p = _try_relax(pos, potential, kind)
        if np.isfinite(e) and _validate_positions(p):
            relaxed_seeds.append((e, p))
            
    if relaxed_seeds:
        # Sort by energy
        relaxed_seeds.sort(key=lambda x: x[0])
        best_energy = relaxed_seeds[0][0]
        best_positions = relaxed_seeds[0][1].copy()
        
        # Run Basin Hopping on the top seeds
        # We have limited time, so we pick the top 5-10 seeds
        top_seeds = relaxed_seeds[:10]
        
        # Estimate time per BH run
        time_per_bh = max(5.0, (deadline - time.time() - 10.0) / len(top_seeds))
        
        for e_seed, p_seed in top_seeds:
            if time.time() > deadline - 10:
                break
            
            e_bh, p_bh = _basin_hopping(p_seed, potential, kind, time_limit=time_per_bh, max_steps=100, noise_scale=0.1)
            
            if np.isfinite(e_bh) and e_bh < best_energy:
                best_energy = e_bh
                best_positions = p_bh.copy()
                
    else:
        # Fallback if no seed relaxed successfully
        pos = _random_compact(n, spacing, seed=999)
        e, best_positions = _try_relax(pos, potential, kind)
        if best_positions is None:
            best_positions = pos

    # Final polish: Ensure we have a valid, relaxed structure
    if best_positions is not None and time.time() < deadline - 2:
        try:
            e2, p2 = relax(best_positions, potential, kind)
            if np.isfinite(e2) and _validate_positions(p2) and e2 < best_energy:
                best_energy = e2
                best_positions = p2.copy()
        except Exception:
            pass

    if best_positions is None:
        pos = _random_compact(n, spacing, seed=999)
        e, best_positions = _try_relax(pos, potential, kind)
        if best_positions is None:
            best_positions = pos

    return best_positions
# EVOLVE-BLOCK-END

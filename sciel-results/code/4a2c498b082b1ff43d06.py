"""Sutton-Chen transition-metal cluster global minima: multi-start from diverse
structured seeds (fcc, hcp, icosahedral, Mackay-like), each relaxed with L-BFGS-B,
followed by basin hopping with Metropolis acceptance on the best candidates.
Strict wall-clock deadline."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


# ---- hard wall-clock deadline (evaluator kills at 330 s; keep a safety margin) ----
_DEADLINE_SECONDS = 315.0
_START_TIME = time.time()


def _remaining():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)


# ---------------------------------------------------------------------------
# Seed generators: produce compact, valid (min-pair-distance) (n,3) arrays.
# ---------------------------------------------------------------------------

def _fcc_seed(n, rng, scale=1.0):
    """Fill the fcc lattice with the n sites closest to the origin."""
    basis = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.5, 0.5],
                      [0.5, 0.0, 0.5],
                      [0.5, 0.5, 0.0]])
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 5
    cand = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                cell = np.array([i, j, k])
                for b in basis:
                    p = (cell + b)
                    cand.append((np.dot(p, p), p))
    cand.sort(key=lambda t: t[0])
    pts = np.array([c[1] for c in cand[:n]], dtype=float)
    pts = pts - pts.mean(axis=0)
    pts = pts * scale
    return pts


def _hcp_seed(n, rng, scale=1.0):
    """Fill an hcp-like lattice (two interpenetrating triangular layers)."""
    a = 1.0
    h = np.sqrt(2.0 / 3.0) * a
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 5
    cand = []
    for layer in range(-max_shell, max_shell + 1):
        z = layer * h
        is_b = (layer % 2 != 0)
        for i in range(-max_shell, max_shell + 1):
            for j in range(-max_shell, max_shell + 1):
                x = a * (i + 0.5 * j)
                y = a * (np.sqrt(3.0) / 2.0) * j
                if is_b:
                    x += a * 0.5
                    y += a * np.sqrt(3.0) / 6.0
                p = np.array([x, y, z])
                cand.append((np.dot(p, p), p))
    cand.sort(key=lambda t: t[0])
    pts = np.array([c[1] for c in cand[:n]], dtype=float)
    pts = pts - pts.mean(axis=0)
    pts = pts * scale
    return pts


def _icosahedral_seed(n, rng, scale=1.0):
    """Start from an icosahedron and fill with fcc points."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    pts_list = [p * 1.0 for p in ico]
    fcc = _fcc_seed(n, rng, scale=1.0)
    all_pts = list(pts_list)
    for p in fcc:
        if len(all_pts) >= n:
            break
        ok = True
        for q in all_pts:
            if np.linalg.norm(p - q) < 0.45:
                ok = False
                break
        if ok:
            all_pts.append(p)
    if len(all_pts) < n:
        for p in fcc:
            if len(all_pts) >= n:
                break
            all_pts.append(p)
    pts = np.array(all_pts[:n], dtype=float)
    pts = pts - pts.mean(axis=0)
    pts = pts * scale
    return pts


def _mackay_seed(n, rng, scale=1.0):
    """Mackay icosahedron-like seed: concentric icosahedral shells."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    pts = [np.zeros(3)]  # center
    
    # First shell: 12 icosahedron vertices
    for v in ico_verts:
        pts.append(v * 1.0)
    
    # Second shell: 30 points (edges of icosahedron midpoints)
    if n > 13:
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                if d < 1.0:  # edge length ~1
                    edges.append((i, j))
        for i, j in edges:
            if len(pts) >= n:
                break
            mid = (ico_verts[i] + ico_verts[j]) / 2.0
            pts.append(mid * 1.62)
    
    # Third shell: 20 points (face centers)
    if n > 43:
        faces = []
        for i in range(12):
            for j in range(i+1, 12):
                for k in range(j+1, 12):
                    d1 = np.linalg.norm(ico_verts[i] - ico_verts[j])
                    d2 = np.linalg.norm(ico_verts[j] - ico_verts[k])
                    d3 = np.linalg.norm(ico_verts[i] - ico_verts[k])
                    if d1 < 1.0 and d2 < 1.0 and d3 < 1.0:
                        faces.append((i, j, k))
        for i, j, k in faces:
            if len(pts) >= n:
                break
            centroid = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3.0
            centroid = centroid / np.linalg.norm(centroid)
            pts.append(centroid * 2.1)
    
    # Fill remaining with fcc
    if len(pts) < n:
        fcc = _fcc_seed(n, rng, scale=1.0)
        for p in fcc:
            if len(pts) >= n:
                break
            ok = True
            for q in pts:
                if np.linalg.norm(p - q) < 0.4:
                    ok = False
                    break
            if ok:
                pts.append(p)
    
    pts = np.array(pts[:n], dtype=float)
    pts = pts - pts.mean(axis=0)
    pts = pts * scale
    return pts


def _random_compact_seed(n, rng, scale=1.0, min_dist=0.55):
    """Random points in a ball with a minimum-pair-distance constraint."""
    radius = scale * n ** (1.0 / 3.0) * 0.85
    pts = []
    tries = 0
    max_tries = 30000
    while len(pts) < n and tries < max_tries:
        tries += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
    if len(pts) < n:
        md = min_dist
        while len(pts) < n:
            md *= 0.95
            if md < 0.36:
                while len(pts) < n:
                    p = rng.uniform(-radius, radius, 3)
                    if np.linalg.norm(p) <= radius:
                        pts.append(p)
                break
            tries = 0
            while len(pts) < n and tries < max_tries:
                tries += 1
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) > radius:
                    continue
                if all(np.linalg.norm(p - q) >= md for q in pts):
                    pts.append(p)
    pts = np.array(pts, dtype=float)
    pts = pts - pts.mean(axis=0)
    return pts


def _jitter(seed, rng, amount):
    return seed + rng.normal(0.0, amount, seed.shape)


def _is_valid(pts, min_dist=0.35):
    if not np.all(np.isfinite(pts)):
        return False
    n = len(pts)
    if n < 2:
        return True
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(d, axis=2)
    np.fill_diagonal(dist, np.inf)
    return dist.min() >= min_dist


def _try_relax(seed, potential, kind, maxiter=3000):
    """Relax a seed; return (energy, positions) or (inf, None) on failure."""
    try:
        energy, positions = relax(seed, potential, kind)
    except Exception:
        return np.inf, None
    if not np.all(np.isfinite(positions)):
        return np.inf, None
    if not _is_valid(positions):
        return np.inf, None
    return float(energy), positions


def _basin_hopping(positions, potential, kind, rng, n_iter=20, sigma_start=0.3, sigma_end=0.05, T_start=0.5, T_end=0.01):
    """Basin hopping with Metropolis acceptance. Returns best (energy, positions)."""
    best_energy, best_pos = _try_relax(positions, potential, kind)
    if best_pos is None:
        return best_energy, best_pos
    
    n = len(positions)
    for it in range(n_iter):
        if _remaining() < 5.0:
            break
        
        # Linearly anneal sigma and T
        frac = it / max(n_iter - 1, 1)
        sigma = sigma_start * (1.0 - frac) + sigma_end * frac
        T = T_start * (1.0 - frac) + T_end * frac
        
        # Propose a random displacement
        delta = rng.normal(0.0, sigma, positions.shape)
        proposed = best_pos + delta
        
        # Relax the proposed structure
        e_prop, pos_prop = _try_relax(proposed, potential, kind)
        
        if pos_prop is None:
            continue
        
        # Metropolis acceptance
        if e_prop < best_energy:
            best_energy = e_prop
            best_pos = pos_prop
        else:
            dE = e_prop - best_energy
            if T > 1e-6:
                accept_prob = np.exp(-dE / T)
                if rng.random() < accept_prob:
                    best_energy = e_prop
                    best_pos = pos_prop
    
    return best_energy, best_pos


def construct(instance):
    global _START_TIME
    _START_TIME = time.time()

    n = int(instance["n"])
    potential = instance["potential"]
    kind = instance.get("kind")
    parameter = kind

    # equilibrium nearest-neighbor spacing ~ 1.0 for these potentials;
    # use a scale that keeps seeds compact.
    base_scale = 1.0

    rng = np.random.default_rng(12345 + n)

    # Build a diverse list of seeds, ordered to spend budget on the most
    # promising first (structured), then jittered structured, then random.
    seeds = []

    # 1. Pure structured seeds (small jitter to escape exact lattice degeneracy)
    for gen in (_fcc_seed, _hcp_seed, _icosahedral_seed, _mackay_seed):
        s = gen(n, rng, scale=base_scale)
        if _is_valid(s, 0.4):
            seeds.append(s)
        # jittered variants
        for amt in (0.05, 0.12, 0.25):
            sj = _jitter(s, rng, amt)
            if _is_valid(sj, 0.36):
                seeds.append(sj)

    # 2. Random compact seeds
    for _ in range(6):
        s = _random_compact_seed(n, rng, scale=base_scale, min_dist=0.6)
        if _is_valid(s, 0.36):
            seeds.append(s)

    best_energy = np.inf
    best_positions = None

    # First, if we have any valid seed, make sure we always have a fallback.
    # Relax seeds one at a time under the deadline.
    for seed in seeds:
        if _remaining() < 8.0:
            break  # safety margin before the hard kill
        e, pos = _try_relax(seed, potential, parameter)
        if e < best_energy:
            best_energy = e
            best_positions = pos

    # If nothing valid yet (shouldn't happen), produce a guaranteed-valid
    # fallback: a compact fcc seed, relaxed once.
    if best_positions is None:
        s = _fcc_seed(n, rng, scale=base_scale)
        if not _is_valid(s, 0.35):
            s = _random_compact_seed(n, rng, scale=base_scale, min_dist=0.36)
        e, pos = _try_relax(s, potential, parameter)
        if pos is not None:
            best_energy = e
            best_positions = pos
        else:
            # absolute last resort: spread points on a coarse grid
            side = int(np.ceil(n ** (1.0 / 3.0)))
            grid = []
            for i in range(side):
                for j in range(side):
                    for k in range(side):
                        grid.append([i - side // 2, j - side // 2, k - side // 2])
            pts = np.array(grid[:n], dtype=float) * 0.75
            best_positions = pts

    # If we still have budget, do basin hopping from the best solution
    if best_positions is not None and _remaining() > 30.0:
        # Determine how many basin hopping iterations we can afford
        # Each iteration takes roughly 1-3 seconds for n~50-80
        # Be conservative: assume ~2s per iteration
        est_time_per_iter = 2.0
        available_time = _remaining() - 10.0  # safety margin
        n_bh_iters = max(5, int(available_time / est_time_per_iter))
        n_bh_iters = min(n_bh_iters, 40)  # cap to avoid too long
        
        e_bh, pos_bh = _basin_hopping(best_positions, potential, parameter, 
                                       rng, n_iter=n_bh_iters,
                                       sigma_start=0.25, sigma_end=0.05,
                                       T_start=0.3, T_end=0.01)
        if pos_bh is not None and e_bh < best_energy:
            best_energy = e_bh
            best_positions = pos_bh
    
    # If we still have budget, do a few extra random restarts from the best
    # solution to try to escape the local minimum (cheap, bounded by deadline).
    if best_positions is not None and _remaining() > 15.0:
        for _ in range(8):
            if _remaining() < 8.0:
                break
            perturbed = _jitter(best_positions, rng, 0.15)
            e, pos = _try_relax(perturbed, potential, parameter)
            if e < best_energy:
                best_energy = e
                best_positions = pos

    return best_positions
# EVOLVE-BLOCK-END

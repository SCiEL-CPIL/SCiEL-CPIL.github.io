"""Sutton-Chen transition-metal cluster global minima search.
Uses parallel multi-start basin hopping from diverse structural seeds
(FCC, BCC, HCP, Icosahedral, random) to explore many basins within
the time budget."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START = time.time()
_DEADLINE = 290.0  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _is_valid(positions, min_dist=0.35):
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _make_fcc(n):
    """Generate a compact FCC-like cluster of n atoms."""
    pts = []
    mc = int(np.ceil(n ** (1.0 / 3.0))) + 2
    a = 1.0
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    if len(pts) < n:
                        pts.append(p)
    if len(pts) < n:
        rng = np.random.default_rng(42)
        while len(pts) < n:
            p = rng.uniform(-mc * a, mc * a, 3)
            pts.append(p)
    return np.array(pts[:n])


def _make_bcc(n):
    """Generate a compact BCC-like cluster of n atoms."""
    pts = []
    mc = int(np.ceil(n ** (1.0 / 3.0))) + 2
    a = 1.0
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                p = np.array([i, j, k]) * a
                if len(pts) < n:
                    pts.append(p)
                pc = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                if len(pts) < n:
                    pts.append(pc)
    if len(pts) < n:
        rng = np.random.default_rng(42)
        while len(pts) < n:
            p = rng.uniform(-mc * a, mc * a, 3)
            pts.append(p)
    return np.array(pts[:n])


def _make_hcp(n):
    """Generate a compact HCP-like cluster of n atoms."""
    pts = []
    a = 1.0
    mc = int(np.ceil(n ** (1.0 / 3.0))) + 2
    h = a * np.sqrt(2.0 / 3.0)
    for layer in range(-mc, mc + 1):
        for i in range(-mc, mc + 1):
            for j in range(-mc, mc + 1):
                if layer % 2 == 0:
                    p = np.array([i * a, j * a, layer * h])
                else:
                    p = np.array([(i + 0.5) * a, (j + 0.5) * a, layer * h])
                if len(pts) < n:
                    pts.append(p)
    if len(pts) < n:
        rng = np.random.default_rng(42)
        while len(pts) < n:
            p = rng.uniform(-mc * a, mc * a, 3)
            pts.append(p)
    return np.array(pts[:n])


def _make_icosahedral(n):
    """Generate an icosahedral/Mackay-like cluster."""
    pts = [np.array([0.0, 0.0, 0.0])]
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    for v in ico:
        if len(pts) < n:
            pts.append(v * 1.0)
    if len(pts) < n:
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(ico[i] - ico[j])
                if 0.9 < d < 1.1:
                    mid = (ico[i] + ico[j]) / 2.0
                    mid = mid / np.linalg.norm(mid) * 1.5
                    edges.append(mid)
        for e in edges:
            if len(pts) < n:
                pts.append(e)
    if len(pts) < n:
        rng = np.random.default_rng(123)
        while len(pts) < n:
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0, np.pi)
            r = 2.0
            p = np.array([r * np.sin(phi) * np.cos(theta),
                         r * np.sin(phi) * np.sin(theta),
                         r * np.cos(phi)])
            pts.append(p)
    return np.array(pts[:n])


def _make_random_compact(n, seed, spacing=0.72):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _basin_hopping(start_pos, kind, max_iter=30, perturb_scale=0.2, seed=0):
    """Run a single basin-hopping chain from start_pos.
    Returns (best_energy, best_positions)."""
    rng = np.random.default_rng(seed)
    pos = start_pos.copy()
    try:
        e, pos = relax(pos, "sutton_chen", kind)
    except Exception:
        return np.inf, start_pos.copy()
    
    if not np.isfinite(e) or not _is_valid(pos):
        return np.inf, start_pos.copy()
    
    best_e = e
    best_pos = pos.copy()
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Adaptive perturbation: larger when stuck
        scale = perturb_scale * (1.0 + 0.3 * it / max_iter)
        perturbed = pos + rng.normal(0, scale, pos.shape)
        
        if not _is_valid(perturbed):
            continue
        
        try:
            e_new, pos_new = relax(perturbed, "sutton_chen", kind)
        except Exception:
            continue
        
        if not np.isfinite(e_new) or not _is_valid(pos_new):
            continue
        
        # Accept if better or with Metropolis-like probability (always accept better)
        if e_new < best_e:
            best_e = e_new
            best_pos = pos_new.copy()
        
        # Move to new position regardless (stochastic hill-climbing)
        pos = pos_new
    
    return best_e, best_pos


def construct(instance):
    size = instance["n"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    
    # Generate diverse starting points
    starts = []
    starts.append(_make_fcc(size))
    starts.append(_make_bcc(size))
    starts.append(_make_hcp(size))
    starts.append(_make_icosahedral(size))
    
    # Deformed FCC variants
    rng_fcc = np.random.default_rng(1)
    fcc_base = _make_fcc(size)
    starts.append(fcc_base + rng_fcc.normal(0, 0.2, fcc_base.shape))
    starts.append(fcc_base + rng_fcc.normal(0, 0.4, fcc_base.shape))
    
    # Random compact seeds
    for s in range(6):
        starts.append(_make_random_compact(size, seed=100 + s))
    
    # Run basin hopping from each seed
    best_energy = np.inf
    best_positions = None
    
    for idx, start in enumerate(starts):
        if _time_left() < 8.0:
            break
        
        if not _is_valid(start):
            continue
        
        # Budget per chain: use remaining time / remaining chains
        remaining_chains = len(starts) - idx
        time_per_chain = max(2.0, _time_left() / (remaining_chains + 1))
        
        # Estimate max iterations based on time budget
        # Each relax call takes roughly 0.1-0.5s for these cluster sizes
        max_iter = int(min(40, max(5, time_per_chain / 0.3)))
        
        e, pos = _basin_hopping(start, kind, max_iter=max_iter,
                                perturb_scale=0.2, seed=1000 + idx)
        
        if np.isfinite(e) and _is_valid(pos) and e < best_energy:
            best_energy = e
            best_positions = pos.copy()
    
    # If time remains, do additional refinement on the best solution
    if best_positions is not None and _time_left() > 20.0:
        for attempt in range(15):
            if _time_left() < 10.0:
                break
            rng = np.random.default_rng(5000 + attempt)
            perturbed = best_positions + rng.normal(0, 0.15, best_positions.shape)
            if not _is_valid(perturbed):
                continue
            try:
                e, pos = relax(perturbed, "sutton_chen", kind)
                if np.isfinite(e) and _is_valid(pos) and e < best_energy:
                    best_energy = e
                    best_positions = pos.copy()
            except Exception:
                continue
    
    if best_positions is None:
        # Fallback
        best_positions = _make_fcc(size)
        try:
            _, best_positions = relax(best_positions, "sutton_chen", kind)
        except Exception:
            pass
    
    if not _is_valid(best_positions):
        best_positions = np.zeros((size, 3))
        for i in range(size):
            best_positions[i] = [i * 0.5, (i // 10) * 0.5, (i % 10) * 0.5]
    
    return best_positions
# EVOLVE-BLOCK-END

"""Portfolio-search program for Sutton-Chen TM cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 295.0  # safety margin: evaluator kills at 330s, we target 295s


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    if positions is None:
        return False
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    for i in range(n):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _get_params(kind):
    """Return (n_exp, m_exp, c) for the given kind string."""
    if kind == '12-6':
        return 12, 6, 144.41
    elif kind == '9-6':
        return 9, 6, 39.432
    elif kind == '10-8':
        return 10, 8, 34.408
    else:
        return 12, 6, 144.41


def _sc_energy_pair(r, n_exp, m_exp, c):
    """Energy contribution for a pair at distance r."""
    if r < 0.1:
        return 1e10
    r_inv = 1.0 / r
    r_inv_n = r_inv ** n_exp
    r_inv_m = r_inv ** m_exp
    return 0.5 * r_inv_n - 0.5 * c * np.sqrt(r_inv_m)


def _energy_grown_cluster(n, kind, seed=0, max_candidates=200):
    """
    Build a cluster atom-by-atom using energy-based candidate selection.
    At each step, generate candidate positions around the existing cluster,
    evaluate the incremental energy, and place the atom at the best position.
    """
    n_exp, m_exp, c = _get_params(kind)
    rng = np.random.default_rng(seed)
    
    # Start with first atom at origin
    pts = [np.zeros(3)]
    
    for step in range(1, n):
        # Generate candidate positions
        # Strategy: place near existing atoms (shell growth) and in random directions
        candidates = []
        
        # Radius estimate for current cluster
        if len(pts) > 1:
            radii = np.linalg.norm(pts, axis=1)
            max_r = np.max(radii)
        else:
            max_r = 0.0
        
        # Base spacing ~1.0 (lattice constant)
        base_spacing = 1.0
        
        # Candidates near each existing atom
        for p in pts:
            # Random directions
            for _ in range(8):
                direction = rng.standard_normal(3)
                direction /= (np.linalg.norm(direction) + 1e-10)
                # Place at various distances
                for dist in [base_spacing * 0.8, base_spacing * 1.0, base_spacing * 1.2]:
                    cand = p + direction * dist
                    candidates.append(cand)
        
        # Candidates in random directions from center
        for _ in range(30):
            direction = rng.standard_normal(3)
            direction /= (np.linalg.norm(direction) + 1e-10)
            for dist in [max_r * 0.5, max_r * 0.8, max_r * 1.0, max_r * 1.2, max_r * 1.5]:
                if dist > 0.1:
                    cand = direction * dist
                    candidates.append(cand)
        
        # Also try positions on a sphere of radius ~max_r
        for _ in range(20):
            theta = rng.uniform(0, 2 * np.pi)
            phi_angle = rng.uniform(0, np.pi)
            direction = np.array([
                np.sin(phi_angle) * np.cos(theta),
                np.sin(phi_angle) * np.sin(theta),
                np.cos(phi_angle)
            ])
            for dist in [max_r * 0.9, max_r * 1.0, max_r * 1.1]:
                if dist > 0.1:
                    cand = direction * dist
                    candidates.append(cand)
        
        # Evaluate candidates
        best_cand = None
        best_energy = np.inf
        
        pts_arr = np.array(pts)
        
        for cand in candidates[:max_candidates]:
            # Check minimum distance
            if len(pts) > 0:
                dists = np.linalg.norm(pts_arr - cand, axis=1)
                if np.min(dists) < 0.38:
                    continue
            
            # Compute incremental energy (interaction with all existing atoms)
            incr_energy = 0.0
            for p in pts:
                d = np.linalg.norm(cand - p)
                if d < 0.35:
                    incr_energy = 1e10
                    break
                r_inv = 1.0 / d
                r_inv_n = r_inv ** n_exp
                r_inv_m = r_inv ** m_exp
                incr_energy += 0.5 * r_inv_n - 0.5 * c * np.sqrt(r_inv_m)
            
            if incr_energy < best_energy:
                best_energy = incr_energy
                best_cand = cand
        
        if best_cand is None:
            # Fallback: place at a random position far away
            best_cand = pts[-1] + rng.standard_normal(3) * base_spacing
        
        pts.append(best_cand)
        
        # Periodic re-centering to avoid drift
        if step % 5 == 0:
            pts = [p - np.mean(pts, axis=0) for p in pts]
    
    result = np.array(pts)
    result -= result.mean(axis=0)
    return result


def _make_fcc(n):
    """Generate an FCC-like cluster by taking atoms from a cubic grid with FCC offsets."""
    pts = []
    mc = int(np.ceil(n ** (1/3))) + 3
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                for off in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + off[0], j + off[1], k + off[2]], dtype=float)
                    pts.append(p)
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break
        if len(pts) >= n:
            break
    pts = np.array(pts[:n])
    pts -= pts.mean(axis=0)
    return pts


def _make_bcc(n):
    """Generate a BCC-like cluster."""
    pts = []
    mc = int(np.ceil(n ** (1/3))) + 3
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                pts.append(np.array([i, j, k], dtype=float))
                pts.append(np.array([i + 0.5, j + 0.5, k + 0.5], dtype=float))
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break
        if len(pts) >= n:
            break
    pts = np.array(pts[:n])
    pts -= pts.mean(axis=0)
    return pts


def _make_icosahedron_shell(n):
    """Generate an icosahedral-like cluster with shells."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    pts = [np.zeros(3)]
    for v in verts:
        pts.append(v * 1.0)
    
    if len(pts) >= n:
        return np.array(pts[:n])
    
    edges = [(0,1),(1,3),(3,2),(2,0),
             (0,4),(1,5),(2,6),(3,7),
             (0,8),(1,9),(2,10),(3,11),
             (4,5),(5,7),(7,6),(6,4),
             (8,9),(9,11),(11,10),(10,8),
             (4,8),(5,9),(6,10),(7,11),
             (4,10),(5,11),(6,9),(7,8)]
    shell2 = []
    for a, b in edges:
        mid = (verts[a] + verts[b]) / 2
        mid = mid / np.linalg.norm(mid) * np.sqrt(2)
        shell2.append(mid)
    for v in shell2:
        pts.append(v)
        if len(pts) >= n:
            break
    
    if len(pts) >= n:
        return np.array(pts[:n])
    
    rng = np.random.default_rng(42)
    while len(pts) < n:
        v = verts[rng.integers(0, 12)]
        p = v * (np.sqrt(2) + rng.uniform(0.1, 0.5))
        pts.append(p)
    
    pts = np.array(pts[:n])
    pts -= pts.mean(axis=0)
    return pts


def _make_random_compact(n, spacing=0.72, seed=None):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    for _ in range(2000):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.7 * spacing for q in points):
                points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points[:n])


def _perturb(positions, amplitude, rng):
    """Perturb positions randomly."""
    return positions + rng.standard_normal(positions.shape) * amplitude


def _basin_hop(positions, kind, potential, n_hops=10, base_amp=0.3):
    """
    Basin hopping: perturb, relax, keep if lower energy.
    Returns best energy and positions found.
    """
    rng = np.random.default_rng(12345)
    energy, pos = sutton_chen_energy_gradient(positions, kind)
    best_energy = energy
    best_pos = pos.copy()
    
    for hop in range(n_hops):
        if _time_left() < 5:
            break
        
        amp = base_amp * (1.0 + 0.1 * hop)
        perturbed = _perturb(pos, amp, rng)
        
        try:
            e2, p2 = relax(perturbed, potential, kind)
        except Exception:
            continue
        
        if not _is_valid(p2):
            continue
        
        if e2 < best_energy:
            best_energy = e2
            best_pos = p2.copy()
            pos = p2
        else:
            # Metropolis-like acceptance with cooling
            T = 0.5 * max(0.01, 1.0 - hop / n_hops)
            if np.exp(-(e2 - best_energy) / T) > rng.random():
                pos = p2
            else:
                pos = best_pos
    
    return best_energy, best_pos


def construct(instance):
    """
    Structural change: use deterministic energy-based growth as primary method,
    then basin hopping to escape local minima. Also keep template starts as fallback.
    """
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    if _time_left() < 10:
        # Emergency: return a simple valid cluster
        pts = _make_random_compact(size, seed=0)
        energy, positions = relax(pts, potential, kind)
        return positions
    
    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Energy-grown clusters (primary structural method)
    # Use multiple seeds for diversity
    n_growth_seeds = 3
    for seed in range(n_growth_seeds):
        if _time_left() < 15:
            break
        try:
            grown = _energy_grown_cluster(size, kind, seed=seed)
            if _is_valid(grown):
                energy, positions = relax(grown, potential, kind)
                if _is_valid(positions) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            continue
    
    # Phase 2: Template-based starts (secondary)
    templates = []
    try:
        fcc_pts = _make_fcc(size)
        if _is_valid(fcc_pts):
            templates.append(fcc_pts)
    except Exception:
        pass
    
    try:
        bcc_pts = _make_bcc(size)
        if _is_valid(bcc_pts):
            templates.append(bcc_pts)
    except Exception:
        pass
    
    try:
        ico_pts = _make_icosahedron_shell(size)
        if _is_valid(ico_pts):
            templates.append(ico_pts)
    except Exception:
        pass
    
    for seed in [0, 1]:
        try:
            rand_pts = _make_random_compact(size, seed=seed)
            if _is_valid(rand_pts):
                templates.append(rand_pts)
        except Exception:
            pass
    
    for start_pts in templates:
        if _time_left() < 15:
            break
        try:
            energy, positions = relax(start_pts, potential, kind)
            if _is_valid(positions) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    if best_positions is None:
        # Fallback
        pts = _make_random_compact(size, seed=99)
        energy, positions = relax(pts, potential, kind)
        best_positions = positions
    
    # Phase 3: Basin hopping on the best result
    if _time_left() > 30:
        try:
            bh_energy, bh_pos = _basin_hop(best_positions, kind, potential, n_hops=15, base_amp=0.25)
            if _is_valid(bh_pos) and bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos
        except Exception:
            pass
    
    # Phase 4: If time remains, try basin hopping from a second-best starting point
    if _time_left() > 20 and len(templates) > 0:
        # Try basin hopping from the first template (different basin)
        try:
            energy2, pos2 = relax(templates[0], potential, kind)
            if _is_valid(pos2):
                bh2_energy, bh2_pos = _basin_hop(pos2, kind, potential, n_hops=8, base_amp=0.3)
                if _is_valid(bh2_pos) and bh2_energy < best_energy:
                    best_energy = bh2_energy
                    best_positions = bh2_pos
        except Exception:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END

"""Dzugutov cluster global minima: parameterized icosahedral manifold + basin hopping."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient, relax

_START = time.time()
TIME_LIMIT = 320.0  # safety margin under the 330 s kill


def _remaining():
    return TIME_LIMIT - (time.time() - _START)


# Dzugutov parameters
_A, _B = 5.82, 1.28
_a, _b = 1.87, 1.94
_c, _d = 1.10, 0.27
_m = 16
_MIN_SEP = 0.5


def _energy(pos):
    """Vectorised Dzugutov energy (flat array, n*3)."""
    p = np.asarray(pos, dtype=float)
    n = p.size // 3
    p = p.reshape(n, 3)
    diff = p[:, None, :] - p[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    ri = r[iu]
    e = np.zeros(ri.shape, dtype=float)
    m1 = ri < _a
    m2 = (ri >= _a) & (ri < _b)
    e[m1] = _A * (ri[m1] ** (-_m) - _B) * np.exp(_c / (ri[m1] - _a))
    e[m2] = _B * np.exp(_d / (ri[m2] - _b))
    return float(e.sum())


def _energy_fn(x):
    return _energy(x)


def _grad_fn(x):
    p = np.asarray(x, dtype=float).reshape(-1, 3)
    try:
        g = dzugutov_energy_gradient(p)
        g = np.asarray(g, dtype=float).reshape(-1)
        if np.all(np.isfinite(g)):
            return g
    except Exception:
        pass
    # finite-difference fallback
    n = p.size
    h = 1e-6
    e0 = _energy(x)
    g = np.empty(n, dtype=float)
    for i in range(n):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (_energy(xp) - _energy(xm)) / (2 * h)
    return g


def _min_sep_ok(pos):
    if pos.shape[0] < 2:
        return True
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(pos.shape[0], k=1)
    return r[iu].min() >= _MIN_SEP - 1e-9


def _make_icosahedron(scale):
    """12 vertices of a regular icosahedron scaled to nearest-neighbour distance ~scale."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    # edge length of this icosahedron is 2
    v = v * (scale / 2.0)
    return v


def _make_dodecahedron(scale):
    """20 vertices of a regular dodecahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    v = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, inv_phi, phi], [0, inv_phi, -phi], [0, -inv_phi, phi], [0, -inv_phi, -phi],
        [inv_phi, phi, 0], [inv_phi, -phi, 0], [-inv_phi, phi, 0], [-inv_phi, -phi, 0],
        [phi, 0, inv_phi], [phi, 0, -inv_phi], [-phi, 0, inv_phi], [-phi, 0, -inv_phi],
    ], dtype=float)
    # Normalize to have edge length ~2
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / norms
    # Scale so that nearest-neighbor distance is approximately scale
    # For a dodecahedron, the edge length relative to circumradius is 2*sin(pi/5)
    edge_to_circum = 2 * np.sin(np.pi / 5)
    v = v * (scale / edge_to_circum)
    return v


def _make_cuboctahedron(scale):
    """12 vertices of a cuboctahedron."""
    v = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / norms
    # Edge length for cuboctahedron with circumradius 1/sqrt(2) is 1
    # So scale by scale
    v = v * (scale / np.sqrt(2.0))
    return v


def _get_icosahedral_directions():
    """Get unit direction vectors from icosahedron, dodecahedron, and cuboctahedron vertices."""
    ico_dirs = _make_icosahedron(1.0)
    ico_dirs = ico_dirs / np.linalg.norm(ico_dirs, axis=1, keepdims=True)
    
    dod_dirs = _make_dodecahedron(1.0)
    dod_dirs = dod_dirs / np.linalg.norm(dod_dirs, axis=1, keepdims=True)
    
    cub_dirs = _make_cuboctahedron(1.0)
    cub_dirs = cub_dirs / np.linalg.norm(cub_dirs, axis=1, keepdims=True)
    
    # Combine all directions
    all_dirs = np.vstack([ico_dirs, dod_dirs, cub_dirs])
    
    # Remove near-duplicates
    unique_dirs = []
    for d in all_dirs:
        is_dup = False
        for ud in unique_dirs:
            if np.dot(d, ud) > 0.99:
                is_dup = True
                break
        if not is_dup:
            unique_dirs.append(d)
    unique_dirs = np.array(unique_dirs)
    return unique_dirs


def _mckay_icosahedron(n, base=1.10):
    """
    Generate a Mackay icosahedral cluster with n atoms.
    Uses concentric shells: 1 (center), 12 (icosahedron), 42 (Mackay), 92 (Mackay).
    """
    unique_dirs = _get_icosahedral_directions()
    
    # Shell definitions: (radius_factor, max_atoms)
    # For n up to 60, we need shells of 1, 12, 42
    shell_defs = [
        (1.0, 1),    # center
        (1.65, 12),  # first shell (icosahedron)
        (2.30, 42),  # second shell (Mackay icosahedron)
        (2.95, 92),  # third shell (Mackay icosahedron)
    ]
    
    pts = [np.zeros(3)]
    placed = 1
    dir_idx = 0
    
    for shell_i, (radius_factor, max_atoms) in enumerate(shell_defs):
        if placed >= n:
            break
        r = base * radius_factor
        count = min(max_atoms, n - placed)
        for _ in range(count):
            if dir_idx >= len(unique_dirs):
                dir_idx = 0
            d = unique_dirs[dir_idx]
            # Add slight angular perturbation for different shells to avoid perfect overlap
            if shell_i > 0:
                angle = 0.03 * shell_i + 0.01 * (dir_idx % 7)
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                # Rotate around y-axis
                d_new = np.array([
                    d[0] * cos_a + d[2] * sin_a,
                    d[1],
                    -d[0] * sin_a + d[2] * cos_a
                ])
                d = d_new / (np.linalg.norm(d_new) + 1e-12)
            pts.append(d * r)
            placed += 1
            dir_idx += 1
    
    if placed < n:
        # Fill remaining with random directions
        rng = np.random.default_rng(n)
        while placed < n:
            d = rng.normal(size=3)
            d = d / (np.linalg.norm(d) + 1e-12)
            r = base * 2.5
            pts.append(d * r)
            placed += 1
    
    cfg = np.array(pts[:n])
    return cfg


def _icosahedral_seeds(n, base=1.10):
    """Generate several icosahedral-shell starting configurations for cluster size n."""
    seeds = []
    
    # Get direction vectors from different polyhedra
    unique_dirs = _get_icosahedral_directions()
    
    # Build seeds with different shell structures
    shell_configs = [
        # (shell_radii_factors, n_atoms_per_shell)
        ([1.0, 1.65, 2.30], [1, 12, 42]),
        ([1.0, 1.70, 2.35], [1, 12, 42]),
        ([1.0, 1.60, 2.25], [1, 12, 42]),
        ([1.0, 1.75, 2.40], [1, 12, 42]),
        ([1.0, 1.65, 2.30, 2.95], [1, 12, 42, 92]),
        ([1.0, 1.70, 2.35, 3.00], [1, 12, 42, 92]),
    ]
    
    for radii_factors, shell_counts in shell_configs:
        radii = [base * f for f in radii_factors]
        pts = [np.zeros(3)]
        placed = 1
        dir_idx = 0
        
        for shell_i, count in enumerate(shell_counts):
            if placed >= n:
                break
            r = radii[min(shell_i, len(radii) - 1)]
            for _ in range(min(count, n - placed)):
                if dir_idx >= len(unique_dirs):
                    dir_idx = 0
                d = unique_dirs[dir_idx]
                # Add slight angular perturbation for different shells
                if shell_i > 0:
                    angle = 0.05 * shell_i + 0.02 * (dir_idx % 5)
                    cos_a = np.cos(angle)
                    sin_a = np.sin(angle)
                    # Rotate around y-axis
                    d_new = np.array([
                        d[0] * cos_a + d[2] * sin_a,
                        d[1],
                        -d[0] * sin_a + d[2] * cos_a
                    ])
                    d = d_new / (np.linalg.norm(d_new) + 1e-12)
                pts.append(d * r)
                placed += 1
                dir_idx += 1
        
        if placed == n:
            cfg = np.array(pts[:n])
            if _min_sep_ok(cfg):
                seeds.append(cfg)
    
    # Add Mackay icosahedron seed
    mckay = _mckay_icosahedron(n, base)
    if _min_sep_ok(mckay):
        seeds.append(mckay)
    
    # Add random-shell seeds
    rng = np.random.default_rng(n)
    for trial in range(6):
        pts = [np.zeros(3)]
        for i in range(n - 1):
            d = rng.normal(size=3)
            d = d / (np.linalg.norm(d) + 1e-12)
            # Radius increases with shell index
            shell = i // 12
            r = base * (1.0 + 0.6 * shell + 0.1 * trial)
            pts.append(d * r)
        cfg = np.array(pts)
        if _min_sep_ok(cfg):
            seeds.append(cfg)
    
    return seeds


def _relax_lbfgsb(pos, maxiter=500):
    """L-BFGS-B relaxation; returns (energy, positions)."""
    x0 = pos.reshape(-1)
    try:
        res = minimize(
            _energy_fn, x0, jac=_grad_fn,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "gtol": 1e-6, "disp": False},
        )
        e = _energy(res.x)
        p = res.x.reshape(-1, 3)
        if _min_sep_ok(p):
            return e, p
    except Exception:
        pass
    return _energy(x0), pos


def _kick_and_relax(pos, kick_scale, n_kicks=3):
    """Apply random kicks and relax; returns best (energy, positions)."""
    best_e = _energy(pos)
    best_p = pos.copy()
    rng = np.random.default_rng(abs(hash(tuple(np.round(pos, 4)))) % (2**31))
    
    for _ in range(n_kicks):
        if _remaining() < 5.0:
            break
        kicked = pos + rng.normal(0.0, kick_scale, size=pos.shape)
        e, p = _relax_lbfgsb(kicked, maxiter=200)
        if e < best_e and _min_sep_ok(p):
            best_e, best_p = e, p
    
    return best_e, best_p


def _basin_hopping(pos, n_iterations=20, kick_scale=0.3):
    """
    Basin hopping: repeatedly kick and relax, accepting worse moves with Metropolis criterion.
    """
    best_e = _energy(pos)
    best_p = pos.copy()
    current_e = best_e
    current_p = pos.copy()
    
    rng = np.random.default_rng(abs(hash(tuple(np.round(pos, 4)))) % (2**31))
    T = 0.5  # Temperature for Metropolis acceptance
    
    for i in range(n_iterations):
        if _remaining() < 5.0:
            break
        
        # Kick
        kicked = current_p + rng.normal(0.0, kick_scale, size=current_p.shape)
        
        # Relax
        e, p = _relax_lbfgsb(kicked, maxiter=200)
        
        if not _min_sep_ok(p):
            continue
        
        # Metropolis acceptance
        delta_e = e - current_e
        if delta_e < 0:
            accept = True
        else:
            accept = rng.random() < np.exp(-delta_e / T)
        
        if accept:
            current_e = e
            current_p = p
        
        # Update best
        if e < best_e:
            best_e = e
            best_p = p.copy()
        
        # Gradually increase kick scale
        kick_scale *= 1.05
    
    return best_e, best_p


def construct(instance):
    n = instance["n"]
    potential = instance.get("potential", "dzugutov")
    
    seeds = _icosahedral_seeds(n)
    if not seeds:
        # Last-resort compact random
        rng = np.random.default_rng(n)
        rad = 1.1 * n ** (1.0 / 3.0)
        pts = []
        while len(pts) < n:
            p = rng.uniform(-rad, rad, 3)
            if np.linalg.norm(p) <= rad and all(np.linalg.norm(p - q) >= 0.8 for q in pts):
                pts.append(p)
        seeds = [np.array(pts)]
    
    best_e = np.inf
    best_pos = seeds[0]
    t0 = time.time()
    
    # Primary: relax each seed with L-BFGS-B
    for s in seeds:
        if _remaining() < 15.0:
            break
        try:
            e, p = _relax_lbfgsb(s)
            if e < best_e and _min_sep_ok(p):
                best_e, best_pos = e, p
        except Exception:
            pass
    
    # Secondary: basin hopping from best structure
    if _remaining() > 30.0:
        try:
            e, p = _basin_hopping(best_pos, n_iterations=15, kick_scale=0.25)
            if e < best_e and _min_sep_ok(p):
                best_e, best_pos = e, p
        except Exception:
            pass
    
    # Tertiary: kick-and-relax from best structure with increasing kick scales
    kick_scales = [0.10, 0.20, 0.35, 0.50]
    for scale in kick_scales:
        if _remaining() < 10.0:
            break
        try:
            e, p = _kick_and_relax(best_pos, scale, n_kicks=2)
            if e < best_e and _min_sep_ok(p):
                best_e, best_pos = e, p
        except Exception:
            pass
    
    # Quaternary: try the built-in relax function if available
    if _remaining() > 15.0:
        try:
            relaxed = relax(best_pos, 'dzugutov')
            e = _energy(relaxed)
            if e < best_e and _min_sep_ok(relaxed):
                best_e, best_pos = e, relaxed
        except Exception:
            pass
    
    return np.asarray(best_pos, dtype=float)
# EVOLVE-BLOCK-END

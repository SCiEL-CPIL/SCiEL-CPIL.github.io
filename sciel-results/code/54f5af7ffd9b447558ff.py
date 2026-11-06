"""Dzugutov cluster global minima: parameterized icosahedral manifold + basin hopping."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient, relax

_START = time.time()
TIME_LIMIT = 310.0  # safety margin under the 330 s kill


def _remaining():
    return TIME_LIMIT - (time.time() - _START)


def _mckay_icosahedron(n):
    """Generate Mackay icosahedral cluster with n atoms using shell radii and directions."""
    # Icosahedral directions: 12 vertices of icosahedron
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Dodecahedral directions: 20 vertices
    inv_phi = 1.0 / phi
    dod_verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, inv_phi, phi], [0, inv_phi, -phi], [0, -inv_phi, phi], [0, -inv_phi, -phi],
        [inv_phi, phi, 0], [inv_phi, -phi, 0], [-inv_phi, phi, 0], [-inv_phi, -phi, 0],
        [phi, 0, inv_phi], [phi, 0, -inv_phi], [-phi, 0, inv_phi], [-phi, 0, -inv_phi],
    ], dtype=float)
    dod_verts = dod_verts / np.linalg.norm(dod_verts, axis=1, keepdims=True)
    
    # Cuboctahedral directions: 12 vertices
    cub_verts = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
    ], dtype=float)
    cub_verts = cub_verts / np.linalg.norm(cub_verts, axis=1, keepdims=True)
    
    # Standard Mackay shells: 1 + 12 + 42 + 92 + ... (icosahedral numbers)
    # For n=38: 1+12+25 (partial 3rd shell)
    # For n=45: 1+12+32 (partial)
    # For n=50: 1+12+37
    # For n=55: 1+12+42
    # For n=57: 1+12+44
    # For n=60: 1+12+47
    
    pts = [np.zeros(3)]
    placed = 1
    
    # Shell 1: 12 icosahedral vertices
    if n > 1:
        for v in ico_verts:
            if placed >= n:
                break
            pts.append(v * 1.0)
            placed += 1
    
    # Shell 2: dodecahedral directions (20) + cuboctahedral (12) = 32
    if placed < n:
        for v in dod_verts:
            if placed >= n:
                break
            pts.append(v * 1.65)
            placed += 1
        for v in cub_verts:
            if placed >= n:
                break
            pts.append(v * 1.75)
            placed += 1
    
    # Shell 3: more directions if needed
    if placed < n:
        # Use combinations of icosahedral directions with slight perturbations
        rng = np.random.default_rng(42)
        for _ in range(n - placed):
            base = ico_verts[rng.integers(0, 12)]
            perturb = rng.normal(0, 0.15, 3)
            d = base + perturb
            d = d / (np.linalg.norm(d) + 1e-12)
            pts.append(d * 2.3)
            placed += 1
    
    return np.array(pts[:n])


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


def _icosahedral_seeds(n, base=1.10):
    """Generate several icosahedral-shell starting configurations for cluster size n."""
    seeds = []
    
    # Primary: Mackay icosahedron
    mckay = _mckay_icosahedron(n)
    if _min_sep_ok(mckay):
        seeds.append(mckay)
    
    # Get direction vectors from different polyhedra
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    inv_phi = 1.0 / phi
    dod_verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, inv_phi, phi], [0, inv_phi, -phi], [0, -inv_phi, phi], [0, -inv_phi, -phi],
        [inv_phi, phi, 0], [inv_phi, -phi, 0], [-inv_phi, phi, 0], [-inv_phi, -phi, 0],
        [phi, 0, inv_phi], [phi, 0, -inv_phi], [-phi, 0, inv_phi], [-phi, 0, -inv_phi],
    ], dtype=float)
    dod_verts = dod_verts / np.linalg.norm(dod_verts, axis=1, keepdims=True)
    
    cub_verts = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
    ], dtype=float)
    cub_verts = cub_verts / np.linalg.norm(cub_verts, axis=1, keepdims=True)
    
    # Try different shell radius combinations
    shell_configs = [
        (1.0, 1.60, 2.20),
        (1.0, 1.65, 2.25),
        (1.0, 1.70, 2.30),
        (1.0, 1.55, 2.15),
        (1.05, 1.65, 2.20),
        (0.95, 1.60, 2.15),
    ]
    
    for r1, r2, r3 in shell_configs:
        pts = [np.zeros(3)]
        placed = 1
        
        # Shell 1: 12 icosahedral
        for v in ico_verts:
            if placed >= n:
                break
            pts.append(v * r1)
            placed += 1
        
        # Shell 2: 20 dodecahedral
        for v in dod_verts:
            if placed >= n:
                break
            pts.append(v * r2)
            placed += 1
        
        # Shell 3: 12 cuboctahedral + more if needed
        for v in cub_verts:
            if placed >= n:
                break
            pts.append(v * r3)
            placed += 1
        
        # Fill remaining with perturbed icosahedral
        rng = np.random.default_rng(hash((r1, r2, r3)) % (2**31))
        while placed < n:
            base_dir = ico_verts[rng.integers(0, 12)]
            perturb = rng.normal(0, 0.2, 3)
            d = base_dir + perturb
            d = d / (np.linalg.norm(d) + 1e-12)
            pts.append(d * r3)
            placed += 1
        
        cfg = np.array(pts[:n])
        if _min_sep_ok(cfg):
            seeds.append(cfg)
    
    # Random compact seeds
    rng = np.random.default_rng(n)
    for trial in range(3):
        rad = 1.2 * n ** (1.0 / 3.0)
        pts = [np.zeros(3)]
        for i in range(n - 1):
            p = rng.uniform(-rad, rad, 3)
            if np.linalg.norm(p) > rad * 0.5:
                p = p * (rad * 0.5) / np.linalg.norm(p)
            pts.append(p)
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


def _basin_hopping(pos, n_steps=50, kick_scale=0.25):
    """Basin hopping: kick, relax, accept if lower energy (with occasional uphill moves)."""
    best_e = _energy(pos)
    best_p = pos.copy()
    current_e = best_e
    current_p = pos.copy()
    rng = np.random.default_rng(12345)
    
    for step in range(n_steps):
        if _remaining() < 10.0:
            break
        
        # Random kick
        kicked = current_p + rng.normal(0.0, kick_scale, size=current_p.shape)
        
        # Relax
        e, p = _relax_lbfgsb(kicked, maxiter=300)
        
        if not _min_sep_ok(p):
            continue
        
        # Metropolis-like acceptance
        delta_e = e - current_e
        if delta_e < 0 or (step > 10 and rng.random() < np.exp(-delta_e / 0.5)):
            current_e = e
            current_p = p
        
        if e < best_e:
            best_e = e
            best_p = p
    
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
    
    # Phase 1: Relax all seeds with L-BFGS-B
    for s in seeds:
        if _remaining() < 20.0:
            break
        try:
            e, p = _relax_lbfgsb(s)
            if e < best_e and _min_sep_ok(p):
                best_e, best_pos = e, p
        except Exception:
            pass
    
    # Phase 2: Basin hopping from best structure
    if _remaining() > 30.0:
        try:
            e, p = _basin_hopping(best_pos, n_steps=30, kick_scale=0.20)
            if e < best_e and _min_sep_ok(p):
                best_e, best_pos = e, p
        except Exception:
            pass
    
    # Phase 3: Kick-and-relax with multiple scales
    if _remaining() > 15.0:
        for scale in [0.15, 0.30, 0.50]:
            if _remaining() < 10.0:
                break
            try:
                e, p = _kick_and_relax(best_pos, scale, n_kicks=3)
                if e < best_e and _min_sep_ok(p):
                    best_e, best_pos = e, p
            except Exception:
                pass
    
    # Phase 4: Try built-in relax
    if _remaining() > 10.0:
        try:
            relaxed = relax(best_pos, 'dzugutov')
            e = _energy(relaxed)
            if e < best_e and _min_sep_ok(relaxed):
                best_e, best_pos = e, relaxed
        except Exception:
            pass
    
    return np.asarray(best_pos, dtype=float)
# EVOLVE-BLOCK-END

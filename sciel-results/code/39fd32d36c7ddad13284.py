"""Dzugutov cluster global minima: Icosahedral and Decahedral seeds with Basin Hopping."""
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
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    v = v * (scale / 2.0)
    return v


def _make_dodecahedron(scale):
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    v = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, inv_phi, phi], [0, inv_phi, -phi], [0, -inv_phi, phi], [0, -inv_phi, -phi],
        [inv_phi, phi, 0], [inv_phi, -phi, 0], [-inv_phi, phi, 0], [-inv_phi, -phi, 0],
        [phi, 0, inv_phi], [phi, 0, -inv_phi], [-phi, 0, inv_phi], [-phi, 0, -inv_phi],
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / norms
    edge_to_circum = 2 * np.sin(np.pi / 5)
    v = v * (scale / edge_to_circum)
    return v


def _generate_mackay_seed(n, rng):
    """Generate a seed based on the 45-atom Mackay icosahedron."""
    # 45-atom Mackay icosahedron:
    # 1 center
    # 12 shell 1
    # 12 shell 2
    # 20 shell 3
    # We will generate a larger icosahedron and trim/extend.
    
    # Generate 13-atom icosahedron
    ico1 = _make_icosahedron(2.0)
    pts = [np.zeros(3)]
    pts.extend(ico1.tolist())
    
    # Add shell 2 (12 atoms) - approximated by dodecahedron vertices scaled
    dod = _make_dodecahedron(2.0 * 1.6) # Approximate radius
    pts.extend(dod.tolist())
    
    # Add shell 3 (20 atoms) - approximated by larger icosahedron
    ico2 = _make_icosahedron(2.0 * 2.3)
    pts.extend(ico2.tolist())
    
    pts = np.array(pts)
    
    # If n < len(pts), remove random outer atoms
    if n < len(pts):
        # Sort by distance from center, remove farthest
        dists = np.linalg.norm(pts, axis=1)
        order = np.argsort(dists)[::-1]
        pts = pts[order][n:] # This keeps the inner ones? No, order is farthest first.
        # We want to keep the inner ones.
        order_inner = np.argsort(dists)
        pts = pts[order_inner][:n]
        
    # If n > len(pts), add random atoms on outer shell
    elif n > len(pts):
        while len(pts) < n:
            # Add atom on a random direction at radius ~3.0
            d = rng.normal(0, 1, 3)
            d = d / np.linalg.norm(d)
            r = 3.0 + rng.normal(0, 0.1)
            new_pt = d * r
            # Check min separation
            if all(np.linalg.norm(new_pt - p) > 0.5 for p in pts):
                pts = np.vstack([pts, [new_pt]])
            else:
                # Retry
                continue
                
    return pts


def _generate_decahedral_seed(n, rng):
    """Generate a seed based on a decahedral cluster (5 icosahedra)."""
    # Central icosahedron (13 atoms)
    ico_c = _make_icosahedron(2.0)
    pts = [np.zeros(3)]
    pts.extend(ico_c.tolist())
    
    # 4 surrounding icosahedra
    # Directions for the 4 surrounding icosahedra (towards the vertices of a tetrahedron?)
    # Decahedral cluster usually has 5 icosahedra, one in the center and 4 around.
    # The 4 outer ones are oriented towards the 4 vertices of a tetrahedron?
    # Actually, the 5 icosahedra in a decahedron are arranged with 5-fold symmetry?
    # No, a decahedron has 5 icosahedra.
    # Let's place 4 icosahedra at the vertices of a regular tetrahedron, scaled.
    
    tet_dirs = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    tet_dirs = tet_dirs / np.linalg.norm(tet_dirs, axis=1, keepdims=True)
    
    for d in tet_dirs:
        center = d * 3.5 # Distance from origin
        ico_o = _make_icosahedron(2.0) + center
        pts.extend(ico_o.tolist())
        
    pts = np.array(pts)
    
    if n < len(pts):
        dists = np.linalg.norm(pts, axis=1)
        order_inner = np.argsort(dists)
        pts = pts[order_inner][:n]
    elif n > len(pts):
        while len(pts) < n:
            d = rng.normal(0, 1, 3)
            d = d / np.linalg.norm(d)
            r = 4.0 + rng.normal(0, 0.1)
            new_pt = d * r
            if all(np.linalg.norm(new_pt - p) > 0.5 for p in pts):
                pts = np.vstack([pts, [new_pt]])
                
    return pts


def _relax_lbfgsb(pos, maxiter=300):
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


def _kick_and_relax(pos, kick_scale, n_kicks=3, rng=None):
    if rng is None:
        rng = np.random.default_rng(abs(hash(tuple(np.round(pos, 4)))) % (2**31))
    
    best_e = _energy(pos)
    best_p = pos.copy()
    
    for _ in range(n_kicks):
        if _remaining() < 5.0:
            break
        kicked = pos + rng.normal(0.0, kick_scale, size=pos.shape)
        e, p = _relax_lbfgsb(kicked, maxiter=150)
        if e < best_e and _min_sep_ok(p):
            best_e, best_p = e, p
    
    return best_e, best_p


def construct(instance):
    n = instance["n"]
    potential = instance.get("potential", "dzugutov")
    
    rng = np.random.default_rng(n)
    
    seeds = []
    
    # Generate Mackay seeds
    for _ in range(5):
        seeds.append(_generate_mackay_seed(n, rng))
        
    # Generate Decahedral seeds
    for _ in range(5):
        seeds.append(_generate_decahedral_seed(n, rng))
        
    best_e = np.inf
    best_pos = seeds[0]
    
    # Phase 1: Relax all seeds
    for s in seeds:
        if _remaining() < 15.0:
            break
        try:
            e, p = _relax_lbfgsb(s)
            if e < best_e and _min_sep_ok(p):
                best_e, best_pos = e, p
        except Exception:
            pass
            
    # Phase 2: Basin Hopping from best
    if _remaining() > 30.0:
        kick_scales = [0.05, 0.1, 0.2, 0.3]
        for scale in kick_scales:
            if _remaining() < 10.0:
                break
            try:
                e, p = _kick_and_relax(best_pos, scale, n_kicks=5, rng=rng)
                if e < best_e and _min_sep_ok(p):
                    best_e, best_pos = e, p
            except Exception:
                pass
                
    # Phase 3: Try built-in relax
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

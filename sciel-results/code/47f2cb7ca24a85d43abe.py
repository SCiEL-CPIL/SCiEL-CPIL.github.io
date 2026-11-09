"""Morse cluster global minima: vectorized L-BFGS-B with many diverse starts."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_DEADLINE = time.time() + 305.0

def _time_left():
    return _DEADLINE - time.time()

def _vec_energy_grad(pos, rho):
    """Vectorized Morse energy and gradient."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(r, 1.0)
    iu = np.triu_indices(n, k=1)
    ri = r[iu]
    xi = rho * (1.0 - ri)
    exi = np.exp(xi)
    e = np.sum(exi * (exi - 2.0))
    dei = rho * exi * (2.0 * exi - 2.0)
    dr = diff[iu] / ri[:, None]
    g = np.zeros_like(pos)
    np.add.at(g, iu[0], dei[:, None] * dr)
    np.add.at(g, iu[1], -dei[:, None] * dr)
    return e, g

def _lbfgs(pos, rho, maxiter=300):
    """L-BFGS-B optimization with vectorized objective."""
    n = len(pos)
    flat = pos.flatten()
    def obj(flat):
        p = flat.reshape(n, 3)
        e, g = _vec_energy_grad(p, rho)
        return e, g.flatten()
    try:
        res = minimize(obj, flat, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10,
                                'maxls': 20, 'maxfun': 5000})
        return res.fun, res.x.reshape(n, 3)
    except Exception:
        return _vec_energy_grad(pos, rho)[0], pos

def _valid(pos, min_d=0.5):
    if len(pos) < 2:
        return True
    r = np.sqrt(np.sum((pos[:, None, :] - pos[None, :, :])**2, axis=2))
    np.fill_diagonal(r, np.inf)
    return np.min(r) >= min_d - 1e-9

def _fibonacci_sphere(n, radius):
    i = np.arange(n)
    phi = np.arccos(1 - 2*(i + 0.5)/n)
    theta = np.pi * (1 + np.sqrt(5.0)) * i
    return radius * np.column_stack([np.sin(phi)*np.cos(theta),
                                      np.sin(phi)*np.sin(theta),
                                      np.cos(phi)])

def _fcc(n, sp=1.0):
    pts = []
    s = int(np.ceil(n**(1/3))) + 2
    offs = np.array([[0,0,0],[.5,.5,0],[.5,0,.5],[0,.5,.5]])*sp
    for i in range(-s, s+1):
        for j in range(-s, s+1):
            for k in range(-s, s+1):
                for o in offs:
                    pts.append(np.array([i,j,k],dtype=float)*sp + o)
                if len(pts)>=n: break
            if len(pts)>=n: break
        if len(pts)>=n: break
    pts = np.array(pts[:n])
    pts -= pts.mean(axis=0)
    return pts

def _hcp(n, sp=1.0):
    pts = []
    a = sp; c = sp*np.sqrt(8/3); dz = c/3
    ml = int(np.ceil(n**(1/3)))+2
    for L in range(-ml, ml+1):
        z = L*dz
        for i in range(-ml, ml+1):
            for j in range(-ml, ml+1):
                x = a*(i + 0.5*(j%2))
                y = a*np.sqrt(3)/2*j
                if L%2: x += a*0.5
                pts.append(np.array([x,y,z]))
    pts = np.array(pts)
    idx = np.argsort(np.linalg.norm(pts, axis=1))[:n]
    pts = pts[idx]
    pts -= pts.mean(axis=0)
    return pts

def _bcc(n, sp=1.0):
    pts = []
    s = int(np.ceil(n**(1/3)))+2
    offs = np.array([[0,0,0],[.5,.5,.5]])*sp
    for i in range(-s, s+1):
        for j in range(-s, s+1):
            for k in range(-s, s+1):
                for o in offs:
                    pts.append(np.array([i,j,k],dtype=float)*sp + o)
                if len(pts)>=n: break
            if len(pts)>=n: break
        if len(pts)>=n: break
    pts = np.array(pts[:n])
    pts -= pts.mean(axis=0)
    return pts

def _icosahedral(n, sp=1.0):
    t = (1+np.sqrt(5))/2
    ico = np.array([[-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],
                    [0,-1,t],[0,1,t],[0,-1,-t],[0,1,-t],
                    [t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]], dtype=float)
    ico /= np.linalg.norm(ico[0])
    pts = [np.zeros(3)]
    for shell_r in [sp, sp*1.8, sp*2.6, sp*3.4]:
        shell = ico * shell_r
        for v in shell:
            if len(pts) < n:
                pts.append(v)
    rng = np.random.default_rng(7)
    while len(pts) < n:
        th = rng.uniform(0, 2*np.pi); ph = rng.uniform(0, np.pi)
        r = sp*(3.5 + rng.uniform(0, 0.5))
        pts.append(np.array([r*np.sin(ph)*np.cos(th), r*np.sin(ph)*np.sin(th), r*np.cos(ph)]))
    return np.array(pts[:n])

def _random_compact(n, sp=1.0, seed=0):
    rng = np.random.default_rng(seed)
    R = sp * n**(1/3) * 0.8
    pts = []
    for _ in range(n*300):
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) <= R and all(np.linalg.norm(p-q)>=0.5 for q in pts):
            pts.append(p)
        if len(pts) == n: break
    while len(pts) < n:
        th = rng.uniform(0,2*np.pi); ph = rng.uniform(0,np.pi)
        r = R*(1+rng.uniform(0,0.5))
        pts.append(np.array([r*np.sin(ph)*np.cos(th), r*np.sin(ph)*np.sin(th), r*np.cos(ph)]))
    return np.array(pts[:n])

def _perturb(pos, rng, scale=0.3):
    return pos + rng.normal(0, scale, pos.shape)

def _basin_hop(pos, rho, rng, n_hops=8, scale=0.35):
    cur = pos.copy()
    cur_e = _vec_energy_grad(cur, rho)[0]
    for _ in range(n_hops):
        if _time_left() < 4: break
        p = _perturb(cur, rng, scale)
        e, polished = _lbfgs(p, rho, maxiter=150)
        if _valid(polished) and e < cur_e:
            cur_e, cur = e, polished
    return cur_e, cur

def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))
    if instance.get("potential", "morse") != "morse":
        cfg = _random_compact(n, 1.0, 0)
        _, pos = relax(cfg, instance["potential"], rho)
        return pos

    best_e = np.inf
    best_p = None
    rng = np.random.default_rng(42)

    # Build diverse initial configs
    configs = []
    if _time_left() > 15:
        configs.append(_icosahedral(n))
    if _time_left() > 15:
        configs.append(_fcc(n))
    if _time_left() > 15:
        configs.append(_hcp(n))
    if _time_left() > 15:
        configs.append(_bcc(n))
    for s in range(8):
        if _time_left() > 15:
            configs.append(_random_compact(n, 1.0, s))

    # Process each config: L-BFGS-B then basin hopping
    for cfg in configs:
        if _time_left() < 8: break
        try:
            e, p = _lbfgs(cfg, rho, maxiter=200)
            if _valid(p) and e < best_e:
                best_e, best_p = e, p.copy()
            if _time_left() > 12:
                e2, p2 = _basin_hop(p, rho, rng, n_hops=6, scale=0.3)
                if _valid(p2) and e2 < best_e:
                    best_e, best_p = e2, p2.copy()
        except Exception:
            continue

    # More random restarts if time allows
    extra = 0
    while _time_left() > 20 and extra < 10:
        extra += 1
        cfg = _random_compact(n, 1.0, 100+extra)
        try:
            e, p = _lbfgs(cfg, rho, maxiter=150)
            if _valid(p) and e < best_e:
                best_e, best_p = e, p.copy()
            if _time_left() > 15:
                e2, p2 = _basin_hop(p, rho, rng, n_hops=5, scale=0.35)
                if _valid(p2) and e2 < best_e:
                    best_e, best_p = e2, p2.copy()
        except Exception:
            continue

    # Final polish
    if best_p is not None and _time_left() > 5:
        for _ in range(3):
            if _time_left() < 3: break
            e, p = _lbfgs(best_p, rho, maxiter=100)
            if _valid(p) and e < best_e:
                best_e, best_p = e, p

    if best_p is None:
        cfg = _random_compact(n, 1.0, 0)
        try:
            _, best_p = relax(cfg, "morse", rho)
        except Exception:
            best_p = cfg

    return best_p
# EVOLVE-BLOCK-END

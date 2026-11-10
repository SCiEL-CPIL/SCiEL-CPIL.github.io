# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = 580.0  # safety margin before 600s


def _hex_vertices(cx, cy, s, a):
    """Vertices of a regular hexagon (center cx,cy, side s, angle a in radians)."""
    th = a + np.arange(6) * (np.pi / 3.0)
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])


def _separation_ok(hexes, tol=1e-6):
    """Return True if every pair of unit hexagons is strictly separated (>tol gap
    on at least one of their 12 edge normals)."""
    n = len(hexes)
    for i in range(n):
        vi = _hex_vertices(hexes[i, 0], hexes[i, 1], 1.0, np.deg2rad(hexes[i, 2]))
        for j in range(i + 1, n):
            vj = _hex_vertices(hexes[j, 0], hexes[j, 1], 1.0, np.deg2rad(hexes[j, 2]))
            ok = False
            for k in range(6):
                p1, p2 = vi[k], vi[(k + 1) % 6]
                edge = p2 - p1
                nrm = np.array([-edge[1], edge[0]])
                nrm /= (np.linalg.norm(nrm) + 1e-15)
                a1 = np.dot(vi, nrm)
                a2 = np.dot(vj, nrm)
                if (a1.max() < a2.min() - tol) or (a2.max() < a1.min() - tol):
                    ok = True
                    break
            if not ok:
                return False
    return True


def _outer_side(hexes):
    """Minimal side length of a regular hexagon (center 0, angle 0) that contains
    all inner hexagons."""
    vs = []
    for h in hexes:
        vs.append(_hex_vertices(h[0], h[1], 1.0, np.deg2rad(h[2])))
    vs = np.vstack(vs)
    lo = 0.0
    hi = 200.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        oev = _hex_vertices(0.0, 0.0, mid, 0.0)
        ok = True
        for k in range(6):
            p1, p2 = oev[k], oev[(k + 1) % 6]
            e = p2 - p1
            rel = vs - p1
            c = e[0] * rel[:, 1] - e[1] * rel[:, 0]
            if np.any(c < -1e-6):
                ok = False
                break
        if ok:
            hi = mid
        else:
            lo = mid
    return hi


def _build_result(hexes):
    hexes = np.asarray(hexes, dtype=float).reshape(11, 3)
    side = _outer_side(hexes)
    inner = hexes.copy()
    outer = np.array([0.0, 0.0, 0.0])
    return inner, outer, float(side)


def _fallback():
    inner = np.array([
        [0.0, 0.0, 0.0],
        [-2.5, 0.0, 0.0], [2.5, 0.0, 0.0],
        [-1.25, 2.17, 0.0], [1.25, 2.17, 0.0],
        [-1.25, -2.17, 0.0], [1.25, -2.17, 0.0],
        [-3.75, 2.17, 0.0], [3.75, 2.17, 0.0],
        [-3.75, -2.17, 0.0], [3.75, -2.17, 0.0],
    ])
    return _build_result(inner)


def _flower_config():
    """Classic 1+6 flower arrangement of 7 hexagons, plus 4 more."""
    centers = [[0.0, 0.0, 0.0]]
    for i in range(6):
        angle = np.deg2rad(60.0 * i)
        cx = 2.5 * np.cos(angle)
        cy = 2.5 * np.sin(angle)
        centers.append([cx, cy, 0.0])
    for i in range(4):
        angle = np.deg2rad(60.0 * i + 30.0)
        cx = 4.33 * np.cos(angle)
        cy = 4.33 * np.sin(angle)
        centers.append([cx, cy, 0.0])
    return np.array(centers)


def _ring_config():
    """Ring of 10 hexagons around a center."""
    centers = [[0.0, 0.0, 0.0]]
    for i in range(10):
        angle = np.deg2rad(36.0 * i)
        cx = 2.5 * np.cos(angle)
        cy = 2.5 * np.sin(angle)
        centers.append([cx, cy, 0.0])
    return np.array(centers)


def _line_config():
    """Linear arrangement."""
    centers = []
    for i in range(11):
        centers.append([(i - 5) * 2.5, 0.0, 0.0])
    return np.array(centers)


def _compact_config():
    """Try to pack more compactly using hexagonal lattice."""
    centers = []
    dx = 2.5
    dy = 2.17
    positions = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1), (2, 0), (0, 2)
    ]
    for px, py in positions:
        cx = px * dx + (0.5 * dx if py % 2 else 0)
        cy = py * dy
        centers.append([cx, cy, 0.0])
    return np.array(centers)


def _objective(x):
    """Objective = outer side length (minimized). Invalid configs -> large penalty."""
    hexes = x.reshape(11, 3)
    if not np.all(np.isfinite(hexes)):
        return 1e6
    if not _separation_ok(hexes):
        return 1e6
    return _outer_side(hexes)


def _nm_refine(x, maxiter=500):
    try:
        res = minimize(_objective, x, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-7, 'fatol': 1e-8})
        return res.x
    except Exception:
        return x


def _simulated_annealing(hexes, rng, n_iter=20000, T0=0.5, T1=0.001, alpha=0.9999):
    """Simulated annealing to refine a configuration."""
    best_x = hexes.copy().ravel()
    best_f = _objective(best_x)
    cur_x = hexes.copy().ravel()
    cur_f = _objective(cur_x)
    
    T = T0
    for i in range(n_iter):
        if (time.time() - _T0) > _DEADLINE - 5:
            break
        
        # Propose a move: perturb one hexagon's position or angle
        idx = rng.integers(0, 11)
        move_type = rng.integers(0, 3)  # 0=x, 1=y, 2=angle
        step = T * 0.5
        
        new_x = cur_x.copy()
        if move_type == 0:
            new_x[idx * 3] += rng.normal(0, step)
        elif move_type == 1:
            new_x[idx * 3 + 1] += rng.normal(0, step)
        else:
            new_x[idx * 3 + 2] += rng.normal(0, step * 10)
        
        new_f = _objective(new_x)
        
        # Acceptance criterion
        if new_f < cur_f or rng.random() < np.exp(-(new_f - cur_f) / max(T, 1e-10)):
            cur_x = new_x
            cur_f = new_f
            if cur_f < best_f:
                best_f = cur_f
                best_x = cur_x.copy()
        
        T *= alpha
    
    return best_x, best_f


def hexagon_packing_11():
    best_x = None
    best_f = 1e6
    
    # Try multiple initial configurations
    configs = [
        _flower_config(),
        _ring_config(),
        _line_config(),
        _compact_config(),
    ]
    
    rng = np.random.default_rng(12345)
    
    for cfg in configs:
        if (time.time() - _T0) > _DEADLINE - 10:
            break
        
        # Add small random perturbations to break symmetry
        for trial in range(5):
            if (time.time() - _T0) > _DEADLINE - 10:
                break
            
            perturbed = cfg.copy()
            perturbed[:, :2] += rng.normal(0, 0.1, (11, 2))
            perturbed[:, 2] = rng.choice([0.0, 30.0, 60.0], size=11) * 0.1
            
            init_x = perturbed.ravel()
            cand = _nm_refine(init_x, maxiter=300)
            cf = _objective(cand)
            
            if cf < best_f:
                best_f = cf
                best_x = cand
    
    # If we have time, do simulated annealing from the best config
    if best_x is not None and (time.time() - _T0) < _DEADLINE - 60:
        sa_x, sa_f = _simulated_annealing(best_x.reshape(11, 3), rng, n_iter=50000)
        if sa_f < best_f:
            best_f = sa_f
            best_x = sa_x
    
    # If we have time, do more random restarts with SA
    if best_f > 3.5 and (time.time() - _T0) < _DEADLINE - 120:
        for restart in range(5):
            if (time.time() - _T0) > _DEADLINE - 10:
                break
            
            # Random initial configuration
            centers = rng.uniform(-5, 5, (11, 2))
            angles = rng.choice([0.0, 30.0, 60.0], size=11)
            init_hexes = np.column_stack([centers, angles])
            
            # Quick NM refinement first
            cand = _nm_refine(init_hexes.ravel(), maxiter=200)
            cf = _objective(cand)
            
            # Then SA
            if cf < 100:
                sa_x, sa_f = _simulated_annealing(cand.reshape(11, 3), rng, n_iter=30000)
                if sa_f < best_f:
                    best_f = sa_f
                    best_x = sa_x
    
    # Final refinement if we have time
    if best_x is not None and (time.time() - _T0) < _DEADLINE - 30:
        refined = _nm_refine(best_x, maxiter=1000)
        rf = _objective(refined)
        if rf < best_f:
            best_f = rf
            best_x = refined
    
    if best_x is None or not np.all(np.isfinite(best_x)):
        return _fallback()
        
    return _build_result(best_x)


# EVOLVE-BLOCK-END

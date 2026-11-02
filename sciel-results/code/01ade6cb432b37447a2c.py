"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26
DEADLINE_FRAC = 0.92


def _radii_from_centers(c, iters=200, tol=1e-10):
    """Greatest fixed point of T(r)_i = min(wall_i, min_j (d_ij - r_j)).
    Returns optimal radii for fixed centers."""
    n = c.shape[0]
    wall = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]),
                      np.minimum(c[:, 1], 1.0 - c[:, 1]))
    wall = np.maximum(wall, 0.0)
    diff = c[:, None, :] - c[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    # Set diagonal to inf so it doesn't constrain
    np.fill_diagonal(d, np.inf)
    r = wall.copy()
    for _ in range(iters):
        # T(r)_i = min(wall_i, min_j (d_ij - r_j))
        # min_j (d_ij - r_j) = d_ij - r_j, take min over j
        mr = d - r[None, :]  # (n, n): d_ij - r_j
        mr = np.min(mr, axis=1)
        new_r = np.minimum(wall, mr)
        # Ensure non-negative
        new_r = np.maximum(new_r, 0.0)
        if np.max(np.abs(new_r - r)) < tol:
            r = new_r
            break
        r = new_r
    return r


def _objective(c_flat):
    """Return (sum_radii, radii, centers)."""
    c = c_flat.reshape(-1, 2)
    # Keep centers in [0,1]
    c = np.clip(c, 0.0, 1.0)
    r = _radii_from_centers(c)
    return float(np.sum(r)), r, c


def construct():
    t0 = time.time()
    deadline = t0 + 120.0 * DEADLINE_FRAC

    best_val = -1.0
    best_c = None
    best_r = None

    def _update(c, r, val):
        nonlocal best_val, best_c, best_r
        if val > best_val:
            best_val = val
            best_c = c.copy()
            best_r = r.copy()

    rng = np.random.default_rng(2024)

    # --- Generate diverse starts ---
    starts = []

    # 5x5 grid + center
    side = 5
    g = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
    g.append((0.5, 0.5))
    starts.append(np.array(g[:N], dtype=float))

    # 6x5 grid
    g2 = [((i + 0.5) / 6.0, (j + 0.5) / 5.0) for i in range(6) for j in range(5)]
    starts.append(np.array(g2[:N], dtype=float))

    # 7x4 grid
    g3 = [((i + 0.5) / 7.0, (j + 0.5) / 4.0) for i in range(7) for j in range(4)]
    starts.append(np.array(g3[:N], dtype=float))

    # Hexagonal-ish packing
    hex_pts = []
    row_h = 0.22
    for row in range(5):
        y = 0.12 + row * row_h
        xoff = 0.0 if row % 2 == 0 else 0.11
        for col in range(6):
            x = 0.11 + xoff + col * 0.145
            if 0 < x < 1:
                hex_pts.append((x, y))
    starts.append(np.array(hex_pts[:N], dtype=float))

    # Ring + center
    ring = [(0.5 + 0.38 * np.cos(a), 0.5 + 0.38 * np.sin(a))
            for a in np.linspace(0, 2 * np.pi, 25, endpoint=False)]
    ring.append((0.5, 0.5))
    starts.append(np.array(ring, dtype=float))

    # Random starts
    for _ in range(20):
        starts.append(rng.uniform(0.05, 0.95, size=(N, 2)))

    # Evaluate starts
    for c0 in starts:
        if time.time() > deadline:
            break
        val, r, c = _objective(c0.ravel())
        _update(c, r, val)

    # Rank starts
    sv = []
    for c0 in starts:
        v, _, _ = _objective(c0.ravel())
        sv.append(v)
    order = np.argsort(-np.array(sv))

    def obj_neg(c_flat):
        v, _, _ = _objective(c_flat)
        return -v

    # --- Local optimization from best starts ---
    for idx in order:
        if time.time() > deadline:
            break
        c0 = starts[idx].copy().ravel()

        # Nelder-Mead
        try:
            res = minimize(obj_neg, c0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8,
                                    'adaptive': True})
            if time.time() > deadline:
                break
            val, r, c = _objective(res.x)
            _update(c, r, val)
        except Exception:
            pass

        # Powell from Nelder-Mead result
        if 'res' in dir() and time.time() < deadline:
            try:
                res2 = minimize(obj_neg, res.x, method='Powell',
                                options={'maxiter': 40, 'xtol': 1e-7, 'ftol': 1e-9})
                if time.time() > deadline:
                    break
                val, r, c = _objective(res2.x)
                _update(c, r, val)
            except Exception:
                pass

    # --- Differential evolution for global exploration ---
    if time.time() < deadline - 15:
        try:
            bounds_de = [(0.0, 1.0)] * (2 * N)
            de_res = differential_evolution(obj_neg, bounds_de, seed=42,
                                            maxiter=40, popsize=20,
                                            tol=1e-7, polish=False,
                                            init='sobol')
            if time.time() > deadline:
                break
            val, r, c = _objective(de_res.x)
            _update(c, r, val)

            # Polish
            try:
                res3 = minimize(obj_neg, de_res.x, method='Nelder-Mead',
                                options={'maxiter': 200, 'adaptive': True})
                if time.time() > deadline:
                    break
                val, r, c = _objective(res3.x)
                _update(c, r, val)
            except Exception:
                pass
        except Exception:
            pass

    # --- Second DE with different seed if time remains ---
    if time.time() < deadline - 10:
        try:
            de_res2 = differential_evolution(obj_neg, bounds_de, seed=99,
                                             maxiter=30, popsize=15,
                                             tol=1e-7, polish=False,
                                             init='latinhypercube')
            if time.time() > deadline:
                break
            val, r, c = _objective(de_res2.x)
            _update(c, r, val)
        except Exception:
            pass

    # --- Finalize ---
    if best_c is None:
        c = starts[0]
        val, r, c = _objective(c.ravel())
        best_c, best_r, best_val = c, r, val

    # Slight shrink for strict validity
    best_r = best_r * 0.99995
    best_c = np.clip(best_c, 0.0, 1.0)
    best_r = np.maximum(best_r, 1e-6)

    out = np.column_stack([best_c, best_r])
    return out

"""LJ-55 global minimum: start from Mackay icosahedron and polish."""
import time
import numpy as np

N = 55
START = time.time()
DEADLINE = 118.0   # safety margin vs 120 s evaluator kill

try:
    from science_chemistry import relax as _ext_relax
except Exception:
    _ext_relax = None


def _energy(pos):
    d = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(r2, 1.0)
    r6 = 1.0 / r2
    r6 = r6 * r6 * r6
    r12 = r6 * r6
    np.fill_diagonal(r12, 0.0)
    np.fill_diagonal(r6, 0.0)
    return 4.0 * (r12.sum() - r6.sum())


def _energy_grad(pos):
    d = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(r2, 1.0)
    r6 = 1.0 / r2
    r6 = r6 * r6 * r6
    r12 = r6 * r6
    # force on i from j: F = 4*(24*r12 - 6*r6)/r2 * (pi - pj)
    coef = 4.0 * (24.0 * r12 - 6.0 * r6) / r2
    np.fill_diagonal(coef, 0.0)
    grad = -np.einsum('ij,ijk->ik', coef, d)
    return 4.0 * (r12.sum() - r6.sum()), grad


def _lbfgs(pos, f0, iters=3000, tol=1e-14):
    m = 20
    x = np.array(pos, dtype=float)
    f = f0
    g, _ = _energy_grad(x)
    s, y, rho = [], [], []
    for _ in range(iters):
        if time.time() > DEADLINE:
            break
        # two-loop recursion
        q = g.copy()
        al = [0.0] * len(s)
        for i in range(len(s) - 1, -1, -1):
            al[i] = rho[i] * np.dot(s[i], q)
            q = q - al[i] * y[i]
        r = np.dot(y[-1], s[-1]) / (np.dot(y[-1], y[-1]) + 1e-30) if s else 1.0
        r = r * q
        for i in range(len(s)):
            be = rho[i] * np.dot(y[i], r)
            r = r + (al[i] - be) * s[i]
        if np.dot(g, r) > 0:
            r = -g
        # backtracking line search
        step = 1.0
        gnorm = np.linalg.norm(g)
        for _ in range(40):
            xn = x + step * r
            fn, gn = _energy_grad(xn)
            if fn < f + 1e-9 * step * np.dot(g, r):
                break
            step *= 0.5
        else:
            break
        if np.linalg.norm(gn) < tol or step * np.linalg.norm(r) < 1e-14:
            x, f, g = xn, fn, gn
            break
        ds = xn - x
        dy = gn - g
        sy = np.dot(ds, dy)
        if sy > 1e-16:
            s.append(ds); y.append(dy); rho.append(1.0 / sy)
            if len(s) > m:
                s.pop(0); y.pop(0); rho.pop(0)
        x, f, g = xn, fn, gn
    return x, f


def _valid(pos):
    if pos.shape != (N, 3):
        return False
    d = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(r2, 1.0)
    return bool(np.all(r2 >= 0.25))


def _icosahedron():
    t = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    v = v / np.linalg.norm(v[0])
    r1 = 1.055          # inner shell radius
    r2 = 2.062          # outer shell radius
    inner = v * r1
    outer = v * r2
    return np.vstack([inner, outer])


def construct():
    best = _icosahedron()
    best_e = _energy(best)
    if _ext_relax is not None:
        try:
            r = _ext_relax(best.copy())
            if _valid(r):
                r = np.asarray(r, dtype=float)
                e = _energy(r)
                if e < best_e:
                    best, best_e = r, e
        except Exception:
            pass

    # Tight custom polish on the best structure found.
    try:
        x, e = _lbfgs(best, best_e)
        if _valid(x) and e < best_e:
            best, best_e = x, e
    except Exception:
        pass

    # If still not great, try a small randomised restart of the icosahedron.
    if best_e > -279.0 and time.time() < DEADLINE - 5.0:
        rng = np.random.default_rng(0)
        for _ in range(3):
            cand = _icosahedron()
            cand += rng.normal(0.0, 0.02, cand.shape)
            ce = _energy(cand)
            if ce < best_e:
                best, best_e = cand, ce
            try:
                x, e = _lbfgs(best, best_e, iters=2000)
                if _valid(x) and e < best_e:
                    best, best_e = x, e
            except Exception:
                pass
            if best_e <= -279.24:
                break

    return np.asarray(best, dtype=float)

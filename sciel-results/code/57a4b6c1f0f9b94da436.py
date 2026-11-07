import numpy as np
import time

_DEADLINE = 150.0  # safety margin under 180s / 210s kill


def _hermite_coeffs(n):
    """Physicists' Hermite H_n coefficients, highest degree first."""
    def sub(a, b):
        return [x - y for x, y in zip(a, b)]

    def scale(a, s):
        return [x * s for x in a]

    H0 = [1.0]
    H1 = [2.0, 0.0]
    if n == 0:
        return H0
    if n == 1:
        return H1
    Hnm1, Hn = H0, H1
    for k in range(1, n):
        term1 = scale(Hn, 2.0) + [0.0]        # 2x * H_k
        term2 = [0.0] + scale(Hnm1, 2.0 * k)  # 2k * H_{k-1}, degree-aligned
        Hn1 = sub(term1, term2)
        Hnm1, Hn = Hn, Hn1
    return Hn


_H0c = _hermite_coeffs(0)
_H4c = _hermite_coeffs(4)
_H8c = _hermite_coeffs(8)
_H12c = _hermite_coeffs(12)
_H0_0 = float(_H0c[-1])
_H4_0 = float(_H4c[-1])
_H8_0 = float(_H8c[-1])
_H12_0 = float(_H12c[-1])


def _align(coeffs, val):
    out = np.zeros(13)
    d = len(coeffs) - 1
    for i, a in enumerate(coeffs):
        out[12 - d + i] = val * a
    return out


def _build_q(c0, c1, c2):
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    g = (_align(_H0c, c0) + _align(_H4c, c1)
         + _align(_H8c, c2) + _align(_H12c, c3))
    return g[:-2]  # q = g / x^2


def _largest_sign_change_root(q):
    roots = np.roots(q)
    real_roots = []
    for rt in roots:
        if abs(rt.imag) < 1e-6 * max(1.0, abs(rt.real)):
            real_roots.append(float(rt.real))
    if not real_roots:
        return 0.0, False

    def qeval(x):
        return np.polyval(q, x)

    candidates = []
    for rt in real_roots:
        for d in (1e-4, 1e-3, 1e-2, 1e-1):
            dd = d * max(1.0, abs(rt))
            lo, hi = rt - dd, rt + dd
            qlo, qhi = qeval(lo), qeval(hi)
            if qlo * qhi < 0:
                for _ in range(50):  # bisection refine
                    mid = 0.5 * (lo + hi)
                    qm = qeval(mid)
                    if qlo * qm <= 0:
                        hi = mid
                    else:
                        lo, qlo = mid, qm
                candidates.append(0.5 * (lo + hi))
                break
    if not candidates:
        return 0.0, False
    r = max(candidates)
    return max(0.0, r), True


def _objective(c):
    q = _build_q(float(c[0]), float(c[1]), float(c[2]))
    r, ok = _largest_sign_change_root(q)
    return r if ok else 1e9


def construct():
    t0 = time.time()
    try:
        seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
        best_c = seed.copy()
        r0, ok0 = _largest_sign_change_root(_build_q(*seed))
        best_r = r0 if ok0 else 1e9

        def _try_update(x):
            nonlocal best_c, best_r
            r, ok = _largest_sign_change_root(
                _build_q(float(x[0]), float(x[1]), float(x[2])))
            if ok and r < best_r:
                best_r = r
                best_c = np.array([float(x[0]), float(x[1]), float(x[2])])

        _try_update(seed)
        bounds = [(-6.0, 6.0), (-3.0, 3.0), (-0.05, 0.05)]

        # Global search
        if time.time() - t0 < _DEADLINE - 40:
            try:
                from scipy.optimize import differential_evolution
                for init in ('sobol', 'latinhypercube'):
                    try:
                        res = differential_evolution(
                            _objective, bounds, seed=12345, maxiter=150,
                            popsize=15, tol=1e-8, polish=True,
                            init=init, updating='deferred')
                        if res.x is not None:
                            _try_update(res.x)
                        break
                    except Exception:
                        continue
            except Exception:
                pass

        # Local polish from diverse starts
        rng = np.random.default_rng(777)
        starts = [best_c]
        for _ in range(14):
            starts.append(rng.uniform([b[0] for b in bounds],
                                      [b[1] for b in bounds]))
        try:
            from scipy.optimize import minimize
            for s in starts:
                if time.time() - t0 > _DEADLINE - 15:
                    break
                try:
                    res = minimize(_objective, s, method='Nelder-Mead',
                                   options={'maxiter': 1500, 'xatol': 1e-9,
                                            'fatol': 1e-11, 'adaptive': True})
                    if res.x is not None:
                        _try_update(res.x)
                except Exception:
                    continue
        except Exception:
            pass

        if best_r >= 1e9:  # no valid config found; fall back to seed
            return seed
        return best_c
    except Exception:
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

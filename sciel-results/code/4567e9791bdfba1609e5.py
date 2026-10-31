import time
import numpy as np

N = 38
_DEADLINE = 145.0
_T0 = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _lj_energy(pos):
    pos = np.asarray(pos, dtype=float)
    d = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(r2, np.inf)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** -3
    return 0.5 * np.sum(4.0 * (r6 * r6 - r6))


def _fcc_truncated_octahedron():
    a = 1.12
    pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                base = np.array([i, j, k]) * a
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    pts.append(base + a * np.array([dx, dy, dz]))
    pts = np.array(pts)
    d = np.sum(pts ** 2, axis=1)
    idx = np.argsort(d)[:N]
    return pts[idx]


def _try_relax(pos):
    try:
        import science_chemistry
        if hasattr(science_chemistry, 'relax'):
            out = np.asarray(science_chemistry.relax(np.asarray(pos, dtype=float)), dtype=float)
            if out.shape == (N, 3) and np.all(np.isfinite(out)):
                return out
    except Exception:
        pass
    return None


def _enforce_min_sep(pos, min_sep=0.5):
    pos = pos.copy()
    for _ in range(100):
        d = pos[:, None, :] - pos[None, :, :]
        r2 = np.einsum('ijk,ijk->ij', d, d)
        np.fill_diagonal(r2, np.inf)
        rmin2 = r2.min()
        if rmin2 >= min_sep ** 2:
            break
        i, j = np.unravel_index(int(np.argmin(r2)), r2.shape)
        if i == j:
            break
        vec = pos[i] - pos[j]
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            vec = np.array([1.0, 0.0, 0.0]); norm = 1.0
        push = (min_sep - np.sqrt(r2[i, j])) * 0.5 * vec / norm
        pos[i] += push
        pos[j] -= push
    return pos


def construct():
    best = None
    best_e = np.inf

    def consider(pos):
        nonlocal best, best_e
        try:
            pos = np.asarray(pos, dtype=float)
            if pos.shape != (N, 3) or not np.all(np.isfinite(pos)):
                return
            e = _lj_energy(pos)
            if np.isfinite(e) and e < best_e:
                best_e = e
                best = pos.copy()
        except Exception:
            pass

    fcc = _fcc_truncated_octahedron()
    consider(fcc)

    relaxed = _try_relax(fcc)
    if relaxed is not None:
        consider(relaxed)

    rng = np.random.default_rng(0)
    base = best if best is not None else fcc
    for k in range(300):
        if _remaining() < 15.0:
            break
        step = 0.05 * (0.98 ** k) + 0.005
        pert = base + rng.normal(0.0, step, size=(N, 3))
        r = _try_relax(pert)
        if r is not None:
            consider(r)
        if best is not None and best is not base:
            base = best

    if best is not None:
        rp = _try_relax(best)
        if rp is not None:
            consider(rp)

    if best is None:
        best = fcc

    return np.asarray(_enforce_min_sep(best, 0.5), dtype=float)

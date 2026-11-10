# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

_DEADLINE = time.time() + 600.0
_MARGIN = 20.0  # safety margin seconds


def _count_convex7_fast_proxy(pts: np.ndarray) -> int:
    """Fast proxy: count convex 4-gons. Fewer convex 4-gons correlates with fewer
    convex 7-gons; O(n^4) vectorized, fast for n=33."""
    n = len(pts)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    P = pts[order]
    total = 0
    for i in range(n):
        Pi = P[i]
        for j in range(i + 1, n):
            Pj = P[j]
            for k in range(j + 1, n):
                if k + 1 >= n:
                    continue
                Pk = P[k]
                # cross(Pi,Pj,Pk)
                c1 = (Pj[0] - Pi[0]) * (Pk[1] - Pi[1]) - (Pj[1] - Pi[1]) * (Pk[0] - Pi[0])
                if c1 <= 1e-9:
                    continue
                l = np.arange(k + 1, n)
                Pl = P[l]
                # cross(Pj,Pk,Pl)
                c2 = (Pk[0] - Pj[0]) * (Pl[:, 1] - Pj[1]) - (Pk[1] - Pj[1]) * (Pl[:, 0] - Pj[0])
                # cross(Pk,Pl,Pi)
                c3 = (Pl[:, 0] - Pk[0]) * (Pi[1] - Pk[1]) - (Pl[:, 1] - Pk[1]) * (Pi[0] - Pk[0])
                # cross(Pl,Pi,Pj)
                c4 = (Pi[0] - Pl[:, 0]) * (Pj[1] - Pi[1]) - (Pi[1] - Pl[:, 1]) * (Pj[0] - Pi[0])
                total += int(np.sum((c2 > 1e-9) & (c3 > 1e-9) & (c4 > 1e-9)))
    return total


def _generate_cup_cap(n1: int, n2: int, cup_exp: float, cap_exp: float,
                      shift: float) -> np.ndarray:
    """Cup-cap: n1 on lower convex cup chain, n2 on upper concave cap chain."""
    cup_x = np.linspace(shift, 1.0 - shift, n1)
    cup_yv = 0.05 + 0.5 * (cup_x ** cup_exp)
    cap_x = np.linspace(shift + 0.05, 1.0 - shift - 0.05, n2)
    cap_yv = 0.95 - 0.5 * ((1.0 - cap_x) ** cap_exp)
    pts = np.vstack([np.column_stack([cup_x, cup_yv]),
                     np.column_stack([cap_x, cap_yv])])
    return pts


def _is_valid(pts: np.ndarray) -> bool:
    if pts.shape != (33, 2):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if not np.all((pts >= -1e-6) & (pts <= 1 + 1e-6)):
        return False
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    if d2.min() < 1e-12:
        return False
    n = 33
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                tri = abs(pts[i, 0] * (pts[j, 1] - pts[k, 1])
                          + pts[j, 0] * (pts[k, 1] - pts[i, 1])
                          + pts[k, 0] * (pts[i, 1] - pts[j, 1]))
                if tri < 1e-9:
                    return False
    return True


def _exact_count_convex7(pts: np.ndarray) -> int:
    """Exact count matching the evaluator's convex-chain DP."""
    n = len(pts)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    P = pts[order]
    total = 0
    for i in range(n):
        later = np.arange(i + 1, n)
        m = len(later)
        if m < 6:
            continue
        pi = P[i]
        ang = np.arctan2(P[later, 1] - pi[1], P[later, 0] - pi[0])
        later = later[np.argsort(ang)]
        m = len(later)
        Lpts = P[later]
        v1 = Lpts - pi
        CA = np.zeros((m, m))
        CA[:m - 1, 1:m] = v1[:-1, 0:1] * v1[1:, 1] - v1[:-1, 1:2] * v1[1:, 0]
        idx = list(range(m))
        for combo in itertools.combinations(idx, 6):
            a1, a2, a3, a4, a5, a6 = combo
            if CA[a1, a2] <= 1e-9:
                continue
            ok = True
            for t in range(1, 5):
                aa, bb, cc = combo[t], combo[t + 1], combo[t + 2]
                cr = ((Lpts[bb, 0] - Lpts[aa, 0]) * (Lpts[cc, 1] - Lpts[aa, 1])
                      - (Lpts[bb, 1] - Lpts[aa, 1]) * (Lpts[cc, 0] - Lpts[aa, 0]))
                if cr <= 1e-9:
                    ok = False
                    break
            if not ok:
                continue
            cr56i = ((Lpts[a6, 0] - Lpts[a5, 0]) * (pi[1] - Lpts[a5, 1])
                     - (Lpts[a6, 1] - Lpts[a5, 1]) * (pi[0] - Lpts[a5, 0]))
            if cr56i <= 1e-9:
                continue
            cr6i1 = ((pi[0] - Lpts[a6, 0]) * (Lpts[a1, 1] - Lpts[a6, 1])
                     - (pi[1] - Lpts[a6, 1]) * (Lpts[a1, 0] - Lpts[a6, 0]))
            if cr6i1 <= 1e-9:
                continue
            total += 1
    return total


def _optimize() -> np.ndarray:
    start = time.time()
    best_proxy = None
    best_proxy_val = np.inf

    param_grid = []
    for cup_exp in [1.5, 2.0, 2.5, 3.0]:
        for cap_exp in [1.5, 2.0, 2.5, 3.0]:
            for n1 in [16, 17]:
                n2 = 33 - n1
                param_grid.append((n1, n2, cup_exp, cap_exp))

    for (n1, n2, cup_exp, cap_exp) in param_grid:
        if time.time() > _DEADLINE - _MARGIN:
            break
        for shift in [0.02, 0.05, 0.1]:
            pts = _generate_cup_cap(n1, n2, cup_exp, cap_exp, shift)
            if not _is_valid(pts):
                continue
            proxy = _count_convex7_fast_proxy(pts)
            if proxy < best_proxy_val:
                best_proxy_val = proxy
                best_proxy = pts.copy()

    if best_proxy is None:
        grid = np.array([[(i / 5.0), (j / 5.0)] for i in range(6) for j in range(6)])
        best_proxy = grid[:33].copy()
        best_proxy_val = _count_convex7_fast_proxy(best_proxy)

    # Hill-climb on proxy
    pts = best_proxy.copy()
    cur_val = best_proxy_val
    np.random.seed(1)
    step = 0.02
    while time.time() < _DEADLINE - _MARGIN and step > 1e-5:
        improved = False
        for _ in range(300):
            if time.time() > _DEADLINE - _MARGIN:
                break
            idx = np.random.randint(0, 33)
            old = pts[idx].copy()
            pts[idx] = old + np.random.normal(0, step, 2)
            pts[idx] = np.clip(pts[idx], 0.0, 1.0)
            if not _is_valid(pts):
                pts[idx] = old
                continue
            val = _count_convex7_fast_proxy(pts)
            if val < cur_val:
                cur_val = val
                improved = True
            else:
                pts[idx] = old
        if not improved:
            step *= 0.7

    # Exact evaluation of a few candidates
    candidates = [pts.copy(), best_proxy.copy()]
    grid = np.array([[(i / 5.0), (j / 5.0)] for i in range(6) for j in range(6)])
    candidates.append(grid[:33].copy())

    best_exact = None
    best_exact_val = np.inf
    for cand in candidates:
        if time.time() > _DEADLINE - _MARGIN:
            break
        if not _is_valid(cand):
            continue
        ev = _exact_count_convex7(cand)
        if ev < best_exact_val:
            best_exact_val = ev
            best_exact = cand.copy()

    if best_exact is not None:
        return best_exact
    return pts


def erdos_szekeres_problem7() -> np.ndarray:
    return _optimize()


# EVOLVE-BLOCK-END

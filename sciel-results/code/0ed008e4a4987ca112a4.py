# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 600.0
_t0 = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _t0)


def _normalize(P):
    c = P.mean(axis=0)
    Q = P - c
    dists = np.linalg.norm(Q, axis=1)
    m = dists.mean()
    if m <= 0.01:
        return Q
    return Q / m


def _is_valid_convex(P):
    n = P.shape[0]
    if n < 3:
        return False
    if not np.all(np.isfinite(P)):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(P[i] - P[j]) < 1e-9:
                return False
    signs = []
    for i in range(n):
        a = P[i]
        b = P[(i + 1) % n]
        c = P[(i + 2) % n]
        cr = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(cr) < 1e-12:
            return False
        signs.append(1 if cr > 0 else -1)
    if len(set(signs)) != 1:
        return False
    return True


def _vertex_score(v, P):
    d = np.sort(np.linalg.norm(P - v, axis=1))
    if d[0] < 0.05:
        return 0.0
    gaps = d[1:] - d[:-1]
    k = int(np.argmin(gaps))
    D = 0.5 * (d[k] + d[k + 1])
    if D < 0.1:
        return 0.0
    if k == 0:
        four = d[0:4]
    elif k == 17:
        four = d[15:19]
    else:
        four = d[k - 1:k + 3]
    ratio = np.maximum(four / D, D / four)
    raw = np.mean(ratio)
    if not np.isfinite(raw) or raw <= 0.0:
        return 0.0
    return float(min(1.0, 1.0 / raw))


def _metric(P):
    s = np.empty(P.shape[0])
    for i in range(P.shape[0]):
        s[i] = _vertex_score(P[i], P)
    return float(np.min(s))


def _make_convex(P, max_iter=200):
    n = P.shape[0]
    adjusted = P.copy()
    for _ in range(max_iter):
        if _is_valid_convex(adjusted):
            return adjusted
        for i in range(n):
            p0 = adjusted[(i - 1) % n]
            p1 = adjusted[i]
            p2 = adjusted[(i + 1) % n]
            e1 = p1 - p0
            e2 = p2 - p1
            cross = e1[0] * e2[1] - e1[1] * e2[0]
            if abs(cross) < 1e-10:
                centroid = adjusted.mean(axis=0)
                direction = p1 - centroid
                norm = np.linalg.norm(direction)
                if norm > 0:
                    adjusted[i] = p1 + 0.02 * direction / norm
        adjusted += np.random.randn(n, 2) * 0.001
    return adjusted


def _local_refine(P, max_iter=500, rng=None):
    n = P.shape[0]
    if rng is None:
        rng = np.random.default_rng()
    best = P.copy()
    best_score = _metric(best)
    for it in range(max_iter):
        if _remaining() < 30.0:
            break
        vi = rng.integers(0, n)
        step = 0.01 * (1.0 + 0.5 * np.cos(0.1 * it))
        delta = rng.normal(0, step, 2)
        candidate = best.copy()
        candidate[vi] += delta
        candidate = _normalize(candidate)
        if _is_valid_convex(candidate):
            cand_score = _metric(candidate)
            if cand_score > best_score:
                best = candidate
                best_score = cand_score
    return best, best_score


def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Start with regular polygon
    P_reg = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    P_reg = _normalize(P_reg)
    best_P = P_reg.copy()
    best_score = _metric(P_reg)
    
    rng = np.random.default_rng(42)
    
    # Multi-scale random search in coordinate space
    scales = [0.05, 0.03, 0.02, 0.01]
    for scale in scales:
        if _remaining() < 60.0:
            break
        n_iter = 3000 if scale >= 0.03 else 5000
        for it in range(n_iter):
            if _remaining() < 60.0:
                break
            # Perturb angles and radii
            angle_perturb = rng.normal(0, scale, n)
            radius_perturb = 1.0 + rng.normal(0, scale * 0.5, n)
            new_angles = angles + angle_perturb
            new_P = np.stack([radius_perturb * np.cos(new_angles),
                              radius_perturb * np.sin(new_angles)], axis=1)
            new_P = _normalize(new_P)
            
            if not _is_valid_convex(new_P):
                new_P = _make_convex(new_P)
                if not _is_valid_convex(new_P):
                    continue
            
            score = _metric(new_P)
            if score > best_score:
                best_score = score
                best_P = new_P.copy()
    
    # Local refinement from best
    if _remaining() > 60.0:
        refined, ref_score = _local_refine(best_P, max_iter=2000, rng=rng)
        if ref_score > best_score:
            best_P = refined
            best_score = ref_score
    
    # Final safety check
    if not _is_valid_convex(best_P):
        best_P = P_reg.copy()
    
    return best_P


# EVOLVE-BLOCK-END

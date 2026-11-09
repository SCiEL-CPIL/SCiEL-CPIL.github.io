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
    # No repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(P[i] - P[j]) < 1e-9:
                return False
    # Cross products all same sign and nonzero
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
    d = np.linalg.norm(P - v, axis=1)
    # Set self-distance to inf so it's sorted last
    idx = np.argmin(d)
    d[idx] = np.inf
    d_sorted = np.sort(d)
    if d_sorted[0] < 0.05:
        return 0.0
    gaps = d_sorted[1:] - d_sorted[:-1]
    k = int(np.argmin(gaps))
    D = 0.5 * (d_sorted[k] + d_sorted[k + 1])
    if D < 0.1:
        return 0.0
    if k == 0:
        four = d_sorted[0:4]
    elif k == 17:
        four = d_sorted[15:19]
    else:
        four = d_sorted[k - 1:k + 3]
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


def _harmonic_perturb(base_angles, amplitudes, n):
    """Create a polygon by perturbing the regular polygon with harmonic modes."""
    k = np.arange(n)
    # Amplitudes for each harmonic mode
    angles = base_angles.copy()
    radii = np.ones(n)
    for mode, amp in amplitudes.items():
        if mode == 0:
            radii *= (1.0 + amp)
        else:
            # Add angular and radial perturbation for this mode
            angles += amp * np.cos(mode * k + mode * np.pi / n)
            radii *= (1.0 + amp * 0.5 * np.sin(mode * k))
    return np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)


def _local_refine_targeted(P, max_iter=3000, rng=None):
    """Local refinement that targets the worst-scoring vertex."""
    n = P.shape[0]
    if rng is None:
        rng = np.random.default_rng()
    best = P.copy()
    best_score = _metric(best)
    step = 0.02
    for it in range(max_iter):
        if _remaining() < 30.0:
            break
        # Find worst vertex
        scores = np.array([_vertex_score(best[i], best) for i in range(n)])
        vi = int(np.argmin(scores))
        # Try to improve this vertex
        improved = False
        for trial in range(5):
            delta = rng.normal(0, step, 2)
            candidate = best.copy()
            candidate[vi] += delta
            candidate = _normalize(candidate)
            if _is_valid_convex(candidate):
                cand_score = _metric(candidate)
                if cand_score > best_score:
                    best = candidate
                    best_score = cand_score
                    improved = True
                    break
        if not improved:
            step *= 0.9
        else:
            step = min(step * 1.05, 0.05)
    return best, best_score


def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Start with regular polygon
    P_reg = np.stack([np.cos(base_angles), np.sin(base_angles)], axis=1)
    P_reg = _normalize(P_reg)
    best_P = P_reg.copy()
    best_score = _metric(P_reg)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Harmonic search - explore low-frequency perturbations
    # Use modes 1-5 with small amplitudes
    modes_to_try = [1, 2, 3, 4, 5]
    for mode in modes_to_try:
        if _remaining() < 60.0:
            break
        for amp_scale in [0.01, 0.02, 0.03, 0.05]:
            if _remaining() < 60.0:
                break
            n_iter = 200
            for it in range(n_iter):
                if _remaining() < 60.0:
                    break
                # Random amplitudes for all modes up to current mode
                amplitudes = {}
                for m in range(1, mode + 1):
                    amplitudes[m] = rng.normal(0, amp_scale / np.sqrt(m))
                # Also try negative
                if rng.random() < 0.5:
                    for m in amplitudes:
                        amplitudes[m] = -amplitudes[m]
                
                new_P = _harmonic_perturb(base_angles, amplitudes, n)
                new_P = _normalize(new_P)
                
                if not _is_valid_convex(new_P):
                    continue
                
                score = _metric(new_P)
                if score > best_score:
                    best_score = score
                    best_P = new_P.copy()
    
    # Phase 2: Multi-scale random search in angle/radius space
    scales = [0.03, 0.02, 0.01, 0.005]
    for scale in scales:
        if _remaining() < 60.0:
            break
        n_iter = 3000 if scale >= 0.02 else 5000
        for it in range(n_iter):
            if _remaining() < 60.0:
                break
            angle_perturb = rng.normal(0, scale, n)
            radius_perturb = 1.0 + rng.normal(0, scale * 0.5, n)
            new_angles = base_angles + angle_perturb
            new_P = np.stack([radius_perturb * np.cos(new_angles),
                              radius_perturb * np.sin(new_angles)], axis=1)
            new_P = _normalize(new_P)
            
            if not _is_valid_convex(new_P):
                continue
            
            score = _metric(new_P)
            if score > best_score:
                best_score = score
                best_P = new_P.copy()
    
    # Phase 3: Targeted local refinement
    if _remaining() > 60.0:
        refined, ref_score = _local_refine_targeted(best_P, max_iter=5000, rng=rng)
        if ref_score > best_score:
            best_P = refined
            best_score = ref_score
    
    # Final safety check
    if not _is_valid_convex(best_P):
        best_P = P_reg.copy()
    
    return best_P


# EVOLVE-BLOCK-END

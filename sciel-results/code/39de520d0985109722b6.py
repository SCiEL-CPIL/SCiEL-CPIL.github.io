# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 630.0
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


def _compute_scores(P):
    n = P.shape[0]
    s = np.empty(n)
    for i in range(n):
        s[i] = _vertex_score(P[i], P)
    return s


def _soft_min(scores, beta=50.0):
    """Soft minimum using log-sum-exp trick."""
    if len(scores) == 0:
        return 0.0
    m = np.min(scores)
    if m < -1e6:
        return m
    # Soft min: -1/beta * log(sum(exp(-beta * s)))
    exp_terms = np.exp(-beta * (scores - m))
    smin = m - np.log(np.sum(exp_terms)) / beta
    return float(smin)


def _local_refine_smooth(P, max_iter=2000, rng=None):
    """Local refinement using smooth surrogate (soft-min) for faster gradient-like search."""
    n = P.shape[0]
    if rng is None:
        rng = np.random.default_rng()
    best = P.copy()
    best_score = _metric(best)
    
    # Start with moderate step, gradually decrease
    step = 0.015
    beta = 30.0  # soft-min sharpness
    
    for it in range(max_iter):
        if _remaining() < 45.0:
            break
        
        # Find worst vertex (use actual min for targeting)
        scores = _compute_scores(best)
        vi = int(np.argmin(scores))
        
        # Try multiple perturbations around worst vertex
        improved = False
        for trial in range(8):
            delta = rng.normal(0, step, 2)
            candidate = best.copy()
            candidate[vi] += delta
            candidate = _normalize(candidate)
            if not _is_valid_convex(candidate):
                continue
            cand_score = _metric(candidate)
            if cand_score > best_score + 1e-10:
                best = candidate
                best_score = cand_score
                improved = True
                break
        
        if not improved:
            step *= 0.92
            beta = max(beta * 0.95, 10.0)
        else:
            step = min(step * 1.03, 0.03)
    
    return best, best_score


def _local_refine_targeted(P, max_iter=3000, rng=None):
    """Local refinement that targets the worst-scoring vertex with adaptive step."""
    n = P.shape[0]
    if rng is None:
        rng = np.random.default_rng()
    best = P.copy()
    best_score = _metric(best)
    step = 0.02
    for it in range(max_iter):
        if _remaining() < 45.0:
            break
        # Find worst vertex
        scores = _compute_scores(best)
        vi = int(np.argmin(scores))
        # Try to improve this vertex
        improved = False
        for trial in range(6):
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


def _coordinate_descent(P, max_iter=5000, rng=None):
    """Coordinate descent: optimize one vertex at a time with line search."""
    n = P.shape[0]
    if rng is None:
        rng = np.random.default_rng()
    best = P.copy()
    best_score = _metric(best)
    
    for it in range(max_iter):
        if _remaining() < 40.0:
            break
        
        improved_any = False
        for vi in range(n):
            if _remaining() < 40.0:
                break
            
            # Try small perturbations of vertex vi
            for step in [0.005, 0.01, 0.02]:
                for _ in range(4):
                    delta = rng.normal(0, step, 2)
                    candidate = best.copy()
                    candidate[vi] += delta
                    candidate = _normalize(candidate)
                    if not _is_valid_convex(candidate):
                        continue
                    cand_score = _metric(candidate)
                    if cand_score > best_score + 1e-10:
                        best = candidate
                        best_score = cand_score
                        improved_any = True
                        break
                if improved_any:
                    break
        
        if not improved_any:
            break
    
    return best, best_score


def _harmonic_perturb(base_angles, amplitudes, n):
    """Create a polygon by perturbing the regular polygon with harmonic modes."""
    k = np.arange(n)
    angles = base_angles.copy()
    radii = np.ones(n)
    for mode, amp in amplitudes.items():
        if mode == 0:
            radii *= (1.0 + amp)
        else:
            angles += amp * np.cos(mode * k + mode * np.pi / n)
            radii *= (1.0 + amp * 0.5 * np.sin(mode * k))
    return np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)


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
    modes_to_try = [1, 2, 3, 4, 5, 6]
    for mode in modes_to_try:
        if _remaining() < 120.0:
            break
        for amp_scale in [0.005, 0.01, 0.015, 0.02, 0.03, 0.04]:
            if _remaining() < 120.0:
                break
            n_iter = 150
            for it in range(n_iter):
                if _remaining() < 120.0:
                    break
                amplitudes = {}
                for m in range(1, mode + 1):
                    amplitudes[m] = rng.normal(0, amp_scale / np.sqrt(m))
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
    scales = [0.02, 0.015, 0.01, 0.005, 0.003]
    for scale in scales:
        if _remaining() < 90.0:
            break
        n_iter = 4000 if scale >= 0.01 else 6000
        for it in range(n_iter):
            if _remaining() < 90.0:
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
    
    # Phase 3: Targeted local refinement (coarse)
    if _remaining() > 60.0:
        refined, ref_score = _local_refine_targeted(best_P, max_iter=4000, rng=rng)
        if ref_score > best_score:
            best_P = refined
            best_score = ref_score
    
    # Phase 4: Coordinate descent for finer optimization
    if _remaining() > 45.0:
        refined, ref_score = _coordinate_descent(best_P, max_iter=4000, rng=rng)
        if ref_score > best_score:
            best_P = refined
            best_score = ref_score
    
    # Phase 5: Final high-precision polish
    if _remaining() > 30.0:
        refined, ref_score = _local_refine_smooth(best_P, max_iter=2000, rng=rng)
        if ref_score > best_score:
            best_P = refined
            best_score = ref_score
    
    # Final safety check
    if not _is_valid_convex(best_P):
        best_P = P_reg.copy()
    
    return best_P


# EVOLVE-BLOCK-END

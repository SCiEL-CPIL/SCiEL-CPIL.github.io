import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# Precompute triangle indices once
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]
_NTRIS = len(_IDX)

def compute_score(points):
    """Compute min triangle area / convex hull area, vectorized."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0

    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    return min_area / hull_area

def neg_score(x):
    points = x.reshape((N, 2))
    return -compute_score(points)

def _smooth_min_score(x, beta=50.0):
    """Smooth approximation of -min_area/hull_area for gradient-based optimization."""
    points = x.reshape((N, 2))
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 1e6
    except Exception:
        return 1e6

    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Smooth min using log-sum-exp
    min_area = np.min(areas)
    if min_area < 1e-10:
        return 1e6
    
    # LSE approximation of min: min(a) ≈ -1/beta * log(sum(exp(-b_i)))
    # But we want to maximize min, so minimize -min
    # Use: -min ≈ (1/beta)*log(sum(exp(-beta*a_i))) ... not quite
    # Better: smooth_min = (1/beta) * log(sum(exp(-beta * a_i)))  gives approx min
    # We want to minimize -smooth_min / hull_area
    # Actually let's use: penalty = (1/beta)*log(sum(exp(-beta*(a_i - min_a)))) + min_a
    # This is the softmin
    shifted = areas - min_area
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_area + np.log(np.sum(np.exp(-beta * shifted))) / beta
    # Wait, that's not right. softmin(a) = -1/beta * log(sum(exp(-beta*a)))
    # Let me redo: softmin = -1/beta * log(sum(exp(-beta * a_i)))
    log_sum_exp = np.log(np.sum(np.exp(-beta * areas)))
    soft_min = -log_sum_exp / beta
    
    return soft_min / hull_area

def _smooth_neg_score(x, beta=50.0):
    return _smooth_min_score(x, beta)

def _smooth_neg_score_beta200(x):
    return _smooth_min_score(x, 200.0)

def construct():
    start_time = time.time()
    deadline = 142.0  # Safety margin before 150s kill

    best_points = None
    best_score = -float('inf')
    rng = np.random.default_rng(42)

    def try_update(points):
        nonlocal best_points, best_score
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    def remaining():
        return deadline - (time.time() - start_time)

    # --- Generate diverse initial points with emphasis on ring structures ---
    inits = []

    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(p_reg)

    # 2. Two concentric rings: 7 outer, 7 inner with various offsets
    a_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    for ratio in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for offset in [0, np.pi/7, np.pi/14, np.pi/3.5]:
            a_in = a_out + offset
            p_out = np.column_stack((np.cos(a_out), np.sin(a_out)))
            p_in = np.column_stack((ratio * np.cos(a_in), ratio * np.sin(a_in)))
            inits.append(np.vstack((p_out, p_in)))

    # 3. Two rings: 6 outer, 8 inner
    a_out6 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    a_in8 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    for ratio in [0.4, 0.5, 0.6, 0.7]:
        for offset in [0, np.pi/8, np.pi/16]:
            a_in = a_in8 + offset
            p_out = np.column_stack((np.cos(a_out6), np.sin(a_out6)))
            p_in = np.column_stack((ratio * np.cos(a_in), ratio * np.sin(a_in)))
            inits.append(np.vstack((p_out, p_in)))

    # 4. Two rings: 8 outer, 6 inner
    a_out8 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    a_in6 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    for ratio in [0.4, 0.5, 0.6, 0.7]:
        for offset in [0, np.pi/6, np.pi/12]:
            a_in = a_in6 + offset
            p_out = np.column_stack((np.cos(a_out8), np.sin(a_out8)))
            p_in = np.column_stack((ratio * np.cos(a_in), ratio * np.sin(a_in)))
            inits.append(np.vstack((p_out, p_in)))

    # 5. Three rings: 5, 5, 4
    r1, r2, r3 = 1.0, 0.6, 0.25
    a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    a2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi/5
    a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4
    inits.append(np.vstack((
        np.column_stack((r1*np.cos(a1), r1*np.sin(a1))),
        np.column_stack((r2*np.cos(a2), r2*np.sin(a2))),
        np.column_stack((r3*np.cos(a3), r3*np.sin(a3)))
    )))

    # 6. Three rings: 6, 4, 4
    a1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    a2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4
    a3 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    inits.append(np.vstack((
        np.column_stack((np.cos(a1), np.sin(a1))),
        np.column_stack((0.55*np.cos(a2), 0.55*np.sin(a2))),
        np.column_stack((0.25*np.cos(a3), 0.25*np.sin(a3)))
    )))

    # 7. Ellipse-based
    for rx, ry in [(1.2, 0.8), (1.5, 0.65), (1.3, 0.75)]:
        a = np.linspace(0, 2 * np.pi, N, endpoint=False)
        inits.append(np.column_stack((rx*np.cos(a), ry*np.sin(a))))

    # 8. Random in disk
    for _ in range(15):
        r = np.sqrt(rng.random(N))
        th = rng.random(N) * 2 * np.pi
        inits.append(np.column_stack((r*np.cos(th), r*np.sin(th))))

    # 9. Jittered regular polygon
    for _ in range(10):
        inits.append(p_reg + rng.normal(0, 0.12, (N, 2)))

    # 10. Random in square
    for _ in range(8):
        inits.append(rng.uniform(-1, 1, (N, 2)))

    # 11. Hexagonal lattice subset
    pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            pts.append([x, y])
    pts = np.array(pts)
    # Select 14 points closest to center
    dists = np.sqrt(pts[:,0]**2 + pts[:,1]**2)
    idx = np.argsort(dists)[:N]
    inits.append(pts[idx])

    # 12. Center + 13 around
    center = np.array([[0, 0]])
    a13 = np.linspace(0, 2*np.pi, 13, endpoint=False)
    ring13 = np.column_stack((np.cos(a13), np.sin(a13)))
    inits.append(np.vstack((center, ring13)))

    # 13. Two centers + 12 around
    c1 = np.array([[-0.3, 0]])
    c2 = np.array([[0.3, 0]])
    a12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    ring12 = np.column_stack((np.cos(a12), np.sin(a12)))
    inits.append(np.vstack((c1, c2, ring12)))

    # Evaluate all inits
    for p in inits:
        try_update(p)

    # Sort by score, keep top for refinement
    scored = [(compute_score(p), p) for p in inits]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Phase 1: Nelder-Mead from top initial points
    for init_score, init_points in scored[:8]:
        if remaining() < 12:
            break
        x0 = init_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'ftol': 1e-10})
            new_p = res.x.reshape((N, 2))
            try_update(new_p)
        except Exception:
            pass

    # Phase 2: Powell refinement from best
    if best_points is not None and remaining() > 18:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Powell',
                           options={'maxiter': 800, 'xtol': 1e-9, 'ftol': 1e-11})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    # Phase 3: Basin hopping with varied parameters
    if best_points is not None and remaining() > 20:
        for step, T in [(0.4, 0.8), (0.6, 1.2), (0.8, 1.5), (0.3, 0.6), (0.5, 1.0)]:
            if remaining() < 10:
                break
            x0 = best_points.flatten()
            try:
                res = minimize(neg_score, x0, method='Nelder-Mead',
                               options={'maxiter': 400, 'xatol': 1e-8, 'ftol': 1e-10})
                # Basin hopping manually
                for _ in range(6):
                    if remaining() < 8:
                        break
                    perturbation = rng.normal(0, step, (N, 2))
                    x_pert = res.x + perturbation.flatten()
                    res2 = minimize(neg_score, x_pert, method='Nelder-Mead',
                                    options={'maxiter': 200, 'xatol': 1e-8, 'ftol': 1e-10})
                    if res2.fun < res.fun:
                        res = res2
                    else:
                        # Accept with probability exp(-delta/T)
                        delta = res2.fun - res.fun
                        if rng.random() < np.exp(-delta / max(T, 1e-10)):
                            res = res2
                try_update(res.x.reshape((N, 2)))
            except Exception:
                pass

    # Phase 4: BFGS with smooth approximation from best
    if best_points is not None and remaining() > 15:
        x0 = best_points.flatten()
        try:
            res = minimize(_smooth_neg_score, x0, method='L-BFGS-B',
                           options={'maxiter': 300, 'gtol': 1e-8})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    # Phase 5: Tight Nelder-Mead polish
    if best_points is not None and remaining() > 10:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead',
                           options={'maxiter': 6000, 'xatol': 1e-10, 'ftol': 1e-12})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    # Phase 6: Powell final polish
    if best_points is not None and remaining() > 8:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Powell',
                           options={'maxiter': 1500, 'xtol': 1e-10, 'ftol': 1e-12})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    # Phase 7: More basin hopping from different starting points
    if remaining() > 12:
        for init_score, init_points in scored[2:6]:
            if remaining() < 8:
                break
            x0 = init_points.flatten()
            try:
                res = minimize(neg_score, x0, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-8, 'ftol': 1e-10})
                for _ in range(4):
                    if remaining() < 6:
                        break
                    perturbation = rng.normal(0, 0.5, (N, 2))
                    x_pert = res.x + perturbation.flatten()
                    res2 = minimize(neg_score, x_pert, method='Nelder-Mead',
                                    options={'maxiter': 150, 'xatol': 1e-8, 'ftol': 1e-10})
                    if res2.fun < res.fun:
                        res = res2
                try_update(res.x.reshape((N, 2)))
            except Exception:
                pass

    # Phase 8: Final ultra-tight polish
    if best_points is not None and remaining() > 5:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-11, 'ftol': 1e-13})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    if best_points is None:
        return p_reg
    return best_points

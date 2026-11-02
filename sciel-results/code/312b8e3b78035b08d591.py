import numpy as np
import time
from scipy.optimize import minimize, basinhopping, differential_evolution
from scipy.spatial import ConvexHull

N = 14

_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

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

def soft_min_score(x, tau):
    """
    Compute -soft_min(areas) / hull_area.
    soft_min(A) = -tau * log(sum(exp(-A/tau)))
    """
    points = x.reshape((N, 2))
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
    
    areas = np.maximum(areas, 1e-12)
        
    scaled = -areas / tau
    max_scaled = np.max(scaled)
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    soft_min_val = -tau * (max_scaled + np.log(sum_exp))
    
    return -soft_min_val / hull_area

def smooth_score_and_grad(x, beta):
    """
    Returns (-score, -grad_score) using log-sum-exp soft-min with explicit gradient.
    """
    points = x.reshape((N, 2))
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    if np.any(areas < 1e-12):
        return 0.0, np.zeros_like(x)
    
    A_max = np.max(areas)
    shifted = areas - A_max
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    
    if sum_exp < 1e-300:
        return 0.0, np.zeros_like(x)
    
    weights = exp_vals / sum_exp
    
    soft_min_val = -A_max - (1.0 / beta) * np.log(sum_exp)
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0, np.zeros_like(x)
    except Exception:
        return 0.0, np.zeros_like(x)
    
    score = soft_min_val / hull_area
    
    signs = np.sign(cross)
    
    d_cross_di_x = v1[:, 1] - v2[:, 1]
    d_cross_di_y = v2[:, 0] - v1[:, 0]
    d_cross_dj_x = v2[:, 1]
    d_cross_dj_y = -v2[:, 0]
    d_cross_dk_x = -v1[:, 1]
    d_cross_dk_y = v1[:, 0]
    
    w = weights * signs * 0.5
    
    grad_points = np.zeros_like(points)
    
    dA_di = w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    dA_dj = w[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y])
    dA_dk = w[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y])
    
    np.add.at(grad_points, _I, dA_di)
    np.add.at(grad_points, _J, dA_dj)
    np.add.at(grad_points, _K, dA_dk)
    
    grad_score = grad_points / hull_area
    
    return -score, -grad_score.flatten()

def active_set_score_and_grad(x, tol=1e-6):
    """
    Score/gradient focusing on triangles near the minimum (active set).
    At the optimum, multiple triangles achieve min area simultaneously.
    """
    points = x.reshape((N, 2))
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0, np.zeros_like(x)
    
    # Active set: triangles within tol of minimum
    active = areas <= (min_area + tol)
    if not np.any(active):
        active = np.zeros(len(areas), dtype=bool)
        active[np.argmin(areas)] = True
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0, np.zeros_like(x)
    except Exception:
        return 0.0, np.zeros_like(x)
    
    score = min_area / hull_area
    
    # Gradient: only from active triangles, equal weight
    signs = np.sign(cross)
    
    d_cross_di_x = v1[:, 1] - v2[:, 1]
    d_cross_di_y = v2[:, 0] - v1[:, 0]
    d_cross_dj_x = v2[:, 1]
    d_cross_dj_y = -v2[:, 0]
    d_cross_dk_x = -v1[:, 1]
    d_cross_dk_y = v1[:, 0]
    
    n_active = np.sum(active)
    w = np.zeros(len(areas))
    w[active] = signs[active] * 0.5 / n_active
    
    grad_points = np.zeros_like(points)
    
    dA_di = w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    dA_dj = w[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y])
    dA_dk = w[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y])
    
    np.add.at(grad_points, _I, dA_di)
    np.add.at(grad_points, _J, dA_dj)
    np.add.at(grad_points, _K, dA_dk)
    
    grad_score = grad_points / hull_area
    
    return -score, -grad_score.flatten()

def heptagon_7_7_configuration(params):
    r_in, th_off = params
    angles_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    angles_in = th_off + np.linspace(0, 2*np.pi, 7, endpoint=False)
    pts_in = r_in * np.column_stack((np.cos(angles_in), np.sin(angles_in)))
    return np.vstack((pts_out, pts_in))

def neg_heptagon_7_7_score(params):
    try:
        points = heptagon_7_7_configuration(params)
        return -compute_score(points)
    except Exception:
        return 1e10

def ring_configuration(params):
    r1, r2, r3, th1, th2, th3, n1, n2, n3 = params
    n1 = int(round(n1))
    n2 = int(round(n2))
    n3 = int(round(n3))
    
    while n1 + n2 + n3 > 14:
        if n3 >= 1: n3 -= 1
        elif n2 >= 1: n2 -= 1
        else: n1 -= 1
    while n1 + n2 + n3 < 14:
        n3 += 1
    
    points = []
    if n1 > 0:
        angles = th1 + np.linspace(0, 2*np.pi, n1, endpoint=False)
        points.append(np.column_stack((r1*np.cos(angles), r1*np.sin(angles))))
    if n2 > 0:
        angles = th2 + np.linspace(0, 2*np.pi, n2, endpoint=False)
        points.append(np.column_stack((r2*np.cos(angles), r2*np.sin(angles))))
    if n3 > 0:
        angles = th3 + np.linspace(0, 2*np.pi, n3, endpoint=False)
        points.append(np.column_stack((r3*np.cos(angles), r3*np.sin(angles))))
    
    if not points:
        return np.zeros((N, 2))
    
    result = np.vstack(points)
    if len(result) < N:
        extra = np.zeros((N - len(result), 2))
        result = np.vstack([result, extra])
    elif len(result) > N:
        result = result[:N]
    return result

def ring_score(params):
    try:
        points = ring_configuration(params)
        return compute_score(points)
    except Exception:
        return 0.0

def neg_ring_score(params):
    return -ring_score(params)

def coordinate_descent_refine(points, max_iters=500, tol=1e-12):
    """
    Coordinate descent: for each point, try small perturbations in x and y
    to improve the min triangle area / hull area ratio.
    """
    current = points.copy()
    current_score = compute_score(current)
    
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            for axis in range(2):
                for step in step_sizes:
                    trial = current.copy()
                    trial[i, axis] += step
                    s = compute_score(trial)
                    if s > current_score + tol:
                        current = trial
                        current_score = s
                        improved = True
                        break
                    trial = current.copy()
                    trial[i, axis] -= step
                    s = compute_score(trial)
                    if s > current_score + tol:
                        current = trial
                        current_score = s
                        improved = True
                        break
        if not improved:
            break
    
    return current, current_score

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin for 150s limit
    
    best_points = None
    best_score = -float('inf')
    rng = np.random.default_rng(12345)
    
    def try_update(points):
        nonlocal best_points, best_score
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    # Phase 0: Quick grid search for 7+7 heptagon structure
    if time.time() - start_time < deadline - 60:
        r_vals = np.arange(0.3, 0.86, 0.03)
        th_vals = np.arange(0, np.pi/7 + 1e-9, np.pi/35)
        
        best_77_params = None
        best_77_score = -1e10
        
        for r in r_vals:
            for th in th_vals:
                s = compute_score(heptagon_7_7_configuration([r, th]))
                if s > best_77_score:
                    best_77_score = s
                    best_77_params = [r, th]
        
        if best_77_params is not None:
            x0 = np.array(best_77_params)
            try:
                res = minimize(neg_heptagon_7_7_score, x0, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
                best_77_points = heptagon_7_7_configuration(res.x)
                try_update(best_77_points)
            except Exception:
                pass

    # Phase 1: Quick DE on 3-ring parameters (reduced)
    if time.time() - start_time < deadline - 50:
        bounds = [
            (0.1, 1.5), (0.05, 1.2), (0.0, 0.8),
            (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi),
            (1, 14), (1, 14), (1, 14)
        ]
        try:
            res = differential_evolution(
                neg_ring_score, bounds,
                seed=rng.integers(0, 2**31),
                maxiter=40, tol=1e-6, popsize=10,
                mutation=(0.3, 1.2), recombination=0.9,
                polish=True, init='latinhypercube'
            )
            if res.fun < 0:
                points = ring_configuration(res.x)
                try_update(points)
        except Exception:
            pass

    # Phase 2: Multi-start soft-min annealing (reduced starts)
    if best_points is not None and time.time() - start_time < deadline - 40:
        starts = [best_points.copy()]
        
        for _ in range(6):
            jittered = best_points + rng.normal(0, 0.15, (N, 2))
            starts.append(jittered)
        
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        starts.append(np.column_stack((np.cos(angles), np.sin(angles))))
        
        a_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
        a_in = a_out + np.pi/7
        starts.append(np.vstack((
            np.column_stack((np.cos(a_out), np.sin(a_out))),
            np.column_stack((0.5*np.cos(a_in), 0.5*np.sin(a_in)))
        )))
        
        taus = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003]
        
        for start in starts:
            if time.time() - start_time > deadline - 30:
                break
            
            x0 = start.flatten()
            current_x = x0
            for tau in taus:
                if time.time() - start_time > deadline - 25:
                    break
                try:
                    res = minimize(
                        soft_min_score, current_x, args=(tau,),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-7}
                    )
                    current_x = res.x
                    new_p = current_x.reshape((N, 2))
                    try_update(new_p)
                except Exception:
                    pass

    # Phase 3: Gradient-based refinement with active set
    if best_points is not None and time.time() - start_time < deadline - 20:
        x0 = best_points.flatten()
        
        # First: soft-min gradient annealing
        betas = [20.0, 100.0, 300.0, 600.0]
        current_x = x0
        
        for beta in betas:
            if time.time() - start_time > deadline - 15:
                break
            try:
                res = minimize(
                    lambda x: smooth_score_and_grad(x, beta),
                    current_x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-9}
                )
                current_x = res.x
                new_p = current_x.reshape((N, 2))
                try_update(new_p)
            except Exception:
                pass
        
        # Then: active set gradient refinement
        if time.time() - start_time < deadline - 10:
            current_x = best_points.flatten()
            for _ in range(3):
                if time.time() - start_time > deadline - 8:
                    break
                try:
                    res = minimize(
                        lambda x: active_set_score_and_grad(x, tol=1e-4),
                        current_x,
                        jac=True,
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9}
                    )
                    current_x = res.x
                    new_p = current_x.reshape((N, 2))
                    try_update(new_p)
                except Exception:
                    pass

    # Phase 4: Basin hopping (shorter)
    if best_points is not None and time.time() - start_time < deadline - 12:
        x0 = best_points.flatten()
        try:
            res = basinhopping(
                neg_score, x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11},
                minsteps=3, T=0.5, stepsize=0.3,
                interval=5, disp=False
            )
            new_p = res.x.reshape((N, 2))
            try_update(new_p)
        except Exception:
            pass

    # Phase 5: Powell refinement
    if best_points is not None and time.time() - start_time < deadline - 8:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Powell',
                          options={'maxiter': 800, 'xtol': 1e-11, 'ftol': 1e-13})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    # Phase 6: High-precision Nelder-Mead
    if best_points is not None and time.time() - start_time < deadline - 4:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 15000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass

    # Phase 7: Coordinate descent polish
    if best_points is not None and time.time() - start_time < deadline - 1:
        refined, refined_score = coordinate_descent_refine(best_points, max_iters=200, tol=1e-13)
        if refined_score > best_score:
            best_points = refined
            best_score = refined_score

    return best_points

import numpy as np
import time
from scipy.optimize import minimize, basinhopping, differential_evolution
from scipy.spatial import ConvexHull

N = 14

# Precompute triangle indices
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
    This is differentiable and approximates -min(A) as tau -> 0.
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

def heptagon_7_7_configuration(params):
    """
    Generate points from two nested heptagons (7 outer, 7 inner).
    params: [r_inner, theta_offset]
    Outer ring is fixed at radius 1.0 and angle 0.
    """
    r_in, th_off = params
    
    # Outer 7 points
    angles_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    
    # Inner 7 points
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
    """
    Generate points from ring parameters.
    params: [r1, r2, r3, r4, theta_offset1, theta_offset2, theta_offset3, theta_offset4,
             n1, n2, n3, n4] where n1+n2+n3+n4=14, r_i >= 0, theta offsets in [0, 2pi)
    """
    r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4 = params
    n1 = int(round(n1))
    n2 = int(round(n2))
    n3 = int(round(n3))
    n4 = int(round(n4))
    
    # Ensure we have exactly 14 points
    while n1 + n2 + n3 + n4 > 14:
        if n4 >= 1:
            n4 -= 1
        elif n3 >= 1:
            n3 -= 1
        elif n2 >= 1:
            n2 -= 1
        else:
            n1 -= 1
    while n1 + n2 + n3 + n4 < 14:
        n4 += 1
    
    points = []
    
    # Ring 1 (outermost)
    if n1 > 0:
        angles = th1 + np.linspace(0, 2*np.pi, n1, endpoint=False)
        points.append(np.column_stack((r1*np.cos(angles), r1*np.sin(angles))))
    
    # Ring 2
    if n2 > 0:
        angles = th2 + np.linspace(0, 2*np.pi, n2, endpoint=False)
        points.append(np.column_stack((r2*np.cos(angles), r2*np.sin(angles))))
    
    # Ring 3
    if n3 > 0:
        angles = th3 + np.linspace(0, 2*np.pi, n3, endpoint=False)
        points.append(np.column_stack((r3*np.cos(angles), r3*np.sin(angles))))
    
    # Ring 4 (innermost)
    if n4 > 0:
        angles = th4 + np.linspace(0, 2*np.pi, n4, endpoint=False)
        points.append(np.column_stack((r4*np.cos(angles), r4*np.sin(angles))))
    
    if not points:
        return np.zeros((N, 2))
    
    result = np.vstack(points)
    
    # Pad or trim to exactly N points if needed
    if len(result) < N:
        # Add points at center
        extra = np.zeros((N - len(result), 2))
        result = np.vstack([result, extra])
    elif len(result) > N:
        result = result[:N]
    
    return result

def ring_score(params):
    """Score for ring parameterization."""
    try:
        points = ring_configuration(params)
        return compute_score(points)
    except Exception:
        return 0.0

def neg_ring_score(params):
    return -ring_score(params)

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
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

    # Phase 0: Optimize the 7+7 Nested Heptagon Structure
    # This is a highly symmetric candidate for N=14
    if time.time() - start_time < deadline - 25:
        # Grid search + Local Optimization for [r_inner, theta_offset]
        # r_inner in [0.2, 0.9], theta_offset in [0, pi/7] (symmetry period)
        
        # Initial coarse grid to find good basin
        r_vals = np.linspace(0.25, 0.85, 15)
        th_vals = np.linspace(0, np.pi/7, 15)
        
        best_77_params = None
        best_77_score = -1e10
        
        for r in r_vals:
            for th in th_vals:
                s = compute_score(heptagon_7_7_configuration([r, th]))
                if s > best_77_score:
                    best_77_score = s
                    best_77_params = [r, th]
        
        if best_77_params is not None:
            # Refine with Nelder-Mead
            x0 = np.array(best_77_params)
            try:
                res = minimize(neg_heptagon_7_7_score, x0, method='Nelder-Mead',
                               options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12})
                best_77_points = heptagon_7_7_configuration(res.x)
                try_update(best_77_points)
            except Exception:
                pass
            
            # Also try a few local perturbations of the best grid point
            for _ in range(10):
                perturbed = best_77_params + rng.normal(0, 0.03, 2)
                perturbed[0] = np.clip(perturbed[0], 0.1, 1.0)
                perturbed[1] = np.clip(perturbed[1], 0, 2*np.pi)
                try:
                    res = minimize(neg_heptagon_7_7_score, perturbed, method='Nelder-Mead',
                                   options={'maxiter': 300})
                    p = heptagon_7_7_configuration(res.x)
                    try_update(p)
                except Exception:
                    pass

    # Phase 1: Differential Evolution on 4-ring parameters
    # Parameters: [r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4]
    bounds = [
        (0.3, 1.5),  # r1 (outermost)
        (0.1, 1.0),  # r2
        (0.0, 0.7),  # r3
        (0.0, 0.4),  # r4 (innermost)
        (0, 2*np.pi),  # th1
        (0, 2*np.pi),  # th2
        (0, 2*np.pi),  # th3
        (0, 2*np.pi),  # th4
        (1, 14),  # n1
        (1, 14),  # n2
        (1, 14),  # n3
        (1, 14),  # n4
    ]
    
    # Try differential evolution with multiple restarts
    for restart in range(6):
        if time.time() - start_time > deadline - 30:
            break
        
        try:
            res = differential_evolution(
                neg_ring_score,
                bounds,
                seed=rng.integers(0, 2**31),
                maxiter=100,
                tol=1e-7,
                popsize=14,
                mutation=(0.3, 1.2),
                recombination=0.9,
                polish=True,
                init='latinhypercube'
            )
            
            if res.success or res.fun < 0:
                best_params = res.x
                points = ring_configuration(best_params)
                s = try_update(points)
        except Exception:
            pass
    
    # Phase 2: Multi-start L-BFGS-B on full 28D space from good ring configs
    if best_points is not None and time.time() - start_time < deadline - 35:
        # Generate diverse starting points
        starts = [best_points]
        
        # Add jittered versions
        for _ in range(8):
            jittered = best_points + rng.normal(0, 0.06, (N, 2))
            starts.append(jittered)
        
        # Add some structured starts
        # Regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        starts.append(np.column_stack((np.cos(angles), np.sin(angles))))
        
        # Two rings 7+7
        a_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
        a_in = a_out + np.pi/7
        starts.append(np.vstack((
            np.column_stack((np.cos(a_out), np.sin(a_out))),
            np.column_stack((0.5*np.cos(a_in), 0.5*np.sin(a_in)))
        )))
        
        # Three rings 5+5+4
        a1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
        a2 = a1 + np.pi/5
        a3 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        starts.append(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.6*np.cos(a2), 0.6*np.sin(a2))),
            np.column_stack((0.25*np.cos(a3), 0.25*np.sin(a3)))
        )))
        
        # Four rings 4+4+3+3
        a1 = np.linspace(0, 2*np.pi, 4, endpoint=False)
        a2 = a1 + np.pi/4
        a3 = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
        a4 = a3 + np.pi/3
        starts.append(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.65*np.cos(a2), 0.65*np.sin(a2))),
            np.column_stack((0.35*np.cos(a3), 0.35*np.sin(a3))),
            np.column_stack((0.1*np.cos(a4), 0.1*np.sin(a4)))
        )))
        
        # Additional structured starts
        # 6+5+3 rings
        a1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
        a3 = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
        starts.append(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.55*np.cos(a2), 0.55*np.sin(a2))),
            np.column_stack((0.2*np.cos(a3), 0.2*np.sin(a3)))
        )))
        
        # 8+6 rings
        a1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
        starts.append(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.45*np.cos(a2), 0.45*np.sin(a2)))
        )))
        
        for start in starts:
            if time.time() - start_time > deadline - 15:
                break
            
            x0 = start.flatten()
            
            # Annealing schedule for tau - more steps, tighter
            taus = [0.1, 0.05, 0.02, 0.005, 0.001, 0.0001, 0.00001]
            
            current_x = x0
            for tau in taus:
                if time.time() - start_time > deadline - 8:
                    break
                
                try:
                    res = minimize(
                        soft_min_score,
                        current_x,
                        args=(tau,),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    current_x = res.x
                    new_p = current_x.reshape((N, 2))
                    try_update(new_p)
                except Exception:
                    pass
    
    # Phase 3: Basin hopping from best
    if best_points is not None and time.time() - start_time < deadline - 12:
        x0 = best_points.flatten()
        try:
            res = basinhopping(
                neg_score, x0,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11},
                minsteps=4, T=0.5, stepsize=0.25,
                interval=5, disp=False
            )
            new_p = res.x.reshape((N, 2))
            try_update(new_p)
        except Exception:
            pass
    
    # Phase 4: Powell refinement
    if best_points is not None and time.time() - start_time < deadline - 8:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Powell',
                          options={'maxiter': 600, 'xtol': 1e-10, 'ftol': 1e-12})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
    
    # Phase 5: Final Nelder-Mead polish with very tight tolerances
    if best_points is not None and time.time() - start_time < deadline - 4:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
    
    # Phase 6: One more round of soft-min polishing from the best point
    if best_points is not None and time.time() - start_time < deadline - 2:
        x0 = best_points.flatten()
        for tau in [0.00005, 0.00001, 0.000001]:
            if time.time() - start_time > deadline - 1:
                break
            try:
                res = minimize(
                    soft_min_score,
                    x0,
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                x0 = res.x
                try_update(x0.reshape((N, 2)))
            except Exception:
                pass
    
    if best_points is None:
        # Fallback: regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_points

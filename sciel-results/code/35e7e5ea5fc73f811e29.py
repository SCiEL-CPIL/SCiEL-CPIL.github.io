import numpy as np
import time
from scipy.optimize import minimize, basinhopping, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 14

# Precompute triangle indices
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

# Precompute edge pairs for convex hull area
def _convex_hull_area(points):
    """Fast convex hull area computation."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def compute_score(points):
    """Compute min triangle area / convex hull area, vectorized."""
    n = len(points)
    if n < 3:
        return 0.0
    
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
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
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
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

def ring_configuration(params):
    """
    Generate points from ring parameters.
    params: [r1, r2, r3, theta_offset1, theta_offset2, theta_offset3, 
             n1, n2, n3] where n1+n2+n3=14, r_i >= 0, theta offsets in [0, 2pi)
    """
    r1, r2, r3, th1, th2, th3, n1, n2, n3 = params
    n1 = max(1, int(round(n1)))
    n2 = max(1, int(round(n2)))
    n3 = max(1, int(round(n3)))
    
    # Ensure we have exactly 14 points by adjusting n3
    total = n1 + n2 + n3
    n3 = 14 - n1 - n2
    if n3 < 1:
        n3 = 1
        n2 = 14 - n1 - n3
    if n2 < 1:
        n2 = 1
        n3 = 14 - n1 - n2
    
    points = []
    
    # Ring 1 (outermost)
    if n1 > 0:
        angles = th1 + np.linspace(0, 2*np.pi, n1, endpoint=False)
        points.append(np.column_stack((r1*np.cos(angles), r1*np.sin(angles))))
    
    # Ring 2
    if n2 > 0:
        angles = th2 + np.linspace(0, 2*np.pi, n2, endpoint=False)
        points.append(np.column_stack((r2*np.cos(angles), r2*np.sin(angles))))
    
    # Ring 3 (innermost)
    if n3 > 0:
        angles = th3 + np.linspace(0, 2*np.pi, n3, endpoint=False)
        points.append(np.column_stack((r3*np.cos(angles), r3*np.sin(angles))))
    
    if not points:
        return np.zeros((N, 2))
    
    result = np.vstack(points)
    
    # Pad or trim to exactly N points if needed
    if len(result) < N:
        extra = np.zeros((N - len(result), 2))
        result = np.vstack([result, extra])
    elif len(result) > N:
        result = result[:N]
    
    return result

def four_ring_configuration(params):
    """
    Generate points from 4-ring parameters.
    params: [r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4]
    where n1+n2+n3+n4=14
    """
    r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4 = params
    n1 = max(1, int(round(n1)))
    n2 = max(1, int(round(n2)))
    n3 = max(1, int(round(n3)))
    n4 = max(1, int(round(n4)))
    
    # Ensure we have exactly 14 points
    total = n1 + n2 + n3 + n4
    if total != 14:
        # Adjust n4 to make total 14
        n4 = 14 - n1 - n2 - n3
        if n4 < 1:
            n4 = 1
            n3 = 14 - n1 - n2 - n4
        if n3 < 1:
            n3 = 1
            n2 = 14 - n1 - n3 - n4
        if n2 < 1:
            n2 = 1
            n1 = 14 - n2 - n3 - n4
    
    points = []
    
    rings = [(r1, th1, n1), (r2, th2, n2), (r3, th3, n3), (r4, th4, n4)]
    for r, th, n in rings:
        if n > 0:
            angles = th + np.linspace(0, 2*np.pi, n, endpoint=False)
            points.append(np.column_stack((r*np.cos(angles), r*np.sin(angles))))
    
    if not points:
        return np.zeros((N, 2))
    
    result = np.vstack(points)
    
    if len(result) < N:
        extra = np.zeros((N - len(result), 2))
        result = np.vstack([result, extra])
    elif len(result) > N:
        result = result[:N]
    
    return result

def four_ring_score(params):
    """Score for 4-ring parameterization."""
    try:
        points = four_ring_configuration(params)
        return compute_score(points)
    except Exception:
        return 0.0

def neg_four_ring_score(params):
    return -four_ring_score(params)

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
    
    # Phase 1: Differential Evolution on 3-ring parameters
    bounds_3 = [
        (0.1, 1.5),  # r1
        (0.05, 1.2),  # r2
        (0.0, 0.8),   # r3
        (0, 2*np.pi),  # th1
        (0, 2*np.pi),  # th2
        (0, 2*np.pi),  # th3
        (1, 14),  # n1
        (1, 14),  # n2
        (1, 14),  # n3
    ]
    
    # Try differential evolution with fewer restarts but more iterations
    for restart in range(3):
        if time.time() - start_time > deadline - 40:
            break
        
        try:
            res = differential_evolution(
                neg_ring_score,
                bounds_3,
                seed=rng.integers(0, 2**31),
                maxiter=80,
                tol=1e-6,
                popsize=12,
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
    
    # Phase 1b: Differential Evolution on 4-ring parameters
    bounds_4 = [
        (0.1, 1.5),  # r1
        (0.05, 1.2),  # r2
        (0.0, 0.8),   # r3
        (0.0, 0.5),   # r4
        (0, 2*np.pi),  # th1
        (0, 2*np.pi),  # th2
        (0, 2*np.pi),  # th3
        (0, 2*np.pi),  # th4
        (1, 14),  # n1
        (1, 14),  # n2
        (1, 14),  # n3
        (1, 14),  # n4
    ]
    
    if time.time() - start_time < deadline - 30:
        for restart in range(2):
            if time.time() - start_time > deadline - 25:
                break
            
            try:
                res = differential_evolution(
                    neg_four_ring_score,
                    bounds_4,
                    seed=rng.integers(0, 2**31),
                    maxiter=60,
                    tol=1e-6,
                    popsize=10,
                    mutation=(0.3, 1.2),
                    recombination=0.9,
                    polish=True,
                    init='latinhypercube'
                )
                
                if res.success or res.fun < 0:
                    best_params = res.x
                    points = four_ring_configuration(best_params)
                    s = try_update(points)
            except Exception:
                pass
    
    # Phase 2: Multi-start L-BFGS-B on full 28D space from good ring configs
    if best_points is not None and time.time() - start_time < deadline - 25:
        # Generate diverse starting points
        starts = [best_points]
        
        # Add jittered versions
        for _ in range(4):
            jittered = best_points + rng.normal(0, 0.1, (N, 2))
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
        a2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
        a3 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        starts.append(np.vstack((
            np.column_stack((np.cos(a1), np.sin(a1))),
            np.column_stack((0.6*np.cos(a2), 0.6*np.sin(a2))),
            np.column_stack((0.3*np.cos(a3), 0.3*np.sin(a3)))
        )))
        
        for start in starts:
            if time.time() - start_time > deadline - 12:
                break
            
            x0 = start.flatten()
            
            # Annealing schedule for tau - more aggressive
            taus = [0.03, 0.01, 0.003, 0.001, 0.0003]
            
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
                        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
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
                options={'maxiter': 150, 'xatol': 1e-7, 'fatol': 1e-9},
                minsteps=3, T=0.5, stepsize=0.25,
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
                          options={'maxiter': 300, 'xtol': 1e-8, 'ftol': 1e-10})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
    
    # Phase 5: Final Nelder-Mead polish
    if best_points is not None and time.time() - start_time < deadline - 4:
        x0 = best_points.flatten()
        try:
            res = minimize(neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-11})
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
    
    # Phase 6: Final soft-min polish with very small tau
    if best_points is not None and time.time() - start_time < deadline - 2:
        x0 = best_points.flatten()
        try:
            res = minimize(
                soft_min_score, x0, args=(0.0001,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
    
    return best_points

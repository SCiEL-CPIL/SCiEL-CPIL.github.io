"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(pts) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    n = len(pts)
    min_area = float('inf')
    
    # Efficient vectorized computation
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
    return min_area / hull_area

def _generate_two_ring_points(r, theta):
    """
    Generate 14 points: 7 on unit circle (outer), 7 on circle of radius r (inner).
    """
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False) + theta
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = r * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    pts = np.vstack([outer_pts, inner_pts])
    return pts

def _fast_analyze(r, theta):
    """
    Fast analytic/semi-analytic score calculation for two-ring configuration.
    """
    pts = _generate_two_ring_points(r, theta)
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-12:
        return 0.0
        
    min_area = float('inf')
    
    outer = pts[:7]
    inner = pts[7:]
    
    # 1. Outer-Outer-Outer (consecutive is min for regular polygon)
    p0, p1, p2 = outer[0], outer[1], outer[2]
    area_ooo = 0.5 * np.abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0]))
    min_area = min(min_area, area_ooo)
    
    # 2. Inner-Inner-Inner (consecutive is min for regular polygon)
    q0, q1, q2 = inner[0], inner[1], inner[2]
    area_iii = 0.5 * np.abs((q1[0]-q0[0])*(q2[1]-q0[1]) - (q1[1]-q0[1])*(q2[0]-q0[0]))
    min_area = min(min_area, area_iii)
    
    # 3. Mixed: OOI and OII
    # OOI: for each inner point k, min area with any pair of outer points
    for k in range(7):
        qk = inner[k]
        diffs = outer - qk
        for i in range(7):
            for j in range(i+1, 7):
                v1 = diffs[i]
                v2 = diffs[j]
                area = 0.5 * np.abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    
    # OII: for each outer point i, min area with any pair of inner points
    for i in range(7):
        pi = outer[i]
        diffs = inner - pi
        for j in range(7):
            for k in range(j+1, 7):
                v1 = diffs[j]
                v2 = diffs[k]
                area = 0.5 * np.abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area

def _three_ring_points(r1, r2, th1, th2):
    """
    Generate 14 points: 5 on unit circle, 5 on radius r1, 4 on radius r2.
    Angles: outer 0, 2pi/5...; middle th1, th1+2pi/5...; inner th2, th2+2pi/4...
    """
    angles_outer = np.linspace(0, 2*np.pi, 5, endpoint=False)
    angles_mid = np.linspace(0, 2*np.pi, 5, endpoint=False) + th1
    angles_inner = np.linspace(0, 2*np.pi, 4, endpoint=False) + th2
    
    p1 = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    p2 = r1 * np.column_stack([np.cos(angles_mid), np.sin(angles_mid)])
    p3 = r2 * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    return np.vstack([p1, p2, p3])

def _fast_analyze_3ring(r1, r2, th1, th2):
    """
    Score calculation for three-ring configuration.
    """
    pts = _three_ring_points(r1, r2, th1, th2)
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-12:
        return 0.0
        
    min_area = float('inf')
    n = len(pts)
    
    # Generic but vectorized where possible? 
    # For 14 points, C(14,3) = 364 triangles.
    # Looping i, j and vectorizing k is efficient enough.
    
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
    return min_area / hull_area

def _optimize_2ring(deadline):
    """
    Optimize two-ring configuration (7 outer, 7 inner).
    """
    best_score = -float('inf')
    best_r = 0.6
    best_theta = np.pi / 7
    
    # 1. Coarse Grid over theta
    # Symmetry: theta in [0, pi/7]
    theta_coarse = np.linspace(0, np.pi/7, 15)
    
    from scipy.optimize import minimize_scalar, minimize
    
    for th in theta_coarse:
        if time.time() > deadline - 30.0:
            break
            
        def neg_score_r(r):
            if r < 0.1 or r > 0.99: return 1e9
            return -_fast_analyze(r, th)
            
        try:
            res_r = minimize_scalar(neg_score_r, bounds=(0.2, 0.95), method='bounded', options={'xatol': 1e-5})
            r_opt = res_r.x
            score_opt = -res_r.fun
        except Exception:
            continue
            
        if score_opt > best_score:
            best_score = score_opt
            best_r = r_opt
            best_theta = th
            
    # 2. Fine Grid / 1D Theta Search
    if time.time() < deadline - 10.0:
        theta_step = np.pi / 7 / 20
        theta_lo = max(0, best_theta - theta_step)
        theta_hi = min(np.pi/7, best_theta + theta_step)
        
        theta_fine = np.linspace(theta_lo, theta_hi, 20)
        
        for th in theta_fine:
            if time.time() > deadline - 10.0:
                break
            
            def neg_score_r_fine(r):
                if r < 0.1 or r > 0.99: return 1e9
                return -_fast_analyze(r, th)
                
            try:
                res_r = minimize_scalar(neg_score_r_fine, bounds=(0.2, 0.95), method='bounded', options={'xatol': 1e-6})
                r_opt = res_r.x
                score_opt = -res_r.fun
            except Exception:
                continue
                
            if score_opt > best_score:
                best_score = score_opt
                best_r = r_opt
                best_theta = th
                
    # 3. 2D Polish
    if time.time() < deadline - 5.0:
        def neg_score_2d(params):
            r, th = params
            if r < 0.1 or r > 0.99: return 1e9
            return -_fast_analyze(r, th)
            
        x0 = [best_r, best_theta]
        
        try:
            res = minimize(neg_score_2d, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200, 'adaptive': True})
            
            if res.fun < -best_score:
                best_r, best_theta = res.x
                best_score = -res.fun
        except Exception:
            pass
            
        if time.time() < deadline - 2.0:
            for _ in range(5):
                if time.time() > deadline - 2.0:
                    break
                r_pert = best_r + np.random.normal(0, 0.02)
                th_pert = best_theta + np.random.normal(0, np.pi/70)
                
                if r_pert < 0.1 or r_pert > 0.99:
                    continue
                    
                try:
                    res = minimize(neg_score_2d, [r_pert, th_pert], method='Nelder-Mead', 
                                   options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 50, 'adaptive': True})
                    
                    if res.fun < -best_score:
                        best_r, best_theta = res.x
                        best_score = -res.fun
                except Exception:
                    pass

    best_pts = _generate_two_ring_points(best_r, best_theta)
    return best_pts, best_score

def _optimize_3ring(deadline):
    """
    Optimize three-ring configuration (5 outer, 5 middle, 4 inner).
    Uses Differential Evolution for robustness in 4D space.
    """
    from scipy.optimize import differential_evolution
    
    def objective(params):
        r1, r2, th1, th2 = params
        # Constraints
        if r1 < 0.2 or r1 > 0.98 or r2 < 0.1 or r2 > 0.98:
            return 1e9
        # Normalize angles to ranges for symmetry reduction?
        # Symmetry: th1 in [0, 2pi/5), th2 in [0, 2pi/4)
        # DE handles bounds, but we can wrap or just let it search the box.
        # Wrapping ensures we don't miss equivalent points due to periodicity, 
        # though the landscape is periodic so it shouldn't matter much for global max.
        return -_fast_analyze_3ring(r1, r2, th1, th2)
        
    bounds = [(0.3, 0.98), (0.2, 0.98), (0, 2*np.pi/5), (0, 2*np.pi/4)]
    
    best_score = -float('inf')
    best_params = None
    
    try:
        # DE is stochastic, so use seed for reproducibility? 
        # Actually, random is fine, but we want to be fast.
        # maxiter and popsize control speed.
        res = differential_evolution(objective, bounds, maxiter=50, popsize=15, tol=1e-6, seed=42, polish=True)
        
        if res.fun < 1e8: # Check if valid
            best_score = -res.fun
            best_params = res.x
    except Exception:
        pass
        
    if best_params is None:
        # Fallback: return a default 3-ring config
        return None, -float('inf')
        
    r1, r2, th1, th2 = best_params
    best_pts = _three_ring_points(r1, r2, th1, th2)
    
    # Final verification
    final_score = _score(best_pts)
    if final_score > best_score:
        best_score = final_score
        
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Strategy:
    # 1. Run 2-ring optimization (proven to be good, ~48 score)
    # 2. Run 3-ring optimization (new structural family, potentially better)
    # 3. Return the best of the two.
    
    pts_2ring, score_2ring = _optimize_2ring(deadline)
    
    pts_3ring, score_3ring = None, -float('inf')
    if time.time() < deadline - 10.0:
        pts_3ring, score_3ring = _optimize_3ring(deadline)
        
    # Compare
    if pts_3ring is not None and score_3ring > score_2ring:
        best_pts = pts_3ring
        # Re-verify best score for safety
        final_score = _score(best_pts)
        # If final_score is significantly lower, something went wrong in optimization vs scoring
        # But _fast_analyze_3ring is identical to _score logic for area, just maybe hull diff?
        # No, _fast_analyze_3ring uses ConvexHull too.
        # So score_3ring should be close to final_score.
        if final_score > score_2ring:
            return best_pts
    else:
        return pts_2ring

# EVOLVE-BLOCK-END

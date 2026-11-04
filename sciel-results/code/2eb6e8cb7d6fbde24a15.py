"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar, minimize

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(pts) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    n = len(pts)
    min_area = float('inf')
    
    # Efficient vectorized computation
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            
            # Cross product of (pj - pi) and (pk - pi)
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
    Outer angles: 0, 2pi/7, ...
    Inner angles: theta, theta + 2pi/7, ...
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
    Avoids full ConvexHull call in inner loop by using known geometry where possible,
    but for accuracy we still might need hull if r is large. 
    However, for r < 1, if r < cos(pi/7) ~ 0.9009, hull is the outer heptagon.
    Area of regular heptagon on unit circle:
    A_7 = 7 * sin(2*pi/7) / 2 * (1/2)^? No.
    Area = (n * a^2) / (4 * tan(pi/n)). Here side length a = 2*sin(pi/7).
    Or simply: 7 * 0.5 * R^2 * sin(2*pi/7) = 3.5 * sin(2*pi/7).
    
    If r > cos(pi/7), hull changes.
    To be safe and robust, we compute hull area only if r > 0.89 or just compute it always?
    Computing hull for 14 points is fast (~10us). Let's just do it to ensure correctness
    if the inner points stick out.
    """
    pts = _generate_two_ring_points(r, theta)
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-12:
        return 0.0
        
    # Compute min triangle area
    # We can be smarter here.
    # Outer points: P[0..6]
    # Inner points: P[7..13]
    
    min_area = float('inf')
    
    # 1. Outer-Outer-Outer
    # Only need to check consecutive triples due to regularity?
    # Min area for 3 points on unit circle is formed by consecutive vertices.
    # Area = 0.5 * sin(2*pi/7) * ... 
    # Let's just compute the specific triple (0,1,2) area, it's the global min for outer ring.
    p0, p1, p2 = pts[0], pts[1], pts[2]
    area_ooo = 0.5 * np.abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0]))
    min_area = min(min_area, area_ooo)
    
    # 2. Inner-Inner-Inner
    q0, q1, q2 = pts[7], pts[8], pts[9]
    area_iii = 0.5 * np.abs((q1[0]-q0[0])*(q2[1]-q0[1]) - (q1[1]-q0[1])*(q2[0]-q0[0]))
    min_area = min(min_area, area_iii)
    
    # 3. Mixed: OOI and OII
    # OOI: 2 Outer, 1 Inner
    # OII: 1 Outer, 2 Inner
    
    # To speed up, we rely on the fact that the minimum usually involves "close" points.
    # But to be rigorous, let's check all mixed triangles.
    # There are C(7,2)*7 = 147 OOI and C(7,1)*C(7,2) = 147 OII.
    # Total 294 triangles.
    # We can vectorize this.
    
    outer = pts[:7]
    inner = pts[7:]
    
    # OOI: for each inner point k, find min area with any pair of outer points.
    # We can precompute outer pairs? Or just loop k.
    for k in range(7):
        qk = inner[k]
        # v1 = outer - qk (7x2)
        # We need min |cross(v1[i], v1[j])| for i < j
        # This is equivalent to finding the minimum area triangle with vertex qk and base in outer ring.
        # Due to symmetry of outer ring, for a fixed qk, the min area is likely with the two outer points
        # that are "closest" to qk in angular sense? Not necessarily, but we can just compute all 21.
        
        # Vectorized for fixed k:
        # diffs = outer - qk
        # cross matrix C[i,j] = diffs[i,0]*diffs[j,1] - diffs[i,1]*diffs[j,0]
        diffs = outer - qk
        # Create cross product matrix
        # C = diffs[:, 0:1] * diffs[1:, 1] - ... no, easier to use broadcasting
        # c = np.abs(diffs[:, 0:1] * diffs[1:, 1] - diffs[:, 1:2] * diffs[1:, 0]) -- shape issues
        
        # Let's just loop i, j for the 21 pairs, it's tiny.
        for i in range(7):
            for j in range(i+1, 7):
                v1 = diffs[i]
                v2 = diffs[j]
                area = 0.5 * np.abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    
    # OII: for each outer point i, find min area with any pair of inner points.
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_score = -float('inf')
    best_r = 0.6
    best_theta = np.pi / 7
    
    # Strategy:
    # 1. Coarse Grid Search
    # 2. 1D Refinement over theta (optimizing r inside)
    # 3. 2D Polish (Nelder-Mead)
    
    # 1. Coarse Grid
    # theta range [0, pi/7] due to symmetry
    # r range [0.3, 0.95]
    
    theta_coarse = np.linspace(0, np.pi/7, 15)
    
    for th in theta_coarse:
        if time.time() > deadline - 30.0:
            break
            
        # Optimize r for this theta
        def neg_score_r(r):
            if r < 0.1 or r > 0.99: return 1e9
            return -_fast_analyze(r, th)
            
        res_r = minimize_scalar(neg_score_r, bounds=(0.2, 0.95), method='bounded', options={'xatol': 1e-5})
        r_opt = res_r.x
        score_opt = -res_r.fun
        
        if score_opt > best_score:
            best_score = score_opt
            best_r = r_opt
            best_theta = th
            
    # 2. Fine Grid / 1D Theta Search
    # Refine theta around best_theta
    if time.time() < deadline - 10.0:
        theta_step = np.pi / 7 / 20 # ~0.022
        theta_lo = max(0, best_theta - theta_step)
        theta_hi = min(np.pi/7, best_theta + theta_step)
        
        theta_fine = np.linspace(theta_lo, theta_hi, 20)
        
        for th in theta_fine:
            if time.time() > deadline - 10.0:
                break
            
            def neg_score_r_fine(r):
                if r < 0.1 or r > 0.99: return 1e9
                return -_fast_analyze(r, th)
                
            res_r = minimize_scalar(neg_score_r_fine, bounds=(0.2, 0.95), method='bounded', options={'xatol': 1e-6})
            r_opt = res_r.x
            score_opt = -res_r.fun
            
            if score_opt > best_score:
                best_score = score_opt
                best_r = r_opt
                best_theta = th
                
    # 3. 2D Polish (Nelder-Mead)
    if time.time() < deadline - 5.0:
        def neg_score_2d(params):
            r, th = params
            if r < 0.1 or r > 0.99: return 1e9
            # Normalize theta to [0, pi/7] range for consistency if needed, 
            # but _fast_analyze handles any theta.
            # However, due to periodicity, we should probably just let it be.
            return -_fast_analyze(r, th)
            
        x0 = [best_r, best_theta]
        
        # Multiple starts for polish? No, just start from best.
        # Maybe add a few perturbations if time permits?
        # For now, single run from best.
        
        try:
            res = minimize(neg_score_2d, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200, 'adaptive': True})
            
            if res.fun < -best_score:
                best_r, best_theta = res.x
                best_score = -res.fun
        except Exception:
            pass
            
        # If time is still available, try a few random restarts near the best point
        if time.time() < deadline - 2.0:
            for _ in range(5):
                if time.time() > deadline - 2.0:
                    break
                # Perturb
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
    
    # Final verification with the exact _score function (which is slower but accurate)
    # _fast_analyze is already quite accurate, but let's double check.
    # Actually, _fast_analyze uses the same logic as _score for min area, just optimized.
    # The only potential discrepancy is hull area if my analytic assumption was wrong, 
    # but I used ConvexHull in _fast_analyze too.
    # So _fast_analyze should be identical to _score in value, just faster.
    
    # Let's do one final check
    final_score = _score(best_pts)
    if final_score > best_score:
        best_score = final_score
        
    return best_pts
# EVOLVE-BLOCK-END

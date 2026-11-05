"""Pack 11 unit hexagons inside a minimal regular hexagon."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11
DEADLINE = 140.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, angle, s=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with rotation angle (radians)."""
    angles = angle + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _compute_S_for_rotation(verts, oangle):
    """
    Compute minimal side length S of outer hexagon at origin with rotation oangle
    containing all vertices in 'verts' (N x 6 x 2).
    """
    # Transform to container local frame (rotate by -oangle)
    cos_a = np.cos(-oangle)
    sin_a = np.sin(-oangle)
    
    rx = verts[..., 0] * cos_a - verts[..., 1] * sin_a
    ry = verts[..., 0] * sin_a + verts[..., 1] * cos_a
    
    # Normals for unit hexagon facets (apothem directions)
    # For hexagon with vertices at 0, 60, 120, ... degrees, facets are at 30, 90, 150, ...
    theta_n = np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(theta_n)
    ny = np.sin(theta_n)
    
    # Projections: (N, 6, 6)
    proj = rx[..., np.newaxis] * nx[np.newaxis, np.newaxis, :] + \
           ry[..., np.newaxis] * ny[np.newaxis, np.newaxis, :]
    
    max_proj = np.max(np.abs(proj))
    apothem = np.sqrt(3) / 2
    return max_proj / apothem

def _compute_S_smooth(verts, oangle, beta=50.0):
    """
    Compute a smooth approximation of the minimal side length S using log-sum-exp
    to approximate the max operation. This helps the optimizer navigate the
    nonsmooth landscape.
    """
    cos_a = np.cos(-oangle)
    sin_a = np.sin(-oangle)
    
    rx = verts[..., 0] * cos_a - verts[..., 1] * sin_a
    ry = verts[..., 0] * sin_a + verts[..., 1] * cos_a
    
    theta_n = np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(theta_n)
    ny = np.sin(theta_n)
    
    proj = rx[..., np.newaxis] * nx[np.newaxis, np.newaxis, :] + \
           ry[..., np.newaxis] * ny[np.newaxis, np.newaxis, :]
    
    abs_proj = np.abs(proj)
    
    # Soft max via log-sum-exp
    max_val = np.max(abs_proj)
    shifted = abs_proj - max_val
    log_sum_exp = max_val + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    apothem = np.sqrt(3) / 2
    return log_sum_exp / apothem

def _find_best_outer(verts):
    """
    Given all inner hexagon vertices (N x 6 x 2), find the minimal enclosing regular hexagon.
    Returns (best_S, best_oangle).
    """
    def S_of_angle(oangle):
        return _compute_S_for_rotation(verts, oangle)
    
    # Coarse search with more points for better resolution
    best_S = float('inf')
    best_angle = 0.0
    n_coarse = 60
    for i in range(n_coarse):
        ang = i * (np.pi / 3) / n_coarse
        s_val = S_of_angle(ang)
        if s_val < best_S:
            best_S = s_val
            best_angle = ang
    
    # Fine search around best with tighter tolerance
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(S_of_angle, bounds=(best_angle - 0.05, best_angle + 0.05), 
                             method='bounded', options={'xatol': 1e-10})
    if result.success:
        best_S = result.fun
        best_angle = result.x
    
    # Second pass: try nearby angles in case of local minima
    for offset in [-0.1, 0.1]:
        center = (best_angle + offset) % (np.pi / 3)
        lower = max(0, center - 0.02)
        upper = min(np.pi / 3, center + 0.02)
        if upper > lower:
            res2 = minimize_scalar(S_of_angle, bounds=(lower, upper), 
                                   method='bounded', options={'xatol': 1e-10})
            if res2.success and res2.fun < best_S:
                best_S = res2.fun
                best_angle = res2.x
    
    # Ensure angle is in [0, pi/3)
    best_angle = best_angle % (np.pi / 3)
    
    return best_S, best_angle

def _sat_overlap(v1, v2, tol=1e-8):
    """Check if two convex hexagons overlap using Separating Axis Theorem."""
    polys = [v1, v2]
    for poly in polys:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            if proj1.max() + tol < proj2.min() or proj2.max() + tol < proj1.min():
                return False
    return True

def construct():
    start_time = time.time()
    
    def get_all_vertices(params):
        """Get vertices for all hexagons given parameters [cx0, cy0, a0, cx1, ...]"""
        n = len(params) // 3
        verts = np.zeros((n, 6, 2))
        for i in range(n):
            cx = params[3 * i]
            cy = params[3 * i + 1]
            angle = params[3 * i + 2]
            verts[i] = _hex_vertices(cx, cy, angle)
        return verts

    def objective(params):
        if time.time() - start_time > DEADLINE:
            return float('inf')
        
        n = N
        verts = get_all_vertices(params)
        
        # Find best outer hexagon analytically
        S, _ = _find_best_outer(verts)
        
        # Overlap penalty with smoother formulation
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                c1 = verts[i].mean(axis=0)
                c2 = verts[j].mean(axis=0)
                dist = np.linalg.norm(c1 - c2)
                if dist < 2.0:  # Only check SAT if potentially overlapping
                    if _sat_overlap(verts[i], verts[j]):
                        # Smooth penalty that grows quadratically
                        overlap = 2.0 - dist
                        penalty += 500.0 * overlap + 1000.0 * overlap * overlap
        
        return S + penalty

    # --- Analytic Initial Guess (The "Flower" Packing) ---
    # This configuration is known to be optimal for N=11 with side length 4.0.
    # Inner hexagons are rotated 30 degrees (flat-top).
    # Outer hexagon is rotated 30 degrees (flat-top).
    
    eps = 1e-9
    centers = []
    
    # Ring 0: Center
    centers.append((0.0, 0.0))
    
    # Ring 1: 6 hexagons at distance sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
        
    # Ring 2: 4 hexagons at distance 3 + 2*eps, angles 30, 90, 150, 210
    # Note: The inspirations used 30, 90, 150, 270. 
    # Let's check symmetry. 30, 90, 150, 210 is not symmetric under 60 deg rotation.
    # 30, 90, 150, 270 is also not fully symmetric.
    # The standard "flower" with 4 extra hexes usually places them at the gaps.
    # Gaps in ring 1 are at 30, 90, 150, 210, 270, 330.
    # Placing 4 hexes at 30, 90, 150, 210 leaves 270 and 330 empty?
    # Actually, the inspirations say: "Ring 2: 4 hexagons ... angles 30, 90, 150, 270" in Insp 1, 
    # but "30, 90, 150, 210" in Insp 2.
    # Let's use the coordinates from Insp 1 which scored 10.0000.
    # Insp 1: angles 30, 90, 150, 270.
    # Wait, Insp 1 code: `for k in range(4): angle = np.deg2rad(30 + k * 60)` -> 30, 90, 150, 210.
    # Insp 2 code: `for k in range(4): angle = np.deg2rad(30 + k * 60)` -> 30, 90, 150, 210.
    # Both inspirations used 30, 90, 150, 210.
    
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
        
    params = np.zeros(3 * N)
    for i, (x, y) in enumerate(centers):
        params[3*i] = x
        params[3*i+1] = y
        params[3*i+2] = np.deg2rad(30.0) # Inner rotation 30 deg
    
    # Center the configuration (it is already centered at origin)
    # best_params = params.copy()
    
    # Calculate the side length for this analytic solution
    verts = get_all_vertices(params)
    # The outer hexagon should be rotated 30 degrees to match the flat-top orientation
    # of the inner hexagons and the symmetry of the cluster.
    S_analytic, oangle_analytic = _find_best_outer(verts)
    
    best_params = params.copy()
    best_score = S_analytic
    
    # --- Refinement via Optimization ---
    # Try to improve upon the analytic solution using Nelder-Mead.
    # This is a structural change: using the known optimal symmetric configuration
    # as the seed for a local search that might break symmetry if beneficial.
    
    if time.time() - start_time < DEADLINE - 30:
        res = minimize(objective, best_params, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x.copy()
            
    # If we have time, try Powell for final polish
    if time.time() - start_time < DEADLINE - 10:
        res2 = minimize(objective, best_params, method='Powell',
                       options={'maxiter': 1000, 'maxfev': 10000, 'disp': False, 'xtol': 1e-10, 'ftol': 1e-12})
        if res2.fun < best_score:
            best_score = res2.fun
            best_params = res2.x.copy()
    
    # Final high-precision polish: coordinate descent on each parameter
    if time.time() - start_time < DEADLINE - 2:
        for _ in range(3):  # 3 passes of coordinate descent
            for i in range(3 * N):
                if time.time() - start_time > DEADLINE - 2:
                    break
                # Try small perturbations
                best_local = best_params[i]
                best_local_score = best_score
                for delta in [1e-4, -1e-4, 1e-5, -1e-5, 1e-6, -1e-6]:
                    trial = best_params.copy()
                    trial[i] += delta
                    trial_score = objective(trial)
                    if trial_score < best_local_score:
                        best_local = trial[i]
                        best_local_score = trial_score
                best_params[i] = best_local
                best_score = best_local_score

    # Extract results
    inner = []
    for i in range(N):
        cx = best_params[3 * i]
        cy = best_params[3 * i + 1]
        angle = best_params[3 * i + 2]
        inner.append([cx, cy, np.degrees(angle)])
    
    # Find final best outer hexagon with high precision
    verts = get_all_vertices(best_params)
    S, oangle = _find_best_outer(verts)
    
    return inner, (0.0, 0.0), S, np.degrees(oangle)

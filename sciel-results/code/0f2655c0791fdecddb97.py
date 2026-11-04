"""Pack 11 unit hexagons inside a minimal regular hexagon."""
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

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

def _find_best_outer(verts):
    """
    Given all inner hexagon vertices (N x 6 x 2), find the minimal enclosing regular hexagon.
    Returns (best_S, best_oangle).
    """
    def S_of_angle(oangle):
        return _compute_S_for_rotation(verts, oangle)
    
    # Coarse search
    best_S = float('inf')
    best_angle = 0.0
    n_coarse = 30
    for i in range(n_coarse):
        ang = i * (np.pi / 3) / n_coarse
        s_val = S_of_angle(ang)
        if s_val < best_S:
            best_S = s_val
            best_angle = ang
    
    # Fine search around best
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(S_of_angle, bounds=(best_angle - 0.1, best_angle + 0.1), 
                             method='bounded', options={'xatol': 1e-8})
    if result.success:
        best_S = result.fun
        best_angle = result.x
    
    # Ensure angle is in [0, pi/3)
    best_angle = best_angle % (np.pi / 3)
    
    return best_S, best_angle

def _min_dist_between_hexagons(v1, v2):
    """Compute minimum distance between vertices of two hexagons (approximation for separation)."""
    # Compute all pairwise distances between vertices
    # v1: (6, 2), v2: (6, 2)
    diff = v1[:, np.newaxis, :] - v2[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    return np.min(dists)

def _smooth_overlap_penalty(verts, threshold=0.01, softness=0.05):
    """
    Compute a smooth overlap penalty based on minimum distance between hexagons.
    Uses a smooth approximation: if min_dist < threshold, penalty increases.
    """
    n = len(verts)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            # Quick center-distance check
            c1 = verts[i].mean(axis=0)
            c2 = verts[j].mean(axis=0)
            center_dist = np.linalg.norm(c1 - c2)
            if center_dist < 2.5:  # Only compute if potentially close
                min_d = _min_dist_between_hexagons(verts[i], verts[j])
                # Smooth penalty: quadratic when overlapping, smooth transition
                if min_d < threshold:
                    # Smooth: (threshold - min_d)^2 / softness
                    penalty += ((threshold - min_d) ** 2) / softness
    return penalty

def _sat_overlap(v1, v2, tol=1e-9):
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

    def _centered_verts(verts):
        """Center the vertices at the origin."""
        centroid = verts.mean(axis=(0, 1))
        return verts - centroid

    def objective_fixed_angle(params, fixed_oangle):
        """
        Fast objective function using a fixed outer hexagon angle.
        This avoids the expensive minimize_scalar call in each iteration.
        """
        if time.time() - start_time > DEADLINE:
            return float('inf')
        
        n = N
        verts = get_all_vertices(params)
        verts = _centered_verts(verts)
        
        # Compute S with fixed angle
        S = _compute_S_for_rotation(verts, fixed_oangle)
        
        # Smooth overlap penalty (differentiable)
        penalty = _smooth_overlap_penalty(verts, threshold=0.02, softness=0.01)
        
        return S + penalty

    # --- Analytic Initial Guess (The "Flower" Packing) ---
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
    
    # Calculate the side length and optimal angle for this analytic solution
    verts = get_all_vertices(params)
    verts_centered = _centered_verts(verts)
    S_analytic, oangle_analytic = _find_best_outer(verts_centered)
    
    best_params = params.copy()
    best_score = S_analytic
    
    # --- Refinement via Optimization ---
    # Use the fixed angle for the main optimization loop to speed it up
    
    start_points = [best_params.copy()]
    
    # Add perturbations of varying magnitude
    np.random.seed(42)
    for scale in [0.05, 0.1, 0.2, 0.5]:
        perturbed = best_params + np.random.randn(3 * N) * scale
        start_points.append(perturbed)
    
    # Try each starting point with Nelder-Mead
    for sp in start_points:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        try:
            # Use the fixed angle from the seed
            res = minimize(lambda p: objective_fixed_angle(p, oangle_analytic), sp, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 
                                  'adaptive': True, 'maxfev': 6000})
            if res.fun < best_score:
                # Note: res.fun includes penalty, so it's not directly comparable to S_analytic
                # But we track best_params and re-evaluate S at the end
                # To be safe, we only update if the penalty is small
                verts_res = get_all_vertices(res.x)
                verts_res_centered = _centered_verts(verts_res)
                S_res = _compute_S_for_rotation(verts_res_centered, oangle_analytic)
                pen_res = _smooth_overlap_penalty(verts_res_centered, threshold=0.02, softness=0.01)
                if pen_res < 1e-5: # Ensure no significant overlap
                    if S_res < best_score:
                        best_score = S_res
                        best_params = res.x.copy()
        except Exception:
            continue
    
    # Final polish with Powell using tighter tolerances
    if time.time() - start_time < DEADLINE - 5:
        try:
            res2 = minimize(lambda p: objective_fixed_angle(p, oangle_analytic), best_params, method='Powell',
                           options={'maxiter': 2000, 'maxfev': 20000, 
                                   'xrtol': 1e-10, 'ftol': 1e-12, 'disp': False})
            verts_res = get_all_vertices(res2.x)
            verts_res_centered = _centered_verts(verts_res)
            S_res = _compute_S_for_rotation(verts_res_centered, oangle_analytic)
            pen_res = _smooth_overlap_penalty(verts_res_centered, threshold=0.02, softness=0.01)
            if pen_res < 1e-5 and S_res < best_score:
                best_score = S_res
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish with Nelder-Mead
    if time.time() - start_time < DEADLINE - 2:
        try:
            res3 = minimize(lambda p: objective_fixed_angle(p, oangle_analytic), best_params, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'adaptive': True})
            verts_res = get_all_vertices(res3.x)
            verts_res_centered = _centered_verts(verts_res)
            S_res = _compute_S_for_rotation(verts_res_centered, oangle_analytic)
            pen_res = _smooth_overlap_penalty(verts_res_centered, threshold=0.02, softness=0.01)
            if pen_res < 1e-5 and S_res < best_score:
                best_score = S_res
                best_params = res3.x.copy()
        except Exception:
            pass

    # Extract results
    inner = []
    for i in range(N):
        cx = best_params[3 * i]
        cy = best_params[3 * i + 1]
        angle = best_params[3 * i + 2]
        inner.append([cx, cy, np.degrees(angle)])
    
    # Find final best outer hexagon with refined search (allowing angle to vary)
    verts = get_all_vertices(best_params)
    verts_centered = _centered_verts(verts)
    S, oangle = _find_best_outer(verts_centered)
    
    # Final verification: ensure no overlaps in the returned configuration
    final_verts = get_all_vertices(best_params)
    has_overlap = False
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_overlap(final_verts[i], final_verts[j]):
                has_overlap = True
                break
        if has_overlap:
            break
    
    if has_overlap:
        # Fall back to the known-good analytic solution
        verts = get_all_vertices(params)
        verts_centered = _centered_verts(verts)
        S, oangle = _find_best_outer(verts_centered)
        inner = []
        for i, (x, y) in enumerate(centers):
            inner.append([x, y, 30.0])
    
    return inner, (0.0, 0.0), S, np.degrees(oangle)

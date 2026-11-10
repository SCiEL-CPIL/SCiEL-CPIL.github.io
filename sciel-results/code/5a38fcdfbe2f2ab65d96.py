# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a_rad = np.deg2rad(a_deg)
    angles = a_rad + np.deg2rad(60 * np.arange(6))
    x = cx + s * np.cos(angles)
    y = cy + s * np.sin(angles)
    return np.column_stack((x, y))

def _fast_separated(centers, angle_deg=0.0):
    """Fast SAT check for 11 unit hexagons all with same orientation.
    Centers are (11,2) array. Returns True if all pairs separated."""
    a_rad = np.deg2rad(angle_deg)
    # For regular hexagons with side 1, the support function in direction n is:
    # max over vertices of dot(vertex, n) = dot(center, n) + max over offsets
    # For a unit hexagon at origin with angle a, vertices are at angles a+60i
    # The 12 edge normals are at angles a+30+60i for i=0..5 (two per edge)
    # Actually for SAT we need 6 edge normals per hexagon, but since all same orientation,
    # we only need 6 normals total (the 6 edge directions)
    
    # Edge normals for a hexagon with angle a: perpendicular to edges
    # Edge i goes from vertex i to vertex i+1
    # Vertex i is at angle a + 60*i
    # Edge direction: v(i+1) - v(i)
    # Normal: perpendicular to edge
    
    # For unit hex, the 6 outward edge normals are at angles a + 30 + 60*i, i=0..5
    normal_angles = a_rad + np.deg2rad(30) + np.deg2rad(60 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    # For each normal direction, compute projection of all 11 centers
    # The support radius of a unit hexagon in direction n is the max dot(v, n) over vertices v
    # For a regular hexagon with circumradius 1, the apothem (inradius) = cos(30°) = sqrt(3)/2
    # The support in any direction n is: max over 6 vertices of dot(v_i, n)
    # Since vertices are at unit distance from center, and normals are at 30°+60i,
    # the support in the normal direction equals the apothem = cos(30°) = sqrt(3)/2
    
    # Actually let's be more careful. The support function h(n) = max_i dot(v_i, n)
    # where v_i are vertices relative to center.
    # For a regular hexagon with circumradius 1, vertices at angles 0,60,...,300 (before rotation)
    # After rotation by a: vertices at angles a, a+60, ..., a+300
    # Normal directions: a+30, a+90, ..., a+330
    # dot(v_i, n_j) where v_i is at angle a+60i, n_j is at angle a+30+60j
    # = cos((a+60i) - (a+30+60j)) = cos(60(i-j) - 30)
    # Max over i for fixed j: when i=j, cos(-30) = cos(30) = sqrt(3)/2
    # So support = sqrt(3)/2 for all normal directions.
    
    support = np.sqrt(3) / 2  # = cos(30°)
    
    # For two hexagons with centers c1, c2 and same orientation:
    # They are separated iff there exists a normal n such that
    # dot(c1, n) + support < dot(c2, n) - support - 1e-6
    # i.e., dot(c2-c1, n) > 2*support + 1e-6
    # OR dot(c1-c2, n) > 2*support + 1e-6
    
    # 2*support = sqrt(3) ≈ 1.732
    # So we need |dot(c2-c1, n)| > sqrt(3) + 1e-6 for some normal n
    
    n_pairs = len(centers)
    for i in range(n_pairs):
        for j in range(i + 1, n_pairs):
            d = centers[j] - centers[i]
            proj = normals @ d
            if np.max(np.abs(proj)) > np.sqrt(3) + 1e-6:
                continue
            else:
                return False
    return True

def _min_enclosing_hexagon_side(centers, angle_deg):
    """Given fixed centers (11,2) and outer hexagon angle, compute minimum side length
    of a regular hexagon with that angle that contains all vertices of the 11 unit hexagons.
    The outer hexagon center is optimized to minimize the side length."""
    a_rad = np.deg2rad(angle_deg)
    
    # Get all vertices of the 11 inner hexagons
    all_verts = []
    for i in range(len(centers)):
        v = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)  # (66, 2)
    
    # The outer hexagon has 6 edges. For a regular hexagon with side S and angle a,
    # centered at (cx, cy), the vertices are at (cx + S*cos(a+60i), cy + S*sin(a+60i))
    # The edge normals point outward at angles a+30+60i
    # A point p is inside iff for all 6 normals n_k: dot(p - center, n_k) <= S*cos(30°) = S*sqrt(3)/2
    # (since the distance from center to each edge is the apothem = S*sqrt(3)/2)
    
    normal_angles = a_rad + np.deg2rad(30) + np.deg2rad(60 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))  # (6,2)
    
    # For a given center (cx, cy), the required S is:
    # S >= max over all vertices p, all normals n: dot(p - (cx,cy), n) / (sqrt(3)/2)
    # = (2/sqrt(3)) * max_{p,n} [dot(p,n) - dot((cx,cy), n)]
    # = (2/sqrt(3)) * [max_p dot(p,n) - dot((cx,cy), n)]
    
    # To minimize S over (cx, cy), we want to maximize dot((cx,cy), n) for the constraining normal.
    # But S is determined by the max over all 6 normals.
    # This is equivalent to finding the minimum enclosing regular hexagon.
    
    # For a fixed angle, the minimum side regular hexagon containing a set of points
    # can be found by noting that the center should be at the "center" of the point set
    # in a sense. Let's use the fact that for each normal direction n_k, we need:
    # dot(center, n_k) >= max_p dot(p, n_k) - S*sqrt(3)/2
    # and also for the opposite direction:
    # -dot(center, n_k) >= max_p dot(-p, n_k) - S*sqrt(3)/2  =>  dot(center, n_k) <= -max_p dot(-p, n_k) + S*sqrt(3)/2
    
    # So: max_p dot(p, n_k) - S*sqrt(3)/2 <= dot(center, n_k) <= -max_p dot(-p, n_k) + S*sqrt(3)/2
    # Let A_k = max_p dot(p, n_k), B_k = max_p dot(-p, n_k) = -min_p dot(p, n_k)
    # Then: A_k - S*sqrt(3)/2 <= dot(center, n_k) <= -B_k + S*sqrt(3)/2
    # For feasibility: A_k - S*sqrt(3)/2 <= -B_k + S*sqrt(3)/2
    # => A_k + B_k <= S*sqrt(3)
    # => S >= (A_k + B_k) / sqrt(3)
    
    # So S_min = max_k (A_k + B_k) / sqrt(3)
    # And the center must satisfy the constraints.
    
    # Compute A_k and B_k for each of the 6 normals
    proj = all_verts @ normals.T  # (66, 6)
    A = np.max(proj, axis=0)  # max dot(p, n_k)
    B = np.max(-proj, axis=0)  # max dot(-p, n_k) = -min dot(p, n_k)
    
    S = np.max(A + B) / np.sqrt(3)
    
    # Now find a valid center. The center must satisfy:
    # A_k - S*sqrt(3)/2 <= dot(center, n_k) <= -B_k + S*sqrt(3)/2  for all k
    # Let's use the centroid of the feasible region. Since the normals span R^2,
    # we can solve for a center that satisfies all constraints.
    # A simple choice: center = centroid of all vertices, then adjust if needed.
    # Actually, let's just use the mean of the tightest constraints.
    
    # For each normal, the constraint interval is [A_k - S*sqrt(3)/2, -B_k + S*sqrt(3)/2]
    # We need dot(center, n_k) to be in this interval for all k.
    # Let's find a center by averaging the midpoints of the tightest constraints.
    
    midpoints = (A - B) / 2.0  # midpoint of each constraint interval
    # We want dot(center, n_k) ≈ midpoints_k
    # Solve: normals @ center.T ≈ midpoints
    # Using least squares: center = (normals.T @ normals)^{-1} @ normals.T @ midpoints
    try:
        center = np.linalg.lstsq(normals.T, midpoints, rcond=None)[0]
    except:
        center = np.mean(all_verts, axis=0)
    
    return S, center

def _golden_section_min(f, a, b, tol=1e-10, max_iter=100):
    """Golden section search for minimum of f on [a, b]."""
    gr = (np.sqrt(5) - 1) / 2  # golden ratio ≈ 0.618
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)
    return f((a + b) / 2), (a + b) / 2

def _validate(inner_hex_data, outer_hex_data, side):
    """Full validation check. Returns True if valid."""
    if not (np.all(np.isfinite(inner_hex_data)) and np.all(np.isfinite(outer_hex_data)) and np.isfinite(side)):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False

    inner_verts = []
    for i in range(11):
        cx, cy, ang = inner_hex_data[i]
        inner_verts.append(_hex_vertices(cx, cy, 1.0, ang))

    # Check separation using SAT
    for i in range(11):
        for j in range(i + 1, 11):
            if not _fast_separated(np.array([inner_hex_data[i,:2], inner_hex_data[j,:2]])):
                return False

    ox, oy, oang = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, side, oang)
    shifted_outer = outer_verts - np.array([ox, oy])
    shifted_inner = []
    for v in inner_verts:
        shifted_inner.append(v - np.array([ox, oy]))

    for iv in shifted_inner:
        for p in iv:
            for k in range(6):
                p1 = shifted_outer[k]
                p2 = shifted_outer[(k + 1) % 6]
                cross_val = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross_val < -1e-6:
                    return False

    for sv in shifted_inner:
        for k in range(6):
            d = np.linalg.norm(sv[(k+1)%6] - sv[k])
            if abs(d - 1.0) > 1e-8:
                return False

    return True

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580.0
    
    h = np.sqrt(3)
    
    # Configuration 1: Standard Flower + 4 in gaps (compact)
    centers1 = np.array([
        [0.0, 0.0],
        [h, 0.0],
        [h*np.cos(np.deg2rad(60)), h*np.sin(np.deg2rad(60))],
        [h*np.cos(np.deg2rad(120)), h*np.sin(np.deg2rad(120))],
        [h*np.cos(np.deg2rad(180)), h*np.sin(np.deg2rad(180))],
        [h*np.cos(np.deg2rad(240)), h*np.sin(np.deg2rad(240))],
        [h*np.cos(np.deg2rad(300)), h*np.sin(np.deg2rad(300))],
        [2*h*np.cos(np.deg2rad(30)), 2*h*np.sin(np.deg2rad(30))],
        [2*h*np.cos(np.deg2rad(150)), 2*h*np.sin(np.deg2rad(150))],
        [2*h*np.cos(np.deg2rad(210)), 2*h*np.sin(np.deg2rad(210))],
        [2*h*np.cos(np.deg2rad(330)), 2*h*np.sin(np.deg2rad(330))],
    ])
    
    # Configuration 2: More compact - move outer ring inward slightly
    # The 4 outer hexagons can be moved toward center if the gaps allow
    # In the standard flower, the gap between adjacent outer hexagons is just touching
    # Let's try a slightly different arrangement
    
    # Configuration 3: 3-4-3 grid pattern
    sp = np.sqrt(3)  # hex spacing
    centers3 = np.array([
        [0, 0], [-sp, 0], [sp, 0],
        [-sp/2, h], [sp/2, h],
        [-sp, h], [0, h], [sp, h],
        [-sp/2, 2*h], [sp/2, 2*h],
        [0, 2*h]
    ])
    
    # Configuration 4: Compact diamond
    centers4 = np.array([
        [0, 0],
        [-h, 0], [h, 0],
        [-h/2, h*np.sqrt(3)/2], [h/2, h*np.sqrt(3)/2],
        [-h/2, -h*np.sqrt(3)/2], [h/2, -h*np.sqrt(3)/2],
        [-3*h/2, h*np.sqrt(3)/2], [3*h/2, h*np.sqrt(3)/2],
        [-3*h/2, -h*np.sqrt(3)/2], [3*h/2, -h*np.sqrt(3)/2],
    ])
    
    configs = [centers1, centers3, centers4]
    
    best_S = float('inf')
    best_inner = None
    best_outer = None
    
    for centers in configs:
        if time.time() > deadline:
            break
        
        # Check if this configuration is valid (all pairs separated)
        if not _fast_separated(centers):
            continue
        
        # Use golden section search over the outer hexagon angle
        def obj_angle(angle_deg):
            S, center = _min_enclosing_hexagon_side(centers, angle_deg)
            return S
        
        # Search over one full period (60 degrees)
        # The function is periodic with period 60 degrees
        best_angle_S = float('inf')
        best_angle = 0.0
        
        # Coarse search first
        for a0 in np.linspace(0, 60, 25):
            S_test, _ = _min_enclosing_hexagon_side(centers, float(a0))
            if S_test < best_angle_S:
                best_angle_S = S_test
                best_angle = float(a0)
        
        # Fine search around best coarse result
        a_lo = max(0, best_angle - 30)
        a_hi = min(60, best_angle + 30)
        
        # Golden section in the refined range
        S_gold, angle_gold = _golden_section_min(obj_angle, a_lo, a_hi, tol=1e-10, max_iter=80)
        
        if S_gold < best_angle_S:
            best_angle_S = S_gold
            best_angle = angle_gold
        
        # Also try the exact best_angle from coarse
        S_test, _ = _min_enclosing_hexagon_side(centers, best_angle)
        if S_test < best_angle_S:
            best_angle_S = S_test
        
        # Get the optimal center for the best angle
        S_final, center_final = _min_enclosing_hexagon_side(centers, best_angle)
        
        if S_final < best_S:
            best_S = S_final
            best_inner = centers.copy()
            best_outer = np.array([center_final[0], center_final[1], best_angle])
    
    if best_inner is None:
        # Fallback
        best_inner = centers1
        centroid = np.mean(best_inner, axis=0)
        best_outer = np.array([centroid[0], centroid[1], 0.0])
        best_S, _ = _min_enclosing_hexagon_side(best_inner, 0.0)
    
    # Local refinement: optimize the positions of all 11 hexagons
    if time.time() < deadline - 60:
        def _separation_penalty(centers):
            """Penalty for overlapping hexagons."""
            pen = 0.0
            n = len(centers)
            for i in range(n):
                for j in range(i + 1, n):
                    d = centers[j] - centers[i]
                    # For two unit hexagons with same orientation, they overlap iff
                    # for all 6 normals, |dot(d, n)| <= sqrt(3)
                    a_rad = 0.0
                    normal_angles = a_rad + np.deg2rad(30) + np.deg2rad(60 * np.arange(6))
                    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
                    proj = normals @ d
                    min_gap = np.sqrt(3) - np.max(np.abs(proj))
                    if min_gap > 0:
                        pen += min_gap ** 2
            return pen
        
        def _full_objective(params):
            centers = params[:22].reshape(11, 2)
            ox, oy, oang = params[22], params[23], params[24]
            
            # Separation penalty
            sep_pen = _separation_penalty(centers)
            
            # Compute required outer hexagon side
            all_verts = []
            for i in range(11):
                v = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0)
                all_verts.append(v)
            all_verts = np.vstack(all_verts)
            
            a_rad = np.deg2rad(oang)
            normal_angles = a_rad + np.deg2rad(30) + np.deg2rad(60 * np.arange(6))
            normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
            
            proj = all_verts @ normals.T
            A = np.max(proj, axis=0)
            B = np.max(-proj, axis=0)
            S = np.max(A + B) / np.sqrt(3)
            
            # Containment penalty: check if all vertices are inside
            outer_verts = _hex_vertices(ox, oy, S, oang)
            shifted_outer = outer_verts - np.array([ox, oy])
            cont_pen = 0.0
            for iv in all_verts.reshape(11, 6, 2):
                for p in iv:
                    ps = p - np.array([ox, oy])
                    for k in range(6):
                        p1 = shifted_outer[k]
                        p2 = shifted_outer[(k + 1) % 6]
                        cross_val = (p2[0] - p1[0]) * (ps[1] - p1[1]) - (p2[1] - p1[1]) * (ps[0] - p1[0])
                        if cross_val < 0:
                            cont_pen += 10.0
                            break
                    else:
                        continue
                    break
            
            return S + sep_pen + cont_pen
        
        x0 = np.concatenate([best_inner.flatten(), best_outer])
        
        try:
            res = minimize(_full_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            new_centers = res.x[:22].reshape(11, 2)
            new_outer = res.x[22:25]
            
            # Verify the refined solution
            inner_hex_data_test = np.zeros((11, 3))
            for i in range(11):
                inner_hex_data_test[i] = [new_centers[i, 0], new_centers[i, 1], 0.0]
            
            if _validate(inner_hex_data_test, new_outer, new_outer[2]):
                if new_outer[2] < best_S:
                    best_inner = new_centers
                    best_outer = new_outer
                    best_S = new_outer[2]
        except Exception:
            pass
    
    # Final golden-section refinement on the best configuration
    if time.time() < deadline - 30:
        def _obj_angle_final(angle_deg):
            S, _ = _min_enclosing_hexagon_side(best_inner, angle_deg)
            return S
        
        # Fine golden section around current best angle
        a0 = best_outer[2] % 60
        a_lo = max(0, a0 - 5)
        a_hi = min(60, a0 + 5)
        S_ref, ang_ref = _golden_section_min(_obj_angle_final, a_lo, a_hi, tol=1e-12, max_iter=60)
        
        if S_ref < best_S:
            S_final, center_final = _min_enclosing_hexagon_side(best_inner, ang_ref)
            if S_final < best_S:
                best_S = S_final
                best_outer = np.array([center_final[0], center_final[1], ang_ref])
    
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [best_inner[i, 0], best_inner[i, 1], 0.0]
    
    outer_hex_data = best_outer
    
    if not _validate(inner_hex_data, outer_hex_data, best_S):
        # Bump side slightly
        best_S += 1e-4
        if not _validate(inner_hex_data, outer_hex_data, best_S):
            best_S += 1e-3
            if not _validate(inner_hex_data, outer_hex_data, best_S):
                # Last resort: use a known valid configuration
                best_inner = np.array([
                    [0, 0], [-2.5, 0], [2.5, 0],
                    [-1.25, 2.17], [1.25, 2.17],
                    [-1.25, -2.17], [1.25, -2.17],
                    [-3.75, 2.17], [3.75, 2.17],
                    [-3.75, -2.17], [3.75, -2.17]
                ])
                inner_hex_data = np.zeros((11, 3))
                for i in range(11):
                    inner_hex_data[i] = [best_inner[i, 0], best_inner[i, 1], 0.0]
                centroid = np.mean(best_inner, axis=0)
                best_outer = np.array([centroid[0], centroid[1], 0.0])
                best_S, _ = _min_enclosing_hexagon_side(best_inner, 0.0)
                best_S += 1e-3
    
    return inner_hex_data, outer_hex_data, best_S

# EVOLVE-BLOCK-END

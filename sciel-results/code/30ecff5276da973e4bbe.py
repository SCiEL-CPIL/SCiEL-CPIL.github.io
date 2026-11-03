import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def hexagon_vertices(cx, cy, angle_deg=0.0):
        """Return 6 vertices of a unit regular hexagon."""
        a = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = a + k * math.pi / 3.0
            verts.append((cx + math.cos(theta), cy + math.sin(theta)))
        return np.array(verts)

    def hexagon_normals(angle_deg=0.0):
        """Return 6 outward unit normals of a unit regular hexagon."""
        a = math.radians(angle_deg)
        normals = []
        for k in range(6):
            theta = a + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(theta), math.sin(theta)))
        return np.array(normals)

    def check_hex_overlap(v1, v2, n1, n2, eps=1e-9):
        """Check if two hexagons overlap using SAT (Separating Axis Theorem).
        v1, v2: vertex arrays (6,2), n1, n2: normal arrays (6,2)
        Returns True if they overlap (including touching with eps tolerance)."""
        # Check all 12 axes (6 from each hexagon)
        for normals in [n1, n2]:
            for n in normals:
                # Project both hexagons onto this axis
                p1 = v1 @ n
                p2 = v2 @ n
                min1, max1 = np.min(p1), np.max(p1)
                min2, max2 = np.min(p2), np.max(p2)
                # Check for separation
                if max1 + eps < min2 or max2 + eps < min1:
                    return False
        return True

    def check_all_no_overlap(positions, angles_deg, eps=1e-9):
        """Check that no two hexagons overlap."""
        n = len(positions)
        verts = []
        normals = []
        for i in range(n):
            cx, cy = positions[i]
            a = angles_deg[i]
            verts.append(hexagon_vertices(cx, cy, a))
            normals.append(hexagon_normals(a))
        
        for i in range(n):
            for j in range(i+1, n):
                if check_hex_overlap(verts[i], verts[j], normals[i], normals[j], eps):
                    return False
        return True

    def compute_containing_hexagon(pts, center, beta_deg):
        """Compute the minimum side length S of a hexagon with given center and rotation
        that contains all points. pts: (n,2) array, center: (cx,cy), beta_deg: rotation."""
        beta = math.radians(beta_deg)
        normals = hexagon_normals(math.degrees(beta))
        centered = pts - np.array(center)
        projs = centered @ normals.T
        D = float(np.max(np.abs(projs)))
        S = (D + APOTHEM) / APOTHEM
        return S, D

    def optimize_container(pts, initial_center=None, initial_beta=0.0):
        """Optimize center and rotation to minimize containing hexagon side length."""
        if initial_center is None:
            initial_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        
        best_S = float('inf')
        best_center = initial_center
        best_beta = initial_beta

        # Phase 1: Coarse grid over beta
        n_coarse = 37
        for i in range(n_coarse):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_coarse - 1)
            
            # LP to find optimal center for this beta
            normals = hexagon_normals(math.degrees(beta))
            n_pts = len(pts)
            
            A_ub = np.zeros((12 * n_pts, 3))
            b_ub = np.zeros(12 * n_pts)
            
            for k in range(6):
                nx, ny = normals[k]
                idx1 = 2 * k * n_pts
                idx2 = idx1 + n_pts
                
                A_ub[idx1:idx1+n_pts, 0] = -nx
                A_ub[idx1:idx1+n_pts, 1] = -ny
                A_ub[idx1:idx1+n_pts, 2] = -1.0
                b_ub[idx1:idx1+n_pts] = -(pts[:, 0] * nx + pts[:, 1] * ny)
                
                A_ub[idx2:idx2+n_pts, 0] = nx
                A_ub[idx2:idx2+n_pts, 1] = ny
                A_ub[idx2:idx2+n_pts, 2] = -1.0
                b_ub[idx2:idx2+n_pts] = pts[:, 0] * nx + pts[:, 1] * ny

            c_obj = np.array([0.0, 0.0, 1.0])
            bounds = [(None, None), (None, None), (0, None)]
            
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.status == 0:
                    D = float(res.x[2])
                    cx, cy = float(res.x[0]), float(res.x[1])
                    S = (D + APOTHEM) / APOTHEM
                    if S < best_S:
                        best_S = S
                        best_center = (cx, cy)
                        best_beta = math.degrees(beta)
            except Exception:
                pass

        # Phase 2: Refine beta with scalar optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def obj_beta(beta_rad):
                    beta_deg = math.degrees(beta_rad)
                    S, _ = compute_containing_hexagon(pts, best_center, beta_deg)
                    return S
                
                lo = max(0.0, math.radians(best_beta) - math.pi / 30)
                hi = min(math.pi / 6, math.radians(best_beta) + math.pi / 30)
                res = minimize_scalar(obj_beta, bounds=(lo, hi), method='bounded',
                                     options={'maxiter': 100, 'xatol': 1e-10})
                if res.fun < best_S:
                    best_beta = math.degrees(res.x)
                    best_S = float(res.fun)
        except Exception:
            pass

        # Phase 3: Joint optimization of center and beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def joint_obj(params):
                    cx, cy, beta_rad = params
                    beta_deg = math.degrees(beta_rad)
                    S, _ = compute_containing_hexagon(pts, (cx, cy), beta_deg)
                    return S
                
                init = [best_center[0], best_center[1], math.radians(best_beta)]
                res = minimize(joint_obj, init, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
                if res.fun < best_S:
                    best_S = float(res.fun)
                    best_center = (float(res.x[0]), float(res.x[1]))
                    best_beta = math.degrees(float(res.x[2]))
        except Exception:
            pass

        return best_S, best_center, best_beta

    # Generate candidate configurations
    # Each config is a list of (dx, dy, angle_deg) relative to a reference point
    candidates = []

    # Config 1: Center + 6 neighbors + 4 in second ring (hexagonal packing)
    config1 = [
        (0, 0, 0),
        (math.sqrt(3), 0, 0),
        (math.sqrt(3)/2, 1.5, 0),
        (-math.sqrt(3)/2, 1.5, 0),
        (-math.sqrt(3), 0, 0),
        (-math.sqrt(3)/2, -1.5, 0),
        (math.sqrt(3)/2, -1.5, 0),
        (math.sqrt(3), 3.0, 0),
        (0, 3.0, 0),
        (-math.sqrt(3), 3.0, 0),
        (2*math.sqrt(3), 1.5, 0),
    ]
    candidates.append(config1)

    # Config 2: 3x3 block + 2 extras
    config2 = []
    for r in range(3):
        for c in range(3):
            x = c * math.sqrt(3) + (r % 2) * math.sqrt(3) / 2
            y = r * 1.5
            config2.append((x, y, 0))
    config2.append((0, 4.5, 0))
    config2.append((math.sqrt(3), 4.5, 0))
    candidates.append(config2)

    # Config 3: 4x3 block
    config3 = []
    for r in range(3):
        for c in range(4):
            x = c * math.sqrt(3) + (r % 2) * math.sqrt(3) / 2
            y = r * 1.5
            config3.append((x, y, 0))
    candidates.append(config3)

    # Config 4: Star pattern - center + 6 + 4 opposite
    config4 = [
        (0, 0, 0),
        (math.sqrt(3), 0, 0),
        (math.sqrt(3)/2, 1.5, 0),
        (-math.sqrt(3)/2, 1.5, 0),
        (-math.sqrt(3), 0, 0),
        (-math.sqrt(3)/2, -1.5, 0),
        (math.sqrt(3)/2, -1.5, 0),
        (2*math.sqrt(3), 0, 0),
        (-2*math.sqrt(3), 0, 0),
        (0, 3.0, 0),
        (0, -3.0, 0),
    ]
    candidates.append(config4)

    # Config 5: 5+6 row pattern
    config5 = []
    for c in range(5):
        config5.append((c * math.sqrt(3), 0, 0))
    for c in range(6):
        x = c * math.sqrt(3) + math.sqrt(3) / 2
        y = 1.5
        config5.append((x, y, 0))
    candidates.append(config5)

    # Config 6: 3x4 block
    config6 = []
    for r in range(4):
        for c in range(3):
            x = c * math.sqrt(3) + (r % 2) * math.sqrt(3) / 2
            y = r * 1.5
            config6.append((x, y, 0))
    candidates.append(config6)

    # Config 7: Cross pattern
    config7 = [
        (0, 0, 0),
        (math.sqrt(3), 0, 0),
        (-math.sqrt(3), 0, 0),
        (math.sqrt(3)/2, 1.5, 0),
        (-math.sqrt(3)/2, 1.5, 0),
        (math.sqrt(3)/2, -1.5, 0),
        (-math.sqrt(3)/2, -1.5, 0),
        (2*math.sqrt(3), 0, 0),
        (-2*math.sqrt(3), 0, 0),
        (0, 3.0, 0),
        (0, -3.0, 0),
    ]
    candidates.append(config7)

    # Config 8: 4+4+3 pattern
    config8 = []
    for c in range(4):
        config8.append((c * math.sqrt(3), 0, 0))
    for c in range(4):
        x = c * math.sqrt(3) + math.sqrt(3) / 2
        y = 1.5
        config8.append((x, y, 0))
    for c in range(3):
        x = c * math.sqrt(3)
        y = 3.0
        config8.append((x, y, 0))
    candidates.append(config8)

    # Config 9: 3+4+3+1 pattern
    config9 = []
    for c in range(3):
        config9.append((c * math.sqrt(3), 0, 0))
    for c in range(4):
        x = c * math.sqrt(3) + math.sqrt(3) / 2
        y = 1.5
        config9.append((x, y, 0))
    for c in range(3):
        x = c * math.sqrt(3)
        y = 3.0
        config9.append((x, y, 0))
    config9.append((math.sqrt(3)/2, 4.5, 0))
    candidates.append(config9)

    # Config 10: 2+3+2+3+1 pattern
    config10 = []
    for c in range(2):
        config10.append((c * math.sqrt(3), 0, 0))
    for c in range(3):
        x = c * math.sqrt(3) + math.sqrt(3) / 2
        y = 1.5
        config10.append((x, y, 0))
    for c in range(2):
        x = c * math.sqrt(3)
        y = 3.0
        config10.append((x, y, 0))
    for c in range(3):
        x = c * math.sqrt(3) + math.sqrt(3) / 2
        y = 4.5
        config10.append((x, y, 0))
    config10.append((math.sqrt(3)/2, 6.0, 0))
    candidates.append(config10)

    # Config 11: Diagonal arrangement
    config11 = []
    for i in range(11):
        x = i * math.sqrt(3) / 2
        y = i * 0.75
        config11.append((x, y, 0))
    candidates.append(config11)

    # Config 12: Compact cluster
    config12 = [
        (0, 0, 0),
        (math.sqrt(3), 0, 0),
        (math.sqrt(3)/2, 1.5, 0),
        (-math.sqrt(3)/2, 1.5, 0),
        (-math.sqrt(3), 0, 0),
        (-math.sqrt(3)/2, -1.5, 0),
        (math.sqrt(3)/2, -1.5, 0),
        (math.sqrt(3)/2, 3.0, 0),
        (-math.sqrt(3)/2, 3.0, 0),
        (math.sqrt(3), -1.5, 0),
        (-math.sqrt(3), -1.5, 0),
    ]
    candidates.append(config12)

    # Config 13: 5+5+1 pattern
    config13 = []
    for c in range(5):
        config13.append((c * math.sqrt(3), 0, 0))
    for c in range(5):
        x = c * math.sqrt(3) + math.sqrt(3) / 2
        y = 1.5
        config13.append((x, y, 0))
    config13.append((math.sqrt(3), 3.0, 0))
    candidates.append(config13)

    # Config 14: 4+3+4 pattern
    config14 = []
    for c in range(4):
        config14.append((c * math.sqrt(3), 0, 0))
    for c in range(3):
        x = c * math.sqrt(3) + math.sqrt(3) / 2
        y = 1.5
        config14.append((x, y, 0))
    for c in range(4):
        x = c * math.sqrt(3)
        y = 3.0
        config14.append((x, y, 0))
    candidates.append(config14)

    # Config 15: 3+3+3+2 pattern
    config15 = []
    for c in range(3):
        config15.append((c * math.sqrt(3), 0, 0))
    for c in range(3):
        x = c * math.sqrt(3) + math.sqrt(3) / 2
        y = 1.5
        config15.append((x, y, 0))
    for c in range(3):
        x = c * math.sqrt(3)
        y = 3.0
        config15.append((x, y, 0))
    for c in range(2):
        x = c * math.sqrt(3) + math.sqrt(3) / 2
        y = 4.5
        config15.append((x, y, 0))
    candidates.append(config15)

    # Config 16: Rotated hexagon pattern
    config16 = [
        (0, 0, 30),
        (math.sqrt(3), 0, 30),
        (math.sqrt(3)/2, 1.5, 30),
        (-math.sqrt(3)/2, 1.5, 30),
        (-math.sqrt(3), 0, 30),
        (-math.sqrt(3)/2, -1.5, 30),
        (math.sqrt(3)/2, -1.5, 30),
        (math.sqrt(3), 3.0, 30),
        (0, 3.0, 30),
        (-math.sqrt(3), 3.0, 30),
        (2*math.sqrt(3), 1.5, 30),
    ]
    candidates.append(config16)

    # Config 17: 2+4+4+1 pattern
    config17 = []
    for c in range(2):
        config17.append((c * math.sqrt(3), 0, 0))
    for c in range(4):
        x = c * math.sqrt(3) + math.sqrt(3) / 2
        y = 1.5
        config17.append((x, y, 0))
    for c in range(4):
        x = c * math.sqrt(3)
        y = 3.0
        config17.append((x, y, 0))
    config17.append((math.sqrt(3), 4.5, 0))
    candidates.append(config17)

    # Config 18: 3+2+3+3 pattern
    config18 = []
    for c in range(3):
        config18.append((c * math.sqrt(3), 0, 0))
    for c in range(2):
        x = c * math.sqrt(3) + math.sqrt(3) / 2
        y = 1.5
        config18.append((x, y, 0))
    for c in range(3):
        x = c * math.sqrt(3)
        y = 3.0
        config18.append((x, y, 0))
    for c in range(3):
        x = c * math.sqrt(3) + math.sqrt(3) / 2
        y = 4.5
        config18.append((x, y, 0))
    candidates.append(config18)

    # Config 19: 4+2+2+2+1 pattern
    config19 = []
    for c in range(4):
        config19.append((c * math.sqrt(3), 0, 0))
    for c in range(2):
        x = c * math.sqrt(3) + math.sqrt(3) / 2
        y = 1.5
        config19.append((x, y, 0))
    for c in range(2):
        x = c * math.sqrt(3)
        y = 3.0
        config19.append((x, y, 0))
    for c in range(2):
        x = c * math.sqrt(3) + math.sqrt(3) / 2
        y = 4.5
        config19.append((x, y, 0))
    config19.append((math.sqrt(3)/2, 6.0, 0))
    candidates.append(config19)

    # Config 20: 2+2+3+2+2 pattern
    config20 = []
    for c in range(2):
        config20.append((c * math.sqrt(3), 0, 0))
    for c in range(2):
        x = c * math.sqrt(3) + math.sqrt(3) / 2
        y = 1.5
        config20.append((x, y, 0))
    for c in range(3):
        x = c * math.sqrt(3)
        y = 3.0
        config20.append((x, y, 0))
    for c in range(2):
        x = c * math.sqrt(3) + math.sqrt(3) / 2
        y = 4.5
        config20.append((x, y, 0))
    config20.append((math.sqrt(3)/2, 6.0, 0))
    candidates.append(config20)

    # Sort candidates by bounding box area (smaller is better)
    def config_area(config):
        xs = [p[0] for p in config]
        ys = [p[1] for p in config]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))
    
    candidates.sort(key=config_area)
    
    # Try top candidates
    best_global_S = float('inf')
    best_global_config = None
    best_global_center = None
    best_global_beta = 0.0

    for config in candidates[:20]:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Convert to absolute positions (centered)
        xs = [p[0] for p in config]
        ys = [p[1] for p in config]
        cx_mean = (max(xs) + min(xs)) / 2
        cy_mean = (max(ys) + min(ys)) / 2
        
        pts = np.array([(p[0] - cx_mean, p[1] - cy_mean) for p in config])
        angles_deg = [p[2] for p in config]
        
        # Check for overlaps
        positions = list(pts)
        if not check_all_no_overlap(positions, angles_deg, eps=1e-8):
            continue
        
        # Optimize container
        S, center, beta_deg = optimize_container(pts)
        
        if S < best_global_S:
            best_global_S = S
            best_global_config = config
            best_global_center = center
            best_global_beta = beta_deg

    # Fallback: use first valid config
    if best_global_config is None:
        for config in candidates:
            xs = [p[0] for p in config]
            ys = [p[1] for p in config]
            cx_mean = (max(xs) + min(xs)) / 2
            cy_mean = (max(ys) + min(ys)) / 2
            pts = np.array([(p[0] - cx_mean, p[1] - cy_mean) for p in config])
            angles_deg = [p[2] for p in config]
            positions = list(pts)
            if check_all_no_overlap(positions, angles_deg, eps=1e-8):
                S, center, beta_deg = optimize_container(pts)
                if S < best_global_S:
                    best_global_S = S
                    best_global_config = config
                    best_global_center = center
                    best_global_beta = beta_deg
                break
    
    # Final fallback
    if best_global_config is None:
        best_global_config = candidates[0]
        xs = [p[0] for p in best_global_config]
        ys = [p[1] for p in best_global_config]
        cx_mean = (max(xs) + min(xs)) / 2
        cy_mean = (max(ys) + min(ys)) / 2
        best_global_center = (cx_mean, cy_mean)
        best_global_beta = 0.0
        pts = np.array([(p[0] - cx_mean, p[1] - cy_mean) for p in best_global_config])
        best_global_S, _, _ = optimize_container(pts)

    # Build final output
    xs = [p[0] for p in best_global_config]
    ys = [p[1] for p in best_global_config]
    cx_mean = (max(xs) + min(xs)) / 2
    cy_mean = (max(ys) + min(ys)) / 2
    
    # Apply center offset
    cx, cy = best_global_center
    offset_x = cx - cx_mean
    offset_y = cy - cy_mean
    
    inner = []
    for p in best_global_config:
        x = p[0] + offset_x
        y = p[1] + offset_y
        angle = p[2]
        inner.append([float(x), float(y), float(angle)])
    
    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(best_global_beta)
    
    return inner, center, side, rotation

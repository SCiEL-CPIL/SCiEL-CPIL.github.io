import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0
_TIME_START = time.time()

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def get_hex_vertices_batch(xys_rots):
    """Batch compute vertices for all hexagons. xys_rots is (N, 3) array."""
    N_pts = len(xys_rots)
    base_angles = np.radians(np.array([0, 60, 120, 180, 240, 300]))
    rots = np.radians(xys_rots[:, 2])
    angles = base_angles[None, :] + rots[:, None]
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    cx = xys_rots[:, 0]
    cy = xys_rots[:, 1]
    x = cx[:, None] + cos_a
    y = cy[:, None] + sin_a
    return np.stack((x, y), axis=-1)

def sat_overlap(v1, v2):
    """SAT for two convex hexagons. Returns separation distance (positive) or negative penetration."""
    axes = []
    for v in (v1, v2):
        for i in range(6):
            e = v[(i+1) % 6] - v[i]
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                axes.append(n)
    
    min_overlap = float('inf')
    for n in axes:
        proj1 = v1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = v2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    return -min_overlap

def check_no_overlap(vertices_list, eps=1e-9):
    """Check all pairs for non-overlap using SAT."""
    for i in range(len(vertices_list)):
        for j in range(i + 1, len(vertices_list)):
            sep = sat_overlap(vertices_list[i], vertices_list[j])
            if sep < -eps:
                return False
    return True

def outer_hex_side_from_points(points, outer_rot_deg):
    """Min side of regular hexagon (centered at origin) with given rotation enclosing all points."""
    angles = np.radians(np.array([90, 150, 210, 270, 330, 30]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    sqrt3 = np.sqrt(3.0)
    
    def make_lattice_pattern(rows, cols, rot_deg=0.0):
        """Create a hex grid of unit hexagons on triangular lattice."""
        pts = []
        for r in range(rows):
            for c in range(cols):
                x = c * sqrt3 + (r % 2) * (sqrt3 / 2)
                y = r * 1.5
                pts.append([x, y, rot_deg])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def make_ring_pattern(rot_deg=0.0):
        """1 center + 6 ring + 5 outer."""
        pts = [[0, 0, rot_deg]]
        for k in range(6):
            ang = np.radians(60 * k)
            pts.append([sqrt3 * np.cos(ang), sqrt3 * np.sin(ang), rot_deg])
        for k in range(5):
            ang = np.radians(60 * k + 30)
            pts.append([2*sqrt3 * np.cos(ang), 2*sqrt3 * np.sin(ang), rot_deg])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    # Generate diverse initial configurations
    configs = []
    configs.append(('grid_3x4', make_lattice_pattern(3, 4)))
    configs.append(('grid_4x3', make_lattice_pattern(4, 3)))
    configs.append(('grid_2x6', make_lattice_pattern(2, 6)))
    configs.append(('grid_6x2', make_lattice_pattern(6, 2)))
    configs.append(('ring', make_ring_pattern()))
    
    # Rotated variants
    for name, base in list(configs):
        rotated = base.copy()
        rotated[:, 2] = 30.0
        configs.append((name + '_r30', rotated))
    
    best_L = float('inf')
    best_params = None
    
    outer_rots = [0.0, 30.0, 15.0, 45.0, 22.5, 37.5]
    
    for cfg_name, cfg in configs:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        for orot in outer_rots:
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Check validity
            verts = get_hex_vertices_batch(cfg)
            if not check_no_overlap(verts, eps=1e-8):
                continue
            
            all_pts = np.vstack(verts)
            L_init = outer_hex_side_from_points(all_pts, orot)
            
            if L_init < best_L:
                best_L = L_init
                best_params = np.concatenate((cfg.flatten(), [orot]))
            
            # Optimize with Nelder-Mead
            def objective(params):
                if time.time() - start_time > DEADLINE - 5:
                    return 1e6
                
                xys_rots = params[:3*N].reshape(N, 3)
                outer_rot = params[3*N]
                verts = get_hex_vertices_batch(xys_rots)
                
                penalty = 0.0
                for i in range(N):
                    for j in range(i + 1, N):
                        sep = sat_overlap(verts[i], verts[j])
                        if sep < 0:
                            penalty += sep ** 2
                
                all_pts = np.vstack(verts)
                L = outer_hex_side_from_points(all_pts, outer_rot)
                return L + 50.0 * penalty
            
            try:
                init_params = np.concatenate((cfg.flatten(), [orot]))
                
                res = minimize(
                    objective,
                    init_params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 80,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
                
                xys_rots = res.x[:3*N].reshape(N, 3)
                outer_rot = res.x[3*N]
                verts = get_hex_vertices_batch(xys_rots)
                
                if check_no_overlap(verts, eps=1e-8):
                    all_pts = np.vstack(verts)
                    L_val = outer_hex_side_from_points(all_pts, outer_rot)
                    
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res.x.copy()
                        
            except Exception:
                pass
    
    # Final refinement
    if best_params is not None and time.time() - start_time < DEADLINE - 25:
        try:
            def final_obj(params):
                if time.time() - start_time > DEADLINE - 5:
                    return 1e6
                
                xys_rots = params[:3*N].reshape(N, 3)
                outer_rot = params[3*N]
                verts = get_hex_vertices_batch(xys_rots)
                
                penalty = 0.0
                for i in range(N):
                    for j in range(i + 1, N):
                        sep = sat_overlap(verts[i], verts[j])
                        if sep < 0:
                            penalty += sep ** 2
                
                all_pts = np.vstack(verts)
                L = outer_hex_side_from_points(all_pts, outer_rot)
                return L + 100.0 * penalty
            
            res = minimize(
                final_obj,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 150,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            xys_rots = res.x[:3*N].reshape(N, 3)
            outer_rot = res.x[3*N]
            verts = get_hex_vertices_batch(xys_rots)
            
            if check_no_overlap(verts, eps=1e-8):
                all_pts = np.vstack(verts)
                L_val = outer_hex_side_from_points(all_pts, outer_rot)
                
                if L_val < best_L:
                    best_L = L_val
                    best_params = res.x.copy()
                    
        except Exception:
            pass
    
    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(N)]
        pts = np.array([[x, y] for x, y, r in inner])
        L_val = outer_hex_side_from_points(pts, 0.0)
        return inner, (0.0, 0.0), L_val, 0.0
    
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(outer_rot)
    
    return inner, center, side, angle

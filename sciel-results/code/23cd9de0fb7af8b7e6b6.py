import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + d * np.cos(angles),
        cy + d * np.sin(angles)
    ])

def _check_separation(hex1, hex2, tol=1e-6):
    """Check if two hexagons are separated using SAT."""
    v1 = _hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
    v2 = _hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])
    
    for verts in [v1, v2]:
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i+1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-15:
                normal = normal / norm
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                if max1 + tol < min2 or max2 + tol < min1:
                    return True
    
    return False

def _vertex_inside_outer(v, outer_verts):
    """Check if vertex v is inside outer hexagon."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i+1) % 6]
        cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
        if cross < -1e-6:
            return False
    return True

def _validate(inner_data, outer_data, s):
    """Validate the packing."""
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    
    # Check outer vertices distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    # Check inner hexagons
    inner_verts_list = []
    for i in range(12):
        verts = _hex_vertices(inner_data[i,0], inner_data[i,1], 1.0, inner_data[i,2])
        inner_verts_list.append(verts)
        
        # Check unit edges
        for j in range(6):
            p1 = verts[j]
            p2 = verts[(j+1) % 6]
            if abs(np.linalg.norm(p2 - p1) - 1.0) > 1e-8:
                return False
        
        # Check inside outer
        for v in verts:
            if not _vertex_inside_outer(v, outer_verts):
                return False
    
    # Check separation
    for i in range(12):
        for j in range(i+1, 12):
            if not _check_separation(inner_data[i], inner_data[j]):
                return False
    
    return True

def _min_enclosing_hex_side(centers, circumradius=1.0):
    """
    Find the minimum side length s of a regular hexagon (centered at origin, angle 0)
    that contains all points at distance circumradius from each center.
    """
    normals = []
    for k in range(6):
        angle = np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(angle), np.sin(angle)]))
    normals = np.array(normals)
    
    unit_verts = _hex_vertices(0, 0, 1.0, 0.0)
    
    max_required_s = 0.0
    for center in centers:
        for n in normals:
            support = np.max(unit_verts @ n)
            required_s = (center @ n + support) / (np.sqrt(3) / 2)
            if required_s > max_required_s:
                max_required_s = required_s
    
    return max_required_s

def _initial_honeycomb():
    """Generate initial honeycomb arrangement of 12 hexagons."""
    sqrt3 = np.sqrt(3)
    col_spacing = 1.5
    row_spacing = sqrt3
    
    positions = []
    for col in range(4):
        for row in range(3):
            x = col * col_spacing
            if col % 2 == 0:
                y = row * row_spacing
            else:
                y = (row + 0.5) * row_spacing
            positions.append([x, y, 0.0])
    
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    for p in positions:
        p[0] -= cx
        p[1] -= cy
    
    return np.array(positions)

def _initial_circular():
    """Generate initial circular arrangement of 12 hexagons."""
    positions = []
    # Center hexagon
    positions.append([0.0, 0.0, 0.0])
    # 6 hexagons around center
    for i in range(6):
        angle = np.radians(60 * i)
        x = 2.0 * np.cos(angle)
        y = 2.0 * np.sin(angle)
        positions.append([x, y, 0.0])
    # 5 more hexagons in outer ring
    for i in range(5):
        angle = np.radians(60 * i + 30)
        x = 3.46 * np.cos(angle)
        y = 3.46 * np.sin(angle)
        positions.append([x, y, 0.0])
    
    return np.array(positions)

def _initial_grid():
    """Generate initial grid arrangement."""
    positions = []
    sqrt3 = np.sqrt(3)
    # 3x4 grid with offset
    for i in range(3):
        for j in range(4):
            x = j * 1.5
            y = i * sqrt3
            if j % 2 == 1:
                y += sqrt3 / 2
            positions.append([x, y, 0.0])
    
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    for p in positions:
        p[0] -= cx
        p[1] -= cy
    
    return np.array(positions)

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s kill
    
    sqrt3 = np.sqrt(3)
    
    # Try multiple initial configurations
    initial_configs = [
        _initial_honeycomb(),
        _initial_circular(),
        _initial_grid()
    ]
    
    best_value = 0.0
    best_result = None
    
    for init_idx, inner_data in enumerate(initial_configs):
        centers = inner_data[:, :2]
        s = _min_enclosing_hex_side(centers)
        s *= 1.001
        outer_data = np.array([0.0, 0.0, 0.0])
        
        if not _validate(inner_data, outer_data, s):
            s = 6.0
        
        if best_result is None or 1.0/s > best_value:
            best_value = 1.0 / s
            best_result = (inner_data.copy(), outer_data.copy(), s)
        
        if time.time() > deadline - 30:
            break
    
    if best_result is None:
        inner_data = _initial_honeycomb()
        s = 6.0
        outer_data = np.array([0.0, 0.0, 0.0])
        return (inner_data, outer_data, s)
    
    # Optimize to minimize s
    def objective(params):
        inner_xy = params[:24].reshape(12, 2)
        s_val = params[24]
        
        if s_val < np.sqrt(12):
            return 1e6
        
        outer_verts = _hex_vertices(0, 0, s_val, 0)
        penalty = 0.0
        
        for i in range(12):
            verts = _hex_vertices(inner_xy[i,0], inner_xy[i,1], 1.0, 0.0)
            
            # Check inside outer
            for v in verts:
                for j in range(6):
                    p1 = outer_verts[j]
                    p2 = outer_verts[(j+1) % 6]
                    cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                    if cross < 0:
                        penalty += abs(cross)
            
            # Check separation from other hexagons
            for j in range(i+1, 12):
                verts2 = _hex_vertices(inner_xy[j,0], inner_xy[j,1], 1.0, 0.0)
                
                separated = False
                for verts in [verts, verts2]:
                    for k in range(6):
                        p1 = verts[k]
                        p2 = verts[(k+1) % 6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-15:
                            normal = normal / norm
                            proj1 = verts @ normal
                            proj2 = verts2 @ normal
                            min1, max1 = proj1.min(), proj1.max()
                            min2, max2 = proj2.min(), proj2.max()
                            if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                                separated = True
                                break
                    if separated:
                        break
                
                if not separated:
                    min_gap = np.inf
                    for verts in [verts, verts2]:
                        for k in range(6):
                            p1 = verts[k]
                            p2 = verts[(k+1) % 6]
                            edge = p2 - p1
                            normal = np.array([-edge[1], edge[0]])
                            norm = np.linalg.norm(normal)
                            if norm > 1e-15:
                                normal = normal / norm
                                proj1 = verts @ normal
                                proj2 = verts2 @ normal
                                min1, max1 = proj1.min(), proj1.max()
                                min2, max2 = proj2.min(), proj2.max()
                                gap = min(min2 - max1, min1 - max2)
                                if gap < 0:
                                    min_gap = min(min_gap, -gap)
                    
                    if min_gap < np.inf:
                        penalty += min_gap ** 2
        
        return s_val + 1000 * penalty
    
    # Optimize from each initial configuration
    for init_idx, inner_data in enumerate(initial_configs):
        if time.time() > deadline - 20:
            break
        
        centers = inner_data[:, :2]
        s_init = _min_enclosing_hex_side(centers) * 1.001
        
        n_restarts = 8
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            if restart == 0:
                params0 = np.zeros(25)
                params0[:24] = inner_data[:, :2].flatten()
                params0[24] = s_init
            else:
                perturbation = 0.2 * np.random.randn(24)
                params0 = np.zeros(25)
                params0[:24] = inner_data[:, :2].flatten() + perturbation
                params0[24] = s_init + 0.2 * np.random.randn()
            
            try:
                result = minimize(
                    objective,
                    params0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-7,
                        'fatol': 1e-7,
                        'adaptive': True
                    }
                )
                
                if result.fun < 100:
                    final_params = result.x
                    inner_xy_final = final_params[:24].reshape(12, 2)
                    s_final = final_params[24]
                    
                    if s_final > np.sqrt(12):
                        inner_final = np.zeros((12, 3))
                        inner_final[:, 0] = inner_xy_final[:, 0]
                        inner_final[:, 1] = inner_xy_final[:, 1]
                        inner_final[:, 2] = 0.0
                        outer_final = np.array([0.0, 0.0, 0.0])
                        
                        s_exact = _min_enclosing_hex_side(inner_xy_final)
                        s_final = s_exact * 1.0001
                        
                        if _validate(inner_final, outer_final, s_final):
                            value = 1.0 / s_final
                            if value > best_value:
                                best_value = value
                                best_result = (inner_final, outer_final, s_final)
            except Exception:
                pass
    
    return best_result

import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _hex_vertices(cx, cy, s, angle_deg=0.0):
    theta = np.radians(angle_deg)
    angles = theta + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _separating_axis_check(v1, v2):
    """Return True if polygons v1, v2 are separated (no overlap)."""
    for v in (v1, v2):
        for i in range(len(v)):
            e = v[(i + 1) % len(v)] - v[i]
            n = np.array([-e[1], e[0]])
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n = n / nn
            proj1 = v1 @ n
            proj2 = v2 @ n
            if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                return True
    return False

def _vertex_in_hexagon(px, py, s, angle_deg=0.0):
    """Check if point (px,py) is inside hexagon centered at origin with side s."""
    theta = -np.radians(angle_deg)
    lx = px * np.cos(theta) - py * np.sin(theta)
    ly = px * np.sin(theta) + py * np.cos(theta)
    inradius = s * np.sqrt(3) / 2
    for i in range(6):
        angle = np.radians(60 * i + 30)
        if lx * np.cos(angle) + ly * np.sin(angle) > inradius:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = np.sqrt(3)
    
    # Compact initial arrangement: hexagonal cluster
    # 12 hexagons: use a 2-wide hexagonal strip
    positions = []
    # Two rows of 3, offset
    # Row 0: 4 hexagons at y=0
    for i in range(4):
        positions.append([(i - 1.5) * sqrt3, 0.0])
    # Row 1: 4 hexagons at y=1.5, offset by sqrt3/2
    for i in range(4):
        positions.append([(i - 1.5) * sqrt3 + sqrt3 / 2, 1.5])
    # Row 2: 4 hexagons at y=3.0
    for i in range(4):
        positions.append([(i - 1.5) * sqrt3, 3.0])
    
    positions = np.array(positions[:N])
    
    def objective(params):
        pos = params[:2 * N].reshape(N, 2)
        outer_s = params[2 * N]
        
        penalty = 0.0
        outer_verts_cache = None
        
        # Check containment in outer hexagon
        for i in range(N):
            cx, cy = pos[i]
            inner_verts = _hex_vertices(cx, cy, 1.0, 0.0)
            for v in inner_verts:
                if not _vertex_in_hexagon(v[0], v[1], outer_s, 0.0):
                    # Compute violation
                    lx, ly = v[0], v[1]
                    inradius = outer_s * np.sqrt(3) / 2
                    for j in range(6):
                        angle = np.radians(60 * j + 30)
                        dist = lx * np.cos(angle) + ly * np.sin(angle) - inradius
                        if dist > 0:
                            penalty += dist * dist
                    break
        
        # Check overlaps
        for i in range(N):
            v1 = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, 0.0)
            for j in range(i + 1, N):
                v2 = _hex_vertices(pos[j, 0], pos[j, 1], 1.0, 0.0)
                if not _separating_axis_check(v1, v2):
                    penalty += 100.0
        
        return outer_s + 0.05 * penalty
    
    # Estimate initial outer size
    max_x = np.max(np.abs(positions[:, 0]))
    max_y = np.max(np.abs(positions[:, 1]))
    init_outer_s = max(max_x * 2, max_y * 2 / (np.sqrt(3) / 2)) * 1.3
    init_outer_s = max(init_outer_s, 5.0)
    
    x0 = np.concatenate([positions.flatten(), [init_outer_s]])
    
    best_score = float('inf')
    best_pos = None
    best_s = None
    
    n_restarts = 4
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            np.random.seed(100 + restart * 7)
            x_start = x0.copy()
            x_start[:2 * N] += np.random.randn(2 * N) * (0.3 + 0.2 * restart)
            x_start[2 * N] = init_outer_s * (1.0 + 0.05 * restart)
        
        try:
            res = minimize(objective, x_start, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30})
            
            pos = res.x[:2 * N].reshape(N, 2)
            s = res.x[2 * N]
            
            # Verify
            ok = True
            for i in range(N):
                inner_verts = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, 0.0)
                for v in inner_verts:
                    if not _vertex_in_hexagon(v[0], v[1], s, 0.0):
                        ok = False
                        break
                if not ok:
                    break
            
            if ok:
                for i in range(N):
                    v1 = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, 0.0)
                    for j in range(i + 1, N):
                        v2 = _hex_vertices(pos[j, 0], pos[j, 1], 1.0, 0.0)
                        if not _separating_axis_check(v1, v2):
                            ok = False
                            break
                    if not ok:
                        break
            
            if ok and s < best_score:
                best_score = s
                best_pos = pos.copy()
                best_s = s
                
        except Exception:
            pass
    
    if best_pos is None:
        pos = positions
        s = init_outer_s
    else:
        pos = best_pos
        s = best_s
    
    inner = [[pos[i, 0], pos[i, 1], 0.0] for i in range(N)]
    return inner, (0.0, 0.0), s, 0.0

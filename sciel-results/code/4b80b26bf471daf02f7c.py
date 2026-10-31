import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    angle = np.radians(angle_deg)
    angles = angle + np.pi * np.arange(6) / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def min_outer_side(inner_verts, outer_cx, outer_cy, outer_angle_deg):
    """Minimum outer hexagon side length to contain all inner vertices."""
    oa = np.radians(outer_angle_deg)
    dirs = np.column_stack([np.cos(oa + np.pi * np.arange(6) / 3.0),
                            np.sin(oa + np.pi * np.arange(6) / 3.0)])
    # For each inner vertex, compute max projection onto each direction
    # All inner vertices flattened
    all_v = inner_verts.reshape(-1, 2)
    rel = all_v - np.array([outer_cx, outer_cy])
    # projections: (n_points, 6)
    proj = rel @ dirs.T
    max_proj = np.max(proj)
    return 2.0 / np.sqrt(3.0) * max_proj

def objective(params):
    """params: [x0,y0,a0, x1,y1,a1, ..., outer_cx, outer_cy, outer_angle]"""
    n = N
    inner_params = params[:3*n]
    outer_cx = params[3*n]
    outer_cy = params[3*n+1]
    outer_angle = params[3*n+2]
    
    # Compute all inner hexagon vertices
    all_verts = []
    for i in range(n):
        x, y, a = inner_params[3*i:3*i+3]
        v = hex_vertices(x, y, a)
        all_verts.append(v)
    all_verts = np.array(all_verts)  # (n, 6, 2)
    
    s = min_outer_side(all_verts, outer_cx, outer_cy, outer_angle)
    return s

def construct():
    start_time = time.time()
    deadline = 140.0  # 150s limit, keep 10s margin
    
    best_result = None
    best_score = float('inf')
    
    # Initial configurations
    configs = []
    
    # Config 1: Linear row (baseline)
    row = []
    for k in range(-(N//2), N - N//2):
        row.append([4.0*k, 0.0, 0.0])
    configs.append(row)
    
    # Config 2: Honeycomb cluster - 1 center, 6 around, 4 more
    # Honeycomb spacing: adjacent hexagons at distance 2*inradius = sqrt(3) if edge-to-edge
    # Actually for two unit hexagons to be disjoint, center distance >= 2 (if aligned) 
    # For honeycomb packing, centers are at distance sqrt(3) * 2 = 2*sqrt(3) ≈ 3.464? No...
    # Two regular hexagons of side 1: if they share an edge, center distance = 2*inradius = 2*(sqrt(3)/2) = sqrt(3) ≈ 1.732
    # But they must be DISJOINT (no overlap), so center distance >= 2 (diameter of inscribed circle... no)
    # Actually, the minimum distance between centers of two disjoint unit hexagons depends on orientation.
    # If aligned, minimum center distance = 2 (they touch at a vertex) or sqrt(3) (they share an edge)?
    # Let me think: a unit hexagon has width 2 (vertex to opposite vertex) and height sqrt(3) (flat to flat).
    # If two hexagons are aligned and placed side by side sharing a flat edge, center distance = sqrt(3).
    # But do they overlap? The inradius is sqrt(3)/2, so two hexagons with centers at distance sqrt(3) 
    # in the flat-to-flat direction: each extends sqrt(3)/2 from center, so total extent = sqrt(3), 
    # they just touch. So center distance = sqrt(3) is valid for flat-edge contact.
    # For vertex-to-vertex (pointy direction), center distance = 2.
    
    # Honeycomb: 7 hexagons in hexagonal arrangement (1+6), then 4 more
    sqrt3 = np.sqrt(3.0)
    # 7-hex cluster: center at origin, 6 at distance sqrt3 in 6 directions (flat-to-flat)
    h7 = [[0.0, 0.0, 0.0]]
    for k in range(6):
        ang = np.radians(60*k)
        h7.append([sqrt3*np.cos(ang), sqrt3*np.sin(ang), 0.0])
    # Add 4 more hexagons around the perimeter
    # The outer ring has 6 hexagons. Add 4 more at positions beyond.
    # Let's place them at the "gaps" or extend the pattern
    # Actually, let's try a different arrangement: 3-4-4 or similar
    
    # Config 3: 3 rows of 4, 4, 3 (staggered)
    # Row spacing in y: sqrt(3) (flat-to-flat)
    # Within row, x spacing: 2 (vertex-to-vertex) or sqrt(3)?
    # For hexagons in a row (all same orientation), if they're side by side with flat edges touching,
    # x-spacing = sqrt(3), y-spacing between staggered rows = sqrt(3)*sqrt(3)/2 = 1.5? 
    # Let me just use reasonable guesses and let optimization fix it.
    
    # Staggered grid:
    staggered = []
    # 4 in top row, 4 in middle, 3 in bottom (or similar)
    for i in range(4):
        staggered.append([i*sqrt3, 0.0, 0.0])
    for i in range(4):
        staggered.append([(i+0.5)*sqrt3, sqrt3*0.5, 0.0])
    for i in range(3):
        staggered.append([(i+1.0)*sqrt3, sqrt3, 0.0])
    # Center it roughly
    cx_avg = np.mean([p[0] for p in staggered])
    cy_avg = np.mean([p[1] for p in staggered])
    staggered = [[p[0]-cx_avg, p[1]-cy_avg, p[2]] for p in staggered]
    configs.append(staggered)
    
    # Config 4: Compact cluster - try to make it more circular
    # 1 center, 6 around at dist sqrt3, then 4 more at dist ~2*sqrt3 in alternating directions
    compact = [[0.0, 0.0, 0.0]]
    for k in range(6):
        ang = np.radians(60*k)
        compact.append([sqrt3*np.cos(ang), sqrt3*np.sin(ang), 0.0])
    for k in range(4):
        ang = np.radians(60*k + 30)  # offset by 30 degrees
        compact.append([2.0*sqrt3*np.cos(ang), 2.0*sqrt3*np.sin(ang), 0.0])
    configs.append(compact)
    
    # Config 5: Rotated variants of honeycomb
    # Try rotating the whole cluster
    for rot in [15, 30, 45]:
        cfg = []
        for x, y, a in compact:
            # Rotate position
            xr = x*np.cos(np.radians(rot)) - y*np.sin(np.radians(rot))
            yr = x*np.sin(np.radians(rot)) + y*np.cos(np.radians(rot))
            cfg.append([xr, yr, a + rot])
        configs.append(cfg)
    
    # Run Nelder-Mead on each config
    for ci, cfg in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Build initial params
        params0 = []
        for x, y, a in cfg:
            params0.extend([x, y, a])
        # Estimate outer center and size
        xs = [p[0] for p in cfg]
        ys = [p[1] for p in cfg]
        ocx = np.mean(xs)
        ocy = np.mean(ys)
        oang = 0.0
        params0.extend([ocx, ocy, oang])
        
        # Compute initial score
        score0 = objective(params0)
        
        # Nelder-Mead optimization
        try:
            res = minimize(objective, params0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
        
        # Also try with different outer angles
        for test_ang in [0, 30, 60]:
            if time.time() > deadline:
                break
            params1 = params0.copy()
            params1[-1] = test_ang
            try:
                res = minimize(objective, params1, method='Nelder-Mead',
                              options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res.x
            except Exception:
                pass
    
    if best_result is None:
        # Fallback to original
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Parse best result
    inner = []
    for i in range(N):
        x, y, a = best_result[3*i:3*i+3]
        inner.append([float(x), float(y), float(a)])
    ocx = float(best_result[3*N])
    ocy = float(best_result[3*N+1])
    oang = float(best_result[3*N+2])
    side = float(best_score)
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_result, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
                inner = []
                for i in range(N):
                    x, y, a = best_result[3*i:3*i+3]
                    inner.append([float(x), float(y), float(a)])
                ocx = float(best_result[3*N])
                ocy = float(best_result[3*N+1])
                oang = float(best_result[3*N+2])
                side = float(best_score)
        except Exception:
            pass
    
    return inner, (ocx, ocy), side, oang

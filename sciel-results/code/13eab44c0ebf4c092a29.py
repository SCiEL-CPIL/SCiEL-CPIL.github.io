import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """Vertices of regular hexagon, center (cx,cy), side s, rotation angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append([cx + s * np.cos(theta), cy + s * np.sin(theta)])
    return np.array(verts)

def _enclosing_hex_side(points, cx, cy, angle_deg):
    """Minimum side length of enclosing hexagon with given center and rotation."""
    a = np.radians(angle_deg)
    # 6 outward normal directions
    normals = np.array([[np.cos(a + i * np.pi / 3), np.sin(a + i * np.pi / 3)] for i in range(6)])
    # For each point and each normal, compute projection
    # (p - c) · n
    diffs = points - np.array([cx, cy])  # (M, 2)
    projs = diffs @ normals.T  # (M, 6)
    max_proj = np.max(projs)
    # side = max_proj / (sqrt(3)/2)
    return max_proj / (np.sqrt(3) / 2)

def _optimize_enclosing(points):
    """Find optimal (cx, cy, angle) minimizing enclosing hexagon side."""
    # Initial guess: center of bounding box, angle 0
    cx0 = np.mean(points[:, 0])
    cy0 = np.mean(points[:, 1])
    
    def objective(params):
        cx, cy, ang = params
        return _enclosing_hex_side(points, cx, cy, ang)
    
    best = None
    best_val = np.inf
    
    # Try multiple initial angles
    for ang0 in range(0, 60, 10):
        x0 = [cx0, cy0, float(ang0)]
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best = res.x
        except Exception:
            continue
    
    # Also try Nelder-Mead from the best
    if best is not None:
        cx, cy, ang = best
        # Refine
        try:
            res = minimize(objective, best, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best = res.x
        except Exception:
            pass
    
    if best is None:
        # Fallback to a rough estimate if optimization fails completely
        # Use bounding box diagonal / 2 as a rough side estimate, centered at mean
        cx, cy = cx0, cy0
        ang = 0.0
        best_val = _enclosing_hex_side(points, cx, cy, ang)
        best = np.array([cx, cy, ang])
        
    return best_val, best

def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin
    
    # Generate hexagonal lattice positions for inner hexagons
    # Flat-top hexagonal lattice: neighbors at distance sqrt(3)
    # Lattice vectors: (sqrt(3), 0) and (sqrt(3)/2, 1.5)
    sq3 = np.sqrt(3)
    
    # Scaling factor to ensure strict disjointness (distance > sqrt(3))
    scale = 1.001
    
    # Generate a grid of lattice positions
    # For 11 hexagons, we need positions in a reasonable range
    # Hex lattice coords: (q, r) -> (q*sq3 + r*sq3/2, r*1.5)
    positions = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = (q * sq3 + r * sq3 / 2) * scale
            y = (r * 1.5) * scale
            positions.append((x, y))
    
    positions = np.array(positions)
    
    # We need to choose 11 positions such that no two hexagons overlap
    # With scale > 1, distance > sqrt(3), so they are strictly disjoint.
    
    # Strategy: try the "hexagonal cluster" arrangements
    # 1 + 6 + 4 = 11: center, first ring (6), and 4 from second ring
    
    # Let me enumerate valid clusters more carefully
    
    # For a hex lattice, the "rings" around center (0,0):
    # Ring 0: (0,0) - 1 point
    # Ring 1: 6 points at hex distance 1
    # Ring 2: 12 points at hex distance 2
    
    # Let me just use Euclidean distance for ring classification
    dists = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    
    # Ring 0: dist == 0
    ring0 = np.where(dists < 0.1)[0]
    # Ring 1: dist ~ sqrt(3) * scale
    # sqrt(3) * 1.001 ~ 1.733
    ring1 = np.where((dists > 1.6) & (dists < 1.9))[0]
    # Ring 2: dist ~ 3 * scale or 2*sqrt(3) * scale
    # 3 * 1.001 ~ 3.003, 2*sqrt(3)*1.001 ~ 3.466
    ring2 = np.where((dists > 2.8) & (dists < 3.8))[0]
    
    # 11 = 1 + 6 + 4: take all of ring0, all of ring1, and 4 from ring2
    # Or 1 + 6 + 3 + 1, etc.
    
    best_side = np.inf
    best_result = None
    
    # Strategy 1: center + 6 ring1 + 4 from ring2
    from itertools import combinations
    
    if len(ring0) >= 1 and len(ring1) >= 6 and len(ring2) >= 4:
        center_idx = ring0[0]
        ring1_idx = ring1[:6]  # take first 6 (should be all 6)
        
        # Try all combinations of 4 from ring2
        # ring2 might have up to 12 points, C(12,4) = 495
        for combo in combinations(range(len(ring2)), 4):
            if time.time() > deadline - 30:
                break
            
            selected = [center_idx] + list(ring1_idx) + [ring2[i] for i in combo]
            
            # Get the positions
            sel_positions = positions[selected]
            
            # For each selected position, it's a hexagon center.
            # Get all vertices of all 11 hexagons
            all_verts = []
            for pos in sel_positions:
                verts = _hex_vertices(pos[0], pos[1], 1.0, 0.0)
                all_verts.append(verts)
            all_verts = np.vstack(all_verts)
            
            # Find enclosing hexagon
            side, params = _optimize_enclosing(all_verts)
            
            if side < best_side:
                best_side = side
                best_result = (sel_positions.copy(), params.copy())
    
    if best_result is not None:
        sel_positions, params = best_result
        cx, cy, ang = params
        
        # Build inner list
        inner = []
        for pos in sel_positions:
            inner.append([pos[0], pos[1], 0.0])
        
        return inner, (cx, cy), best_side, ang
    
    # Fallback: just return a valid (if not optimal) configuration
    # Place 11 hexagons in a line
    inner = [[sq3 * scale * k, 0.0, 0.0] for k in range(N)]
    # Enclosing hexagon for a line of hexagons
    # Use a large hexagon to be safe
    return inner, (sq3 * scale * (N-1) / 2, 0.0), 20.0, 0.0

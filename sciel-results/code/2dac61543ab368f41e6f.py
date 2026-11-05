"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(center, side, angle_deg):
    """Get 6 vertices of a regular hexagon."""
    cx, cy = center
    a = np.radians(angle_deg)
    # Vertices at angles a, a+60, ..., a+300
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _support_hex(center, side, angle_deg, direction):
    """Support function of a hexagon in a given direction."""
    verts = _hex_vertices(center, side, angle_deg)
    return np.max(verts @ direction)

def _enclosing_hex_side(positions, angles_deg, outer_center, outer_angle):
    """
    Compute the minimal side length of a regular hexagon at (outer_center) with rotation outer_angle
    that contains all inner hexagons.
    """
    cx, cy = outer_center
    oa = np.radians(outer_angle)
    # 6 outward normal directions of the outer hexagon
    # For a hexagon with vertex at angle oa, the edge normals are at oa + 30°, oa + 90°, etc.
    # Actually, the support function approach: the outer hexagon has vertices at oa, oa+60,...
    # The 6 edge normals are perpendicular to edges, at oa+30, oa+90, oa+150, oa+210, oa+270, oa+330
    normal_angles = oa + np.pi/6 + np.arange(6) * np.pi / 3
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # For each normal direction, compute max support of all inner hexagons
    max_supports = []
    for n in normals:
        max_s = -np.inf
        for i in range(len(positions)):
            px, py = positions[i]
            pa = np.radians(angles_deg[i])
            verts = _hex_vertices((px, py), 1.0, angles_deg[i])
            s = np.max(verts @ n)
            max_s = max(max_s, s)
        max_supports.append(max_s)
    
    # The outer hexagon with side s and center (cx,cy) has support in direction n:
    # h_outer(n) = s * cos(angle between n and nearest vertex direction) / cos(30°)
    # More precisely, for a regular hexagon with circumradius s, the support in direction at angle phi
    # from the nearest vertex is s * cos(phi) / cos(30°) where phi is in [-30°, 30°]
    # The vertex directions are oa, oa+60, ...
    # For normal at angle theta, find nearest vertex direction
    s_required = 0
    for k, n in enumerate(normals):
        theta = np.arctan2(n[1], n[0])
        # Nearest vertex direction
        vertex_angles = oa + np.arange(6) * np.pi / 3
        diffs = np.abs((theta - vertex_angles + np.pi) % (2*np.pi) - np.pi)
        min_diff = np.min(diffs)
        # Support of outer hexagon in this direction = s * cos(min_diff) / cos(pi/6)
        # We need: max_supports[k] <= s * cos(min_diff) / cos(pi/6)
        # So: s >= max_supports[k] * cos(pi/6) / cos(min_diff)
        # But also need to account for the center offset
        # Actually the support is relative to center, so:
        # max_supports[k] - (outer_center @ n) <= s * cos(min_diff) / cos(pi/6)
        offset = cx * n[0] + cy * n[1]
        required = (max_supports[k] - offset) * np.cos(np.pi/6) / np.cos(min_diff)
        s_required = max(s_required, required)
    return s_required

def _total_cost(params, positions, angles_deg):
    """Cost function: outer hexagon side length."""
    cx, cy, outer_angle = params
    return _enclosing_hex_side(positions, angles_deg, (cx, cy), outer_angle)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin
    
    # Strategy: Try various lattice-based arrangements, then optimize
    best_result = None
    best_side = np.inf
    
    # Generate candidate arrangements
    # Hexagonal lattice: basis vectors at 60°
    # For flat-top hexagons, the lattice spacing is 2*side*sqrt(3)/2 = sqrt(3) for flat-to-flat
    # Actually for unit hexagons on a hex lattice, centers are at distance 2*apothem = sqrt(3) apart
    
    # Try different arrangement patterns
    patterns = []
    
    # Pattern 1: Hexagonal cluster (center + rings)
    # 1 + 6 = 7, need 4 more. Add them to extend one direction.
    # Lattice: a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 1.5) for flat-top
    sqrt3 = np.sqrt(3)
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2, 1.5])
    
    # Hex cluster positions (i,j) in axial coordinates
    # Center at (0,0), ring 1: 6 positions, ring 2: 12 positions
    # For 11: center + 6 ring1 + 4 from ring2
    
    # Axial to cartesian for flat-top hexagons:
    # x = sqrt(3) * (q + r/2), y = 1.5 * r
    def axial_to_cart(q, r):
        return np.array([sqrt3 * (q + r/2.0), 1.5 * r])
    
    # Generate all positions in a hexagon of radius 2 (19 positions), pick best 11
    all_axial = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                all_axial.append((q, r))
    
    # We need to choose 11 from these 19 that minimize the enclosing hexagon
    # For simplicity, try a few subsets
    
    # Subset 1: center + all 6 ring1 + 4 adjacent ring2 positions
    ring0 = [(0,0)]
    ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-2),(2,-1)]
    
    # Try: center + ring1 + 4 specific ring2
    # Best to add ring2 positions that are "closest" to keep compact
    # Add (2,0), (1,1), (0,2), (-1,2) - one arc
    subset1 = ring0 + ring1 + [(2,0),(1,1),(0,2),(-1,2)]
    
    # Try another: center + ring1 + (2,0), (2,-1), (1,-2), (0,-2)
    subset2 = ring0 + ring1 + [(2,0),(2,-1),(1,-2),(0,-2)]
    
    # Try: center + ring1 + 4 spread out
    subset3 = ring0 + ring1 + [(2,0),(0,2),(-2,2),(-2,0)]
    
    # Try: center + ring1 + 4 in two opposite arcs
    subset4 = ring0 + ring1 + [(2,0),(1,1),(-2,0),(-1,-1)]
    
    candidate_subsets = [subset1, subset2, subset3, subset4]
    
    # Also try non-hexagonal arrangements
    # Linear-ish: rows of 3,4,3,1 or similar
    # For flat-top hexagons, "rows" are horizontal
    # Row of k hexagons: centers at x = 0, sqrt3, 2*sqrt3, ...
    
    # Try a 3-4-3-1 arrangement
    # Row 1 (top): 1 hex at y=3.0
    # Row 2: 3 hex at y=1.5
    # Row 3: 4 hex at y=0
    # Row 4: 3 hex at y=-1.5
    # Total: 11
    
    # Actually let's just use the lattice subsets and optimize
    
    arrangements = []
    for subset in candidate_subsets:
        positions = [axial_to_cart(q, r) for q, r in subset]
        angles = [0.0] * 11
        arrangements.append((positions, angles))
    
    # Also try with 30° rotation of inner hexagons
    for subset in candidate_subsets[:2]:
        positions = [axial_to_cart(q, r) for q, r in subset]
        angles = [30.0] * 11
        arrangements.append((positions, angles))
    
    # For each arrangement, optimize the outer hexagon
    for positions, angles in arrangements:
        if time.time() > deadline:
            break
        
        # Initial guess for outer center and angle
        pos_arr = np.array(positions)
        init_center = pos_arr.mean(axis=0)
        
        # Try a few initial outer angles
        for init_oe in [0, 30, 60, 90, 120, 150]:
            if time.time() > deadline:
                break
            x0 = [init_center[0], init_center[1], float(init_oe)]
            
            def cost(params):
                return _total_cost(params, positions, angles)
            
            try:
                result = minimize(cost, x0, method='Nelder-Mead', 
                                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
                if result.fun < best_side:
                    best_side = result.fun
                    best_result = (positions, angles, result.x)
            except:
                pass
    
    # If we found a good result, use it
    if best_result is not None:
        positions, angles, outer_params = best_result
        cx, cy, oa = outer_params
        inner = [[float(p[0]), float(p[1]), float(a)] for p, a in zip(positions, angles)]
        return inner, (float(cx), float(cy)), float(best_side), float(oa)
    else:
        # Fallback: simple row arrangement
        inner = [[sqrt3 * k, 0.0, 0.0] for k in range(-5, 6)]
        return inner, (0.0, 0.0), 15.0, 0.0
# EVOLVE-BLOCK-END

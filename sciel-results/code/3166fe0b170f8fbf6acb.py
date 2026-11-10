import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, s, a_deg):
    """Get 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _min_gap(v1, v2, a1, a2):
    """
    Compute the minimum gap between two hexagons.
    v1, v2: vertex arrays (6, 2)
    a1, a2: angles in degrees
    Returns the minimum separation over all 12 normal directions.
    """
    min_sep = np.inf
    # Edge normals for hex1
    n1 = np.radians(a1) + np.radians(60) * (np.arange(6) + 0.5)
    # Edge normals for hex2
    n2 = np.radians(a2) + np.radians(60) * (np.arange(6) + 0.5)
    
    for i in range(6):
        n = np.array([np.cos(n1[i]), np.sin(n1[i])])
        proj1 = v1[:, 0] * n[0] + v1[:, 1] * n[1]
        proj2 = v2[:, 0] * n[0] + v2[:, 1] * n[1]
        gap = proj2.min() - proj1.max()
        min_sep = min(min_sep, gap)
    
    for i in range(6):
        n = np.array([np.cos(n2[i]), np.sin(n2[i])])
        proj1 = v1[:, 0] * n[0] + v1[:, 1] * n[1]
        proj2 = v2[:, 0] * n[0] + v2[:, 1] * n[1]
        gap = proj1.min() - proj2.max()
        min_sep = min(min_sep, gap)
    
    return min_sep

def _check_all_separated(inner_data):
    """Check that all pairs of inner hexagons are separated."""
    n = len(inner_data)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, a1 = inner_data[i]
            x2, y2, a2 = inner_data[j]
            v1 = _hex_vertices(x1, y1, 1.0, a1)
            v2 = _hex_vertices(x2, y2, 1.0, a2)
            gap = _min_gap(v1, v2, a1, a2)
            if gap <= 1e-6:
                return False
    return True

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner_hex_data, outer_hex_data, outer_hex_side_length)
    """
    start_time = time.time()
    deadline = start_time + 600
    
    sqrt3 = np.sqrt(3)
    
    # Initial configuration: compact honeycomb arrangement
    # Using a=0 (pointy-right hexagons)
    # Hex grid: horizontal spacing 1.5, vertical spacing sqrt(3)/2 for diagonal
    # Let's build a symmetric 11-hexagon cluster
    
    # Center + 6 in first ring + 4 in second ring (at alternating positions)
    inner = np.array([
        [0.0, 0.0, 0.0],              # center
        [1.5, 0.0, 0.0],              # E
        [0.75, sqrt3/2, 0.0],         # NE
        [-0.75, sqrt3/2, 0.0],        # NW
        [-1.5, 0.0, 0.0],             # W
        [-0.75, -sqrt3/2, 0.0],       # SW
        [0.75, -sqrt3/2, 0.0],        # SE
        [3.0, 0.0, 0.0],              # far E
        [2.25, sqrt3/2, 0.0],         # far NE
        [2.25, -sqrt3/2, 0.0],        # far SE
        [0.0, sqrt3, 0.0],            # far N
    ])
    
    # Verify separation
    if not _check_all_separated(inner):
        # Try to fix by small perturbations
        np.random.seed(123)
        for _ in range(1000):
            inner_perturbed = inner.copy()
            inner_perturbed[:, :2] += np.random.randn(11, 2) * 0.01
            if _check_all_separated(inner_perturbed):
                inner = inner_perturbed
                break
    
    # Compute the minimum enclosing regular hexagon
    # For a regular hexagon with center (ox, oy), angle oa, side s:
    # The 6 inward normals are at angles oa + 60*i + 210 (for i=0..5)
    # A point p is inside if for all i: dot(p - (ox,oy), inward_normal_i) <= s*sqrt(3)/2
    
    def _min_side_for_angle(inner_data, oa_deg, ox=0.0, oy=0.0):
        """Compute minimum side of outer hexagon with given angle and center."""
        oa = np.radians(oa_deg)
        inward_normals = oa + np.radians(60) * np.arange(6) + np.radians(210)
        
        max_needed = 0
        for i in range(len(inner_data)):
            x, y, a = inner_data[i]
            verts = _hex_vertices(x, y, 1.0, a)
            # Shift by outer center
            verts_shifted = verts - np.array([ox, oy])
            for j in range(6):
                n = np.array([np.cos(inward_normals[j]), np.sin(inward_normals[j])])
                proj = np.max(verts_shifted[:, 0] * n[0] + verts_shifted[:, 1] * n[1])
                if proj > 0:
                    needed = 2 * proj / np.sqrt(3)
                    max_needed = max(max_needed, needed)
        
        return max_needed
    
    def _objective(params):
        """Optimize outer hex center and angle."""
        if time.time() > deadline - 30:
            return 1e10
        
        ox, oy, oa = params
        oa_deg = np.degrees(oa)
        
        side = _min_side_for_angle(inner, oa_deg, ox, oy)
        
        if side < np.sqrt(11) - 1e-6:
            return 1e10
        
        return side
    
    best_side = np.inf
    best_params = None
    
    # Try multiple initial angles
    for oa_init in [0, 30, 60, 90, 120, 150]:
        if time.time() > deadline - 60:
            break
        
        res = minimize(
            _objective,
            [0.0, 0.0, np.radians(oa_init)],
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7}
        )
        
        if res.fun < best_side:
            best_side = res.fun
            best_params = res.x
    
    if best_params is None:
        # Fallback
        side = _min_side_for_angle(inner, 0, 0, 0)
        outer_data = np.array([0.0, 0.0, 0.0])
        return inner, outer_data, side
    
    ox, oy, oa = best_params
    oa_deg = np.degrees(oa)
    side = _min_side_for_angle(inner, oa_deg, ox, oy)
    
    # Add small margin
    side *= 1.0001
    
    outer_data = np.array([ox, oy, oa_deg])
    
    return inner, outer_data, side

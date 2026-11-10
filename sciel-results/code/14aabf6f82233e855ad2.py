# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _hexes_separated(v1, v2):
    """Check if two unit hexagons (given as 6x2 vertex arrays) are separated."""
    # 12 edge normals: 6 from each hexagon
    # For hexagon with vertices v[0..5], edge i goes from v[i] to v[(i+1)%6]
    # Normal is perpendicular to edge, pointing outward
    # For a regular hexagon centered at origin, the outward normal of edge i (from vertex i to i+1)
    # is at angle (60*i + 30) degrees... wait, let me think.
    # Vertices are at angles 0, 60, 120, 180, 240, 300 degrees.
    # Edge 0: from angle 0 to 60. Midpoint at 30 deg. Outward normal at 30 deg.
    # Edge i: outward normal at (60*i + 30) degrees.
    
    for hex_verts in [v1, v2]:
        for i in range(6):
            p1 = hex_verts[i]
            p2 = hex_verts[(i + 1) % 6]
            # Edge direction
            edge = p2 - p1
            # Outward normal (rotate edge by -90 deg for CCW polygon)
            # For CCW polygon, outward normal is (edge_y, -edge_x) normalized
            # But we need to be consistent. Let's compute the normal pointing away from center.
            center = np.mean(hex_verts, axis=0)
            mid = (p1 + p2) / 2
            n = np.array([edge[1], -edge[0]])
            if np.dot(n, mid - center) < 0:
                n = -n
            n = n / np.linalg.norm(n)
            
            # Project both hexagons onto this normal
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            # Check if intervals are apart with gap > 1e-6
            if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                return True
    
    return False

def _is_inside_hexagon(p, outer_verts):
    """Check if point p is inside or on the boundary of convex polygon outer_verts."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        cross = np.cross(p2 - p1, p - p1)
        if cross < -1e-6:
            return False
    return True

def _validate(inner_hex_data, outer_hex_data, side):
    """Validate the packing configuration."""
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    
    # Shift by outer center
    outer_cx, outer_cy, outer_angle = outer_hex_data
    shifted_inner = inner_hex_data.copy()
    shifted_inner[:, 0] -= outer_cx
    shifted_inner[:, 1] -= outer_cy
    
    # Check edge lengths of inner hexagons
    for i in range(11):
        cx, cy, a = shifted_inner[i]
        verts = _hex_vertices(cx, cy, 1.0, a)
        for j in range(6):
            v1 = verts[j]
            v2 = verts[(j + 1) % 6]
            length = np.linalg.norm(v2 - v1)
            if abs(length - 1.0) > 1e-8:
                return False
    
    # Get all inner hexagon vertices
    all_inner_verts = []
    for i in range(11):
        cx, cy, a = shifted_inner[i]
        verts = _hex_vertices(cx, cy, 1.0, a)
        all_inner_verts.append(verts)
    
    # Check all pairs are separated
    for i in range(11):
        for j in range(i + 1, 11):
            if not _hexes_separated(all_inner_verts[i], all_inner_verts[j]):
                return False
    
    # Check all inner vertices are inside outer hexagon
    outer_verts = _hex_vertices(0, 0, side, outer_angle)
    for verts in all_inner_verts:
        for p in verts:
            if not _is_inside_hexagon(p, outer_verts):
                return False
    
    return True

def _compute_side(inner_hex_data, outer_hex_data):
    """Compute the minimum side length of a hexagon that can contain all inner hexagons."""
    outer_cx, outer_cy, outer_angle = outer_hex_data
    shifted_inner = inner_hex_data.copy()
    shifted_inner[:, 0] -= outer_cx
    shifted_inner[:, 1] -= outer_cy
    
    min_side = np.sqrt(11)
    
    for i in range(11):
        cx, cy, a = shifted_inner[i]
        verts = _hex_vertices(cx, cy, 1.0, a)
        
        for p in verts:
            # For a regular hexagon with side s centered at origin with angle outer_angle,
            # the distance from center to edge in direction theta is s * cos(pi/6) = s * sqrt(3)/2
            # But we need the apothem in the direction of p.
            # The hexagon boundary in direction phi from center is at distance s * cos(pi/6) / cos((phi - outer_angle) mod 60 - 30)
            # Actually, the apothem is s * sqrt(3)/2, and the hexagon is the intersection of 6 half-planes.
            # For a hexagon with vertices at angles outer_angle + 60*i, the edge opposite to vertex i
            # has normal at angle outer_angle + 60*i + 180... 
            # Let me think differently. The hexagon with side s, center at origin, angle a has
            # vertices at s * exp(i*(a + 60*k)). The edges are between consecutive vertices.
            # Edge k goes from vertex k to vertex k+1. The outward normal of edge k is at angle a + 60*k + 30.
            # The distance from center to edge k is s * cos(30 deg) = s * sqrt(3)/2.
            # So the hexagon is: for all k, dot(p, n_k) <= s * sqrt(3)/2, where n_k is unit normal.
            
            phi = np.arctan2(p[1], p[0])
            # Find which edge constraint is most restrictive
            # The constraint is: for each k, dot(p, n_k) <= s * sqrt(3)/2
            # So s >= dot(p, n_k) / (sqrt(3)/2) for the k that gives the max
            # n_k is at angle outer_angle + 60*k + 30 degrees
            
            max_ratio = 0
            for k in range(6):
                n_angle = np.radians(outer_angle + 60 * k + 30)
                n = np.array([np.cos(n_angle), np.sin(n_angle)])
                proj = np.dot(p, n)
                if proj > 0:
                    ratio = proj / (np.sqrt(3) / 2)
                    if ratio > max_ratio:
                        max_ratio = ratio
            
            if max_ratio > min_side:
                min_side = max_ratio
    
    return min_side

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Initial configuration: hexagonal arrangement
    # Center hexagon at origin, 6 around it, 4 more
    # For unit hexagons, the distance between centers of adjacent hexagons is 2 * apothem = 2 * sqrt(3)/2 = sqrt(3)
    # Wait, no. For two unit hexagons to be just touching, the distance between centers is 2 * apothem = sqrt(3)?
    # Actually, the apothem of a unit hexagon is sqrt(3)/2. Two hexagons sharing an edge have centers
    # separated by 2 * apothem = sqrt(3). But they share an edge, so they're not separated.
    # For them to be separated (gap > 1e-6), centers need to be more than sqrt(3) apart.
    
    # Let's use a slightly larger spacing
    spacing = np.sqrt(3) + 0.01  # Small gap
    
    # Hexagonal lattice positions
    # In a hexagonal lattice with spacing d between centers:
    # Row 0: (0,0)
    # Row 1: (d, 0), (-d, 0)  -- wait, hexagonal lattice
    # Let me use axial coordinates. For a hexagonal tiling, the centers are at:
    # (i * d, j * d * sqrt(3) + (i % 2) * d * sqrt(3) / 2)
    # Actually, simpler: use the fact that neighbors are at 60-degree angles.
    
    positions = []
    # Center
    positions.append([0, 0])
    # 6 neighbors
    for k in range(6):
        angle = np.radians(60 * k)
        positions.append([spacing * np.cos(angle), spacing * np.sin(angle)])
    # 4 more: extend some of the neighbors
    # Position 6: further out along angle 0
    positions.append([2 * spacing, 0])
    # Position 7: further out along angle 60
    positions.append([spacing * np.cos(np.radians(60)), spacing * np.sin(np.radians(60)) + spacing * np.cos(np.radians(60))])
    # Hmm, this is getting complicated. Let me just use a grid.
    
    # Simpler: use a 3x4 grid with hexagonal offset
    # Actually, let me just start with the seed's arrangement and optimize.
    
    inner_hex_data = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ]
    )
    
    outer_hex_data = np.array([0, 0, 0])
    side = _compute_side(inner_hex_data, outer_hex_data)
    
    best_side = side
    best_inner = inner_hex_data.copy()
    best_outer = outer_hex_data.copy()
    
    # Optimize positions to minimize the enclosing hexagon side
    # Parameters: 11*2 positions + 11 angles = 33 parameters
    # But we can fix angles to 0 for simplicity first
    
    def objective(params):
        if time.time() > deadline:
            return 1e6
        x = params[:22].reshape(11, 2)
        angles = params[22:]
        
        # Check separation
        verts_list = []
        for i in range(11):
            verts = _hex_vertices(x[i, 0], x[i, 1], 1.0, angles[i])
            verts_list.append(verts)
        
        for i in range(11):
            for j in range(i + 1, 11):
                if not _hexes_separated(verts_list[i], verts_list[j]):
                    return 1e6 + np.sum(np.abs(params))
        
        # Compute side
        inner = np.column_stack([x, angles])
        s = _compute_side(inner, outer_hex_data)
        return s
    
    # Start from current positions
    x0 = np.concatenate([inner_hex_data[:, :2].flatten(), inner_hex_data[:, 2]])
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
    
    if result.fun < best_side and time.time() < deadline:
        params = result.x
        x = params[:22].reshape(11, 2)
        angles = params[22:]
        inner_hex_data = np.column_stack([x, angles])
        side = _compute_side(inner_hex_data, outer_hex_data)
        if side < best_side:
            best_side = side
            best_inner = inner_hex_data.copy()
            best_outer = outer_hex_data.copy()
    
    # Try optimizing with only positions fixed, angles at 0
    def objective_pos_only(params):
        if time.time() > deadline:
            return 1e6
        x = params.reshape(11, 2)
        
        verts_list = []
        for i in range(11):
            verts = _hex_vertices(x[i, 0], x[i, 1], 1.0, 0)
            verts_list.append(verts)
        
        for i in range(11):
            for j in range(i + 1, 11):
                if not _hexes_separated(verts_list[i], verts_list[j]):
                    return 1e6 + np.sum(np.abs(params))
        
        inner = np.column_stack([x, np.zeros(11)])
        s = _compute_side(inner, outer_hex_data)
        return s
    
    x0_pos = inner_hex_data[:, :2].flatten()
    result2 = minimize(objective_pos_only, x0_pos, method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
    
    if result2.fun < best_side and time.time() < deadline:
        x = result2.x.reshape(11, 2)
        inner_hex_data = np.column_stack([x, np.zeros(11)])
        side = _compute_side(inner_hex_data, outer_hex_data)
        if side < best_side:
            best_side = side
            best_inner = inner_hex_data.copy()
            best_outer = outer_hex_data.copy()
    
    # Final validation
    if not _validate(best_inner, best_outer, best_side):
        # Fallback to seed
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        best_side = _compute_side(inner_hex_data, outer_hex_data)
        if not _validate(inner_hex_data, outer_hex_data, best_side):
            # Very large fallback
            best_side = 10.0
    
    return best_inner, best_outer, best_side


# EVOLVE-BLOCK-END

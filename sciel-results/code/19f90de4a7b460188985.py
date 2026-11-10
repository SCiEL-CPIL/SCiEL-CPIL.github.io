# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, s, a_deg):
    """Compute vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _check_separation(inner_hexes, tol=1e-6):
    """Check that all pairs of inner hexagons are separated."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            vi = _hex_vertices(inner_hexes[i, 0], inner_hexes[i, 1], 1.0, inner_hexes[i, 2])
            vj = _hex_vertices(inner_hexes[j, 0], inner_hexes[j, 1], 1.0, inner_hexes[j, 2])
            
            # Check all 12 edge normals (6 from each hexagon)
            separated = False
            for v in [vi, vj]:
                for k in range(6):
                    p1 = v[k]
                    p2 = v[(k + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    normal = normal / (np.linalg.norm(normal) + 1e-15)
                    
                    # Project both hexagons onto this normal
                    proj_i = vi @ normal
                    proj_j = vj @ normal
                    
                    max_i = np.max(proj_i)
                    min_j = np.min(proj_j)
                    max_j = np.max(proj_j)
                    min_i = np.min(proj_i)
                    
                    if max_i < min_j - tol or max_j < min_i - tol:
                        separated = True
                        break
                if separated:
                    break
            if not separated:
                return False
    return True

def _inner_in_outer(inner_hexes, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check that all inner hexagon vertices are inside the outer hexagon."""
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    for i in range(len(inner_hexes)):
        vi = _hex_vertices(inner_hexes[i, 0], inner_hexes[i, 1], 1.0, inner_hexes[i, 2])
        for p in vi:
            # Check if p is inside outer hexagon using cross product test
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                edge = p2 - p1
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < -tol:
                    inside = False
                    break
            if not inside:
                return False
    return True

def _min_enclosing_hex_side(points, max_iters=100, tol=1e-8):
    """Find the minimum side length of a regular hexagon (centered at origin) that contains all points."""
    # The hexagon is defined by 6 half-planes
    # For a regular hexagon with side s and angle a, the constraint for each edge is:
    # n_k . p <= s * apothem where apothem = s * sqrt(3)/2
    # Actually, for a regular hexagon with circumradius R (= side length s), 
    # the apothem is R * sqrt(3)/2
    
    # A regular hexagon with side s has circumradius s and inradius s*sqrt(3)/2
    # The 6 supporting lines are at distance s*sqrt(3)/2 from center, at angles a + 30 + 60*k
    
    # We want to minimize s such that for all points p and all k:
    # p . n_k <= s * sqrt(3)/2
    # where n_k = (cos(a + 30 + 60k), sin(a + 30 + 60k))
    
    # So s >= max_k (p . n_k) / (sqrt(3)/2) for all p
    # s >= max_p max_k (p . n_k) / (sqrt(3)/2)
    
    # For a given angle a, the minimum s is:
    # s(a) = max_p max_k (p . n_k(a)) / (sqrt(3)/2)
    
    # We can optimize over a
    
    sqrt3 = np.sqrt(3)
    best_s = np.inf
    best_a = 0
    
    # Try many angles
    for a_deg in np.linspace(0, 60, 121):  # 60-degree symmetry
        a = np.radians(a_deg)
        max_proj = 0
        for k in range(6):
            angle = a + np.radians(30) + np.radians(60) * k
            n = np.array([np.cos(angle), np.sin(angle)])
            proj = points @ n
            max_proj = max(max_proj, np.max(proj))
        
        s = max_proj / (sqrt3 / 2)
        if s < best_s:
            best_s = s
            best_a = a_deg
    
    return best_s, best_a

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    n = 11
    sqrt3 = np.sqrt(3)
    d = sqrt3  # distance between centers of touching unit hexagons
    
    # Use a compact hexagonal arrangement
    # 7 hexagons in a hexagonal cluster (center + 6)
    # Plus 4 more in the outer ring
    
    # Standard positions for 11 hexagons in a hexagonal lattice
    # Ring 0: 1 hexagon
    # Ring 1: 6 hexagons  
    # Ring 2: 12 hexagons (we use 4 of them)
    
    # Let's use a known good pattern:
    # The 11 hexagons can be arranged as a "hexagonal number" pattern
    
    # Positions in a hexagonal coordinate system
    # Convert to Cartesian
    def hex_to_cart(q, r):
        x = d * (q + r/2)
        y = d * (sqrt3/2) * r
        return x, y
    
    # 11 hexagons: use a compact arrangement
    # Center: (0,0)
    # Ring 1: 6 hexagons
    # Ring 2: 4 hexagons (choose the ones that minimize the bounding hexagon)
    
    positions = []
    # Center
    positions.append((0, 0))
    
    # Ring 1: 6 hexagons at distance d
    for k in range(6):
        angle = np.radians(60 * k)
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        positions.append((x, y))
    
    # Ring 2: 12 hexagons, choose 4
    # The 12 positions in ring 2 are at:
    # 6 at distance 2d (continuing the 6 directions)
    # 6 at distance d*sqrt(3) (the "corner" positions)
    
    # Let's try different sets of 4 from ring 2
    
    # Option 1: 4 at the "corner" positions (distance d*sqrt(3))
    corner_positions = []
    for k in range(6):
        angle = np.radians(60 * k + 30)
        x = d * sqrt3 * np.cos(angle)
        y = d * sqrt3 * np.sin(angle)
        corner_positions.append((x, y))
    
    # Try all combinations of 4 from the 6 corner positions
    from itertools import combinations
    
    best_side = np.inf
    best_config = None
    
    for combo in combinations(range(6), 4):
        if time.time() > deadline - 60:
            break
        
        pos = positions + [corner_positions[i] for i in combo]
        
        # Create inner hex data
        inner = np.zeros((11, 3))
        for i, (x, y) in enumerate(pos):
            inner[i, 0] = x
            inner[i, 1] = y
            inner[i, 2] = 0  # aligned
        
        # Check separation
        if not _check_separation(inner):
            continue
        
        # Get all vertices
        all_verts = []
        for i in range(11):
            vi = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
            all_verts.append(vi)
        
        all_verts = np.vstack(all_verts)
        
        # Find minimum enclosing hexagon
        side, angle = _min_enclosing_hex_side(all_verts)
        
        if side < best_side:
            best_side = side
            best_config = (inner.copy(), angle)
    
    # If no valid config found, use a fallback
    if best_config is None:
        # Use the 7-hexagon cluster + 4 at ring 2 straight positions
        positions = []
        positions.append((0, 0))
        for k in range(6):
            angle = np.radians(60 * k)
            x = d * np.cos(angle)
            y = d * np.sin(angle)
            positions.append((x, y))
        # Add 4 more at distance 2d
        for k in [0, 1, 2, 3]:
            angle = np.radians(60 * k)
            x = 2 * d * np.cos(angle)
            y = 2 * d * np.sin(angle)
            positions.append((x, y))
        
        inner = np.zeros((11, 3))
        for i, (x, y) in enumerate(positions):
            inner[i, 0] = x
            inner[i, 1] = y
            inner[i, 2] = 0
        
        all_verts = []
        for i in range(11):
            vi = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
            all_verts.append(vi)
        all_verts = np.vstack(all_verts)
        
        best_side, angle = _min_enclosing_hex_side(all_verts)
        best_config = (inner, angle)
    
    inner, angle = best_config
    
    # The outer hexagon is centered at origin
    outer = np.array([0, 0, angle])
    
    return inner, outer, best_side

# EVOLVE-BLOCK-END

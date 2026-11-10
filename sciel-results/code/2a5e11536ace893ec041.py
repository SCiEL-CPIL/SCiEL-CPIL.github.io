# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _outer_vertices(cx, cy, d, angle_deg):
    return _hex_vertices(cx, cy, d, angle_deg)

def _check_validity(inner, outer, s):
    """Check all validity constraints. Returns True if valid."""
    # Finite check
    if not (np.all(np.isfinite(inner)) and np.all(np.isfinite(outer)) and np.isfinite(s)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    inner_verts = []
    for i in range(12):
        x, y, ang = inner[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        inner_verts.append(verts)
        # Check unit edge length
        for j in range(6):
            p1 = verts[j]
            p2 = verts[(j + 1) % 6]
            edge_len = np.linalg.norm(p2 - p1)
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    outer_v = _outer_vertices(outer[0], outer[1], s, outer[2])
    
    # Outer vertices distinct
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_v[i] - outer_v[j]) < 1e-12:
                return False
    
    # Inner hexagons must be separated (SAT)
    for i in range(12):
        for j in range(i + 1, 12):
            vi = inner_verts[i]
            vj = inner_verts[j]
            # Test all edge normals from both hexagons
            separated = False
            for v in [vi, vj]:
                for k in range(6):
                    p1 = v[k]
                    p2 = v[(k + 1) % 6]
                    edge = p2 - p1
                    # Normal to edge
                    n = np.array([-edge[1], edge[0]])
                    n_len = np.linalg.norm(n)
                    if n_len < 1e-15:
                        continue
                    n = n / n_len
                    # Project all vertices of both hexagons onto normal
                    proj_i = np.dot(vi, n)
                    proj_j = np.dot(vj, n)
                    min_i = np.min(proj_i)
                    max_i = np.max(proj_i)
                    min_j = np.min(proj_j)
                    max_j = np.max(proj_j)
                    # Check if intervals are separated (more than 1e-6 apart)
                    if max_i + 1e-6 < min_j or max_j + 1e-6 < min_i:
                        separated = True
                        break
                if separated:
                    break
            if not separated:
                return False
    
    # Inner vertices must be inside outer hexagon
    for i in range(12):
        vi = inner_verts[i]
        for v in vi:
            inside = True
            for k in range(6):
                p1 = outer_v[k]
                p2 = outer_v[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    return True

def _min_enclosing_hex_side(inner_verts, outer_angle_deg):
    """
    Given 12 hexagons' vertices and the outer hexagon's angle,
    find the minimum side length s of a regular hexagon (with that angle)
    centered optimally that contains all inner vertices.
    
    We optimize over center (cx, cy) and side s.
    """
    # Collect all 72 vertices
    all_pts = np.vstack(inner_verts)  # (72, 2)
    
    def objective(params):
        cx, cy, s = params
        if s <= 0:
            return 1e10
        ov = _outer_vertices(cx, cy, s, outer_angle_deg)
        # Check containment: for each inner vertex, check against all outer edges
        # cross(p2-p1, v-p1) >= 0 for all edges
        max_violation = 0.0
        for v in all_pts:
            for k in range(6):
                p1 = ov[k]
                p2 = ov[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < 0:
                    max_violation = max(max_violation, -cross)
        # Penalize violations heavily
        if max_violation > 0:
            return s + 1000.0 * max_violation
        return s
    
    # Initial guess
    cx0 = np.mean(all_pts[:, 0])
    cy0 = np.mean(all_pts[:, 1])
    # Estimate needed size
    dx = np.max(all_pts[:, 0]) - np.min(all_pts[:, 0])
    dy = np.max(all_pts[:, 1]) - np.min(all_pts[:, 1])
    s0 = max(dx, dy) / 1.5 + 2
    
    result = minimize(objective, [cx0, cy0, s0], method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    cx, cy, s = result.x
    if s <= 0:
        s = s0
    return cx, cy, s

def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon using a honeycomb topology
    with optimized global rotation and local adjustments.
    """
    start_time = time.time()
    deadline = 620.0  # 10s safety margin
    
    best_s = np.inf
    best_inner = None
    best_outer = None
    
    # Honeycomb 3-4-3 topology: base positions in a hex grid
    # Using axial coordinates converted to Cartesian
    # Hex grid spacing: for unit hexagons, center-to-center distance = sqrt(3)
    # in the flat-top orientation, or 2 for pointy-top
    # Let's use pointy-top hexagons (angle 0)
    # For pointy-top: horizontal spacing = sqrt(3), vertical spacing = 1.5
    # Actually, for regular hexagons with side 1:
    # Pointy-top: width = sqrt(3), height = 2
    # Flat-top: width = 2, height = sqrt(3)
    
    # Let's define base positions for a 3-4-3 honeycomb cluster
    # Using flat-top hexagons (angle 0 means flat-top per the vertex formula)
    # Vertices at angle 0, 60, 120, 180, 240, 300 from center
    # This gives flat-top orientation
    
    # For flat-top hexagons with side 1:
    # Horizontal distance between adjacent centers = 1.5 * 2 = 3? No.
    # Actually, for flat-top: the distance between centers of adjacent hexagons
    # in a honeycomb is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3) in the "column" direction
    # and the row spacing is 1.5 (half of 3)
    
    # Let me just define explicit positions for a compact 3-4-3 arrangement
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    
    # For flat-top hexagons, adjacent in same row: dx = 2 (touching sides)
    # Wait, two flat-top hexagons side by side touching: center distance = 2 * side = 2
    # No, that's not right either. Let me think carefully.
    
    # A regular hexagon with side d has:
    # - Width (flat-top): 2d
    # - Height (flat-top): sqrt(3) * d
    # Two flat-top hexagons touching side-by-side: center distance = 2d? 
    # No. The flat sides are at x = ±d from center. So two touching flat-to-flat:
    # center distance = 2d in x direction.
    # But in a honeycomb, they're not flat-to-flat; they interlock.
    
    # For a honeycomb of flat-top hexagons:
    # Same row neighbors: dx = 2, dy = 0? No, that would be edge-to-edge.
    # Actually in a honeycomb, adjacent hexagons share an edge.
    # For flat-top: sharing a vertical edge means dx = 2, dy = 0? 
    # No, the vertical edges are at x = ±d, so yes dx = 2 for left-right neighbors.
    # For diagonal neighbors (up-right, up-left): 
    # The hexagon has vertices at angles 0,60,120,180,240,300
    # Vertex at 0°: (d, 0)
    # Vertex at 60°: (d/2, d*sqrt(3)/2)
    # Vertex at 120°: (-d/2, d*sqrt(3)/2)
    # So the top edge connects (d/2, d*sqrt(3)/2) to (-d/2, d*sqrt(3)/2)
    # The upper-right edge connects (d, 0) to (d/2, d*sqrt(3)/2)
    # A neighbor sharing the upper-right edge would be at:
    # midpoint of that edge is (3d/4, d*sqrt(3)/4), and the neighbor center is
    # at distance d*sqrt(3) in the normal direction... 
    
    # This is getting complicated. Let me just use known honeycomb positions.
    # For a hex grid with flat-top hexagons of side 1:
    # Column spacing: 1.5 (horizontal distance between column centers)
    # Row spacing: sqrt(3) (vertical distance between row centers)
    # Odd columns are offset by sqrt(3)/2 vertically
    
    # Actually, let me just place them and verify with the validity check.
    # I'll use a simpler approach: place hexagons in a compact cluster and optimize.
    
    # Base positions for 12 hexagons in a roughly circular/hexagonal cluster
    # Using a hexagonal lattice with spacing that allows touching
    # For unit hexagons, the minimum center distance for non-overlapping is 2 (for flat-to-flat)
    # but in honeycomb it's sqrt(3) for some directions
    
    # Let me use pointy-top hexagons (rotate by 90° from flat-top)
    # Pointy-top: vertices at 30, 90, 150, 210, 270, 330
    # Width = sqrt(3), Height = 2
    # Honeycomb: horizontal spacing = sqrt(3), vertical spacing = 1.5
    # Odd rows offset by sqrt(3)/2
    
    # For pointy-top (angle = 90 degrees in our convention, since default 0 is flat-top)
    # Actually, let's keep angle 0 and just use the positions that work.
    
    # I'll define base positions in a local coordinate system, then apply global rotation.
    # The base cluster: 12 hexagons in a 3-4-3 pattern
    
    # Using flat-top hexagons (angle 0):
    # Positions (approximate honeycomb):
    base_positions = np.array([
        # Row 0 (top): 3 hexagons
        [-1.5, 0.866], [0, 0.866], [1.5, 0.866],
        # Row 1 (middle): 4 hexagons
        [-2.25, 0], [-0.75, 0], [0.75, 0], [2.25, 0],
        # Row 2 (bottom): 3 hexagons
        [-1.5, -0.866], [0, -0.866], [1.5, -0.866],
    ])
    
    # That's only 10. Need 12. Let me use a different topology.
    # 4-4-4 or 2-4-4-2 or just a 3x4 grid
    
    # Let me try a 3x4 rectangular grid of flat-top hexagons
    # Horizontal spacing: 2 (edge-to-edge), vertical spacing: sqrt(3)
    # But that's not very compact.
    
    # Better: use a hexagonal cluster. For 12 hexagons, a good shape is:
    # Center + 6 around it + 5 more = 12
    # Or: two rows of 4 and two rows of 2? 
    
    # Let me just use a 4x3 grid with honeycomb offset:
    # Row 0: 4 hexagons at y=0
    # Row 1: 4 hexagons at y=sqrt(3), offset by 1 in x
    # Row 2: 4 hexagons at y=2*sqrt(3)
    
    # For flat-top hexagons with side 1:
    # Same-row adjacent: dx = 2 (they share a vertical edge)
    # Adjacent rows: dy = sqrt(3), dx offset = 1
    
    base_positions = np.array([
        # Row 0
        [-3, 0], [-1, 0], [1, 0], [3, 0],
        # Row 1 (offset)
        [-2, np.sqrt(3)], [0, np.sqrt(3)], [2, np.sqrt(3)],
        # Row 2
        [-3, 2*np.sqrt(3)], [-1, 2*np.sqrt(3)], [1, 2*np.sqrt(3)], [3, 2*np.sqrt(3)],
    ])
    
    # That's 11. Need one more. Let me add a center one.
    # Actually, let me just use 12 positions in a compact arrangement.
    
    # New approach: use a hexagonal number arrangement
    # 1 + 6 + 5 = 12? No, 1+6=7, need 5 more.
    # Let me just place them manually for a compact shape.
    
    # Compact 12-hexagon cluster (flat-top):
    base_positions = np.array([
        # Center
        [0, 0],
        # Ring 1 (6 hexagons)
        [2, 0], [1, np.sqrt(3)], [-1, np.sqrt(3)],
        [-2, 0], [-1, -np.sqrt(3)], [1, -np.sqrt(3)],
        # Ring 2 partial (5 more to make 12)
        [3, np.sqrt(3)], [0, 2*np.sqrt(3)], [-3, np.sqrt(3)],
        [0, -2*np.sqrt(3)], [-3, -np.sqrt(3)],
    ])
    
    # Hmm, that's 12 but might not be optimal. Let me just try several topologies
    # and optimize each.
    
    topologies = []
    
    # Topology 1: 3-4-3 honeycomb
    top1 = np.array([
        [-1.5, np.sqrt(3)], [0, np.sqrt(3)], [1.5, np.sqrt(3)],
        [-2.5, 0], [-0.833, 0], [0.833, 0], [2.5, 0],
        [-1.5, -np.sqrt(3)], [0, -np.sqrt(3)], [1.5, -np.sqrt(3)],
    ])
    # That's 10. Add 2 more:
    top1 = np.vstack([top1, [[-2.5, 2*np.sqrt(3)], [2.5, 2*np.sqrt(3)]]])
    topologies.append(top1)
    
    # Topology 2: 4-4-4 grid
    top2 = np.array([
        [-3, 0], [-1, 0], [1, 0], [3, 0],
        [-2, np.sqrt(3)], [0, np.sqrt(3)], [2, np.sqrt(3)], [4, np.sqrt(3)],
        [-3, 2*np.sqrt(3)], [-1, 2*np.sqrt(3)], [1, 2*np.sqrt(3)], [3, 2*np.sqrt(3)],
    ])
    topologies.append(top2)
    
    # Topology 3: Hexagonal cluster (center + ring)
    top3 = np.array([
        [0, 0],
        [2, 0], [1, np.sqrt(3)], [-1, np.sqrt(3)],
        [-2, 0], [-1, -np.sqrt(3)], [1, -np.sqrt(3)],
        [3, np.sqrt(3)], [0, 2*np.sqrt(3)], [-3, np.sqrt(3)],
        [0, -2*np.sqrt(3)], [-3, -np.sqrt(3)],
    ])
    topologies.append(top3)
    
    # Topology 4: 2-4-4-2
    top4 = np.array([
        [-1, 0], [1, 0],
        [-3, np.sqrt(3)], [-1, np.sqrt(3)], [1, np.sqrt(3)], [3, np.sqrt(3)],
        [-3, 2*np.sqrt(3)], [-1, 2*np.sqrt(3)], [1, 2*np.sqrt(3)], [3, 2*np.sqrt(3)],
        [-1, 3*np.sqrt(3)], [1, 3*np.sqrt(3)],
    ])
    topologies.append(top4)
    
    # For each topology, try different global rotations and find best enclosing hex
    for topo_idx, base_pos in enumerate(topologies):
        if time.time() > deadline:
            break
        
        # Try different rotations of the entire cluster
        for global_angle in [0, 30, 60, 90, 15, 45, 75]:
            if time.time() > deadline:
                break
            
            # Rotate base positions
            theta = np.radians(global_angle)
            R = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
            rotated_pos = base_pos @ R.T
            
            # Each inner hexagon also rotated by global_angle
            inner = np.zeros((12, 3))
            for i in range(12):
                inner[i, 0] = rotated_pos[i, 0]
                inner[i, 1] = rotated_pos[i, 1]
                inner[i, 2] = global_angle
            
            # Compute all vertices
            inner_verts = []
            for i in range(12):
                x, y, ang = inner[i]
                verts = _hex_vertices(x, y, 1.0, ang)
                inner_verts.append(verts)
            
            # Try different outer hexagon angles
            for outer_angle in [0, 30, 60, 90, 15, 45, 75]:
                if time.time() > deadline:
                    break
                
                cx, cy, s = _min_enclosing_hex_side(inner_verts, outer_angle)
                
                if s < best_s:
                    # Verify validity
                    outer = np.array([cx, cy, outer_angle])
                    if _check_validity(inner, outer, s):
                        best_s = s
                        best_inner = inner.copy()
                        best_outer = outer.copy()
    
    # If no valid solution found, return a default large one
    if best_inner is None:
        inner = np.zeros((12, 3))
        for i in range(12):
            inner[i, 0] = (i % 4 - 1.5) * 3
            inner[i, 1] = (i // 4 - 0.5) * 3
        outer = np.array([0, 0, 0])
        s = 15
        return inner, outer, s
    
    return best_inner, best_outer, best_s

# EVOLVE-BLOCK-END

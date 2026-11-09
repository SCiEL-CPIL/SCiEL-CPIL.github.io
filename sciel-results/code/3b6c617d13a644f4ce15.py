# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import itertools


def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon (center, side, angle in deg)."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _separated(v1, v2, eps=1e-6):
    """Check if two convex hexagons (6 vertices each) are separated on at least one of the 12 edge normals."""
    for i in range(6):
        # Edge of v1
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        # Project v1 and v2 onto this normal
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        if proj1.max() < proj2.min() - eps or proj2.max() < proj1.min() - eps:
            return True
        # Edge of v2
        q1 = v2[i]
        q2 = v2[(i + 1) % 6]
        edge2 = q2 - q1
        normal2 = np.array([-edge2[1], edge2[0]])
        proj1b = v1 @ normal2
        proj2b = v2 @ normal2
        if proj1b.max() < proj2b.min() - eps or proj2b.max() < proj1b.min() - eps:
            return True
    return False


def _all_separated(vertices_list, eps=1e-6):
    """Check all pairs of hexagons are separated."""
    n = len(vertices_list)
    for i in range(n):
        for j in range(i + 1, n):
            if not _separated(vertices_list[i], vertices_list[j], eps):
                return False
    return True


def _minimal_enclosing_hexagon(inner_vertices_list, n_inner=11):
    """
    Find the minimal side length s and angle a such that all inner hexagon vertices
    are inside the outer regular hexagon centered at origin with side s and angle a.
    
    Uses linear programming: for a given angle a, the constraint that point p is inside
    hexagon (center 0, side s, angle a) is: for each of 6 edge normals n_k,
    p . n_k <= s * cos(30°) = s * sqrt(3)/2.
    
    So for fixed angle a, we need s >= max over all inner vertices p and all 6 normals n_k of (p . n_k) / (sqrt(3)/2).
    We minimize over angle a.
    
    Actually, the outer hexagon with side s and angle a has vertices at s*cos(a+60k), s*sin(a+60k).
    The edge from vertex k to k+1 has outward normal direction at angle a + 60k + 30°.
    The distance from center to each edge is s * cos(30°) = s * sqrt(3)/2.
    
    So constraint: for each inner vertex p and each k in 0..5:
        p . (cos(a+60k+30°), sin(a+60k+30°)) <= s * sqrt(3)/2
    
    For fixed a: s >= max_{p,k} (p . n_k(a)) / (sqrt(3)/2)
    We want to minimize s over a.
    
    This is a 1D optimization over a in [0, 60°) (periodicity).
    """
    # Collect all inner vertices
    all_pts = np.vstack(inner_vertices_list)
    n_pts = len(all_pts)
    
    def min_side_for_angle(a_deg):
        a = np.radians(a_deg)
        min_s = 0.0
        for k in range(6):
            n_k = np.array([np.cos(a + np.radians(60 * k + 30)), np.sin(a + np.radians(60 * k + 30))])
            proj = all_pts @ n_k
            max_proj = proj.max()
            s_needed = max_proj / (np.sqrt(3) / 2)
            if s_needed > min_s:
                min_s = s_needed
        return max(min_s, 0.0)
    
    # Search over angle in [0, 60) with fine resolution
    best_s = np.inf
    best_a = 0.0
    # Coarse search
    for a_deg in np.linspace(0, 60, 601, endpoint=False):
        s = min_side_for_angle(a_deg)
        if s < best_s:
            best_s = s
            best_a = a_deg
    
    # Fine search around best
    for a_deg in np.linspace(best_a - 2, best_a + 2, 401):
        if a_deg < 0 or a_deg >= 60:
            a_deg = a_deg % 60
        s = min_side_for_angle(a_deg)
        if s < best_s:
            best_s = s
            best_a = a_deg % 60
    
    return best_s, best_a


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon's side length.
    """
    # Hexagonal lattice basis for unit hexagons
    # A unit hexagon has "width" 2 (flat-to-flat = sqrt(3), point-to-point = 2)
    # For flat-top hexagons (angle 0), the centers in a hex grid are spaced by:
    #   horizontal: 3 (for adjacent columns)
    #   vertical offset: sqrt(3)/2 for adjacent rows, with x offset 1.5
    
    # Actually for unit side hexagons:
    # - point-to-point distance = 2
    # - flat-to-flat distance = sqrt(3)
    # - For flat-top orientation (vertex at angle 0), adjacent centers:
    #   East: (3, 0)  -- wait, let me think more carefully
    
    # A regular hexagon with side 1, flat-top (vertex at 0°):
    # Vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
    # Two such hexagons sharing an edge: the edge from (0.5, sqrt(3)/2) to (1,0) has midpoint (0.75, sqrt(3)/4)
    # The adjacent hexagon center is at distance sqrt(3) from this hexagon's center, in the direction perpendicular to the shared edge.
    
    # For flat-top hexagons, the 6 neighbor centers are at:
    # (0, sqrt(3)), (sqrt(3)/2, sqrt(3)/2), (sqrt(3)/2, -sqrt(3)/2), (0, -sqrt(3)), (-sqrt(3)/2, -sqrt(3)/2), (-sqrt(3)/2, sqrt(3)/2)
    
    # Wait, let me reconsider. For a unit hexagon with vertices at angles 0,60,120,180,240,300:
    # v0 = (1, 0), v1 = (0.5, sqrt(3)/2)
    # Edge v0-v1: midpoint = (0.75, sqrt(3)/4), direction = (-0.5, sqrt(3)/2)
    # Normal (outward) = (sqrt(3)/2, 0.5) -- perpendicular to edge, pointing away from center
    # Neighbor center = center + 2 * (outward normal to edge) * (apothem) ... 
    # Actually, the distance between centers of two adjacent unit hexagons = 2 * apothem = 2 * sqrt(3)/2 = sqrt(3)
    # Direction from center to neighbor = outward normal of shared edge
    # Edge v0-v1 outward normal: edge direction = v1-v0 = (-0.5, sqrt(3)/2), normal = (sqrt(3)/2, 0.5) (rotate 90° CCW... let's check: (-0.5,sqrt(3)/2) rotated 90° CW = (sqrt(3)/2, 0.5))
    # So neighbor at sqrt(3) * (sqrt(3)/2, 0.5) = (1.5, sqrt(3)/2)
    
    # So for flat-top unit hexagons, neighbor offsets are:
    # (1.5, sqrt(3)/2), (0, sqrt(3)), (-1.5, sqrt(3)/2), (-1.5, -sqrt(3)/2), (0, -sqrt(3)), (1.5, -sqrt(3)/2)
    
    sqrt3 = np.sqrt(3)
    # Neighbor offsets for flat-top unit hexagons
    offsets = np.array([
        [1.5, sqrt3/2],
        [0, sqrt3],
        [-1.5, sqrt3/2],
        [-1.5, -sqrt3/2],
        [0, -sqrt3],
        [1.5, -sqrt3/2]
    ])
    
    # Generate candidate positions on a hex grid
    # Use axial coordinates (q, r) to generate positions
    # position = q * offsets[0] + r * offsets[1]  (using two basis vectors)
    basis1 = offsets[0]  # (1.5, sqrt3/2)
    basis2 = offsets[1]  # (0, sqrt3)
    
    # Search over a range of grid positions
    # We need 11 hexagons. Let's try configurations where they're roughly in a compact cluster.
    # Generate all subsets of size 11 from a bounded set of grid points.
    
    # Define a set of candidate grid positions
    max_coord = 3  # search q, r in [-max_coord, max_coord]
    candidates = []
    for q in range(-max_coord, max_coord + 1):
        for r in range(-max_coord, max_coord + 1):
            pos = q * basis1 + r * basis2
            candidates.append(pos)
    candidates = np.array(candidates)
    
    n_cand = len(candidates)
    
    # Precompute vertices for each candidate position (angle 0)
    cand_vertices = []
    for pos in candidates:
        verts = _hex_vertices(pos[0], pos[1], 1.0, 0.0)
        cand_vertices.append(verts)
    
    # We need to choose 11 positions from candidates such that all pairs are separated.
    # Then compute minimal enclosing hexagon.
    
    # This is a combinatorial problem. Let's use a smarter approach:
    # Start with a known good configuration and try local moves.
    
    # Initial configuration: a compact cluster of 11 hexagons
    # The "flower" pattern: center + 6 neighbors + 4 more
    # Axial coordinates for a compact 11-hexagon cluster:
    # Center: (0,0)
    # 6 neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # 4 more: (2,0), (0,2), (-2,2), (-2,0) -- or other choices
    
    # Let's define several initial configurations and try to optimize
    
    best_result = None
    best_side = np.inf
    
    # Try multiple initial configurations
    configs = [
        # Config 1: center + 6 neighbors + 4 more
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (0,2), (-2,2), (-2,0)],
        # Config 2: different arrangement
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,1), (-2,1), (-2,-1), (2,-1)],
        # Config 3: more spread
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (1,1), (-1,2), (-2,1)],
    ]
    
    for config in configs:
        # Convert axial coords to Cartesian
        positions = []
        for (q, r) in config:
            pos = q * basis1 + r * basis2
            positions.append(pos)
        
        # Compute vertices
        verts_list = []
        for pos in positions:
            verts = _hex_vertices(pos[0], pos[1], 1.0, 0.0)
            verts_list.append(verts)
        
        # Check separation
        if not _all_separated(verts_list):
            continue
        
        # Compute minimal enclosing hexagon
        s, a = _minimal_enclosing_hexagon(verts_list)
        
        if s < best_side:
            best_side = s
            best_result = (positions, s, a)
    
    if best_result is None:
        # Fallback: use a simple grid that's guaranteed to work
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
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    positions, best_s, best_a = best_result
    
    # Build inner hex data
    inner_hex_data = np.zeros((11, 3))
    for i, pos in enumerate(positions):
        inner_hex_data[i] = [pos[0], pos[1], 0.0]
    
    outer_hex_data = np.array([0.0, 0.0, best_a])
    outer_hex_side_length = float(best_s)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END

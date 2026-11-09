"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    # Hexagonal lattice positions for 11 unit hexagons
    # Using axial coordinates (q, r) -> Cartesian for flat-top hexagons
    # Center-to-center distance for touching hexagons: 2 * apothem = sqrt(3)
    # For flat-top: x = sqrt(3) * (q + r/2), y = 1.5 * r
    
    # Compact arrangement: center + 6 ring1 + 4 ring2
    # Axial coords for ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # For ring 2, pick 4 that minimize bounding hexagon
    # Let's try: (2,0), (1,1), (-1,2), (-2,1) -- but need to check which 4 of 12
    
    # Actually, let me use a known compact shape. For 11 hexagons, 
    # a good arrangement is a hexagon of radius 1 (7 hexes) plus 4 more.
    
    # Axial to cartesian for flat-top hexagons with side 1:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    def axial_to_cart(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)
    
    # Ring 0: center
    positions_axial = [(0, 0)]
    # Ring 1: 6 hexagons
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    positions_axial.extend(ring1)
    # Ring 2: need 4 more. Try several sets and pick best.
    ring2_all = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -1), (3, -1), (2, 1)]
    # Actually ring 2 has 12 hexes. Let me list them properly.
    # Ring 2 axial coords:
    ring2_all = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -1), (3, -1), (2, 1)]
    # Hmm, let me just generate ring 2 properly.
    # Ring 2: all axial coords with max(|q|, |r|, |q+r|) = 2
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                ring2.append((q, r))
    # ring2 should have 12 elements
    
    best_score = float('inf')
    best_inner = None
    best_outer = None
    
    # Try all combinations of 4 from ring 2 (there are C(12,4)=495)
    from itertools import combinations
    
    def compute_min_side(inner_positions):
        """Compute minimal outer hexagon side length for given inner hex positions (all angle 0)."""
        # Collect all vertices of all inner hexagons
        all_verts = []
        for (cx, cy) in inner_positions:
            for k in range(6):
                angle = np.pi / 3 * k
                vx = cx + np.cos(angle)
                vy = cy + np.sin(angle)
                all_verts.append((vx, vy))
        all_verts = np.array(all_verts)
        
        # For outer hexagon centered at origin with side s, rotated by phi:
        # Edge normals are at angles phi + k*pi/3 + pi/6 for k=0..5 (pointing outward from edges)
        # A point p is inside iff p . n_k <= s * sqrt(3)/2 for all k
        # So required s for point p and rotation phi: max_k (p . n_k) / (sqrt(3)/2)
        # Overall required s: max over all points, max over k
        
        # Try 6 outer rotations (0, 30, 60, 90, 120, 150 degrees) to find best
        best_s = float('inf')
        for phi_deg in [0, 30, 60, 90, 120, 150]:
            phi = np.radians(phi_deg)
            # Edge normals for hexagon with vertex at angle phi:
            # Vertices at phi + k*pi/3, edges between consecutive vertices
            # Edge normal (outward) for edge between vertex k and k+1:
            # The edge is perpendicular to the direction from center to midpoint of edge
            # Midpoint of edge k-k+1 is at angle phi + (k+0.5)*pi/3
            # So normal direction is phi + (k+0.5)*pi/3
            normals = []
            for k in range(6):
                n_angle = phi + (k + 0.5) * np.pi / 3
                normals.append((np.cos(n_angle), np.sin(n_angle)))
            normals = np.array(normals)
            
            # For each vertex, compute max projection onto normals
            projections = all_verts @ normals.T  # shape (num_verts, 6)
            max_proj = np.max(projections, axis=1)  # max over 6 normals for each vertex
            required_s = np.max(max_proj) / (np.sqrt(3) / 2)
            if required_s < best_s:
                best_s = required_s
        
        return best_s
    
    # Generate candidate arrangements
    # First, try the 7-hex center arrangement with different 4-hex additions from ring 2
    base_positions = [axial_to_cart(q, r) for (q, r) in positions_axial]
    
    for combo in combinations(range(len(ring2)), 4):
        if time.time() > deadline - 5.0:
            break
        extra = [ring2[i] for i in combo]
        all_axial = positions_axial + extra
        cart_positions = [axial_to_cart(q, r) for (q, r) in all_axial]
        s = compute_min_side(cart_positions)
        if s < best_score:
            best_score = s
            best_inner = cart_positions
            best_outer = (0.0, 0.0, s, 0.0)
    
    # Also try some alternative compact arrangements
    # Try a more linear/strip arrangement
    # And try shifting the whole cluster
    
    if best_inner is None:
        # Fallback: just use the 7-hex cluster + 4 arbitrary
        extra = ring2[:4]
        all_axial = positions_axial + extra
        cart_positions = [axial_to_cart(q, r) for (q, r) in all_axial]
        s = compute_min_side(cart_positions)
        best_score = s
        best_inner = cart_positions
        best_outer = (0.0, 0.0, s, 0.0)
    
    # Build inner list
    inner = []
    for (cx, cy) in best_inner:
        inner.append([cx, cy, 0.0])
    
    outer_center = (0.0, 0.0)
    outer_side = best_score
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END


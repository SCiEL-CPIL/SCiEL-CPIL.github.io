"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5s safety margin
    
    # 1. Generate hexagonal lattice points for N hexagons
    # We want a compact cluster. A hexagonal number pattern is optimal for lattice packing.
    # For N=11, we can use a center point + 6 neighbors + 4 more, or a specific subset of a larger hex.
    # Let's generate a dense hexagonal grid and pick the N closest to the origin.
    
    # Lattice vectors for triangular lattice (hexagon centers)
    # Distance between centers of non-overlapping unit hexagons is 2 * sqrt(3) / 2 * 2? 
    # No. Side length s=1.
    # Width of hex (flat-to-flat) = sqrt(3).
    # To tile, centers are spaced by sqrt(3) in perpendicular directions?
    # Let's check geometry.
    # Regular hexagon side 1.
    # Vertices at distance 1 from center.
    # Flat-to-flat distance = sqrt(3).
    # If we place hexagons in a grid, the centers form a triangular lattice.
    # The distance between adjacent centers in a tiling is sqrt(3).
    # Wait, if they are disjoint, they can be closer? No, "disjoint" usually implies no overlap.
    # If they touch, distance = sqrt(3).
    # Let's use d = sqrt(3) + epsilon to be safe.
    
    d = math.sqrt(3) + 1e-6
    
    # Generate points in a hexagonal grid
    # Basis vectors:
    # v1 = (d, 0)
    # v2 = (d/2, d*sqrt(3)/2)
    
    v1 = (d, 0.0)
    v2 = (d/2.0, d * math.sqrt(3) / 2.0)
    
    points = []
    # Generate enough points to cover N
    # Max radius in grid units needed for N points ~ sqrt(N)
    R = 10 
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            # Calculate cartesian coords
            x = i * v1[0] + j * v2[0]
            y = i * v1[1] + j * v2[1]
            points.append((x, y))
            
    # Sort by distance to origin to get the most compact cluster
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take the first N points
    selected = points[:N]
    
    # Calculate the bounding hexagon side length
    # The outer hexagon must contain all inner hexagons.
    # An inner hexagon has "radius" (center to vertex) = 1.
    # So the effective radius of the cluster is max_dist_to_center + 1.
    # The outer hexagon has side length S. Its inradius (center to flat) is S*sqrt(3)/2.
    # Its circumradius (center to vertex) is S.
    # For a regular hexagon, if we align them, the condition for containment is:
    # The maximum distance from the outer center to any point in the inner set must be <= S.
    # However, the shape matters. A point at distance r from center is contained if r <= S.
    # But if the point is near a flat, the boundary is closer? No, the vertex is the furthest.
    # Actually, for a regular hexagon, the distance to the boundary varies from inradius to circumradius.
    # To be safe, we should ensure the convex hull of the inner hexagons fits inside the outer hexagon.
    
    # A simpler robust check:
    # The outer hexagon is defined by 6 half-planes.
    # For a hexagon of side S centered at origin, aligned with axes (flat top/bottom? or pointy top?)
    # Let's assume standard orientation: vertices at angles 0, 60, ...
    # Wait, standard "pointy top" or "flat top"?
    # The problem allows rotation of the outer hexagon.
    # Let's find the minimal S for a fixed orientation, then try a few orientations.
    
    def min_side_for_orientation(angle_deg):
        # Rotate all selected points by -angle to align outer hexagon with axes
        # Outer hexagon vertices at angles 0, 60, ...
        # Inradius = S * sqrt(3) / 2
        # The constraint is that for every inner hexagon (which is a hexagon of side 1),
        # it must be inside the outer hexagon.
        
        # Approximation: Treat inner hexagons as circles of radius 1?
        # No, hexagons are larger than circles.
        # The maximum extent of a unit hexagon from its center is 1 (to vertices).
        # So if we place the center of the outer hexagon at the centroid of the inner set,
        # and we ensure that the maximum distance from the centroid to any vertex of any inner hexagon
        # is <= S, is that sufficient?
        # Not quite, because the boundary of the outer hexagon is not circular.
        
        # Let's calculate the required S for a specific orientation.
        # The outer hexagon H_out has side S.
        # The inner hexagons H_in have side 1.
        # H_in must be a subset of H_out.
        
        # Let's sample the vertices of the inner hexagons.
        # For each inner hexagon center c, and rotation r (we can set r=0 for simplicity, 
        # or optimize r? The task says "each inner hexagon may translate and rotate".
        # To minimize the outer size, we should probably align all inner hexagons 
        # to minimize the convex hull, or just use the lattice alignment).
        # Given the lattice structure, aligning them is natural.
        
        # Let's assume all inner hexagons are aligned with the outer hexagon (angle 0 relative to outer).
        # We need to find S such that all inner hexagons are inside.
        
        # The vertices of a unit hexagon centered at (x,y) with rotation 0 (flat top? no, let's use standard):
        # Let's define the outer hexagon with "flat" sides horizontal?
        # If outer is flat-top, vertices at 30, 90, 150...
        # Let's use the property: A convex polygon P is inside Q if all vertices of P are inside Q.
        
        # Generate vertices for one inner hexagon (side 1, rot 0)
        # Let's use "pointy-top" for both to match lattice symmetry?
        # Lattice vectors were based on 60 deg symmetry.
        
        # Vertices of unit hexagon (pointy top, center 0,0):
        # Angles: 90, 30, -30, -90, -150, 150 degrees?
        # Let's use angles 0, 60, 120... for simplicity in code, then rotate.
        # Actually, the lattice spacing d=sqrt(3) corresponds to flat-to-flat distance.
        # This implies the hexagons are oriented such that their flats are parallel to the lattice vectors?
        # If centers are at distance sqrt(3), and hexagons are flat-top, the distance between parallel flats is sqrt(3).
        # So if we place them in a grid with spacing sqrt(3) in 2 orthogonal directions (60 deg apart), they touch.
        
        # So, inner hexagons should be "flat-top" (or flat-bottom) to match the lattice spacing of sqrt(3).
        # Let's define the inner hexagon vertices for a flat-top hexagon of side 1:
        # Width (horizontal) = 2. Height (vertical) = sqrt(3).
        # Vertices:
        # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        
        inner_verts_rel = [
            (1.0, 0.0),
            (0.5, math.sqrt(3)/2),
            (-0.5, math.sqrt(3)/2),
            (-1.0, 0.0),
            (-0.5, -math.sqrt(3)/2),
            (0.5, -math.sqrt(3)/2)
        ]
        
        # For each selected center, generate its 6 vertices in global coords.
        # All vertices must be inside the outer hexagon.
        
        # Outer hexagon: Side S, centered at (0,0) (we can shift inner set to center at 0).
        # Let's center the inner set at the origin.
        cx = sum(p[0] for p in selected) / N
        cy = sum(p[1] for p in selected) / N
        selected_centered = [(x - cx, y - cy) for x, y in selected]
        
        # We can rotate the outer hexagon. Let's try 6 orientations (0, 30, 60... effectively 0 and 30 due to symmetry).
        # Actually, the lattice has 6-fold symmetry.
        
        all_verts = []
        for x, y in selected_centered:
            for vx, vy in inner_verts_rel:
                all_verts.append((x + vx, y + vy))
                
        # Find min S for a given rotation of the outer hexagon.
        # Outer hexagon (flat-top) of side S has vertices at:
        # (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
        # The boundary is defined by 6 lines.
        # Line 1 (Right): x = S
        # Line 2 (Top-Right): x/2 + y*sqrt(3)/2 = S*sqrt(3)/2  => x + y*sqrt(3) = S*sqrt(3)
        # Line 3 (Top-Left): -x + y*sqrt(3) = S*sqrt(3)
        # Line 4 (Left): x = -S
        # Line 5 (Bottom-Left): x - y*sqrt(3) = -S*sqrt(3)
        # Line 6 (Bottom-Right): -x - y*sqrt(3) = -S*sqrt(3)
        
        # For a point (px, py) to be inside:
        # px <= S
        # px + py*sqrt(3) <= S*sqrt(3)
        # -px + py*sqrt(3) <= S*sqrt(3)
        # px >= -S
        # px - py*sqrt(3) >= -S*sqrt(3)
        # -px - py*sqrt(3) >= -S*sqrt(3)
        
        # This implies:
        # S >= px
        # S >= (px + py*sqrt(3)) / sqrt(3)
        # S >= (-px + py*sqrt(3)) / sqrt(3)
        # S >= -px
        # S >= -(px - py*sqrt(3)) / sqrt(3)
        # S >= -(-px - py*sqrt(3)) / sqrt(3)
        
        # So S must be at least the max of these 6 values for all points.
        
        def required_S_for_rot(angle_deg):
            # Rotate all vertices by -angle to align outer hexagon with axes
            rad = math.radians(-angle_deg)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            
            max_s = 0.0
            for px, py in all_verts:
                # Rotate point
                rx = px * cos_a - py * sin_a
                ry = px * sin_a + py * cos_a
                
                # Calculate constraints
                c1 = rx
                c2 = (rx + ry * math.sqrt(3)) / math.sqrt(3)
                c3 = (-rx + ry * math.sqrt(3)) / math.sqrt(3)
                c4 = -rx
                c5 = -(rx - ry * math.sqrt(3)) / math.sqrt(3)
                c6 = -(-rx - ry * math.sqrt(3)) / math.sqrt(3)
                
                local_max = max(c1, c2, c3, c4, c5, c6)
                if local_max > max_s:
                    max_s = local_max
                    
            return max_s
            
        # Try a few rotations to find the minimum S
        best_s = float('inf')
        # The lattice is symmetric, but the cluster might not be perfectly centered or symmetric if N is not a hex number.
        # Try 0 to 30 degrees in small steps, or just 0, 15, 30.
        for ang in range(0, 31, 1):
            if time.time() > deadline:
                break
            s = required_S_for_rot(ang)
            if s < best_s:
                best_s = s
                
        # Add a small epsilon to ensure strict disjointness/containment due to floating point
        final_s = best_s + 1e-6
        
        # Construct the output
        # inner: list of [x, y, angle_degrees]
        # We used angle 0 for inner hexagons in our calculation.
        # So angle_degrees = 0.0
        inner = []
        for x, y in selected:
            # Note: 'selected' are original coords, not centered.
            # We need to return coords relative to the outer center (0,0).
            # So we must use the centered coords.
            cx_sel, cy_sel = (x - cx, y - cy)
            inner.append([cx_sel, cy_sel, 0.0])
            
        return inner, (0.0, 0.0), final_s, 0.0

# EVOLVE-BLOCK-END

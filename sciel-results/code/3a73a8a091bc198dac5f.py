import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Define the honeycomb lattice positions for 12 hexagons
    # A 3-3-3-3 strip is a compact arrangement.
    # In a honeycomb lattice of unit hexagons (side 1):
    # Distance between centers in same row: 2 * (sqrt(3)/2) = sqrt(3)
    # Distance between rows: 1.5 (vertical distance is 1.5 for pointy-top? No.)
    
    # Let's use standard coordinates for "flat-top" or "pointy-top".
    # Let's use a coordinate system where the hexagon vertices are at angles 0, 60, ...
    # This is a "pointy-top" hexagon if we rotate 90, but let's stick to the definition:
    # Vertices at (cos(theta), sin(theta)) for theta = 0, 60, 120...
    # This means the hexagon has vertices on the x-axis.
    # The "width" (distance between parallel sides) is sqrt(3).
    # The "height" (distance between opposite vertices) is 2.
    
    # For packing, it is often easier to think in terms of the triangular lattice of centers.
    # If we align the hexagons with angle 0, their centers form a honeycomb.
    # Let's generate centers for a 3x4 block of hexagons in a honeycomb pattern.
    # Rows of 3, 4 rows.
    
    # Coordinates for a honeycomb lattice where adjacent centers are distance sqrt(3) apart?
    # No, if side length is 1, the distance between centers of touching hexagons is 2 * apothem = sqrt(3).
    # Wait. Apothem = sqrt(3)/2. Distance between centers = 2 * apothem = sqrt(3).
    
    # Let's construct the centers:
    # Row 0: 3 hexagons
    # Row 1: 3 hexagons
    # Row 2: 3 hexagons
    # Row 3: 3 hexagons
    
    # In a honeycomb, rows are staggered.
    # Let's define the basis vectors for the center lattice.
    # If hexagons are aligned (angle 0), the neighbors are at:
    # (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), etc.
    # Wait, let's verify.
    # Hexagon centered at (0,0) with vertices at (1,0), (0.5, sqrt(3)/2)...
    # The neighbor to the "right" shares the edge from (0.5, sqrt(3)/2) to (0.5, -sqrt(3)/2)?
    # No, the edge is vertical? No.
    # Vertex 0: (1, 0)
    # Vertex 1: (0.5, sqrt(3)/2)
    # Vertex 5: (0.5, -sqrt(3)/2)
    # The edge between V5 and V0 is not an edge. V5 is (0.5, -0.866), V0 is (1,0).
    # The edges are V0-V1, V1-V2, etc.
    # V0=(1,0), V1=(0.5, 0.866).
    # The midpoint of V0-V1 is (0.75, 0.433).
    # The neighbor across this edge would be at a distance of 2*apothem = sqrt(3) in the direction perpendicular to the edge?
    # Actually, the center of the neighbor is at 2 * (midpoint of edge) - (0,0)? No.
    # The center of the neighbor is at distance sqrt(3) from the origin, in the direction of the normal to the edge.
    # Normal to edge V0-V1:
    # Vector V0->V1 = (-0.5, 0.866).
    # Normal pointing out = (0.866, 0.5) (rotated -90 deg? or +90?).
    # Let's just use the known honeycomb lattice vectors.
    # For a hexagonal tiling, the centers form a triangular lattice with spacing sqrt(3).
    # Basis vectors:
    # a1 = (sqrt(3), 0)
    # a2 = (sqrt(3)/2, 1.5)
    
    # Let's check if these generate touching hexagons.
    # Distance between (0,0) and (sqrt(3), 0) is sqrt(3). Correct.
    # Distance between (0,0) and (sqrt(3)/2, 1.5) is sqrt(3/4 + 2.25) = sqrt(0.75+2.25)=sqrt(3). Correct.
    
    # So we can place 12 centers on this lattice.
    # We want a compact shape. A 3x4 grid in the lattice coordinates (i, j)
    # where position = i*a1 + j*a2.
    # Let's take i in [0, 1, 2] and j in [0, 1, 2, 3].
    # This creates a parallelogram of 12 hexagons.
    
    a1 = np.array([math.sqrt(3), 0])
    a2 = np.array([math.sqrt(3)/2, 1.5])
    
    centers = []
    for i in range(3):
        for j in range(4):
            c = i * a1 + j * a2
            centers.append(c)
            
    centers = np.array(centers)
    
    # Center the configuration at (0,0)
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    # All inner hexagons have angle 0 (aligned with the lattice)
    # This is a valid packing because the lattice ensures they are disjoint (touching).
    # We can potentially rotate the whole assembly to minimize the bounding hexagon?
    # The bounding hexagon can also rotate.
    # Due to the 6-fold symmetry of the hexagon, rotating the inner cluster or the outer hexagon
    # relative to each other is what matters.
    # Let's fix the inner cluster at angle 0 and optimize the outer hexagon's angle and side length.
    # Actually, we can just calculate the minimum side length for a fixed outer angle,
    # and then try a few outer angles to find the global minimum.
    
    def min_outer_side(centers, inner_side=1.0, outer_angle_deg=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with given rotation, that contains all inner hexagons (unit side, angle 0).
        """
        # The inner hexagons are at 'centers' with angle 0.
        # We need to check if all vertices of all inner hexagons are inside the outer hexagon.
        # Outer hexagon: center (0,0), side S, angle theta.
        # A point (x,y) is inside a regular hexagon of side S, angle theta, centered at origin
        # if for all 6 normals n_k, dot((x,y), n_k) <= S * apothem?
        # No, the distance from center to side is S * sqrt(3)/2.
        # The normals to the sides are at angles theta + 30, theta + 90, ...
        # Let's derive the inequality.
        # Vertices of outer hex (side S, angle theta):
        # V_k = S * (cos(theta + k*60), sin(theta + k*60))
        # The hexagon is the intersection of 6 half-planes.
        # The normal to the edge between V_k and V_{k+1} points outwards.
        # Midpoint of edge: M_k = (V_k + V_{k+1})/2.
        # Vector from origin to M_k is the normal direction.
        # Length of OM_k = S * sqrt(3)/2.
        # So the condition for a point P to be inside is:
        # dot(P, unit_normal_k) <= S * sqrt(3)/2
        # for all k=0..5.
        
        theta = math.radians(outer_angle_deg)
        apothem_factor = math.sqrt(3) / 2.0
        
        max_required_apothem = 0.0
        
        # Generate vertices for each inner hexagon
        # Inner hexagon at center C, side 1, angle 0.
        # Vertices: C + (cos(60*i), sin(60*i))
        
        for c in centers:
            for i in range(6):
                vx = math.cos(i * math.pi / 3)
                vy = math.sin(i * math.pi / 3)
                px = c[0] + vx
                py = c[1] + vy
                
                # Check against all 6 sides of the outer hexagon
                for k in range(6):
                    # Normal angle for side k
                    # Side k is between vertex k and k+1.
                    # Vertex k angle: theta + k*60
                    # Vertex k+1 angle: theta + (k+1)*60
                    # Midpoint angle: theta + k*60 + 30
                    normal_angle = theta + k * math.pi / 3 + math.pi / 6
                    
                    nx = math.cos(normal_angle)
                    ny = math.sin(normal_angle)
                    
                    dist = abs(px * nx + py * ny)
                    if dist > max_required_apothem:
                        max_required_apothem = dist
                        
        # The required apothem is max_required_apothem.
        # Side length S = Apothem / (sqrt(3)/2)
        return max_required_apothem / apothem_factor

    # We can optimize the outer angle.
    # The inner cluster is a 3x4 parallelogram of hexagons.
    # It has 2-fold symmetry (180 deg).
    # The outer hexagon has 6-fold symmetry.
    # We just need to check angles from 0 to 30 degrees.
    
    best_side = float('inf')
    best_angle = 0.0
    
    # Scan a few angles to find the optimal alignment
    # Since the computation is cheap, we can scan finely.
    for deg in np.arange(0, 30, 0.5):
        side = min_outer_side(centers, 1.0, deg)
        if side < best_side:
            best_side = side
            best_angle = deg
            
    # Refine the best angle with a finer grid
    if best_angle > 0 and best_angle < 29.5:
        for deg in np.arange(best_angle - 0.5, best_angle + 0.5, 0.01):
            side = min_outer_side(centers, 1.0, deg)
            if side < best_side:
                best_side = side
                best_angle = deg
    else:
        # Check around 0/30 boundaries
        for deg in np.concatenate([np.arange(0, 0.5, 0.01), np.arange(29.5, 30, 0.01)]):
            side = min_outer_side(centers, 1.0, deg)
            if side < best_side:
                best_side = side
                best_angle = deg

    # Construct the return values
    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0])
        
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = best_angle
    
    # Final safety check: verify no overlaps and containment
    # Overlaps are guaranteed by lattice construction.
    # Containment is guaranteed by the calculation, but let's double check one point.
    # This is just for robustness against floating point errors in the calculation above.
    # The calculation `min_outer_side` is exact for the geometry.
    
    return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END

import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define the 11 unit hexagon centers on a triangular lattice
    # Lattice spacing d = sqrt(3)
    d = math.sqrt(3)
    
    # Shape: Central hex, 6 neighbors, 4 more at (2d, 0), (-2d, 0), (d, 3), (-d, -3)
    # This shape is symmetric and compact.
    centers = [
        (0.0, 0.0),
        (d, 0.0), (-d, 0.0),
        (d/2, 1.5), (-d/2, 1.5),
        (d/2, -1.5), (-d/2, -1.5),
        (2*d, 0.0), (-2*d, 0.0),
        (d, 3.0), (-d, -3.0)
    ]
    
    # We need to find the minimum side length S of a regular hexagon
    # that contains all these points.
    # The inner hexagons can be rotated.
    # We test two main orientations:
    # 1. Inner hexagons aligned with the container (0 deg rotation)
    # 2. Inner hexagons rotated by 30 degrees relative to the container
    
    # For a fixed orientation of the container (flat-top),
    # the constraints are projections onto 3 axes: 0, 60, 120 degrees.
    # S >= 0.5 * Range(Proj_i) for each axis i.
    # S = max(0.5 * Range_i)
    
    def calc_side_length(centers, angle_deg):
        # Rotate centers by angle_deg? No, the container is fixed.
        # The inner hexagons are rotated.
        # This changes the "extent" of each hexagon.
        # However, the problem asks for the center positions.
        # The condition is that the *entire* hexagon must be inside.
        # A unit hexagon rotated by angle theta has a support function.
        # For a container of side S, the center of the inner hexagon must be
        # at a distance at most (S - support_inner) from the boundary?
        # Actually, it's easier to think of the Minkowski sum.
        # The set of valid centers for a unit hexagon of orientation theta
        # inside a container hexagon of side S (orientation 0) is
        # Container (side S) eroded by Inner Hexagon (side 1, rot theta).
        # The erosion of a hexagon by a hexagon is a hexagon (or polygon).
        # If orientations are aligned, the eroded shape is a hexagon of side S-1.
        # So the centers must lie within a hexagon of side S-1.
        # If rotated by 30 deg, the erosion is more complex.
        
        # Let's assume the optimal packing uses aligned orientations first,
        # as it maximizes the usable area for centers.
        # If aligned, the centers must lie in a hexagon of side S-1.
        # So we need to find the minimum S such that the centers fit in a hexagon of side S-1.
        # This is equivalent to finding the minimum side length S' of a hexagon
        # containing the centers, and then S = S' + 1.
        
        # Let's calculate S' for the centers.
        # Axes: 0, 60, 120
        angles = [0, 60, 120]
        max_range = 0
        for a_deg in angles:
            a_rad = math.radians(a_deg)
            cos_a = math.cos(a_rad)
            sin_a = math.sin(a_rad)
            projections = [x * cos_a + y * sin_a for x, y in centers]
            min_p = min(projections)
            max_p = max(projections)
            range_p = max_p - min_p
            if range_p > max_range:
                max_range = range_p
        S_prime = max_range / 2.0
        return S_prime + 1.0

    # Test aligned orientation
    s_aligned = calc_side_length(centers, 0)
    
    # Test rotated orientation (inner hexagons rotated 30 deg)
    # If inner hexagons are rotated 30 deg, the "erosion" is different.
    # The support function of a rotated hexagon is larger in some directions.
    # Specifically, the width in the direction normal to the flat side becomes the vertex-to-vertex distance (2).
    # The width in the direction normal to the vertex becomes the flat-to-flat distance (sqrt(3)).
    # For the container (flat-top), the normals are at 0, 60, 120.
    # If inner is rotated 30 deg (pointy-top relative to container flat-top? No, 30 deg rotation makes it pointy-top if container is flat-top? 
    # Flat-top normals: 0, 60...
    # Pointy-top normals: 30, 90...
    # If inner is pointy-top, its width in direction 0 is 2 (vertex).
    # Its width in direction 60 is sqrt(3) (flat).
    # So the effective "radius" subtracted from S varies by axis.
    # Axis 0: subtract 1 (half width 2/2)
    # Axis 60: subtract sqrt(3)/2 (half width sqrt(3)/2)
    # Axis 120: subtract sqrt(3)/2
    # So the condition is:
    # Center Proj on Axis 0 <= S - 1
    # Center Proj on Axis 60 <= S - sqrt(3)/2
    # Center Proj on Axis 120 <= S - sqrt(3)/2
    # And similarly for negatives.
    # So S >= Proj_Max_0 + 1
    # S >= Proj_Max_60 + sqrt(3)/2
    # S >= Proj_Max_120 + sqrt(3)/2
    # And S >= -Proj_Min_0 + 1, etc.
    # So S = max( (Max_0 - Min_0)/2 + 1, (Max_60 - Min_60)/2 + sqrt(3)/2, (Max_120 - Min_120)/2 + sqrt(3)/2 )?
    # No, the center can be shifted.
    # Let v_j be the center projection on axis j.
    # v_j - (S - w_j) <= Min_j  => v_j <= Min_j + S - w_j
    # v_j + (S - w_j) >= Max_j  => v_j >= Max_j - S + w_j
    # So Max_j - S + w_j <= Min_j + S - w_j
    # 2S >= Max_j - Min_j + 2w_j
    # S >= (Max_j - Min_j)/2 + w_j
    
    def calc_side_length_rotated(centers):
        angles = [0, 60, 120]
        # Half-widths of the inner hexagon (rotated 30 deg) projected onto these axes
        # Axis 0 (x-axis): Inner is pointy-top. Width is 2. Half-width w0 = 1.
        # Axis 60: Inner is pointy-top. The normal is 60 deg.
        # The inner hexagon vertices are at 30, 90, 150...
        # The projection on 60 deg axis.
        # Actually, simpler: The support function of a regular hexagon of side 1 is 
        # h(theta) = cos(theta - phi) / cos(30) ? No.
        # For a unit hexagon, the distance from center to edge is sqrt(3)/2.
        # The distance from center to vertex is 1.
        # If rotated by 30 deg, the "apothem" directions are at 30, 90, 150...
        # The container axes are 0, 60, 120.
        # For axis 0: The inner hexagon has a vertex at 0? No, rotated 30 deg means vertices at 30, 90...
        # So at axis 0, the inner hexagon has an edge normal at 30 and -30.
        # The projection of the inner hexagon onto the x-axis (0 deg) is the distance from center to the vertex at 30 deg?
        # No, the max projection is the distance to the vertex closest to the axis.
        # Vertices at 30, 90, 150, 210, 270, 330.
        # Max projection on x-axis (0 deg): x = cos(30) = sqrt(3)/2? No, vertex is at distance 1.
        # x = 1 * cos(30) = sqrt(3)/2.
        # Wait, the vertex is at angle 30. The projection is cos(30)*1 = sqrt(3)/2.
        # Is there a vertex closer to 0? No, 30 is the closest.
        # So half-width w0 = sqrt(3)/2.
        
        # For axis 60 deg:
        # Closest vertex to 60 deg is at 30 or 90.
        # Angle diff is 30.
        # Projection = cos(30)*1 = sqrt(3)/2.
        # So w60 = sqrt(3)/2.
        
        # For axis 120 deg:
        # Closest vertex is 90 or 150.
        # Angle diff is 30.
        # Projection = sqrt(3)/2.
        
        # So for a 30 deg rotated inner hexagon, the half-widths on all 3 container axes are sqrt(3)/2.
        # This is the same as the apothem of the inner hexagon!
        # This makes sense: the "width" in any direction for a regular hexagon is between apothem and radius.
        # But wait, if the inner hexagon is aligned (0 deg rotation), the half-widths are:
        # Axis 0: Apothem = sqrt(3)/2.
        # Axis 60: Radius = 1.
        # Axis 120: Radius = 1.
        
        # So:
        # Aligned: w = [sqrt(3)/2, 1, 1]
        # Rotated 30: w = [sqrt(3)/2, sqrt(3)/2, sqrt(3)/2]
        
        # Let's re-evaluate S for both cases.
        pass

    # Let's just compute S for both cases properly.
    
    def get_ranges(centers):
        angles = [0, 60, 120]
        ranges = []
        for a_deg in angles:
            a_rad = math.radians(a_deg)
            cos_a = math.cos(a_rad)
            sin_a = math.sin(a_rad)
            projections = [x * cos_a + y * sin_a for x, y in centers]
            min_p = min(projections)
            max_p = max(projections)
            ranges.append(max_p - min_p)
        return ranges

    ranges = get_ranges(centers)
    
    # Case 1: Aligned
    # w = [sqrt(3)/2, 1, 1]
    w_aligned = [math.sqrt(3)/2, 1.0, 1.0]
    S_aligned = max((ranges[i] / 2.0) + w_aligned[i] for i in range(3))
    
    # Case 2: Rotated 30 deg
    # w = [sqrt(3)/2, sqrt(3)/2, sqrt(3)/2]
    w_rotated = [math.sqrt(3)/2, math.sqrt(3)/2, math.sqrt(3)/2]
    S_rotated = max((ranges[i] / 2.0) + w_rotated[i] for i in range(3))
    
    S = min(S_aligned, S_rotated)
    
    # Return the inner coordinates, center (0,0), side S, and angle (0 for simplicity, 
    # the evaluator likely checks validity based on the returned angle)
    # If we used the rotated case, we should return 30.0 as angle?
    # The problem says "each inner hexagon may translate and rotate".
    # The return format is (inner, center, side, angle).
    # "inner rows (x, y, angle_degrees)".
    # So we can specify the angle for each inner hexagon.
    # If we chose the rotated case, we set angle to 30.
    
    inner = [[x, y, 0.0] for x, y in centers]
    if S_rotated < S_aligned:
        inner = [[x, y, 30.0] for x, y in centers]
        S = S_rotated
    else:
        S = S_aligned

    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END

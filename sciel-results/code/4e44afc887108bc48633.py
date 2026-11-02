"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the 12 centers for the compact hexagonal cluster
    # 7 central hexagons (dist 0 and 2)
    # 5 from the next shell (dist approx 3.46)
    
    # Center
    centers = [(0.0, 0.0)]
    
    # 6 neighbors (dist 2)
    neighbors = [
        (2.0, 0.0),
        (1.0, math.sqrt(3)),
        (-1.0, math.sqrt(3)),
        (-2.0, 0.0),
        (-1.0, -math.sqrt(3)),
        (1.0, -math.sqrt(3))
    ]
    centers.extend(neighbors)
    
    # 5 points from the next shell (dist sqrt(12) approx 3.46)
    # These are (2, -1), (1, 2), (-1, 2), (-2, 1), (-1, -2) in (i,j) coords
    # Converted to (x,y): x = 2i + j, y = j*sqrt(3)
    extra_points = [
        (3.0, -math.sqrt(3)),      # (2, -1)
        (4.0, 2*math.sqrt(3)),     # (1, 2) -> x=2+2=4, y=2sqrt3? No. i=1, j=2 -> x=2+2=4, y=2sqrt3. Dist sqrt(16+12)=sqrt(28)=5.29. 
                                   # Wait, (1,2) is dist 5.29.
                                   # Let's re-verify the dist 3.46 points.
                                   # d^2 = 4(i^2+ij+j^2) = 12 -> i^2+ij+j^2=3.
                                   # Solutions: (1,1), (1,-2), (-1,2), (-1,-1), (2,-1), (-2,1).
                                   # (1,1) -> x=3, y=sqrt(3). Dist sqrt(9+3)=sqrt(12).
                                   # (1,-2) -> x=0, y=-2sqrt(3). Dist sqrt(0+12)=sqrt(12).
                                   # (-1,2) -> x=-4, y=2sqrt(3). Dist sqrt(16+12)=sqrt(28). NO.
                                   # Let's check (-1,2): i=-1, j=2. x = -2+2=0. y=2sqrt3. Dist sqrt(12).
                                   # So (-1,2) is also dist 3.46.
                                   # Let's list all 6:
                                   # (1,1) -> (3, sqrt(3))
                                   # (1,-2) -> (0, -2sqrt(3))
                                   # (-1,2) -> (0, 2sqrt(3))
                                   # (-1,-1) -> (-3, -sqrt(3))
                                   # (2,-1) -> (3, -sqrt(3))
                                   # (-2,1) -> (-3, sqrt(3))
                                   
    ]
    
    # Correct list of 6 points at dist sqrt(12):
    p1 = (3.0, math.sqrt(3))
    p2 = (0.0, -2.0*math.sqrt(3))
    p3 = (0.0, 2.0*math.sqrt(3))
    p4 = (-3.0, -math.sqrt(3))
    p5 = (3.0, -math.sqrt(3))
    p6 = (-3.0, math.sqrt(3))
    
    # We need 5 of these. We pick any 5.
    # To be symmetric, we might pick 5, but it doesn't matter for S.
    # Let's pick p1, p2, p3, p4, p5.
    centers.extend([p1, p2, p3, p4, p5])
    
    # Format inner rows: (x, y, angle_degrees)
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Outer hexagon
    # Side length S = 5.0
    # Center (0,0)
    # Angle 0.0
    return inner, (0.0, 0.0), 5.0, 0.0
# EVOLVE-BLOCK-END

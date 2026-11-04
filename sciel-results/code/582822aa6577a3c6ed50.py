import numpy as np

N = 11

def construct():
    # 7 hexagons in a flower pattern
    # Center at (0,0)
    # Neighbors at distance sqrt(3) at angles 30, 90, 150, 210, 270, 330
    inner = []
    
    # Center
    inner.append([0.0, 0.0, 0.0])
    
    # 6 surrounding
    for i in range(6):
        angle = np.deg2rad(30 + 60 * i)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        inner.append([x, y, 0.0])
        
    # 4 more hexagons in the side gaps
    # The container is a regular hexagon.
    # We place them in the gaps between the core and the container sides.
    # The core touches the container sides at L=3.
    # At L=4, there is a gap.
    # We place 4 hexagons at angles 90, 150, 210, 270 (or similar)
    # Actually, the side gaps are centered at the midpoints of the container sides.
    # If container vertices are at 0, 60..., sides are at 30, 90...
    # The core vertices are at 30, 90...
    # So the core vertices point towards the container vertices?
    # No, earlier I said core vertices at 30, 90...
    # If core vertices point to container vertices, the sides are "empty" near the vertices?
    # Let's just place them at distance R_extra from center.
    # To fit in L=4, they must be inside.
    # Max radius of container is 4.
    # Hex radius is 1.
    # So center must be at R <= 3.
    # Core outer vertices are at 2.732.
    # We can place extras at R=3.0?
    # If we place at R=3.0, they might overlap with core?
    # Core vertices at 2.732.
    # If we place at 3.0, distance is 0.268. Overlap.
    # We need to place them further out or in the side gaps.
    # Side gaps are at angles 0, 60, 120, 180, 240, 300?
    # If core vertices are at 30, 90..., then side midpoints are at 0, 60...
    # So we can place extras at angles 0, 60, 120, 180.
    # Distance?
    # They need to touch the core and the container.
    # Container side at angle 0: x = L * sqrt(3)/2 = 4 * 0.866 = 3.464.
    # Core side at angle 0: x = 1.5 * sqrt(3) = 2.598.
    # Gap width: 0.866.
    # Hex apothem: 0.866.
    # So we can fit one hex in the gap, centered at x = (3.464 + 2.598)/2 = 3.031?
    # Let's check if it fits.
    # Center at (3.031, 0).
    # Left vertex at 3.031 - 1 = 2.031.
    # Core rightmost point at angle 0?
    # Core has vertices at 30, 90...
    # So at angle 0, the core boundary is the midpoint of the edge between vertex 330 and 30?
    # Vertex 330: (2.732 * cos(-30), 2.732 * sin(-30)).
    # Vertex 30: (2.732 * cos(30), 2.732 * sin(30)).
    # Midpoint x: 2.732 * cos(30) = 2.732 * 0.866 = 2.366?
    # Wait, core apothem is 2.598.
    # So core boundary at angle 0 is x=2.598.
    # Hex left vertex at 2.031.
    # Overlap! 2.031 < 2.598.
    # So we cannot fit a hex in the side gap at L=4 if it's centered on the axis.
    
    # We need a larger L.
    # Let's try L=5.0.
    
    return inner, (0.0, 0.0), 5.0, 0.0

# EVOLVE-BLOCK-START
def construct():
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END

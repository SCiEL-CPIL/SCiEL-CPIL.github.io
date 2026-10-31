import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Define positions for 11 hexagons
    # 1 at center
    # 6 at distance sqrt(3) (Ring 1)
    # 4 at distance 2*sqrt(3) (Ring 2)
    
    inner = []
    
    # Center
    inner.append([0.0, 0.0, 0.0])
    
    # Ring 1
    dist1 = math.sqrt(3.0)
    for i in range(6):
        angle_rad = math.radians(60 * i)
        x = dist1 * math.cos(angle_rad)
        y = dist1 * math.sin(angle_rad)
        inner.append([x, y, 0.0])
        
    # Ring 2: Pick 4 positions symmetrically
    # Angles 0, 120, 240, 300? 
    # Let's use 0, 90, 180, 270? No, hex symmetry is better.
    # Let's use 0, 120, 240, and 300 is not symmetric with 0,120,240.
    # Let's use 0, 60, 180, 240?
    # Actually, any 4 non-adjacent or even adjacent will work if S is big enough.
    # Let's just pick 4 specific angles: 0, 120, 240, and 300? 
    # To be safe and symmetric, let's pick 0, 120, 240, and ... wait, 4 is not divisible by 3.
    # Let's pick 0, 90, 180, 270? This creates a cross.
    # Let's stick to hex grid points.
    # Grid points at Ring 2 are at angles 0, 60, 120, 180, 240, 300, 360(0), 420(60)...
    # There are 12 positions? No, in a hex grid, the "Ring 2" has 12 hexes?
    # No, Ring 1 has 6. Ring 2 has 12.
    # The centers of Ring 2 hexes are at distance 2*sqrt(3).
    # Angles are 0, 30, 60, 90...? No.
    # The hex grid is triangular lattice.
    # Neighbors of (0,0) are at dist sqrt(3).
    # Next layer centers are at dist 2*sqrt(3).
    # The angles for the 12 centers of Ring 2 are 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330.
    # We need to pick 4.
    # Let's pick 0, 90, 180, 270 for simplicity?
    # Or 0, 120, 240, and 300?
    # Let's just pick 0, 120, 240, and 300.
    
    dist2 = 2.0 * math.sqrt(3.0)
    angles_deg = [0, 120, 240, 300]
    for deg in angles_deg:
        angle_rad = math.radians(deg)
        x = dist2 * math.cos(angle_rad)
        y = dist2 * math.sin(angle_rad)
        inner.append([x, y, 0.0])
        
    # Total 1 + 6 + 4 = 11 hexagons
    
    # Calculate required outer side length
    # Max distance from center to any vertex of inner hexagons
    # Inner hex circumradius is 1.
    # Max center dist is dist2 = 2*sqrt(3) ~ 3.464
    # Max vertex dist is dist2 + 1 ~ 4.464
    # Outer hex circumradius must be >= 4.464
    # So S >= 4.464
    # Let's use S = 4.5
    
    side = 4.5
    center = (0.0, 0.0)
    angle_outer = 0.0
    
    return inner, center, side, angle_outer
# EVOLVE-BLOCK-END

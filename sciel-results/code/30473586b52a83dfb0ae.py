"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate 12 unit hexagons in a compact hexagonal cluster
    # Center at (0,0)
    # Layer 1: 6 hexagons at distance sqrt(3)
    # Layer 2: 5 hexagons at distance 2*sqrt(3) (selected from the 12 possible)
    
    hexes = []
    
    # Center hexagon
    hexes.append((0.0, 0.0, 0.0))
    
    # Layer 1: 6 hexagons
    dist1 = math.sqrt(3)
    for i in range(6):
        angle = math.radians(60 * i)
        x = dist1 * math.cos(angle)
        y = dist1 * math.sin(angle)
        hexes.append((x, y, 0.0))
        
    # Layer 2: Add 5 more hexagons. 
    # In a full hexagonal grid of radius 2, there are 12 hexagons in layer 2.
    # We pick 5 of them. To minimize the bounding box, we should pick them 
    # such that they don't extend the maximum radius further than necessary, 
    # but since they are all at the same radius, the span is determined by the 
    # existence of any hexagon in layer 2.
    # To be safe and compact, we just pick the first 5.
    dist2 = 2 * math.sqrt(3)
    # The angles for layer 2 hexagons in a standard hex grid are 
    # 0, 60, ... for the "corners" and 30, 90, ... for the "edges"?
    # Actually, the 12 hexagons in layer 2 are at angles:
    # 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330
    # Let's add 5 of these.
    angles_layer2 = [math.radians(a) for a in [0, 60, 120, 180, 240]]
    for angle in angles_layer2:
        x = dist2 * math.cos(angle)
        y = dist2 * math.sin(angle)
        hexes.append((x, y, 0.0))
        
    # Total 1 + 6 + 5 = 12 hexagons.
    
    # Calculate minimal outer hexagon side length
    # The farthest point from the center is at distance dist2 + 1 (if vertex aligned)
    # or dist2 + sqrt(3)/2 (if side aligned).
    # We use vertex aligned (angle 0.0) so the vertex points outwards.
    # Max radius R = dist2 + 1 = 2*sqrt(3) + 1
    # For a regular hexagon of side L, the distance to vertex is L.
    # So L must be at least R.
    # However, we must also ensure the "width" fits.
    # With 6-fold symmetry, if the vertices fit, the sides fit?
    # The bounding hexagon should have the same orientation.
    # L = 2*sqrt(3) + 1 + small epsilon
    
    L = 2 * math.sqrt(3) + 1 + 0.1 # Add small buffer
    
    return hexes, (0.0, 0.0), L, 0.0
# EVOLVE-BLOCK-END

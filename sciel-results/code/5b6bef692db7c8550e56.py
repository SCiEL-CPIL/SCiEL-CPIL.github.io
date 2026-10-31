"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Place N unit hexagons on a hexagonal lattice
    # Using flat-top orientation for all hexagons
    # Flat-top hexagon: vertices at 0, 60, 120, 180, 240, 300 degrees
    # Distance between adjacent hexagon centers: sqrt(3)
    
    # Axial to cartesian for flat-top hexagons
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = math.sqrt(3) * (r + q / 2.0)
        return (x, y)
    
    # Generate hexagonal spiral positions
    positions_axial = []
    
    # Ring 0
    positions_axial.append((0, 0))
    
    # Generate rings
    def generate_ring(ring):
        if ring == 0:
            return [(0, 0)]
        
        # 6 directions in axial coordinates for flat-top
        directions = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        
        hexes = []
        q, r = 0, 0
        # Move to start of ring
        for _ in range(ring):
            q += directions[0][0]
            r += directions[0][1]
        
        for i in range(6):
            for _ in range(ring):
                hexes.append((q, r))
                q += directions[i][0]
                r += directions[i][1]
        
        return hexes
    
    for ring in range(1, 10):
        ring_hexes = generate_ring(ring)
        for hex_coord in ring_hexes:
            if len(positions_axial) < N:
                positions_axial.append(hex_coord)
            else:
                break
        if len(positions_axial) >= N:
            break
    
    # Convert to cartesian
    cartesian_positions = [axial_to_cartesian(q, r) for q, r in positions_axial]
    
    # All inner hexagons have angle 0 (flat-top, aligned with container)
    inner = [[x, y, 0.0] for x, y in cartesian_positions]
    
    # Calculate minimum container side length
    # Container is flat-top regular hexagon centered at origin
    # For a point (px, py) to be inside: for each side normal theta in [30,90,150,210,270,330]:
    #   px*cos(theta) + py*sin(theta) <= S * sqrt(3)/2
    
    S_min = 0.0
    for (cx, cy) in cartesian_positions:
        # Vertices of unit flat-top hexagon centered at (cx, cy)
        for k in range(6):
            theta = k * math.pi / 3
            vx = cx + math.cos(theta)
            vy = cy + math.sin(theta)
            # Check all 6 side constraints
            for m in range(6):
                normal_theta = (m + 0.5) * math.pi / 3
                proj = vx * math.cos(normal_theta) + vy * math.sin(normal_theta)
                S_needed = 2.0 * proj / math.sqrt(3)
                if S_needed > S_min:
                    S_min = S_needed
    
    S_min += 1e-6
    
    return inner, (0.0, 0.0), S_min, 0.0
# EVOLVE-BLOCK-END

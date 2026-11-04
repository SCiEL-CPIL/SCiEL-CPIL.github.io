import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate positions for a hexagonal grid of side 3 (which contains 19 hexagons)
    # Axial coordinates (q, r)
    positions = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                # Convert to Cartesian
                # x = sqrt(3) * (q + r/2)
                # y = 1.5 * r
                x = np.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                positions.append((x, y))
    
    # Sort by distance from origin to pick the most central 12
    positions.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take the first 12
    selected = positions[:12]
    
    inner = []
    for x, y in selected:
        inner.append([x, y, 0.0])
        
    # Calculate the required side length of the outer hexagon
    # For a grid of side S_grid, the geometric side length L is (S_grid - 1)*sqrt(3) + 1
    # Here we use the max distance from the selected hexagons.
    # The selected hexagons include those at grid distance 2.
    # The distance from center to the vertex of a hexagon at grid distance 2 is 2*sqrt(3) + 1.
    side = 2 * np.sqrt(3) + 1
    
    return inner, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END

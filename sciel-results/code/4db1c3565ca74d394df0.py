import math
import time

N = 12

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 10.0 # 10 seconds is plenty for this static solution
    
    # Configuration: 3 rows, 4 columns of Flat-Top hexagons
    # Inner Hexagons are Flat-Top (Angle 0)
    # Outer Hexagon is Flat-Top (Angle 0)
    
    # Grid parameters for Flat-Top hexagons (Side 1)
    # Width = 2, Height = sqrt(3)
    # Horizontal spacing = 1.5
    # Vertical spacing = sqrt(3)/2
    # Horizontal offset for alternating rows = 0.75
    
    h_spacing = 1.5
    v_spacing = math.sqrt(3) / 2.0
    h_offset = 0.75
    
    inner = []
    
    # 3 rows (y indices 0, 1, 2)
    # 4 cols (x indices 0, 1, 2, 3)
    # Center the grid
    # Total width of centers: 3 * 1.5 = 4.5. Center x = 0.
    # x centers: -2.25, -0.75, 0.75, 2.25
    # Total height of centers: 2 * v_spacing = sqrt(3). Center y = 0.
    # y centers: -sqrt(3)/2, 0, sqrt(3)/2
    
    y_centers = [-v_spacing, 0.0, v_spacing]
    
    for i, y in enumerate(y_centers):
        # Determine x offset
        # If row is even, offset 0?
        # Let's define row 0 (bottom) as offset 0.
        # Row 1 (middle) offset h_offset.
        # Row 2 (top) offset 0.
        # Wait, in a hex grid, if row 0 is at x=0, row 1 is at x=0.75.
        # Let's center the whole block.
        # Row 0 (4 hexes): x = -2.25, -0.75, 0.75, 2.25
        # Row 1 (4 hexes): x = -1.5, 0.0, 1.5, 3.0 ? No.
        # If Row 0 is at -2.25, -0.75, 0.75, 2.25
        # Row 1 should be shifted by 0.75.
        # -1.5, 0.0, 1.5, 3.0
        # This makes the block asymmetric in X?
        # Min X: -2.25 (Row 0) or -1.5 (Row 1).
        # Max X: 2.25 (Row 0) or 3.0 (Row 1).
        # This is not centered.
        
        # Let's center each row independently? No, that breaks the lattice.
        # Let's shift the whole lattice to be centered at (0,0).
        
        # Generate all centers
        centers = []
        for r in range(3):
            for c in range(4):
                x = c * h_spacing
                y = r * v_spacing
                # Apply offset for odd rows
                if r % 2 != 0:
                    x += h_offset
                centers.append((x, y))
        
        # Calculate bounding box of centers to center the lattice
        min_x = min(p[0] for p in centers)
        max_x = max(p[0] for p in centers)
        min_y = min(p[1] for p in centers)
        max_y = max(p[1] for p in centers)
        
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        
        for x, y in centers:
            inner.append([x - cx, y - cy, 0.0])
            
    # Outer Hexagon
    # Side length S
    # We determined S >= 3.75 for this configuration
    S = 3.75
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, S, angle

import numpy as np
import itertools

N = 12

def construct():
    # Define candidate layouts
    # Each layout is a list of (x, y, angle) for the 12 hexagons
    # We will test a few common packings and find the minimal container side length.
    
    sqrt3 = np.sqrt(3)
    
    layouts = []
    
    # Layout 1: 3x4 Grid (flat-top)
    # Rows at y = -sqrt3, 0, sqrt3
    # Cols at x = -3, -1, 1, 3
    l1 = []
    for r in range(3):
        y = (r - 1) * sqrt3
        for c in range(4):
            x = (c - 1.5) * 2.0
            l1.append([x, y, 0.0])
    layouts.append(l1)
    
    # Layout 2: 2-4-4-2 Cluster (flat-top)
    # Rows at y = -1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3
    # Row 1 (2): x = -1, 1
    # Row 2 (4): x = -3, -1, 1, 3
    # Row 3 (4): x = -3, -1, 1, 3
    # Row 4 (2): x = -1, 1
    l2 = []
    l2.extend([[-1, -1.5*sqrt3, 0.0], [1, -1.5*sqrt3, 0.0]])
    l2.extend([[-3, -0.5*sqrt3, 0.0], [-1, -0.5*sqrt3, 0.0], [1, -0.5*sqrt3, 0.0], [3, -0.5*sqrt3, 0.0]])
    l2.extend([[-3, 0.5*sqrt3, 0.0], [-1, 0.5*sqrt3, 0.0], [1, 0.5*sqrt3, 0.0], [3, 0.5*sqrt3, 0.0]])
    l2.extend([[-1, 1.5*sqrt3, 0.0], [1, 1.5*sqrt3, 0.0]])
    layouts.append(l2)
    
    # Layout 3: 3x4 Grid (pointy-top, rotated 30 deg)
    # Hex width sqrt(3), height 2.
    # Spacing x: sqrt(3), y: 1.5
    l3 = []
    for r in range(3):
        y = (r - 1) * 1.5
        for c in range(4):
            x = (c - 1.5) * sqrt3
            l3.append([x, y, 30.0])
    layouts.append(l3)
    
    # Layout 4: 2-4-4-2 Cluster (pointy-top)
    l4 = []
    # y spacing 1.5
    # Row 1 (2): x = -sqrt3/2, sqrt3/2 ? No, need to fit 4 in middle.
    # Middle row width: 3*sqrt3 + sqrt3 = 4*sqrt3? No, 4 hexes side by side.
    # Width of 4 hexes (pointy): 4*sqrt(3)? No, first and last touch?
    # Distance between centers: sqrt(3).
    # x = -1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3
    # Top/Bottom rows (2 hexes): x = -sqrt3, 0? Or -0.5, 0.5?
    # Let's nestle:
    # Row 2 (4): x = -1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3
    # Row 3 (4): x = -1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3
    # Row 1 (2): x = -sqrt3, 0  (offset by sqrt3/2)
    # Row 4 (2): x = -sqrt3, 0
    l4.extend([[-sqrt3, -2.25, 30.0], [0, -2.25, 30.0]])
    l4.extend([[-1.5*sqrt3, -0.75, 30.0], [-0.5*sqrt3, -0.75, 30.0], [0.5*sqrt3, -0.75, 30.0], [1.5*sqrt3, -0.75, 30.0]])
    l4.extend([[-1.5*sqrt3, 0.75, 30.0], [-0.5*sqrt3, 0.75, 30.0], [0.5*sqrt3, 0.75, 30.0], [1.5*sqrt3, 0.75, 30.0]])
    l4.extend([[-sqrt3, 2.25, 30.0], [0, 2.25, 30.0]])
    layouts.append(l4)

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    for layout in layouts:
        # Find minimal container side length for this layout
        # We can optimize the container rotation as well, but let's start with 0.
        # The container is a regular hexagon.
        # We need to find the smallest S such that all vertices of all inner hexagons
        # are inside the container of side S centered at (0,0) with rotation 0.
        
        # Get all vertices of all inner hexagons
        vertices = []
        for x, y, ang in layout:
            rad = np.deg2rad(ang)
            # Vertices of unit hexagon centered at (x,y) with rotation ang
            for i in range(6):
                theta = rad + i * np.pi / 3
                vx = x + np.cos(theta)
                vy = y + np.sin(theta)
                vertices.append((vx, vy))
        
        vertices = np.array(vertices)
        
        # For a container hexagon with side S and rotation 0 (flat-top):
        # Vertices at (S, 0), (S/2, S*sqrt3/2), etc.
        # The half-width is S. The half-height is S*sqrt3/2.
        # The supporting lines are:
        # 1. x <= S
        # 2. x >= -S
        # 3. y <= S*sqrt3/2
        # 4. y >= -S*sqrt3/2
        # 5. x + y/sqrt3 <= S  (Top-Right edge? No, Top-Right vertex is (S/2, S*sqrt3/2). Edge to (S,0).)
        #    Line through (S,0) and (S/2, S*sqrt3/2):
        #    Slope = (S*sqrt3/2 - 0) / (S/2 - S) = (S*sqrt3/2) / (-S/2) = -sqrt3.
        #    y - 0 = -sqrt3(x - S) => y = -sqrt3 x + S*sqrt3 => sqrt3 x + y = S*sqrt3 => x + y/sqrt3 = S.
        # 6. x - y/sqrt3 <= S (Bottom-Right edge)
        # 7. -x + y/sqrt3 <= S (Top-Left edge)
        # 8. -x - y/sqrt3 <= S (Bottom-Left edge)
        
        # We need S >= max of all these projections.
        # proj1 = x
        # proj2 = -x
        # proj3 = y / (sqrt3/2) = 2y/sqrt3
        # proj4 = -2y/sqrt3
        # proj5 = x + y/sqrt3
        # proj6 = x - y/sqrt3
        # proj7 = -x + y/sqrt3
        # proj8 = -x - y/sqrt3
        
        # Actually, the condition for a point (px, py) to be inside a regular hexagon of side S (flat-top, centered at origin) is:
        # |px| <= S
        # |py| <= S * sqrt3 / 2
        # |px + py/sqrt3| <= S
        # |px - py/sqrt3| <= S
        
        # So S must be >= max( |px|, |py|*2/sqrt3, |px + py/sqrt3|, |px - py/sqrt3| ) for all points.
        
        S_required = 0
        for px, py in vertices:
            c1 = abs(px)
            c2 = abs(py) * 2 / sqrt3
            c3 = abs(px + py / sqrt3)
            c4 = abs(px - py / sqrt3)
            S_required = max(S_required, c1, c2, c3, c4)
            
        if S_required < best_side:
            best_side = S_required
            best_inner = layout
            best_center = (0.0, 0.0)
            best_angle = 0.0

    # If no layout found (should not happen), fallback
    if best_inner is None:
        return layouts[0], (0.0, 0.0), 5.0, 0.0

    return best_inner, best_center, best_side, best_angle

import numpy as np
import time

N = 11

def hexagon_vertices(center, angle_deg=0):
    """Generate vertices of a unit hexagon."""
    cx, cy = center
    rad = np.deg2rad(angle_deg)
    angles = np.arange(6) * np.pi / 3 + rad
    return np.array([cx + np.cos(angles), cy + np.sin(angles)]).T

def point_in_hexagon(p, center, side, angle_deg=0):
    """Check if point p is inside hexagon at center with side length and rotation."""
    cx, cy = center
    px, py = p
    rad = np.deg2rad(angle_deg)
    # Rotate point relative to center
    dx = px - cx
    dy = py - cy
    x = dx * np.cos(rad) + dy * np.sin(rad)
    y = -dx * np.sin(rad) + dy * np.cos(rad)
    
    # Hexagon inequalities for side S (apothem a = S*sqrt(3)/2)
    # |x| <= S
    # |x/2 + y*sqrt(3)/2| <= S
    # |-x/2 + y*sqrt(3)/2| <= S
    if abs(x) > side: return False
    if abs(0.5*x + (np.sqrt(3)/2)*y) > side: return False
    if abs(-0.5*x + (np.sqrt(3)/2)*y) > side: return False
    return True

def check_validity(positions, angle_deg=0):
    """Check if all hexagons are disjoint."""
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            # Check if any vertex of i is inside j or vice versa
            # Approximation: distance between centers > 2 for aligned
            # For robustness, check vertices
            vi = hexagon_vertices(positions[i], angle_deg)
            for p in vi:
                if point_in_hexagon(p, positions[j], 1.0, angle_deg):
                    return False
            vj = hexagon_vertices(positions[j], angle_deg)
            for p in vj:
                if point_in_hexagon(p, positions[i], 1.0, angle_deg):
                    return False
    return True

def min_enclosing_hexagon_side(positions, angle_deg=0):
    """Calculate minimal side length of enclosing hexagon (aligned with inner)."""
    # We assume outer hexagon is centered at (0,0) and aligned.
    # We can optimize center later, but for now (0,0) is a good guess.
    max_dist = 0
    for pos in positions:
        verts = hexagon_vertices(pos, angle_deg)
        for v in verts:
            d = np.linalg.norm(v)
            if d > max_dist:
                max_dist = d
    # For aligned hexagon, max dist to boundary is S (at vertices).
    # But the shape is not a circle.
    # The condition is max(|x|, |x/2+y*sqrt(3)/2|, |-x/2+y*sqrt(3)/2|) <= S
    max_s = 0
    for pos in positions:
        verts = hexagon_vertices(pos, angle_deg)
        for v in verts:
            x, y = v
            s1 = abs(x)
            s2 = abs(0.5*x + (np.sqrt(3)/2)*y)
            s3 = abs(-0.5*x + (np.sqrt(3)/2)*y)
            max_s = max(max_s, s1, s2, s3)
    return max_s

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    # Initial positions
    positions = [[0.0, 0.0]]
    
    # Grid for search
    # Use a hexagonal grid step
    step = 0.5
    radius = 10
    
    for _ in range(N - 1):
        if time.time() > deadline:
            break
            
        best_pos = None
        best_score = float('inf')
        
        # Generate candidates
        # Center of mass
        cx = np.mean([p[0] for p in positions])
        cy = np.mean([p[1] for p in positions])
        
        # Search in a box around current cluster
        for dx in np.arange(-radius, radius, step):
            for dy in np.arange(-radius, radius, step):
                cand = [cx + dx, cy + dy]
                
                # Quick check: distance to existing centers
                valid = True
                for p in positions:
                    d = np.linalg.norm(np.array(cand) - np.array(p))
                    if d < 1.9: # Must be > 2 for aligned
                        valid = False
                        break
                if not valid:
                    continue
                    
                # Full validity check
                if not check_validity(positions + [cand]):
                    continue
                
                # Score: minimal enclosing side
                s = min_enclosing_hexagon_side(positions + [cand])
                
                if s < best_score:
                    best_score = s
                    best_pos = cand
        
        if best_pos is not None:
            positions.append(best_pos)
        else:
            # Fallback: add far away
            positions.append([20.0, 0.0])
            
    # Final score
    final_side = min_enclosing_hexagon_side(positions)
    
    # Return
    inner = [[p[0], p[1], 0.0] for p in positions]
    return inner, (0.0, 0.0), final_side, 0.0

# EVOLVE-BLOCK-START
def construct():
    # Use the greedy algorithm defined above
    return _construct_impl()

def _construct_impl():
    import time
    start_time = time.time()
    deadline = start_time + 100.0
    
    positions = [[0.0, 0.0]]
    
    # Grid for search
    step = 0.5
    radius = 8
    
    for _ in range(N - 1):
        if time.time() > deadline:
            break
            
        best_pos = None
        best_score = float('inf')
        
        cx = np.mean([p[0] for p in positions])
        cy = np.mean([p[1] for p in positions])
        
        for dx in np.arange(-radius, radius, step):
            for dy in np.arange(-radius, radius, step):
                cand = [cx + dx, cy + dy]
                
                valid = True
                for p in positions:
                    d = np.linalg.norm(np.array(cand) - np.array(p))
                    if d < 1.9:
                        valid = False
                        break
                if not valid:
                    continue
                    
                if not check_validity(positions + [cand]):
                    continue
                
                s = min_enclosing_hexagon_side(positions + [cand])
                
                if s < best_score:
                    best_score = s
                    best_pos = cand
        
        if best_pos is not None:
            positions.append(best_pos)
        else:
            positions.append([20.0, 0.0])
            
    final_side = min_enclosing_hexagon_side(positions)
    inner = [[p[0], p[1], 0.0] for p in positions]
    return inner, (0.0, 0.0), final_side, 0.0
# EVOLVE-BLOCK-END

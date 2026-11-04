import numpy as np
import time
import random

N = 12

def hex_support_same_orientation(alpha, phi):
    """
    Support function of unit hexagon (same orientation as outer) in direction alpha.
    Outer hexagon has vertices at phi + k*pi/3.
    Inner hexagon also has vertices at phi + k*pi/3 (same orientation).
    Support in direction alpha: cos of min angular distance from alpha to vertex directions.
    """
    beta = (alpha - phi) % (np.pi / 3)
    if beta > np.pi / 6:
        beta = np.pi / 3 - beta
    return np.cos(beta)

def min_side_distance(c, O, S, phi):
    """
    Minimum signed distance from point c to the sides of outer hexagon.
    Outer hex: center O, side S, vertices at phi + k*pi/3.
    Side normals at phi + pi/6 + k*pi/3, inradius = S*sqrt(3)/2.
    """
    min_dist = float('inf')
    for k in range(6):
        n_angle = phi + np.pi / 6 + k * np.pi / 3
        nx = np.cos(n_angle)
        ny = np.sin(n_angle)
        dist = (c[0] - O[0]) * nx + (c[1] - O[1]) * ny
        # Distance from side = inradius - dist
        side_dist = S * np.sqrt(3) / 2 - dist
        if side_dist < min_dist:
            min_dist = side_dist
    return min_dist

def is_valid_same_orient(positions, O, S, phi):
    """
    Check validity assuming all inner hexagons have same orientation as outer.
    Non-overlap: |c_i - c_j| >= 2
    Containment: min_side_distance >= sqrt(3)/2
    """
    sqrt3 = np.sqrt(3)
    
    # Containment
    for i in range(N):
        if min_side_distance(positions[i], O, S, phi) < sqrt3 / 2 - 1e-10:
            return False
    
    # Non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 2.0 - 1e-10:
                return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    random.seed(42)
    np.random.seed(42)
    
    sqrt3 = np.sqrt(3)
    
    # Initial arrangement: hexagonal cluster
    # Try different row patterns
    # Pattern: 2,3,4,3 or 3,4,3,2 or 4,4,4 etc.
    
    best_S = 40.0
    best_positions = None
    best_O = None
    best_phi = 0.0
    
    # Try several initial configurations
    configs = []
    
    # Config 1: 2-3-4-3 rows
    rows = [2, 3, 4, 3]
    positions = []
    total_rows = len(rows)
    for r in range(total_rows):
        n = rows[r]
        y = (r - (total_rows - 1) / 2.0) * sqrt3
        x_offset = -(n - 1) * 2.0 / 2.0
        for c in range(n):
            x = x_offset + c * 2.0
            if r % 2 == 1:
                x += 1.0
            positions.append([x, y])
    configs.append(positions)
    
    # Config 2: 3-4-3-2 rows
    rows = [3, 4, 3, 2]
    positions = []
    total_rows = len(rows)
    for r in range(total_rows):
        n = rows[r]
        y = (r - (total_rows - 1) / 2.0) * sqrt3
        x_offset = -(n - 1) * 2.0 / 2.0
        for c in range(n):
            x = x_offset + c * 2.0
            if r % 2 == 1:
                x += 1.0
            positions.append([x, y])
    configs.append(positions)
    
    # Config 3: 4-4-4 rows
    rows = [4, 4, 4]
    positions = []
    total_rows = len(rows)
    for r in range(total_rows):
        n = rows[r]
        y = (r - (total_rows - 1) / 2.0) * sqrt3
        x_offset = -(n - 1) * 2.0 / 2.0
        for c in range(n):
            x = x_offset + c * 2.0
            if r % 2 == 1:
                x += 1.0
            positions.append([x, y])
    configs.append(positions)
    
    # Config 4: 5-4-3 rows
    rows = [5, 4, 3]
    positions = []
    total_rows = len(rows)
    for r in range(total_rows):
        n = rows[r]
        y = (r - (total_rows - 1) / 2.0) * sqrt3
        x_offset = -(n - 1) * 2.0 / 2.0
        for c in range(n):
            x = x_offset + c * 2.0
            if r % 2 == 1:
                x += 1.0
            positions.append([x, y])
    configs.append(positions)
    
    # Config 5: 2-4-4-2 rows
    rows = [2, 4, 4, 2]
    positions = []
    total_rows = len(rows)
    for r in range(total_rows):
        n = rows[r]
        y = (r - (total_rows - 1) / 2.0) * sqrt3
        x_offset = -(n - 1) * 2.0 / 2.0
        for c in range(n):
            x = x_offset + c * 2.0
            if r % 2 == 1:
                x += 1.0
            positions.append([x, y])
    configs.append(positions)
    
    for config in configs:
        O = np.array([0.0, 0.0])
        phi = 0.0
        
        # Find minimal S for this config
        lo, hi = 0.0, 20.0
        # First check if hi is valid
        while not is_valid_same_orient(config, O, hi, phi):
            hi *= 2
        
        # Binary search for minimal S
        for _ in range(60):
            mid = (lo + hi) / 2
            if is_valid_same_orient(config, O, mid, phi):
                hi = mid
            else:
                lo = mid
        
        S = hi
        if S < best_S:
            best_S = S
            best_positions = [p[:] for p in config]
            best_O = O.copy()
            best_phi = phi
    
    # Local search to improve
    if best_positions is not None:
        current_positions = [p[:] for p in best_positions]
        current_O = best_O.copy()
        current_phi = best_phi
        current_S = best_S
        
        step_size = 0.3
        for iteration in range(50000):
            if time.time() > deadline:
                break
            
            # Try to reduce S
            new_S = current_S - 0.005
            if is_valid_same_orient(current_positions, current_O, new_S, current_phi):
                current_S = new_S
                if current_S < best_S:
                    best_S = current_S
                    best_positions = [p[:] for p in current_positions]
                    best_O = current_O.copy()
                    best_phi = current_phi
                continue
            
            # Random perturbation
            move_type = random.random()
            
            if move_type < 0.7:
                # Move a random hexagon
                i = random.randint(0, N - 1)
                dx = random.gauss(0, step_size)
                dy = random.gauss(0, step_size)
                old_x, old_y = current_positions[i][0], current_positions[i][1]
                current_positions[i][0] += dx
                current_positions[i][1] += dy
                
                if is_valid_same_orient(current_positions, current_O, current_S, current_phi):
                    if current_S < best_S:
                        best_S = current_S
                        best_positions = [p[:] for p in current_positions]
                        best_O = current_O.copy()
                        best_phi = current_phi
                else:
                    current_positions[i][0] = old_x
                    current_positions[i][1] = old_y
            
            elif move_type < 0.85:
                # Move outer center
                dx = random.gauss(0, 0.1)
                dy = random.gauss(0, 0.1)
                old_Ox, old_Oy = current_O[0], current_O[1]
                current_O[0] += dx
                current_O[1] += dy
                
                if is_valid_same_orient(current_positions, current_O, current_S, current_phi):
                    if current_S < best_S:
                        best_S = current_S
                        best_positions = [p[:] for p in current_positions]
                        best_O = current_O.copy()
                        best_phi = current_phi
                else:
                    current_O[0] = old_Ox
                    current_O[1] = old_Oy
            
            else:
                # Rotate outer
                dphi = random.gauss(0, 0.05)
                old_phi = current_phi
                current_phi = (old_phi + dphi) % (2 * np.pi)
                
                if is_valid_same_orient(current_positions, current_O, current_S, current_phi):
                    if current_S < best_S:
                        best_S = current_S
                        best_positions = [p[:] for p in current_positions]
                        best_O = current_O.copy()
                        best_phi = current_phi
                else:
                    current_phi = old_phi
            
            # Occasionally reduce step size
            if iteration % 5000 == 0:
                step_size *= 0.95
    
    # Final validity check
    if best_positions is None:
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
    
    if not is_valid_same_orient(best_positions, best_O, best_S, best_phi):
        # Fallback
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
    
    # All angles are 0 (same orientation as outer)
    inner = [[p[0], p[1], 0.0] for p in best_positions]
    return inner, (best_O[0], best_O[1]), best_S, np.degrees(best_phi) % 360.0

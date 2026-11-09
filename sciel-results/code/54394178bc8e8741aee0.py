import numpy as np
import time
import math

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side=1.0, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    theta = np.radians(angle_deg)
    angles = theta + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def _polygon_area(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _polygon_intersection_area(v1, v2):
    """Sutherland-Hodgman polygon clipping to get intersection area."""
    def clip(subject, clipper):
        if len(subject) == 0:
            return []
        output = list(subject)
        for i in range(len(clipper)):
            if len(output) == 0:
                break
            input_list = output
            output = []
            S = input_list[0]
            for j in range(len(input_list)):
                E = input_list[j]
                C1 = clipper[i]
                C2 = clipper[(i + 1) % len(clipper)]
                # Determine which side of edge C1->C2
                def side(p):
                    return (C2[0]-C1[0])*(p[1]-C1[1]) - (C2[1]-C1[1])*(p[0]-C1[0])
                s_side = side(S)
                e_side = side(E)
                if s_side >= 0:
                    if e_side >= 0:
                        output.append(E)
                    else:
                        # S inside, E outside: add intersection
                        # S + t*(E-S) where t = s_side/(s_side-e_side)
                        t = s_side / (s_side - e_side)
                        ix = S[0] + t*(E[0]-S[0])
                        iy = S[1] + t*(E[1]-S[1])
                        output.append((ix, iy))
                else:
                    if e_side >= 0:
                        # S outside, E inside: add intersection then E
                        t = s_side / (s_side - e_side)
                        ix = S[0] + t*(E[0]-S[1])  # fix below
                        ix = S[0] + t*(E[0]-S[0])
                        iy = S[1] + t*(E[1]-S[1])
                        output.append((ix, iy))
                        output.append(E)
                S = E
        return output

    # Use Sutherland-Hodgman
    clipped = clip([tuple(v) for v in v1], [tuple(v) for v in v2])
    if len(clipped) < 3:
        return 0.0
    clipped = np.array(clipped)
    return _polygon_area(clipped)

def _overlap_area(hex1_center, hex1_angle, hex2_center, hex2_angle, side=1.0):
    v1 = _hex_vertices(hex1_center[0], hex1_center[1], side, hex1_angle)
    v2 = _hex_vertices(hex2_center[0], hex2_center[1], side, hex2_angle)
    return _polygon_intersection_area(v1, v2)

def _hex_in_hexagon(cx, cy, angle_deg, outer_center, outer_side, outer_angle, side=1.0):
    """Check if hexagon at (cx,cy) with angle is fully inside outer hexagon."""
    inner_verts = _hex_vertices(cx, cy, side, angle_deg)
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    # Check each inner vertex is inside outer hexagon
    for v in inner_verts:
        if not _point_in_polygon(v, outer_verts):
            return False
    return True

def _point_in_polygon(p, poly):
    """Ray casting point in polygon."""
    x, y = p
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill
    
    best_side = 40.0
    best_config = None
    best_inner = None
    
    # Strategy: use force-directed relaxation with simulated annealing
    # Multiple initial configurations
    
    def evaluate_config(positions, angles, outer_center, outer_side, outer_angle):
        """Check validity and return objective (outer side length)."""
        # Check all inner hexagons are inside outer hexagon
        for i in range(len(positions)):
            if not _hex_in_hexagon(positions[i][0], positions[i][1], angles[i],
                                   outer_center, outer_side, outer_angle):
                return float('inf')
        # Check no overlaps between inner hexagons
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                overlap = _overlap_area(positions[i], angles[i], positions[j], angles[j])
                if overlap > 1e-8:
                    return float('inf')
        return outer_side
    
    def relax(positions, angles, outer_center, outer_side, outer_angle, max_iters=5000):
        """Force-directed relaxation to push apart overlapping hexagons."""
        positions = [list(p) for p in positions]
        angles = list(angles)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            forces = [[0.0, 0.0] for _ in range(len(positions))]
            torque = [0.0 for _ in range(len(positions))]
            
            # Repulsion between overlapping hexagons
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    dx = positions[j][0] - positions[i][0]
                    dy = positions[j][1] - positions[i][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 2.0:  # potential overlap range
                        overlap = _overlap_area(positions[i], angles[i], positions[j], angles[j])
                        if overlap > 1e-10:
                            # Push apart proportional to overlap
                            if dist > 1e-8:
                                fx = overlap * dx / dist * 0.5
                                fy = overlap * dy / dist * 0.5
                            else:
                                fx, fy = 0.1, 0.1
                            forces[i][0] -= fx
                            forces[i][1] -= fy
                            forces[j][0] += fx
                            forces[j][1] += fy
                            # Torque to reduce overlap by rotating
                            torque[i] += overlap * 0.01
                            torque[j] -= overlap * 0.01
            
            # Attraction to keep hexagons inside outer boundary
            for i in range(len(positions)):
                for v in _hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i]):
                    if not _point_in_polygon(v, _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)):
                        # Push back towards center
                        dx = outer_center[0] - positions[i][0]
                        dy = outer_center[1] - positions[i][1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist > 1e-8:
                            forces[i][0] += dx / dist * 0.1
                            forces[i][1] += dy / dist * 0.1
            
            # Apply forces with damping
            step = 0.05
            for i in range(len(positions)):
                positions[i][0] += forces[i][0] * step
                positions[i][1] += forces[i][1] * step
                angles[i] += torque[i] * step * 180.0 / math.pi
            
            # Check convergence
            max_force = max(math.sqrt(f[0]**2 + f[1]**2) for f in forces)
            if max_force < 1e-6:
                break
        
        return positions, angles
    
    # Generate initial configurations
    def lattice_init():
        """Hexagonal lattice arrangement."""
        positions = []
        angles = [0.0] * N
        # Hex lattice spacing
        dx = 2.0 * math.sqrt(3) / 2  # distance between centers in hex lattice
        dy = 2.0
        # Place in hex pattern
        count = 0
        for row in range(-3, 4):
            for col in range(-3, 4):
                if count >= N:
                    break
                x = col * dx + (row % 2) * dx / 2
                y = row * dy * 0.75
                positions.append([x, y])
                count += 1
        while len(positions) < N:
            positions.append([0.0, 0.0])
        return positions[:N], angles
    
    def ring_init():
        """Ring arrangement."""
        positions = []
        angles = [0.0] * N
        for i in range(N):
            angle = 2 * math.pi * i / N
            r = 2.0
            positions.append([r * math.cos(angle), r * math.sin(angle)])
        return positions, angles
    
    def random_init():
        """Random positions."""
        positions = []
        angles = []
        for i in range(N):
            x = (np.random.random() - 0.5) * 6.0
            y = (np.random.random() - 0.5) * 6.0
            a = np.random.random() * 60.0
            positions.append([x, y])
            angles.append(a)
        return positions, angles
    
    # Try multiple configurations
    configs = []
    configs.append(('lattice', lattice_init()))
    configs.append(('ring', ring_init()))
    for _ in range(5):
        configs.append(('random', random_init()))
    
    for config_name, (positions, angles) in configs:
        if time.time() > deadline:
            break
        
        # Try different outer side lengths
        for outer_side in [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0]:
            if time.time() > deadline:
                break
            
            for outer_angle in [0.0, 30.0, 15.0, 45.0]:
                if time.time() > deadline:
                    break
                
                # Relax
                pos_relaxed, ang_relaxed = relax(positions, angles, (0.0, 0.0), outer_side, outer_angle, max_iters=2000)
                
                # Evaluate
                obj = evaluate_config(pos_relaxed, ang_relaxed, (0.0, 0.0), outer_side, outer_angle)
                if obj < best_side:
                    best_side = obj
                    best_inner = [[p[0], p[1], a] for p, a in zip(pos_relaxed, ang_relaxed)]
                    best_config = (pos_relaxed, ang_relaxed, outer_side, outer_angle)
    
    # If we found a valid configuration, try to minimize outer side
    if best_config is not None:
        pos_best, ang_best, side_best, angle_best = best_config
        
        # Binary search on outer side length
        lo, hi = 4.0, side_best
        for _ in range(30):
            if time.time() > deadline:
                break
            mid = (lo + hi) / 2
            # Try to fit with this outer side
            pos_test, ang_test = relax(pos_best, ang_best, (0.0, 0.0), mid, angle_best, max_iters=3000)
            obj = evaluate_config(pos_test, ang_test, (0.0, 0.0), mid, angle_best)
            if obj == mid:
                hi = mid
                best_side = mid
                best_inner = [[p[0], p[1], a] for p, a in zip(pos_test, ang_test)]
            else:
                lo = mid
    
    # Fallback: return a trivially valid configuration
    if best_inner is None:
        # Place hexagons in a line with large spacing
        best_inner = [[i * 3.0, 0.0, 0.0] for i in range(N)]
        best_side = 3.0 * (N - 1) + 2.0
    
    return best_inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END

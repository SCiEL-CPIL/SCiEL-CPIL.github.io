import numpy as np
from scipy.optimize import minimize
import time
import math
import random

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3)
    
    def hex_vertices(cx, cy, side, angle_deg=0.0):
        """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
        angle = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append([x, y])
        return np.array(verts)
    
    def point_in_hex(px, py, cx, cy, side, angle_deg=0.0):
        """Check if point (px,py) is inside regular hexagon centered at (cx,cy)."""
        angle = math.radians(angle_deg)
        # Transform to hexagon's local frame
        dx = px - cx
        dy = py - cy
        # Rotate back
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        # For a regular hexagon with side s and center at origin, angle 0:
        # Vertices at (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
        # The hexagon is the intersection of 6 half-planes
        # For direction k*pi/3, the support is s*cos(k*pi/3 - angle_to_normal)
        # More directly: for each of the 6 edge normals, check the half-plane constraint
        
        # The 6 edge normals for a hexagon with vertices at angles 0,60,...,300
        # Edge between vertex i and i+1 has normal at angle (i+0.5)*60 degrees
        for i in range(6):
            normal_angle = angle + (i + 0.5) * math.pi / 3
            # The distance from center to this edge (apothem) = side * cos(pi/6) = side * sqrt(3)/2
            apothem = side * math.sqrt(3) / 2
            # Project the point onto the normal direction
            proj = dx * math.cos(normal_angle) + dy * math.sin(normal_angle)
            if proj > apothem + 1e-9:
                return False
        return True
    
    def segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4 (proper intersection only)."""
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        
        if ((d1 > 1e-12 and d2 < -1e-12) or (d1 < -1e-12 and d2 > 1e-12)) and \
           ((d3 > 1e-12 and d4 < -1e-12) or (d3 < -1e-12 and d4 > 1e-12)):
            return True
        return False
    
    def hexagon_overlap(v1, v2, eps=1e-8):
        """Check if two hexagons have overlapping interiors."""
        c1 = (float(np.mean(v1[:, 0])), float(np.mean(v1[:, 1])))
        c2 = (float(np.mean(v2[:, 0])), float(np.mean(v2[:, 1])))
        s1 = float(np.linalg.norm(v1[1] - v1[0]))
        s2 = float(np.linalg.norm(v2[1] - v2[0]))
        # Compute angles
        a1 = math.degrees(math.atan2(v1[0, 1] - c1[1], v1[0, 0] - c1[0]))
        a2 = math.degrees(math.atan2(v2[0, 1] - c2[1], v2[0, 0] - c2[0]))
        
        # Check if any vertex of one is strictly inside the other
        for i in range(6):
            if point_in_hex(v1[i, 0], v1[i, 1], c2[0], c2[1], s2, a2):
                return True
        for i in range(6):
            if point_in_hex(v2[i, 0], v2[i, 1], c1[0], c1[1], s1, a1):
                return True
        # Check edge-edge intersections
        for i in range(6):
            for j in range(6):
                if segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                    return True
        return False
    
    def all_inside_and_disjoint(inner_positions, outer_center, outer_side, outer_angle):
        """Check all inner hexagons are inside outer and mutually disjoint."""
        # Check all inside outer
        for pos in inner_positions:
            x, y, ang = pos
            v = hex_vertices(x, y, 1.0, ang)
            for i in range(6):
                if not point_in_hex(v[i, 0], v[i, 1], outer_center[0], outer_center[1], outer_side, outer_angle):
                    return False
        
        # Check mutual disjointness
        inner_verts = []
        for pos in inner_positions:
            x, y, ang = pos
            v = hex_vertices(x, y, 1.0, ang)
            inner_verts.append(v)
        
        for i in range(len(inner_verts)):
            for j in range(i+1, len(inner_verts)):
                if hexagon_overlap(inner_verts[i], inner_verts[j]):
                    return False
        return True
    
    def min_enclosing_side(positions, outer_angle=0.0):
        """Compute minimum outer hexagon side that contains all inner hexagons."""
        # For each inner hexagon, we need all its vertices inside the outer hexagon
        # The outer hexagon constraint: for each of 6 directions, the projection must be <= apothem
        # Apothem = side * sqrt(3)/2
        # So we need: max over all vertices of all inner hexagons of projection onto each normal <= side * sqrt(3)/2
        # Thus: side >= (2/sqrt(3)) * max_projection
        
        max_proj = 0.0
        for pos in positions:
            x, y, ang = pos
            v = hex_vertices(x, y, 1.0, ang)
            for i in range(6):
                px, py = v[i, 0], v[i, 1]
                # Project onto each of the 6 outer hexagon normals
                for k in range(6):
                    normal_angle = math.radians(outer_angle) + (k + 0.5) * math.pi / 3
                    proj = px * math.cos(normal_angle) + py * math.sin(normal_angle)
                    if proj > max_proj:
                        max_proj = proj
        
        side = max_proj * 2.0 / sqrt3
        return side
    
    def objective(params):
        """Minimize outer side length. params: 3*N values (x, y, angle for each hexagon)."""
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            return 1e10
        
        positions = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            ang = params[3*i+2]
            positions.append([x, y, ang])
        
        # Check disjointness
        inner_verts = []
        for pos in positions:
            x, y, ang = pos
            v = hex_vertices(x, y, 1.0, ang)
            inner_verts.append(v)
        
        for i in range(N):
            for j in range(i+1, N):
                if hexagon_overlap(inner_verts[i], inner_verts[j]):
                    return 1e10
        
        outer_side = min_enclosing_side(positions, 0.0)
        return outer_side
    
    def make_initial_lattice():
        """Create initial hexagonal lattice arrangement."""
        positions = []
        d = sqrt3  # center-to-center for touching same-orientation hexagons
        dy_row = d * sqrt3 / 2  # = 3/2
        
        # 4-3-4 pattern
        for i in range(4):
            positions.append([i * d - 1.5 * d, 0.0, 0.0])
        for i in range(3):
            positions.append([i * d - d, dy_row, 0.0])
        for i in range(4):
            positions.append([i * d - 1.5 * d, 2 * dy_row, 0.0])
        
        return positions
    
    def make_initial_ring():
        """Create initial ring arrangement: 1 center + 6 around + 4 more."""
        positions = []
        positions.append([0.0, 0.0, 0.0])  # center
        # 6 around at distance sqrt(3)
        for i in range(6):
            ang = i * math.pi / 3
            positions.append([sqrt3 * math.cos(ang), sqrt3 * math.sin(ang), 0.0])
        # 4 more in a second ring
        for i in range(4):
            ang = i * math.pi / 2 + math.pi / 4
            r = 2 * sqrt3 * 0.8
            positions.append([r * math.cos(ang), r * math.sin(ang), 0.0])
        return positions
    
    def make_initial_random(seed=42):
        """Create initial random arrangement."""
        rng = random.Random(seed)
        positions = []
        for i in range(N):
            x = rng.uniform(-3, 3)
            y = rng.uniform(-3, 3)
            ang = rng.choice([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
            positions.append([x, y, ang])
        return positions
    
    # Generate initial configurations
    initial_configs = [
        make_initial_lattice(),
        make_initial_ring(),
        make_initial_random(42),
        make_initial_random(123),
        make_initial_random(999),
    ]
    
    best_side = float('inf')
    best_positions = None
    
    for config in initial_configs:
        elapsed = time.time() - start_time
        if elapsed > deadline - 10.0:
            break
        
        # Evaluate initial config
        side = min_enclosing_side(config, 0.0)
        if all_inside_and_disjoint(config, (0.0, 0.0), side, 0.0):
            if side < best_side:
                best_side = side
                best_positions = [list(p) for p in config]
        
        # Try Nelder-Mead optimization
        x0 = []
        for pos in config:
            x0.extend([pos[0], pos[1], pos[2]])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 0.005, 'fatol': 0.005,
                                      'adaptive': True})
            
            if result.fun < 1e9:
                opt_positions = []
                for i in range(N):
                    opt_positions.append([result.x[3*i], result.x[3*i+1], result.x[3*i+2]])
                
                opt_side = min_enclosing_side(opt_positions, 0.0)
                if all_inside_and_disjoint(opt_positions, (0.0, 0.0), opt_side, 0.0):
                    if opt_side < best_side:
                        best_side = opt_side
                        best_positions = [list(p) for p in opt_positions]
        except Exception:
            pass
    
    # Try a second round of optimization from the best found so far
    if best_positions is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 15.0:
            x0 = []
            for pos in best_positions:
                x0.extend([pos[0], pos[1], pos[2]])
            
            try:
                result = minimize(objective, x0, method='Nelder-Mead',
                                 options={'maxiter': 1000, 'xatol': 0.001, 'fatol': 0.001,
                                          'adaptive': True})
                
                if result.fun < 1e9:
                    opt_positions = []
                    for i in range(N):
                        opt_positions.append([result.x[3*i], result.x[3*i+1], result.x[3*i+2]])
                    
                    opt_side = min_enclosing_side(opt_positions, 0.0)
                    if all_inside_and_disjoint(opt_positions, (0.0, 0.0), opt_side, 0.0):
                        if opt_side < best_side:
                            best_side = opt_side
                            best_positions = [list(p) for p in opt_positions]
            except Exception:
                pass
    
    # Fallback if no valid configuration found
    if best_positions is None:
        # Simple line arrangement
        best_positions = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        best_side = 40.0
    
    # Final verification
    final_side = min_enclosing_side(best_positions, 0.0)
    if not all_inside_and_disjoint(best_positions, (0.0, 0.0), final_side, 0.0):
        # Emergency fallback
        best_positions = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        final_side = 40.0
    
    return best_positions, (0.0, 0.0), final_side, 0.0
# EVOLVE-BLOCK-END

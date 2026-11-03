import math
import time
import random
import numpy as np

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Helper: Get vertices of a regular hexagon
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    # Helper: Check if two convex polygons are disjoint using SAT
    def sat_disjoint(v1, v2, eps=1e-5):
        # v1, v2 are lists of (x,y)
        axes = []
        for verts in [v1, v2]:
            for i in range(len(verts)):
                x1, y1 = verts[i]
                x2, y2 = verts[(i+1)%len(verts)]
                dx = x2 - x1
                dy = y2 - y1
                length = math.sqrt(dx*dx + dy*dy)
                if length < 1e-10: continue
                # Normal
                nx = -dy / length
                ny = dx / length
                axes.append((nx, ny))
        
        for nx, ny in axes:
            min1, max1 = float('inf'), float('-inf')
            min2, max2 = float('inf'), float('-inf')
            for px, py in v1:
                proj = px * nx + py * ny
                if proj < min1: min1 = proj
                if proj > max1: max1 = proj
            for px, py in v2:
                proj = px * nx + py * ny
                if proj < min2: min2 = proj
                if proj > max2: max2 = proj
            
            # If there is a gap, they are disjoint
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        return False

    def hexagons_disjoint(inner, eps=1e-5):
        n = len(inner)
        for i in range(n):
            v1 = get_hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            for j in range(i+1, n):
                v2 = get_hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                if not sat_disjoint(v1, v2, eps):
                    return False
        return True

    def compute_min_outer_side(inner, cx, cy, theta_deg):
        """
        Compute the minimum side length of a regular hexagon with center (cx, cy)
        and rotation theta_deg that contains all inner hexagons.
        """
        theta = math.radians(theta_deg)
        # Normals of the outer hexagon edges
        # For a regular hexagon, the distance from center to edge is R * cos(pi/6) = R * sqrt(3)/2
        # Side length S = 2 * R / sqrt(3)
        # So R = S * sqrt(3) / 2
        # Distance from center to boundary along normal n is R * cos(0) ? No.
        # The support function of a regular hexagon with side S, rotated by theta, 
        # in direction (nx, ny) is h(n) = R * max_k (cos(theta + k*pi/3 - angle_n)) * (S*sqrt(3)/2 / R) ?
        
        # Let's use the projection method.
        # The outer hexagon is defined by 6 inequalities: n_k . (x - c) <= h_k
        # where n_k are normals to the sides, and h_k is the apothem (distance from center to side).
        # Apothem A = side * sqrt(3) / 2.
        # So side = 2 * A / sqrt(3).
        
        # We need to find the minimum A such that for all inner vertices p,
        # max_k ( n_k . (p - c) ) <= A
        
        max_proj = 0.0
        for k in range(6):
            # Angle of normal k
            # A regular hexagon has normals at angles theta, theta + 60, ...
            # Wait, if vertices are at angle theta, normals are at theta + 30?
            # Let's define vertices of outer hexagon at angles theta + i*pi/3.
            # Then the edge between vertex i and i+1 has a normal at angle theta + (i+0.5)*pi/3.
            normal_angle = theta + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            for x, y, a in inner:
                verts = get_hex_vertices(x, y, 1.0, a)
                for px, py in verts:
                    proj = (px - cx) * nx + (py - cy) * ny
                    if proj > max_proj:
                        max_proj = proj
        
        # max_proj is the required apothem A
        return 2.0 * max_proj / math.sqrt(3)

    def objective(inner, cx, cy, theta_deg):
        """
        Objective function for optimization.
        Returns side length + penalty for overlaps.
        """
        side = compute_min_outer_side(inner, cx, cy, theta_deg)
        
        # Calculate overlap penalty
        penalty = 0.0
        n = len(inner)
        # We can speed this up by only checking close pairs, but N=12 is small.
        # However, SAT is expensive. Let's use a simpler distance check first?
        # No, SAT is robust. With N=12, pairs = 66.
        # To make it differentiable/smooth-ish for SA, we can't easily get penalty from SAT.
        # We will use a heuristic penalty: sum of squared distances < min_dist?
        # Min distance between centers for non-overlapping unit hexagons is sqrt(3) ~ 1.732.
        # But they can overlap if rotated? No, convex bodies.
        # If centers are closer than sqrt(3), they MUST overlap?
        # Yes, for regular hexagons of side 1, the minimum distance between centers 
        # such that they can be disjoint is sqrt(3). 
        # Wait, if one is rotated, the "radius" changes?
        # The circumradius is 1. The inradius is sqrt(3)/2 ~ 0.866.
        # If centers are distance d apart, they are disjoint if d >= r1 + r2? No, that's for circles.
        # For hexagons, the minimum distance between centers for any orientation is sqrt(3) (touching at edges).
        # The maximum distance for *guaranteed* overlap is ... well, if d < sqrt(3), do they always overlap?
        # Yes. The "width" of a unit hexagon in any direction is between sqrt(3) and 2.
        # Actually, the minimum width is sqrt(3).
        # So if the distance between centers is less than sqrt(3), the projections on the line connecting them 
        # might not overlap? 
        # Let's stick to SAT for validity, but use a soft penalty for the optimization loop to guide the search.
        # Soft penalty: sum over pairs max(0, min_dist - dist)^2
        # This is a standard trick. It doesn't guarantee non-overlap, but it pushes them apart.
        
        for i in range(n):
            x1, y1, _ = inner[i]
            for j in range(i+1, n):
                x2, y2, _ = inner[j]
                dx = x1 - x2
                dy = y1 - y2
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = math.sqrt(3.0)
                if dist < min_dist:
                    # Penalty increases as they get closer
                    # This acts as a repulsive force
                    penalty += (min_dist - dist)**2
        
        return side + 10.0 * penalty

    # Initialize with 3-2-2-2-3 topology
    # Spacing in x: sqrt(3) + eps
    # Spacing in y: 1.5 + eps (for offset rows)
    # Let's use exact touching distances initially and let optimizer push them apart
    # or start slightly apart.
    
    eps = 0.01
    dx = math.sqrt(3.0) + eps
    dy = 1.5 + eps
    
    # Column 0: 3 hexagons at y = -1.5, 0, 1.5
    # Column 1: 2 hexagons at y = -0.75, 0.75
    # Column 2: 2 hexagons at y = -0.75, 0.75
    # Column 3: 3 hexagons at y = -1.5, 0, 1.5
    
    # Note: Hexagons are "pointy-top" (vertex at top) if angle=0?
    # get_hex_vertices uses theta = angle + i*pi/3.
    # If angle=0, vertices are at 0, 60, 120... degrees.
    # This is "pointy-top" relative to x-axis? No, vertex at (1,0).
    # Flat-top would be angle=30.
    # Let's use angle=0 for all initially.
    
    centers = []
    # Col 0 (x = -1.5*dx)
    centers.extend([(-1.5*dx, -1.5), (-1.5*dx, 0.0), (-1.5*dx, 1.5)])
    # Col 1 (x = -0.5*dx)
    centers.extend([(-0.5*dx, -0.75), (-0.5*dx, 0.75)])
    # Col 2 (x = 0.5*dx)
    centers.extend([(0.5*dx, -0.75), (0.5*dx, 0.75)])
    # Col 3 (x = 1.5*dx)
    centers.extend([(1.5*dx, -1.5), (1.5*dx, 0.0), (1.5*dx, 1.5)])
    
    # Center the configuration roughly
    avg_x = sum(c[0] for c in centers) / len(centers)
    avg_y = sum(c[1] for c in centers) / len(centers)
    
    inner = []
    for x, y in centers:
        inner.append([x - avg_x, y - avg_y, 0.0])
    
    # Initial outer hexagon
    cx, cy = 0.0, 0.0
    theta_deg = 0.0
    
    # Calculate initial side
    best_side = compute_min_outer_side(inner, cx, cy, theta_deg)
    best_inner = [list(h) for h in inner]
    best_cx, best_cy, best_theta = cx, cy, theta_deg
    
    # Simulated Annealing
    T = 0.1
    T_min = 0.0001
    cooling_rate = 0.995
    
    current_inner = [list(h) for h in inner]
    current_cx, current_cy, current_theta = cx, cy, theta_deg
    current_obj = objective(current_inner, current_cx, current_cy, current_theta)
    
    step_size = 0.1
    
    while T > T_min and time.time() < deadline - 10.0:
        # Perturb
        new_inner = [list(h) for h in current_inner]
        new_cx, new_cy, new_theta = current_cx, current_cy, current_theta
        
        # Randomly choose what to perturb
        r = random.random()
        if r < 0.6:
            # Move one hexagon
            idx = random.randint(0, N-1)
            dx = random.gauss(0, step_size)
            dy = random.gauss(0, step_size)
            new_inner[idx][0] += dx
            new_inner[idx][1] += dy
        elif r < 0.7:
            # Rotate one hexagon
            idx = random.randint(0, N-1)
            da = random.gauss(0, step_size * 10)
            new_inner[idx][2] += da
        elif r < 0.8:
            # Move outer center
            dx = random.gauss(0, step_size * 0.5)
            dy = random.gauss(0, step_size * 0.5)
            new_cx += dx
            new_cy += dy
        elif r < 0.9:
            # Rotate outer hexagon
            da = random.gauss(0, step_size * 5)
            new_theta += da
        else:
            # Move two hexagons (for better topology changes)
            idx1 = random.randint(0, N-1)
            idx2 = random.randint(0, N-1)
            if idx1 != idx2:
                new_inner[idx1][0] += random.gauss(0, step_size)
                new_inner[idx1][1] += random.gauss(0, step_size)
                new_inner[idx2][0] += random.gauss(0, step_size)
                new_inner[idx2][1] += random.gauss(0, step_size)
        
        new_obj = objective(new_inner, new_cx, new_cy, new_theta)
        
        # Acceptance
        delta = new_obj - current_obj
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_inner = new_inner
            current_cx, current_cy, current_theta = new_cx, new_cy, new_theta
            current_obj = new_obj
            
            # Check if this is a valid configuration (no overlaps)
            # We only update best if it's strictly valid.
            # To save time, we check validity less frequently or only if obj is promising.
            # But we need to ensure we return a valid result.
            # Let's check validity every few steps or if obj is lower than best_side + small margin.
            if current_obj < best_side + 0.1:
                if hexagons_disjoint(current_inner):
                    side_val = compute_min_outer_side(current_inner, current_cx, current_cy, current_theta)
                    if side_val < best_side:
                        best_side = side_val
                        best_inner = [list(h) for h in current_inner]
                        best_cx, best_cy, best_theta = current_cx, current_cy, current_theta
        
        T *= cooling_rate
        # Adapt step size
        if T < 0.01:
            step_size *= 0.98
    
    # Final Polish: Try to tighten the fit
    # Use a steeper gradient descent / local search with very small steps
    # And strictly enforce disjointness.
    
    final_inner = [list(h) for h in best_inner]
    final_cx, final_cy, final_theta = best_cx, best_cy, best_theta
    final_side = best_side
    
    # Local optimization with strict constraints
    step_size = 0.01
    for _ in range(500):
        if time.time() > deadline - 5.0: break
        
        improved = False
        for _ in range(10):
            new_inner = [list(h) for h in final_inner]
            new_cx, new_cy, new_theta = final_cx, final_cy, final_theta
            
            r = random.random()
            if r < 0.7:
                idx = random.randint(0, N-1)
                new_inner[idx][0] += random.uniform(-step_size, step_size)
                new_inner[idx][1] += random.uniform(-step_size, step_size)
            elif r < 0.85:
                new_cx += random.uniform(-step_size*0.5, step_size*0.5)
                new_cy += random.uniform(-step_size*0.5, step_size*0.5)
            else:
                new_theta += random.uniform(-step_size*5, step_size*5)
            
            # Strict check
            if not hexagons_disjoint(new_inner):
                continue
            
            new_side = compute_min_outer_side(new_inner, new_cx, new_cy, new_theta)
            
            if new_side < final_side - 1e-8:
                final_inner = new_inner
                final_cx, final_cy, final_theta = new_cx, new_cy, new_theta
                final_side = new_side
                improved = True
                break
        
        if not improved:
            step_size *= 0.9
            if step_size < 1e-4:
                break
    
    # Final validation
    if not hexagons_disjoint(final_inner):
        # Fallback to the best valid one found earlier
        final_inner = best_inner
        final_cx, final_cy, final_theta = best_cx, best_cy, best_theta
        final_side = best_side
    else:
        # Recalculate exact side
        final_side = compute_min_outer_side(final_inner, final_cx, final_cy, final_theta)

    return final_inner, (final_cx, final_cy), final_side, final_theta


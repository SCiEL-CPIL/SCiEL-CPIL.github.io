"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from itertools import combinations

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    sqrt3 = math.sqrt(3.0)
    
    # Normal directions for flat-top outer hexagon
    normal_angles = np.array([math.radians(30 + 60 * j) for j in range(6)])
    normal_cos = np.cos(normal_angles)
    normal_sin = np.sin(normal_angles)
    
    def min_L_for_hexagon(cx, cy, theta):
        vertex_angles = np.array([math.radians(60 * k) + theta for k in range(6)])
        vx = cx + np.cos(vertex_angles)
        vy = cy + np.sin(vertex_angles)
        
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)
        
    def check_disjoint(centers):
        # centers is list of (x,y)
        # min distance sqrt(3)
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                if dx*dx + dy*dy < 3.0 - 1e-9:
                    return False
        return True

    # Generate fine grid
    # Hex grid with step 0.25
    # Axial coords q, r. x = 1.5*q*step, y = sqrt(3)*(r + q/2)*step
    step = 0.25
    max_r = 18 # 18 * 0.25 = 4.5
    
    grid_points = []
    for q in range(-max_r, max_r + 1):
        for r in range(-max_r, max_r + 1):
            if abs(q + r) <= max_r:
                x = 1.5 * q * step
                y = sqrt3 * (r + q / 2.0) * step
                grid_points.append((x, y))
                
    # Map grid points to indices for fast lookup
    # We'll just use the list and check distances.
    
    # Initial configuration: 12 centers from coarse grid (radius 2, step 1)
    # Coarse grid points:
    coarse_points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                coarse_points.append((x, y))
                
    # Select 12 points from coarse grid that are disjoint and have min max L
    # We know the 12 outer+middle points are disjoint.
    # Let's just pick the 12 with largest distance from center?
    # Or simply use the 12 from the previous best subset if we can find it.
    # For now, pick the 12 coarse points with largest norm.
    coarse_points_sorted = sorted(coarse_points, key=lambda p: p[0]*p[0] + p[1]*p[1], reverse=True)
    initial_centers = coarse_points_sorted[:12]
    
    # Snap to fine grid
    centers = []
    for px, py in initial_centers:
        # Find nearest fine grid point
        best_d = float('inf')
        best_p = None
        for gx, gy in grid_points:
            d = (gx-px)**2 + (gy-py)**2
            if d < best_d:
                best_d = d
                best_p = (gx, gy)
        centers.append(best_p)
        
    # Ensure disjointness. If not, try to find a valid set.
    # The snapped points might overlap if the coarse points were close.
    # Coarse points are dist >= sqrt(3). Fine grid step 0.25.
    # Snapping error <= 0.125.
    # So distance can decrease by 0.25.
    # Min coarse dist 1.732. 1.732 - 0.25 = 1.482 < 1.732.
    # So they might overlap.
    
    # If not disjoint, try to fix by moving points to neighbors.
    # This is complex. Let's just try to find a valid independent set on the fine grid
    # that is close to the coarse configuration.
    
    # Simpler: Start with the 12 coarse points.
    # If they are disjoint (they are), use them.
    # But we want to use the fine grid for optimization.
    # So we start with the coarse points (which are on the fine grid? No, step 1 vs 0.25. Yes, 1 is multiple of 0.25).
    # So the coarse points ARE on the fine grid.
    # So initial centers are valid.
    
    centers = initial_centers
    angles = [0.0] * 12 # Flat-top
    
    current_L = max(min_L_for_hexagon(centers[i][0], centers[i][1], angles[i]) for i in range(12))
    
    best_L = current_L
    best_centers = [c[:] for c in centers]
    best_angles = angles[:]
    
    # Local Search
    # 1. Position optimization
    # For each center, try moving to 6 neighbors on fine grid.
    
    # Precompute neighbors for each point in grid_points?
    # Or just generate neighbors on the fly.
    # Neighbor of (q,r) in axial: (q+1,r), (q-1,r), (q,r+1), (q,r-1), (q+1,r-1), (q-1,r+1)
    # We need to map (x,y) to (q,r) or just use x,y offsets.
    # Offsets for step s:
    # (1.5s, 0), (-1.5s, 0), (0.75s, sqrt(3)s), (-0.75s, -sqrt(3)s), (0.75s, -sqrt(3)s), (-0.75s, sqrt(3)s)
    
    offsets = [
        (1.5 * step, 0),
        (-1.5 * step, 0),
        (0.75 * step, sqrt3 * step),
        (-0.75 * step, -sqrt3 * step),
        (0.75 * step, -sqrt3 * step),
        (-0.75 * step, sqrt3 * step)
    ]
    
    while time.time() < deadline - 5.0:
        improved = False
        
        # Try position moves
        for i in range(12):
            cx, cy = centers[i]
            for ox, oy in offsets:
                nx, ny = cx + ox, cy + oy
                # Check if new position is disjoint from others
                valid = True
                for j in range(12):
                    if i == j: continue
                    dx = nx - centers[j][0]
                    dy = ny - centers[j][1]
                    if dx*dx + dy*dy < 3.0 - 1e-9:
                        valid = False
                        break
                if not valid: continue
                
                # Check if new position is within bounds?
                # We should stay within the outer hexagon?
                # The outer hexagon size is determined by the positions.
                # If we move a point out, L increases.
                # We want to minimize L.
                
                # Calculate new L
                new_L = max(min_L_for_hexagon(centers[k][0], centers[k][1], angles[k]) for k in range(12) if k != i)
                L_i_new = min_L_for_hexagon(nx, ny, angles[i])
                total_L = max(new_L, L_i_new)
                
                if total_L < current_L - 1e-6:
                    centers[i] = (nx, ny)
                    current_L = total_L
                    improved = True
                    
                    if current_L < best_L:
                        best_L = current_L
                        best_centers = [c[:] for c in centers]
                        best_angles = angles[:]
                    break # Restart loop for this i? Or just continue?
                    # Let's continue to try other offsets if possible, but usually one move is enough.
                    # Actually, if we moved, the distances to others changed.
                    # So we should restart the check for this i or just move to next i.
                    # Moving to next i is fine.
        
        # Try rotation moves
        for i in range(12):
            cx, cy = centers[i]
            # Try 12 angles
            for k in range(12):
                new_angle = k * math.pi / 6.0 # 30 deg steps?
                # Or 15 deg steps? 24 steps.
                # Let's try 15 deg steps.
                # Actually, the optimal angle is likely 0 or 30.
                # Let's try 0, 15, 30, 45, 60.
                pass
            
            # Let's just try 0 and 30 for now, then refine.
            # If current angle is 0, try 30. If 30, try 0.
            # And maybe 15, 45.
            
            candidates = [0.0, math.radians(15), math.radians(30), math.radians(45)]
            for new_angle in candidates:
                if abs(new_angle - angles[i]) < 1e-6: continue
                
                # Disjointness is not affected by rotation.
                
                new_L = max(min_L_for_hexagon(centers[k][0], centers[k][1], angles[k]) for k in range(12) if k != i)
                L_i_new = min_L_for_hexagon(cx, cy, new_angle)
                total_L = max(new_L, L_i_new)
                
                if total_L < current_L - 1e-6:
                    angles[i] = new_angle
                    current_L = total_L
                    improved = True
                    if current_L < best_L:
                        best_L = current_L
                        best_centers = [c[:] for c in centers]
                        best_angles = angles[:]
                    break
        
        if not improved:
            # Try finer rotation steps
            # Or break
            # Let's try finer rotation steps for all
            fine_improved = False
            for i in range(12):
                cx, cy = centers[i]
                current_ang = angles[i]
                for delta_deg in [-5, 5, -1, 1]:
                    new_angle = current_ang + math.radians(delta_deg)
                    new_L = max(min_L_for_hexagon(centers[k][0], centers[k][1], angles[k]) for k in range(12) if k != i)
                    L_i_new = min_L_for_hexagon(cx, cy, new_angle)
                    total_L = max(new_L, L_i_new)
                    if total_L < current_L - 1e-6:
                        angles[i] = new_angle
                        current_L = total_L
                        fine_improved = True
                        if current_L < best_L:
                            best_L = current_L
                            best_centers = [c[:] for c in centers]
                            best_angles = angles[:]
                        break
            if not fine_improved:
                break

    # Final polish
    # Use the best found
    centers = best_centers
    angles = best_angles
    
    inner = []
    for i in range(12):
        inner.append([centers[i][0], centers[i][1], math.degrees(angles[i])])
        
    final_L = max(min_L_for_hexagon(centers[i][0], centers[i][1], angles[i]) for i in range(12))
    
    return inner, (0.0, 0.0), final_L, 0.0


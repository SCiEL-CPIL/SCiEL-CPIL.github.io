import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _min_enclosing_hexagon(inner_hexes):
    points = []
    for h in inner_hexes:
        verts = _hex_vertices(h[0], h[1], 1.0, h[2])
        points.extend(verts)
    
    points = np.array(points)
    if len(points) == 0:
        return 0, 0, 1.0, 0.0

    best_side = float('inf')
    best_params = (0, 0, 1.0, 0.0)
    
    # Scan outer orientation
    # The optimal orientation of the container for a set of points is usually
    # aligned with one of the "extreme" directions of the convex hull.
    # Scanning 0-60 with 0.5 deg steps is robust.
    
    # Optimization: We can restrict the search space for the container center (cx, cy)
    # and side (s) using linear constraints for each angle.
    
    # To save time, we first do a coarse scan, then a fine scan around the best.
    
    # Coarse scan
    for angle_deg in np.arange(0, 60, 0.5):
        a = math.radians(angle_deg)
        normals = []
        for k in range(3):
            theta_n = a + math.pi/6 + k * math.pi/3
            normals.append((math.cos(theta_n), math.sin(theta_n)))
        
        A = []
        b = []
        
        for nx, ny in normals:
            for px, py in points:
                A.append([nx, ny, -math.sqrt(3)/2])
                b.append(nx*px + ny*py)
                A.append([-nx, -ny, -math.sqrt(3)/2])
                b.append(-nx*px - ny*py)
        
        bounds = [(-100, 100), (-100, 100), (0, None)]
        
        try:
            res = linprog(c=[0, 0, 1], A_ub=A, b_ub=b, bounds=bounds, method='highs')
            if res.success:
                s_val = res.x[2]
                if s_val < best_side:
                    best_side = s_val
                    best_params = (res.x[0], res.x[1], s_val, angle_deg)
        except Exception:
            pass
            
    # Fine scan around best
    if best_params[3] > 0.5 and best_params[3] < 59.5:
        best_ang = best_params[3]
        for angle_deg in np.arange(best_ang - 0.5, best_ang + 0.5, 0.05):
            if angle_deg < 0: continue
            a = math.radians(angle_deg)
            normals = []
            for k in range(3):
                theta_n = a + math.pi/6 + k * math.pi/3
                normals.append((math.cos(theta_n), math.sin(theta_n)))
                
            A = []
            b = []
            for nx, ny in normals:
                for px, py in points:
                    A.append([nx, ny, -math.sqrt(3)/2])
                    b.append(nx*px + ny*py)
                    A.append([-nx, -ny, -math.sqrt(3)/2])
                    b.append(-nx*px - ny*py)
                    
            bounds = [(-100, 100), (-100, 100), (0, None)]
            try:
                res = linprog(c=[0, 0, 1], A_ub=A, b_ub=b, bounds=bounds, method='highs')
                if res.success:
                    s_val = res.x[2]
                    if s_val < best_side:
                        best_side = s_val
                        best_params = (res.x[0], res.x[1], s_val, angle_deg)
            except Exception:
                pass
                
    return best_params

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Strategy:
    # 1. Assume all inner hexagons are aligned (angle 0). This is WLOG for the set of disjoint
    #    unit hexagons if we allow translation, because any rotated hexagon can be "snapped"
    #    to a lattice position without increasing the bounding box significantly, and often
    #    decreasing it. More importantly, it reduces the search dimension from 3*N to 2*N.
    # 2. The optimal packing of regular hexagons is often related to the hexagonal lattice.
    #    We search for positions on a fine grid.
    
    # Initial guess: A compact cluster.
    # Hex numbers: 1, 7, 19. 11 is close to 7.
    # Let's start with a 7-hex cluster (center + 6 neighbors) and add 4 more.
    
    # Grid parameters
    # The spacing between centers of touching hexagons is sqrt(3) * side = 1.732.
    # We will search on a grid with step 0.05.
    
    # To make the search tractable, we will use a greedy constructive approach
    # followed by a local search (simulated annealing or hill climbing) on the grid.
    
    # Actually, a simple high-resolution grid search over a small region for each hexagon
    # sequentially (greedy) might get stuck in local minima.
    
    # Better: 
    # 1. Generate a set of candidate "patterns" (relative positions) for 11 hexagons.
    # 2. For each pattern, calculate the min enclosing hexagon.
    # 3. Keep the best.
    
    # What patterns?
    # The convex hull of the centers should be compact.
    # We can use a "jittered" hexagonal lattice.
    
    # Let's try a specific heuristic: 
    # Place 1 in center.
    # Place 6 around it (dist sqrt(3)).
    # Place 4 more in the "gaps" or extending the cluster.
    
    # Since N=11 is small, we can do a multi-start local search.
    
    best_side = float('inf')
    best_inner = []
    
    # Define a function to evaluate a configuration
    def evaluate_config(centers):
        # centers: list of (x, y)
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                # Min distance between centers of disjoint unit hexagons is sqrt(3) * 1 * sqrt(3) ?
                # No. Distance between centers of touching hexagons is sqrt(3).
                # If dist < sqrt(3) - epsilon, they overlap.
                # Actually, the minimum distance between centers of two disjoint unit hexagons
                # is sqrt(3) if they are aligned. If rotated, it can be smaller?
                # No, the "width" of a hexagon is 2*sqrt(3)/2 * 1 * 2? 
                # Width = 2 * apothem = sqrt(3).
                # So if centers are closer than sqrt(3), they MUST overlap?
                # Yes, for aligned hexagons. For rotated, the minimal distance is still related to apothem.
                # The minimum distance between centers of two disjoint convex bodies is the sum of their support functions in the direction of the connecting line.
                # For regular hexagons, the min center distance for disjointness is sqrt(3) (when faces touch).
                # If dist < sqrt(3) - 1e-6, overlap.
                if dx*dx + dy*dy < (math.sqrt(3) - 0.001)**2:
                    return float('inf')
        
        inner_hexes = [(c[0], c[1], 0.0) for c in centers]
        _, _, side, _ = _min_enclosing_hexagon(inner_hexes)
        return side

    # Multi-start search
    # Start 1: Hexagonal lattice subset
    # Start 2: Random jittered lattice
    # Start 3: Random
    
    num_starts = 50
    steps_per_start = 100 # Local search steps
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        # Generate initial configuration
        centers = []
        if start_idx == 0:
            # Perfect hex lattice 7 + 4
            # 0,0
            centers.append((0,0))
            # 6 neighbors
            for i in range(6):
                angle = math.pi/3 * i
                centers.append((math.sqrt(3)*math.cos(angle), math.sqrt(3)*math.sin(angle)))
            # 4 more: place them at distance 2*sqrt(3) in 4 directions?
            # Or fill in the next ring?
            # Next ring has 12 spots. We pick 4.
            # Let's pick spots that are "closest" to the existing cluster to keep it compact.
            # Actually, just append 4 random valid spots from the next ring.
            # Next ring centers: distance 2*sqrt(3) * cos/sin? No.
            # Lattice vectors: v1=(sqrt(3), 0), v2=(sqrt(3)/2, 1.5)
            # Let's just use a greedy placement for the first few starts.
            
            # Greedy placement for initial guess
            placed = [(0,0)]
            for k in range(1, N):
                best_pos = None
                best_cl = -1
                # Search in a box around current cluster
                max_coord = 5
                for x in np.arange(-max_coord, max_coord, 0.1):
                    for y in np.arange(-max_coord, max_coord, 0.1):
                        min_cl = float('inf')
                        valid = True
                        for p in placed:
                            dx = x - p[0]
                            dy = y - p[1]
                            d = math.sqrt(dx*dx + dy*dy)
                            if d < math.sqrt(3) - 0.01:
                                valid = False
                                break
                            min_cl = min(min_cl, d)
                        if valid and min_cl > best_cl:
                            best_cl = min_cl
                            best_pos = (x, y)
                if best_pos:
                    placed.append(best_pos)
                else:
                    placed.append((0,0))
            centers = placed
        else:
            # Random jittered lattice
            # Start with a base lattice and jitter
            base_centers = []
            # Generate a random compact set of N points
            # Use a random walk or just random points in a small box and reject overlaps
            # To ensure compactness, generate in a box of size ~4x4
            for _ in range(1000):
                cand = [(np.random.uniform(-2, 2), np.random.uniform(-2, 2)) for _ in range(N)]
                # Check overlaps
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        dx = cand[i][0] - cand[j][0]
                        dy = cand[i][1] - cand[j][1]
                        if dx*dx + dy*dy < (math.sqrt(3) - 0.01)**2:
                            valid = False
                            break
                    if not valid: break
                if valid:
                    centers = cand
                    break
            else:
                # Fallback to greedy
                placed = [(0,0)]
                for k in range(1, N):
                    best_pos = None
                    best_cl = -1
                    for x in np.arange(-5, 5, 0.2):
                        for y in np.arange(-5, 5, 0.2):
                            min_cl = float('inf')
                            valid = True
                            for p in placed:
                                dx = x - p[0]
                                dy = y - p[1]
                                d = math.sqrt(dx*dx + dy*dy)
                                if d < math.sqrt(3) - 0.01:
                                    valid = False
                                    break
                                min_cl = min(min_cl, d)
                            if valid and min_cl > best_cl:
                                best_cl = min_cl
                                best_pos = (x, y)
                    if best_pos:
                        placed.append(best_pos)
                    else:
                        placed.append((0,0))
                centers = placed

        # Local Search (Hill Climbing)
        # Move one hexagon at a time to a nearby grid point if it improves the score.
        current_side = evaluate_config(centers)
        
        for step in range(steps_per_start):
            if time.time() > deadline - 5:
                break
                
            improved = False
            # Pick a random hexagon to move
            idx = np.random.randint(N)
            old_x, old_y = centers[idx]
            
            # Try moving to a nearby grid point
            # Grid step 0.1
            found_better = False
            best_new_side = current_side
            best_new_pos = (old_x, old_y)
            
            # Sample neighbors
            # Instead of full grid, sample random directions
            for _ in range(20):
                angle = np.random.uniform(0, 2*math.pi)
                dist = np.random.uniform(0.05, 0.5)
                nx = old_x + dist * math.cos(angle)
                ny = old_y + dist * math.sin(angle)
                
                # Snap to grid? No, continuous is better if we have time, but grid is safer for disjointness check speed.
                # Let's just check validity and score.
                new_centers = centers.copy()
                new_centers[idx] = (nx, ny)
                
                # Quick disjointness check
                valid = True
                for j in range(N):
                    if j == idx: continue
                    dx = nx - new_centers[j][0]
                    dy = ny - new_centers[j][1]
                    if dx*dx + dy*dy < (math.sqrt(3) - 0.005)**2:
                        valid = False
                        break
                if not valid:
                    continue
                
                side = evaluate_config(new_centers)
                if side < best_new_side:
                    best_new_side = side
                    best_new_pos = (nx, ny)
                    found_better = True
            
            if found_better:
                centers[idx] = best_new_pos
                current_side = best_new_side
                improved = True
                
                if current_side < best_side:
                    best_side = current_side
                    best_inner = centers.copy()
            
            if not improved and step > 20:
                # Stagnation, break early?
                # Let's allow a few more steps to escape local minima?
                # For now, break to save time for other starts.
                break

    # Final refinement of the best solution found
    if best_inner:
        # Do a finer local search on the best solution
        current_side = best_side
        centers = best_inner
        for step in range(200):
            if time.time() > deadline - 2:
                break
            improved = False
            idx = np.random.randint(N)
            old_x, old_y = centers[idx]
            
            best_new_side = current_side
            best_new_pos = (old_x, old_y)
            
            # Finer steps
            for _ in range(50):
                angle = np.random.uniform(0, 2*math.pi)
                dist = np.random.uniform(0.01, 0.2)
                nx = old_x + dist * math.cos(angle)
                ny = old_y + dist * math.sin(angle)
                
                new_centers = centers.copy()
                new_centers[idx] = (nx, ny)
                
                valid = True
                for j in range(N):
                    if j == idx: continue
                    dx = nx - new_centers[j][0]
                    dy = ny - new_centers[j][1]
                    if dx*dx + dy*dy < (math.sqrt(3) - 0.001)**2:
                        valid = False
                        break
                if not valid:
                    continue
                
                side = evaluate_config(new_centers)
                if side < best_new_side:
                    best_new_side = side
                    best_new_pos = (nx, ny)
                    improved = True
            
            if improved:
                centers[idx] = best_new_pos
                current_side = best_new_side
                if current_side < best_side:
                    best_side = current_side
                    best_inner = centers.copy()
            else:
                break

    # Construct final result
    if best_inner:
        inner_hexes = [(c[0], c[1], 0.0) for c in best_inner]
    else:
        # Fallback: Greedy placement
        placed = [(0.0, 0.0, 0.0)]
        for k in range(1, N):
            best_pos = None
            best_cl = -1
            for x in np.arange(-5, 5, 0.2):
                for y in np.arange(-5, 5, 0.2):
                    min_cl = float('inf')
                    valid = True
                    for p in placed:
                        dx = x - p[0]
                        dy = y - p[1]
                        d = math.sqrt(dx*dx + dy*dy)
                        if d < math.sqrt(3) - 0.01:
                            valid = False
                            break
                        min_cl = min(min_cl, d)
                    if valid and min_cl > best_cl:
                        best_cl = min_cl
                        best_pos = (x, y, 0.0)
            if best_pos:
                placed.append(best_pos)
            else:
                placed.append((0,0,0))
        while len(placed) < N:
            placed.append((0,0,0))
        inner_hexes = placed

    ocx, ocy, os, oang = _min_enclosing_hexagon(inner_hexes)
    
    # Convert to required format
    inner_list = [[h[0], h[1], h[2]] for h in inner_hexes]
    
    return inner_list, (ocx, ocy), os, oang

if __name__ == "__main__":
    r = construct()
    print(r)

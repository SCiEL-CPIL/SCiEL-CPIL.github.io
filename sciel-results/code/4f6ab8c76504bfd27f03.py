import numpy as np
import time
import random

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    def compute_objective(pts, radii):
        """Compute sum of radii for a valid packing, or -1 if invalid."""
        # Check boundaries
        for i in range(N):
            x, y = pts[i]
            r = radii[i]
            if r > x + 1e-9 or r > 1 - x + 1e-9 or r > y + 1e-9 or r > 1 - y + 1e-9:
                return -1.0
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((pts[i][0] - pts[j][0])**2 + (pts[i][1] - pts[j][1])**2)
                if radii[i] + radii[j] > d + 1e-9:
                    return -1.0
        return np.sum(radii)
    
    def get_max_feasible_radius(pts, i, eps=1e-9):
        """Given fixed positions of all circles, find max radius for circle i that maintains feasibility."""
        # Boundary constraint
        r_max = min(pts[i][0], 1 - pts[i][0], pts[i][1], 1 - pts[i][1]) - eps
        # Overlap constraints with each other circle j
        for j in range(N):
            if j == i:
                continue
            d = np.sqrt((pts[i][0] - pts[j][0])**2 + (pts[i][1] - pts[j][1])**2)
            r_max = min(r_max, d - radii[j] - eps)
        return max(0, r_max)
    
    def grow_radii(pts, radii):
        """Grow each circle's radius to its maximum feasible value, iteratively."""
        improved = True
        while improved:
            improved = False
            for i in range(N):
                new_r = get_max_feasible_radius(pts, i)
                if new_r > radii[i] + 1e-10:
                    radii[i] = new_r
                    improved = True
        return radii
    
    def push_algorithm(pts, radii, time_limit):
        """
        Hard-constrained coordinate descent:
        1. Grow radii to max feasible.
        2. Pick random circle, try to increase its radius slightly.
        3. If infeasible, push overlapping circles away.
        4. Repeat until time limit.
        """
        best_sum = compute_objective(pts, radii)
        if best_sum < 0:
            # Make initial feasible
            for i in range(N):
                radii[i] = 0.01
            # Then grow
            radii = grow_radii(pts, radii.copy())
            best_sum = compute_objective(pts, radii)
            if best_sum < 0:
                return pts, radii, 0.0
        
        best_pts = pts.copy()
        best_radii = radii.copy()
        best_val = best_sum
        
        # Precompute pairs for speed
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        
        iteration = 0
        while time.time() < time_limit:
            iteration += 1
            if time.time() > time_limit:
                break
            
            # Randomly pick a circle to try to grow
            i = random.randint(0, N - 1)
            current_r = radii[i]
            
            # Try to increase radius by a small amount
            grow_amount = max(0.0005, current_r * 0.02)
            new_r = current_r + grow_amount
            
            # Check if this new radius is feasible with current positions
            feasible = True
            for j in range(N):
                if j == i:
                    continue
                d = np.sqrt((pts[i][0] - pts[j][0])**2 + (pts[i][1] - pts[j][1])**2)
                if new_r + radii[j] > d + 1e-9:
                    feasible = False
                    break
            # Boundary check
            if (new_r > pts[i][0] + 1e-9 or new_r > 1 - pts[i][0] + 1e-9 or 
                new_r > pts[i][1] + 1e-9 or new_r > 1 - pts[i][1] + 1e-9):
                feasible = False
            
            if feasible:
                radii[i] = new_r
            else:
                # Try to push circles away to make room
                # Identify circles that would overlap with the new radius
                overlapping = []
                for j in range(N):
                    if j == i:
                        continue
                    d = np.sqrt((pts[i][0] - pts[j][0])**2 + (pts[i][1] - pts[j][1])**2)
                    if new_r + radii[j] > d + 1e-9:
                        overlapping.append(j)
                
                # Also check boundary - if too close to boundary, push inward
                boundary_violation = False
                if new_r > pts[i][0] + 1e-9 or new_r > 1 - pts[i][0] + 1e-9 or \
                   new_r > pts[i][1] + 1e-9 or new_r > 1 - pts[i][1] + 1e-9:
                    boundary_violation = True
                
                if not overlapping and not boundary_violation:
                    # Actually feasible, maybe numerical issue
                    radii[i] = new_r
                    continue
                
                # Push overlapping circles away from i
                if overlapping:
                    # Sort by how much they overlap (most overlap first)
                    def overlap_amount(j):
                        d = np.sqrt((pts[i][0] - pts[j][0])**2 + (pts[i][1] - pts[j][1])**2)
                        return max(0, new_r + radii[j] - d)
                    
                    overlapping.sort(key=overlap_amount, reverse=True)
                    
                    pushed = False
                    for j in overlapping:
                        d = np.sqrt((pts[i][0] - pts[j][0])**2 + (pts[i][1] - pts[j][1])**2)
                        if d < 1e-12:
                            # Circles are on top of each other, random push
                            angle = random.uniform(0, 2 * np.pi)
                            push_dist = 0.001
                            pts[j][0] += push_dist * np.cos(angle)
                            pts[j][1] += push_dist * np.sin(angle)
                            pushed = True
                            continue
                        
                        # Direction from i to j
                        dir_x = (pts[j][0] - pts[i][0]) / d
                        dir_y = (pts[j][1] - pts[i][1]) / d
                        
                        # Amount to push
                        overlap = new_r + radii[j] - d
                        push_amount = overlap + 0.0005  # small extra gap
                        push_amount = min(push_amount, 0.05)  # limit push size
                        
                        new_x = pts[j][0] + push_amount * dir_x
                        new_y = pts[j][1] + push_amount * dir_y
                        
                        # Check if this new position keeps j feasible (boundary and overlaps with others)
                        # Simple check: just update and let next iteration handle it
                        pts[j][0] = new_x
                        pts[j][1] = new_y
                        pushed = True
                    
                    if pushed:
                        # After pushing, check if circle i is now within boundaries
                        # If not, pull it back
                        for coord in range(2):
                            pts[i][coord] = np.clip(pts[i][coord], 0, 1)
                
                # Try to set the new radius
                # Recheck feasibility
                feasible = True
                for j in range(N):
                    if j == i:
                        continue
                    d = np.sqrt((pts[i][0] - pts[j][0])**2 + (pts[i][1] - pts[j][1])**2)
                    if new_r + radii[j] > d + 1e-9:
                        feasible = False
                        break
                if (new_r > pts[i][0] + 1e-9 or new_r > 1 - pts[i][0] + 1e-9 or 
                    new_r > pts[i][1] + 1e-9 or new_r > 1 - pts[i][1] + 1e-9):
                    feasible = False
                
                if feasible:
                    radii[i] = new_r
            
            # After any change, check if current state is valid and better
            current_sum = compute_objective(pts, radii)
            if current_sum > best_val:
                best_val = current_sum
                best_pts = pts.copy()
                best_radii = radii.copy()
            
            # Occasionally, regrow all radii to max feasible
            if iteration % 50 == 0:
                radii = grow_radii(pts, radii.copy())
                current_sum = compute_objective(pts, radii)
                if current_sum > best_val:
                    best_val = current_sum
                    best_pts = pts.copy()
                    best_radii = radii.copy()
        
        return best_pts, best_radii, best_val
    
    def hex_init():
        """Hexagonal lattice initialization."""
        r0 = 0.08
        dx = 2 * r0
        dy = r0 * np.sqrt(3)
        positions = []
        for j in range(5):
            for i in range(6):
                x = (i + 0.5 * (j % 2)) * dx
                y = j * dy
                positions.append([x, y])
        pos = np.array(positions[:N])
        pos[:, 0] -= pos[:, 0].mean()
        pos[:, 1] -= pos[:, 1].mean()
        x_max, x_min = pos[:, 0].max(), pos[:, 0].min()
        y_max, y_min = pos[:, 1].max(), pos[:, 1].min()
        scale_x = 0.8 / (x_max - x_min) if x_max > x_min else 1.0
        scale_y = 0.8 / (y_max - y_min) if y_max > y_min else 1.0
        scale = min(scale_x, scale_y)
        pos *= scale
        pos[:, 0] += 0.5
        pos[:, 1] += 0.5
        pts = pos.tolist()
        # Initial radii: small, will grow
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                min_dist = min(min_dist, d)
        r_init = min_dist * 0.4
        radii = np.full(N, r_init)
        return pts, radii
    
    def ring_init():
        """Ring-based initialization: 1 center, 6 inner ring, 19 outer ring."""
        pts = []
        radii = []
        
        # Center
        center = [0.5, 0.5]
        pts.append(center)
        radii.append(0.1)
        
        # Ring 1: 6 circles
        r1 = 0.12
        d1 = 0.25
        for k in range(6):
            angle = k * np.pi / 3
            x = 0.5 + d1 * np.cos(angle)
            y = 0.5 + d1 * np.sin(angle)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            pts.append([x, y])
            radii.append(r1)
        
        # Ring 2: need 19 more to reach 26
        r2 = 0.08
        # Place 12 on a larger ring
        d2 = 0.42
        for k in range(12):
            angle = k * 2 * np.pi / 12
            x = 0.5 + d2 * np.cos(angle)
            y = 0.5 + d2 * np.sin(angle)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            pts.append([x, y])
            radii.append(r2)
        
        # Place 7 more near corners/edges
        extra_positions = [
            [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],
            [0.5, 0.08], [0.5, 0.92], [0.08, 0.5]
        ]
        for pos in extra_positions[:7]:
            pts.append(pos)
            radii.append(0.06)
        
        return pts, np.array(radii)
    
    def random_init():
        """Random initialization with minimum distance guarantee."""
        pts = []
        radii = []
        min_dist = 0.15
        attempts = 0
        while len(pts) < N and attempts < 10000:
            attempts += 1
            x = random.uniform(0.1, 0.9)
            y = random.uniform(0.1, 0.9)
            ok = True
            for px, py in pts:
                if np.sqrt((x - px)**2 + (y - py)**2) < min_dist:
                    ok = False
                    break
            if ok:
                pts.append([x, y])
                radii.append(0.05)
        
        # If we didn't get enough, add more with smaller min_dist
        if len(pts) < N:
            min_dist = 0.1
            while len(pts) < N and attempts < 20000:
                attempts += 1
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                ok = True
                for px, py in pts:
                    if np.sqrt((x - px)**2 + (y - py)**2) < min_dist:
                        ok = False
                        break
                if ok:
                    pts.append([x, y])
                    radii.append(0.04)
        
        # Pad if necessary
        while len(pts) < N:
            x = random.uniform(0.1, 0.9)
            y = random.uniform(0.1, 0.9)
            pts.append([x, y])
            radii.append(0.03)
        
        return pts, np.array(radii[:N])
    
    best_overall_val = 0.0
    best_overall_pts = None
    best_overall_radii = None
    
    # Time allocation: try different inits, spend time proportionally
    time_per_init = 35  # 3 inits * 35s = 105s, leaving 40s buffer
    
    inits = [
        ("hex", hex_init),
        ("ring", ring_init),
        ("random", random_init),
    ]
    
    for init_name, init_func in inits:
        if time.time() > deadline - 10:
            break
        
        init_start = time.time()
        pts, radii = init_func()
        
        # Make initial state feasible
        # Shrink radii until feasible
        for _ in range(100):
            if compute_objective(pts, radii) >= 0:
                break
            radii = radii * 0.8
        if compute_objective(pts, radii) < 0:
            radii = np.full(N, 0.01)
        
        # Grow to max feasible
        radii = grow_radii(pts, radii.copy())
        
        # Run push algorithm
        init_time_left = deadline - (time.time() - init_start) - 5
        if init_time_left < 5:
            init_time_left = 5
        result_pts, result_radii, result_val = push_algorithm(
            pts, radii, start_time + init_start - time.time() + init_time_left
        )
        
        if result_val > best_overall_val:
            best_overall_val = result_val
            best_overall_pts = result_pts
            best_overall_radii = result_radii
    
    # Final fallback
    if best_overall_pts is None or best_overall_val <= 0:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side * 0.9
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    # Convert to numpy and final validation
    pts_arr = np.array(best_overall_pts)
    radii_arr = np.array(best_overall_radii)
    
    # Final safety: ensure all constraints are met with a small tolerance
    # If any violation, shrink radii slightly
    for _ in range(10):
        valid = True
        for i in range(N):
            x, y = pts_arr[i]
            r = radii_arr[i]
            if r > x + 1e-7 or r > 1 - x + 1e-7 or r > y + 1e-7 or r > 1 - y + 1e-7:
                valid = False
                break
            for j in range(i + 1, N):
                d = np.sqrt((pts_arr[i][0] - pts_arr[j][0])**2 + (pts_arr[i][1] - pts_arr[j][1])**2)
                if radii_arr[i] + radii_arr[j] > d + 1e-7:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            break
        radii_arr = radii_arr * 0.95
    
    # Ensure non-negative
    radii_arr = np.maximum(radii_arr, 0)
    
    return np.column_stack((pts_arr[:, 0], pts_arr[:, 1], radii_arr))

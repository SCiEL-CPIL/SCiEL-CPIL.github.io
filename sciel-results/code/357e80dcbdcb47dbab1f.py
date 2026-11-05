import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s buffer for 60s limit
    
    N = 30
    RANGE = 30
    
    # Precompute candidate moves for speed
    # For each point, we want to try moving to nearby positions.
    # We'll generate a list of (dx, dy) offsets to try.
    # Small steps for local search, larger for global.
    local_offsets = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4) if dx != 0 or dy != 0]
    random.shuffle(local_offsets)
    
    def gen_grid(w, h, offset_x=0, offset_y=0):
        pts = []
        for i in range(w):
            for j in range(h):
                pts.append((i + offset_x, j + offset_y))
        return pts

    def gen_dual_lattice(shift_x, shift_y, scale=1, center=(0, 0)):
        """Generate 30 points as two 15-point triangular lattices shifted apart."""
        pts = []
        cx, cy = center
        # First lattice: 15 points in compact hex arrangement
        lattice1 = []
        lattice1.append((cx, cy))
        # Ring 1: 6 points
        for dx, dy in [(1,0),(1,1),(0,1),(-1,0),(-1,-1),(0,-1)]:
            lattice1.append((cx + dx*scale, cy + dy*scale))
        # Ring 2: need 8 more, take from hex ring
        ring2_candidates = []
        for dx, dy in [(2,0),(2,1),(1,2),(0,2),(-1,2),(-2,1),(-2,0),(-2,-1),(-1,-2),(0,-2),(1,-2),(2,-1)]:
            ring2_candidates.append((cx + dx*scale, cy + dy*scale))
        for p in ring2_candidates:
            if p not in lattice1:
                lattice1.append(p)
                if len(lattice1) == 15:
                    break
        pts.extend(lattice1)
        
        # Second lattice: shifted by (shift_x, shift_y)
        lattice2 = []
        for p in lattice1:
            p2 = (p[0] + shift_x, p[1] + shift_y)
            if -RANGE <= p2[0] <= RANGE and -RANGE <= p2[1] <= RANGE and p2 not in pts:
                lattice2.append(p2)
        pts.extend(lattice2)
        
        # If we don't have 30, fill with more shifted points
        while len(pts) < 30:
            for p in lattice1:
                for sx in range(-2, 3):
                    for sy in range(-2, 3):
                        p2 = (p[0] + shift_x + sx, p[1] + shift_y + sy)
                        if -RANGE <= p2[0] <= RANGE and -RANGE <= p2[1] <= RANGE and p2 not in pts:
                            pts.append(p2)
                            if len(pts) == 30:
                                return pts
        return pts[:30]

    # Initial configurations
    inits = []
    inits.append(gen_grid(6, 5, -2, -2))
    inits.append(gen_grid(5, 6, -2, -2))
    inits.append(gen_grid(10, 3, -4, -1))
    inits.append(gen_grid(3, 10, -1, -4))
    inits.append(gen_grid(15, 2, -6, 0))
    inits.append(gen_grid(2, 15, 0, -6))
    
    # Add dual-lattice configurations
    for sx in [2, 3, 4, 5, 6]:
        for sy in [0, 1, 2, 3]:
            pts = gen_dual_lattice(sx, sy, scale=1, center=(0, 0))
            if len(set(pts)) == N:
                inits.append(pts)
    
    # Add a few random ones
    for _ in range(5):
        pts = []
        while len(pts) < N:
            p = (random.randint(-RANGE, RANGE), random.randint(-RANGE, RANGE))
            if p not in pts:
                pts.append(p)
        inits.append(pts)

    best_overall_score = 0
    best_points = inits[0]
    
    # Helper to compute full score and structures
    def init_state(points):
        area_counts = Counter()
        freq_counts = Counter()
        
        p = points
        for i in range(N-2):
            xi, yi = p[i]
            for j in range(i+1, N-1):
                xj, yj = p[j]
                for k in range(j+1, N):
                    xk, yk = p[k]
                    det = abs((xj-xi)*(yk-yi) - (xk-xi)*(yj-yi))
                    if det > 0:
                        area_counts[det] += 1
        
        for area, cnt in area_counts.items():
            freq_counts[cnt] += 1
            
        max_score = max(freq_counts.keys()) if freq_counts else 0
        return area_counts, freq_counts, max_score

    def apply_move(points, idx, old_pt, new_pt, area_counts, freq_counts, max_score):
        # Apply the move to points
        points[idx] = new_pt
        
        delta_area_counts = {}
        p = points
        
        # Remove triangles with old_pt and add with new_pt
        for j in range(N):
            if j == idx: continue
            xj, yj = p[j]
            for k in range(j+1, N):
                if k == idx: continue
                xk, yk = p[k]
                
                det_old = abs((xj - old_pt[0]) * (yk - old_pt[1]) - (xk - old_pt[0]) * (yj - old_pt[1]))
                if det_old > 0:
                    delta_area_counts[det_old] = delta_area_counts.get(det_old, 0) - 1
                
                det_new = abs((xj - new_pt[0]) * (yk - new_pt[1]) - (xk - new_pt[0]) * (yj - new_pt[1]))
                if det_new > 0:
                    delta_area_counts[det_new] = delta_area_counts.get(det_new, 0) + 1
        
        new_max_score = max_score
        for area, delta in delta_area_counts.items():
            if delta == 0:
                continue
                
            old_count = area_counts.get(area, 0)
            new_count = old_count + delta
            
            # Update freq_counts
            if old_count > 0:
                freq_counts[old_count] -= 1
                if freq_counts[old_count] == 0:
                    del freq_counts[old_count]
            
            if new_count > 0:
                freq_counts[new_count] += 1
                area_counts[area] = new_count
                if new_count > new_max_score:
                    new_max_score = new_count
            else:
                if area in area_counts:
                    del area_counts[area]
        
        # Recalculate max if the current max count is no longer present
        if freq_counts.get(max_score, 0) == 0 and max_score > 0:
             # Find new max
             new_max_score = max(freq_counts.keys()) if freq_counts else 0
        elif new_max_score < max_score:
             # If we only incremented, max_score can't decrease.
             # If we decremented the max area, it might decrease.
             # The logic above handles the case where the count of 'max_score' becomes 0.
             # But what if max_score was 10, and we decremented an area from 10 to 9,
             # but there was another area with 10? Then freq_counts[10] is still > 0.
             # So max_score stays 10.
             # If we decremented the *only* area with 10, freq_counts[10] becomes 0.
             # Then we recalculate.
             # If we incremented an area to 11, new_max_score becomes 11.
             pass

        return new_max_score

    def revert_move(points, idx, new_pt, old_pt, area_counts, freq_counts, max_score):
        # Revert the move
        return apply_move(points, idx, new_pt, old_pt, area_counts, freq_counts, max_score)

    # Local search function
    def local_search(initial_points, time_limit, T_init=1.0):
        current_points = initial_points[:]
        area_counts, freq_counts, current_score = init_state(current_points)
        
        best_local_score = current_score
        best_local_points = current_points[:]
        
        T = T_init
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < time_limit and no_improve < max_no_improve:
            idx = random.randint(0, N-1)
            old_pt = current_points[idx]
            
            # Generate candidate moves
            moves = []
            
            # Local moves
            for dx, dy in local_offsets:
                new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                if -RANGE <= new_pt[0] <= RANGE and -RANGE <= new_pt[1] <= RANGE:
                    # Check distinctness
                    valid = True
                    for j in range(N):
                        if j != idx and current_points[j] == new_pt:
                            valid = False
                            break
                    if valid:
                        moves.append(new_pt)
                        
            # Add some random global moves
            for _ in range(2):
                new_pt = (random.randint(-RANGE, RANGE), random.randint(-RANGE, RANGE))
                valid = True
                for j in range(N):
                    if j != idx and current_points[j] == new_pt:
                        valid = False
                        break
                if valid:
                    moves.append(new_pt)
                    
            if not moves:
                no_improve += 1
                continue
                
            random.shuffle(moves)
            
            improved = False
            for new_pt in moves:
                new_score = apply_move(current_points, idx, old_pt, new_pt, area_counts, freq_counts, current_score)
                delta_score = new_score - current_score
                
                # Acceptance
                if delta_score > 0:
                    accept = True
                else:
                    if delta_score == 0:
                        accept = random.random() < 0.5
                    else:
                        accept = random.random() < (2.718 ** (delta_score / T))
                
                if accept:
                    current_score = new_score
                    improved = True
                    no_improve = 0
                    if current_score > best_local_score:
                        best_local_score = current_score
                        best_local_points = current_points[:]
                    break
                else:
                    revert_move(current_points, idx, new_pt, old_pt, area_counts, freq_counts, current_score)
                    
            if not improved:
                no_improve += 1
                
            # Cool down
            T *= 0.99
            if T < 0.1:
                T = 0.1

        return best_local_points, best_local_score

    # Main loop over initial configurations
    # We allocate time budget to each
    num_inits = len(inits)
    time_per_init = (deadline - start_time - 5.0) / num_inits if num_inits > 0 else 0
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
            
        # Validate init
        if len(set(init_pts)) != N:
            continue
        valid = True
        for p in init_pts:
            if p[0] < -RANGE or p[0] > RANGE or p[1] < -RANGE or p[1] > RANGE:
                valid = False
                break
        if not valid:
            continue
            
        # Run local search
        limit = min(deadline - 5.0, time.time() + time_per_init)
        best_pts, best_score = local_search(init_pts, limit)
        
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_points = best_pts[:]

    # Final validation
    if len(best_points) != N or len(set(best_points)) != N:
        return gen_grid(6, 5, -2, -2)
    for p in best_points:
        if p[0] < -30 or p[0] > 30 or p[1] < -30 or p[1] > 30:
            return gen_grid(6, 5, -2, -2)

    return best_points

import random
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Grid parameters
    GRID_SIZE = 12
    HALF = GRID_SIZE // 2
    grid_points = []
    for x in range(-HALF, HALF):
        for y in range(-HALF, HALF):
            grid_points.append((x, y))
            
    N_GRID = len(grid_points)
    
    # Precompute lines
    # line_key: (a, b, c) normalized
    # We map (a,b,c) to a unique ID
    line_map = {}
    line_points = [] # list of sets of point indices
    point_line_ids = [[] for _ in range(N_GRID)]
    
    for i in range(N_GRID):
        for j in range(i+1, N_GRID):
            x1, y1 = grid_points[i]
            x2, y2 = grid_points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction
            if dx == 0:
                dy = 1
            elif dy == 0:
                dx = 1
            else:
                g = math.gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                if dx < 0:
                    dx = -dx
                    dy = -dy
                elif dx == 0 and dy < 0:
                    dy = -dy
            
            # Normal vector (a, b)
            a = dy
            b = -dx
            c = a * x1 + b * y1
            
            key = (a, b, c)
            
            if key not in line_map:
                line_map[key] = len(line_points)
                line_points.append(set())
                
            lid = line_map[key]
            line_points[lid].add(i)
            line_points[lid].add(j)
            
            point_line_ids[i].append(lid)
            point_line_ids[j].append(lid)
            
    # Deduplicate point_line_ids (a point might be added multiple times? No, each pair adds to its endpoints. But a line with k points adds k*(k-1)/2 pairs. The set `line_points` handles uniqueness for the line's content. `point_line_ids` will have duplicates if we just append. We need unique LIDs per point.)
    for i in range(N_GRID):
        point_line_ids[i] = list(set(point_line_ids[i]))
        
    num_lines = len(line_points)
    
    def evaluate_initial(points_indices):
        # points_indices: list of 16 grid indices
        counts = [0] * num_lines
        score = 0
        invalid = 0
        for p_idx in points_indices:
            for lid in point_line_ids[p_idx]:
                counts[lid] += 1
        for c in counts:
            if c == 4:
                score += 1
            elif c >= 5:
                invalid += 1
        return counts, score, invalid

    def get_score(score, invalid):
        if invalid > 0:
            return -10000
        return score

    best_score = -1
    best_points = None
    
    # SA Parameters
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    
    # Multiple restarts
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize
        # Try a structured start: 4x4 grid
        # Map 4x4 grid to indices in 12x12
        # 4x4 points: x in [0,3], y in [0,3]
        # In 12x12 (-6..5), let's pick x in [-2, 1], y in [-2, 1]
        start_indices = []
        for x in range(-2, 2):
            for y in range(-2, 2):
                # Find index of (x,y)
                # grid_points is generated x outer, y inner?
                # range(-6, 6) for x, range(-6, 6) for y
                # Index = (x - (-6)) * 12 + (y - (-6))
                idx = (x + HALF) * GRID_SIZE + (y + HALF)
                start_indices.append(idx)
                
        # Randomize slightly? Or just use this.
        # Let's mix: 50% structured, 50% random
        if r % 2 == 0:
            current_indices = start_indices[:]
        else:
            current_indices = random.sample(range(N_GRID), 16)
            
        # Ensure distinct
        if len(set(current_indices)) < 16:
            continue
            
        counts, current_score, current_invalid = evaluate_initial(current_indices)
        current_score_val = get_score(current_score, current_invalid)
        
        if current_score_val > best_score:
            best_score = current_score_val
            best_points = [grid_points[i] for i in current_indices]
            
        T = 5.0
        steps = 0
        while time.time() < deadline - 5 and T > T_min:
            # Pick point to move
            pos = random.randint(0, 15)
            p_out = current_indices[pos]
            
            # Pick new point
            # Random from grid, not in current
            # Optimization: maintain a set of active indices for O(1) check
            active_set = set(current_indices)
            
            p_in = None
            for _ in range(100): # Retry loop
                candidate = random.randint(0, N_GRID - 1)
                if candidate not in active_set:
                    p_in = candidate
                    break
            
            if p_in is None:
                continue
                
            # Update counts
            # Remove p_out
            for lid in point_line_ids[p_out]:
                c_old = counts[lid]
                c_new = c_old - 1
                counts[lid] = c_new
                
                if c_old == 4 and c_new != 4:
                    current_score -= 1
                elif c_old != 4 and c_new == 4:
                    current_score += 1
                elif c_old < 5 and c_new >= 5:
                    current_invalid += 1
                elif c_old >= 5 and c_new < 5:
                    current_invalid -= 1
                    
            # Add p_in
            for lid in point_line_ids[p_in]:
                c_old = counts[lid]
                c_new = c_old + 1
                counts[lid] = c_new
                
                if c_old == 4 and c_new != 4:
                    current_score -= 1
                elif c_old != 4 and c_new == 4:
                    current_score += 1
                elif c_old < 5 and c_new >= 5:
                    current_invalid += 1
                elif c_old >= 5 and c_new < 5:
                    current_invalid -= 1
                    
            new_score_val = get_score(current_score, current_invalid)
            
            # Acceptance
            if new_score_val >= current_score_val:
                # Accept
                current_indices[pos] = p_in
                active_set.remove(p_out)
                active_set.add(p_in)
                current_score_val = new_score_val
                
                if current_score_val > best_score:
                    best_score = current_score_val
                    best_points = [grid_points[i] for i in current_indices]
            else:
                # Reject probability
                delta = new_score_val - current_score_val
                if random.random() < math.exp(delta / T):
                    current_indices[pos] = p_in
                    active_set.remove(p_out)
                    active_set.add(p_in)
                    current_score_val = new_score_val
                else:
                    # Revert
                    # Add back p_out
                    for lid in point_line_ids[p_out]:
                        c_old = counts[lid]
                        c_new = c_old + 1
                        counts[lid] = c_new
                        if c_old == 4 and c_new != 4:
                            current_score -= 1
                        elif c_old != 4 and c_new == 4:
                            current_score += 1
                        elif c_old < 5 and c_new >= 5:
                            current_invalid += 1
                        elif c_old >= 5 and c_new < 5:
                            current_invalid -= 1
                            
                    # Remove p_in
                    for lid in point_line_ids[p_in]:
                        c_old = counts[lid]
                        c_new = c_old - 1
                        counts[lid] = c_new
                        if c_old == 4 and c_new != 4:
                            current_score -= 1
                        elif c_old != 4 and c_new == 4:
                            current_score += 1
                        elif c_old < 5 and c_new >= 5:
                            current_invalid += 1
                        elif c_old >= 5 and c_new < 5:
                            current_invalid -= 1

            T *= cooling
            steps += 1
            
    # Fallback
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(4)]
        
    return best_points

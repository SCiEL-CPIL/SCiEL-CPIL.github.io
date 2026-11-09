import time
import random
from math import gcd
from itertools import combinations

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    a, b = dy, -dx
    d = gcd(abs(a), abs(b))
    if d > 0:
        a //= d
        b //= d
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = -(a * x1 + b * y1)
    return (a, b, c)

def fast_count(points):
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
    four_count = 0
    valid = True
    for count in line_counts.values():
        if count >= 5:
            valid = False
            break
        if count == 4:
            four_count += 1
    if not valid:
        return -1
    return four_count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Strategy: Constructive line-packing with repair.
    # 1. Generate random candidate points.
    # 2. Use a greedy algorithm to identify and "lock" 4-point lines.
    # 3. Move remaining points to complete new 4-lines without breaking existing ones or creating 5-lines.
    # 4. Local search on the "free" points.
    
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = fast_count(best_points)
    
    if best_score < 0:
        return best_points

    def random_initial():
        pts = []
        used = set()
        while len(pts) < 16:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            if (x, y) not in used:
                used.add((x, y))
                pts.append([x, y])
        return pts

    def evaluate_and_repair(pts):
        """
        Attempts to increase the number of 4-point lines by moving points.
        Returns (score, pts).
        """
        n = 16
        # Track lines and their points
        lines = {} # key -> set of indices
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key:
                    if key not in lines:
                        lines[key] = set()
                    lines[key].add(i)
                    lines[key].add(j)
        
        def get_score():
            s = 0
            for k, v in lines.items():
                if len(v) == 4:
                    s += 1
                elif len(v) >= 5:
                    return -1
            return s
        
        current_score = get_score()
        
        # Greedy completion: Find 3-point lines, try to move a point to complete them.
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Find 3-lines
            three_lines = [k for k, v in lines.items() if len(v) == 3]
            random.shuffle(three_lines)
            
            for line_key in three_lines:
                if time.time() > deadline:
                    break
                a, b, c = line_key
                triple = lines[line_key]
                # Find a point NOT in triple to move onto this line
                candidates_to_move = [i for i in range(n) if i not in triple]
                random.shuffle(candidates_to_move)
                
                found_move = False
                for i in candidates_to_move:
                    old_x, old_y = pts[i]
                    # We need to find a new position for i on line (a,b,c)
                    # such that it doesn't collide and doesn't create 5-lines.
                    # Also, moving i away from old lines might break 4-lines.
                    
                    # Heuristic: Pick a position on the line close to the existing triple
                    # but far from other points to avoid accidental 5-collinearity.
                    # Let's try a few random integer points on the line.
                    # a*x + b*y + c = 0
                    
                    # Find a particular solution.
                    # If a != 0, x = ( -b*y - c ) / a. We need divisibility.
                    # Simpler: just try a grid of points near the centroid of the triple.
                    xs = [pts[t][0] for t in triple]
                    ys = [pts[t][1] for t in triple]
                    cx = sum(xs) // 3
                    cy = sum(ys) // 3
                    
                    # Try points in a box around (cx, cy)
                    # Expand box if needed
                    for radius in range(5, 50, 5):
                        if found_move:
                            break
                        for dx in range(-radius, radius + 1):
                            for dy in range(-radius, radius + 1):
                                nx, ny = cx + dx, cy + dy
                                if abs(nx) > 1000 or abs(ny) > 1000:
                                    continue
                                if a * nx + b * ny + c != 0:
                                    continue
                                
                                # Check collision with existing points
                                if any(pts[k][0] == nx and pts[k][1] == ny for k in range(n) if k != i):
                                    continue
                                
                                # Simulate move
                                old_pos = [old_x, old_y]
                                pts[i] = [nx, ny]
                                
                                # Update lines structure for simulation
                                # Remove i from all lines it was on
                                lines_to_remove = []
                                for k, v in lines.items():
                                    if i in v:
                                        v.discard(i)
                                        if not v:
                                            lines_to_remove.append(k)
                                for k in lines_to_remove:
                                    del lines[k]
                                
                                # Add i to new lines
                                for j in range(n):
                                    if j == i:
                                        continue
                                    new_key = get_line_key(pts[i], pts[j])
                                    if new_key:
                                        if new_key not in lines:
                                            lines[new_key] = set()
                                        lines[new_key].add(i)
                                        lines[new_key].add(j)
                                
                                new_score = get_score()
                                
                                # Revert if not better
                                if new_score > current_score and new_score >= 0:
                                    # Accept
                                    current_score = new_score
                                    improved = True
                                    found_move = True
                                    # Note: lines structure is already updated to new state
                                    break
                                else:
                                    # Revert
                                    # This is tricky with the in-place modification.
                                    # Let's rebuild lines from scratch for simplicity in the revert path?
                                    # Or just revert pts and rebuild lines.
                                    pts[i] = old_pos
                                    # Rebuild lines is O(n^2) which is fine for n=16
                                    lines.clear()
                                    for p_idx in range(n):
                                        for q_idx in range(p_idx + 1, n):
                                            k = get_line_key(pts[p_idx], pts[q_idx])
                                            if k:
                                                if k not in lines:
                                                    lines[k] = set()
                                                lines[k].add(p_idx)
                                                lines[k].add(q_idx)
                                    break # Try next candidate point i for this line
                            
            # If no improvement in this pass, do a random perturbation to escape local minima
            if not improved:
                # Random move
                i = random.randint(0, 15)
                old_x, old_y = pts[i]
                nx = old_x + random.randint(-10, 10)
                ny = old_y + random.randint(-10, 10)
                if abs(nx) <= 1000 and abs(ny) <= 1000 and not any(pts[k][0] == nx and pts[k][1] == ny for k in range(n) if k != i):
                    pts[i] = [nx, ny]
                    # Rebuild lines
                    lines.clear()
                    for p_idx in range(n):
                        for q_idx in range(p_idx + 1, n):
                            k = get_line_key(pts[p_idx], pts[q_idx])
                            if k:
                                if k not in lines:
                                    lines[k] = set()
                                lines[k].add(p_idx)
                                lines[k].add(q_idx)
                    new_score = get_score()
                    if new_score > current_score:
                        current_score = new_score
                    # Accept even if worse to escape? No, let's stick to hill climbing for now, 
                    # but the random initial generation provides diversity.
        
        return current_score, pts

    # Main loop: Multiple restarts with random initial configurations
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Generate a diverse random start
        # Mix of grid-like and random
        if restart % 2 == 0:
            # Perturbed grid
            pts = [[x + random.randint(-5, 5), y + random.randint(-5, 5)] for x in range(4) for y in range(4)]
            # Ensure distinct
            valid_pts = []
            used = set()
            for p in pts:
                if (p[0], p[1]) not in used:
                    used.add((p[0], p[1]))
                    valid_pts.append(p)
            while len(valid_pts) < 16:
                x = random.randint(-50, 50)
                y = random.randint(-50, 50)
                if (x, y) not in used:
                    used.add((x, y))
                    valid_pts.append([x, y])
            pts = valid_pts
        else:
            pts = random_initial()
        
        score, final_pts = evaluate_and_repair(pts)
        
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in final_pts]
            
        # Early stop if we hit the theoretical max or close
        if best_score >= 10:
            break

    # Final validation
    final_score = fast_count(best_points)
    if final_score < 0:
        # Fallback to grid
        return [[x, y] for x in range(4) for y in range(4)]
        
    return best_points

if __name__ == "__main__":
    # Quick self-test
    pts = construct()
    print(f"Score: {fast_count(pts)}")
    print(pts)

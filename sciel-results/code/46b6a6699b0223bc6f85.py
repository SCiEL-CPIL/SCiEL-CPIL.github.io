import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initial configuration: 4x4 grid
    # Points: (0,0) to (3,3)
    # We can scale or translate, but 0-3 is fine.
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    
    def score(points):
        """
        Returns the number of lines with exactly 4 points.
        Returns -1 if any line has 5 or more points.
        """
        n = len(points)
        lines = {}
        
        # Precompute points as tuples for hashing
        pts = [tuple(p) for p in points]
        
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i + 1, n):
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                
                # Normalize direction
                if dx == 0 and dy == 0:
                    continue # Should not happen as points are distinct
                g = math.gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                
                # Ensure consistent sign
                if dx < 0:
                    dx = -dx
                    dy = -dy
                elif dx == 0 and dy < 0:
                    dy = -dy
                
                # Line equation: dy*x - dx*y = C
                # C = dy*x1 - dx*y1
                C = dy * x1 - dx * y1
                
                # Further normalize C?
                # The line is uniquely identified by (dx, dy, C) with (dx, dy) reduced.
                # However, (dx, dy, C) and (-dx, -dy, -C) are the same line.
                # Our sign normalization handles the direction.
                # Does it handle C?
                # If we flip dx, dy, C flips sign.
                # We fixed dx, dy sign. So C is fixed for this direction.
                # So (dx, dy, C) is a unique key.
                
                key = (dx, dy, C)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
        
        score_val = 0
        for pts_on_line in lines.values():
            count = len(pts_on_line)
            if count == 4:
                score_val += 1
            elif count >= 5:
                return -1
        return score_val

    # Initial score
    best_points = [p[:] for p in initial_points]
    best_score = score(best_points)
    
    current_points = [p[:] for p in initial_points]
    current_score = best_score
    
    # Local search
    # We will try to move one point at a time.
    # To escape local optima, we use a "temperature" or just random restarts if stuck.
    
    stuck_counter = 0
    max_stuck = 5000  # If no improvement for 5000 steps, try a bigger jump or restart
    
    while time.time() < deadline:
        # Check time frequently
        if time.time() > deadline:
            break
            
        if stuck_counter > max_stuck:
            # Random restart or large perturbation
            # Let's try a large perturbation: move 2-3 points randomly
            for _ in range(random.randint(2, 4)):
                idx = random.randint(0, 15)
                new_x = random.randint(-20, 20)
                new_y = random.randint(-20, 20)
                # Ensure distinct
                if any(current_points[i][0] == new_x and current_points[i][1] == new_y for i in range(16) if i != idx):
                    continue
                current_points[idx] = [new_x, new_y]
            
            new_score = score(current_points)
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
                stuck_counter = 0
            else:
                # Revert if bad? Or keep if it's a random walk?
                # For restart, we should probably revert to best if this didn't help,
                # but keep exploring from current if we want to wander.
                # Let's revert to best to start fresh local search.
                current_points = [p[:] for p in best_points]
                current_score = best_score
                stuck_counter = 0
                continue

        # Pick a random point to move
        idx = random.randint(0, 15)
        
        # Pick a new location
        # Small steps first, larger if stuck?
        # Let's use a random step size.
        step_size = random.choice([1, 2, 3, 5, 10])
        new_x = current_points[idx][0] + random.randint(-step_size, step_size)
        new_y = current_points[idx][1] + random.randint(-step_size, step_size)
        
        # Bounds check
        if new_x < -1000 or new_x > 1000 or new_y < -1000 or new_y > 1000:
            continue
            
        # Distinctness check
        is_distinct = True
        for i in range(16):
            if i == idx:
                continue
            if current_points[i][0] == new_x and current_points[i][1] == new_y:
                is_distinct = False
                break
        if not is_distinct:
            continue
            
        # Apply move
        old_point = current_points[idx][:]
        current_points[idx] = [new_x, new_y]
        
        new_score = score(current_points)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
            stuck_counter = 0
        else:
            # Revert
            current_points[idx] = old_point
            stuck_counter += 1
            # If stuck_counter is high, we will trigger the restart logic above
            
    return best_points

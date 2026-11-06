import time
import random
from math import gcd
from collections import defaultdict

def _line_key(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dx * p1[0] - dy * p1[1]
    return (dx, dy, c)

def _score(points):
    """Calculate the number of lines with exactly 4 points."""
    line_map = defaultdict(set)
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            key = _line_key(points[i], points[j])
            if key:
                line_map[key].add(i)
                line_map[key].add(j)
    
    score = 0
    valid = True
    for s in line_map.values():
        if len(s) == 4:
            score += 1
        elif len(s) >= 5:
            valid = False
    
    if not valid:
        return -1000  # Penalty for invalid configuration (5+ collinear)
    
    return score

def _try_move(points, idx, new_pos):
    """
    Try to move point idx to new_pos.
    Returns (is_valid, delta_score, new_points)
    """
    old_pos = points[idx]
    if old_pos == new_pos:
        return False, 0, None
    
    # Check if new_pos is already occupied
    for i in range(len(points)):
        if i != idx and points[i] == new_pos:
            return False, 0, None
    
    # Check bounds
    if not (-1000 <= new_pos[0] <= 1000 and -1000 <= new_pos[1] <= 1000):
        return False, 0, None

    # Calculate old score
    old_score = _score(points)
    if old_score < 0:
        # If current is invalid, we just want to get to a valid state
        # But _score returns -1000. Let's handle invalid states carefully.
        # For simplicity in this rewrite, we assume we start valid and try to stay valid.
        # If we are invalid, we accept any move that makes it valid or improves score.
        pass

    # Create new points list
    new_points = [list(p) for p in points]
    new_points[idx] = list(new_pos)
    
    new_score = _score(new_points)
    
    # Check validity: no 5 collinear
    # _score returns -1000 if invalid. 
    # We need to know if it's valid.
    line_map = defaultdict(set)
    n = len(new_points)
    for i in range(n):
        for j in range(i+1, n):
            key = _line_key(new_points[i], new_points[j])
            if key:
                line_map[key].add(i)
                line_map[key].add(j)
                
    is_valid = True
    for s in line_map.values():
        if len(s) >= 5:
            is_valid = False
            break
            
    if not is_valid:
        return False, 0, None
        
    delta = new_score - old_score
    return True, delta, new_points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_points = None
    best_score = -1
    rng = random.Random(42)
    
    # Strategy: 
    # 1. Generate diverse initial configurations
    # 2. Use Simulated Annealing to refine
    # 3. Keep the best found
    
    # Initial configurations to try
    # A. 4x4 Grid
    # B. Random points in a small box
    # C. Points on a circle (perturbed)
    # D. "Orchard" construction: Start with lines, pick intersections
    
    def generate_initial_config():
        # Mix of strategies
        r = rng.random()
        if r < 0.3:
            # 4x4 Grid with random shifts
            offset_x = rng.randint(-10, 10)
            offset_y = rng.randint(-10, 10)
            scale = rng.randint(1, 5)
            pts = []
            for x in range(4):
                for y in range(4):
                    pts.append([x * scale + offset_x, y * scale + offset_y])
            return pts
        elif r < 0.6:
            # Random in box
            pts = []
            seen = set()
            while len(pts) < 16:
                x = rng.randint(-50, 50)
                y = rng.randint(-50, 50)
                if (x, y) not in seen:
                    seen.add((x, y))
                    pts.append([x, y])
            return pts
        else:
            # Structured: Two parallel lines with 8 points each? No, that creates many 5+ lines.
            # Let's try a "star" or "grid-like" but perturbed
            pts = []
            seen = set()
            # 4x4 grid base
            for x in range(4):
                for y in range(4):
                    # Add noise
                    nx = x * 10 + rng.randint(-2, 2)
                    ny = y * 10 + rng.randint(-2, 2)
                    if (nx, ny) not in seen:
                        seen.add((nx, ny))
                        pts.append([nx, ny])
            # Fill remaining if any duplicates
            while len(pts) < 16:
                x = rng.randint(-100, 100)
                y = rng.randint(-100, 100)
                if (x, y) not in seen:
                    seen.add((x, y))
                    pts.append([x, y])
            return pts
    
    def simulated_annealing(points, time_limit):
        current_points = [list(p) for p in points]
        current_score = _score(current_points)
        if current_score < 0:
            return current_points, current_score
            
        best_local_points = [list(p) for p in current_points]
        best_local_score = current_score
        
        T = 10.0
        cooling = 0.999
        min_T = 0.1
        
        while time.time() < time_limit and T > min_T:
            idx = rng.randint(0, 15)
            # Generate candidate
            # Move to a random point in a neighborhood
            # Or swap with another point?
            # Let's do a random jump to a new location
            # To make it effective, we should try to land on lines with 3 points.
            # But that's complex. Let's try random local moves first.
            
            # Strategy: Pick a random point, move it to a random position
            # Or pick a random point and move it to a position that shares a line with 2 other points?
            
            # Let's try a "smart" random move:
            # 1. Pick a random point P_i
            # 2. Pick a random other point P_j
            # 3. Move P_i to a position that is on the line P_j-P_k for some P_k?
            # This is getting complex. Let's stick to random moves with larger steps.
            
            new_pos = None
            # 50% chance: local move
            # 50% chance: global random move
            if rng.random() < 0.5:
                dx = rng.randint(-10, 10)
                dy = rng.randint(-10, 10)
                new_pos = (current_points[idx][0] + dx, current_points[idx][1] + dy)
            else:
                new_pos = (rng.randint(-100, 100), rng.randint(-100, 100))
            
            is_valid, delta, new_points = _try_move(current_points, idx, new_pos)
            
            if is_valid:
                # Acceptance probability
                if delta > 0 or rng.random() < pow(2.71828, delta / T):
                    current_points = new_points
                    current_score += delta
                    if current_score > best_local_score:
                        best_local_score = current_score
                        best_local_points = [list(p) for p in current_points]
            T *= cooling
            
        return best_local_points, best_local_score

    # Main loop: Multiple restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        init_pts = generate_initial_config()
        # Ensure init is valid
        init_score = _score(init_pts)
        if init_score < 0:
            # Try to fix or skip
            # For now, skip if invalid
            continue
            
        # Run SA for a fixed amount of time per restart
        # 150s / 50 restarts = 3s per restart.
        # Let's allocate time dynamically.
        time_per_restart = (deadline - time.time()) / (num_restarts - i)
        # Cap time per restart to avoid one taking too long if we have many left
        time_per_restart = min(time_per_restart, 2.0)
        
        sa_pts, sa_score = simulated_annealing(init_pts, time.time() + time_per_restart)
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts
            
        # Also check if init was better (unlikely)
        if init_score > best_score:
            best_score = init_score
            best_points = init_pts

    # If we didn't find anything (should not happen), return a valid grid
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(4)]
        
    # Final verification
    final_score = _score(best_points)
    if final_score < 0:
         # Fallback to 4x4 grid
         best_points = [[x, y] for x in range(4) for y in range(4)]
         
    return [list(p) for p in best_points]

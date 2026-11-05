import time
import random
from math import gcd
from collections import Counter

def _evaluate(points):
    """
    Evaluate a list of 16 [x,y] points.
    Returns (is_valid, score).
    is_valid: 16 distinct integer points, |x|,|y|<=1000, no 5 collinear.
    score: number of lines containing exactly 4 points.
    """
    n = len(points)
    if n != 16:
        return False, 0
    
    # Check bounds and distinctness
    point_set = set()
    for p in points:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return False, 0
        pt = (x, y)
        if pt in point_set:
            return False, 0
        point_set.add(pt)
        
    lines = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            g = gcd(abs(dx), abs(dy))
            ndx = dx // g
            ndy = dy // g
            if ndx < 0 or (ndx == 0 and ndy < 0):
                ndx = -ndx
                ndy = -ndy
            c = ndy * x1 - ndx * y1
            key = (ndx, ndy, c)
            lines[key] = lines.get(key, 0) + 1
            
    score = 0
    for count in lines.values():
        if count >= 10:  # C(5,2)=10 means 5+ collinear
            return False, 0
        if count == 6:   # C(4,2)=6 means exactly 4 collinear
            score += 1
            
    return True, score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial configuration: 4x4 grid
    current_points = [[x, y] for x in range(-1, 3) for y in range(-1, 3)]
    is_valid, current_score = _evaluate(current_points)
    if not is_valid:
        current_points = [[x, y] for x in range(4) for y in range(4)]
        is_valid, current_score = _evaluate(current_points)
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # Local search with restarts and simulated annealing-like acceptance
    # We will run multiple "annealing" cycles.
    
    while time.time() < deadline:
        # Reset current state to best found so far for each new cycle?
        # Or continue from current?
        # Let's try a strategy of restarting from best if stuck, or just continuing.
        # The prompt suggests "Simulated annealing-like random walk with restarts".
        
        # Let's define a temperature schedule for this run?
        # Or just a fixed probability of accepting worse moves.
        
        # To implement restarts properly, we need a loop structure that allows
        # starting a new walk from the best point found so far.
        
        # Let's use a simpler approach:
        # 1. Start from best_points.
        # 2. Perform a sequence of moves.
        # 3. If no improvement for X steps, restart from best_points.
        
        current_points = [p[:] for p in best_points]
        current_score = best_score
        
        no_improve_steps = 0
        max_no_improve = 200  # Restart threshold
        
        while time.time() < deadline:
            prev_points = [p[:] for p in current_points]
            
            # Generate a move
            r = random.random()
            if r < 0.7:
                # Small local move
                idx = random.randint(0, 15)
                ox, oy = current_points[idx]
                # Step size can be adjusted, but let's keep it small
                nx = ox + random.randint(-5, 5)
                ny = oy + random.randint(-5, 5)
                if abs(nx) <= 1000 and abs(ny) <= 1000:
                    current_points[idx] = [nx, ny]
                else:
                    continue
            elif r < 0.85:
                # Medium move
                idx = random.randint(0, 15)
                nx = random.randint(-100, 100)
                ny = random.randint(-100, 100)
                current_points[idx] = [nx, ny]
            else:
                # Large move / Restart-like
                idx = random.randint(0, 15)
                nx = random.randint(-1000, 1000)
                ny = random.randint(-1000, 1000)
                current_points[idx] = [nx, ny]
            
            is_valid, new_score = _evaluate(current_points)
            
            if is_valid:
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
                    no_improve_steps = 0
                elif new_score == current_score:
                    no_improve_steps += 1
                    # Accept equal score to walk
                else:
                    # Worse score
                    # Accept with some probability (simulated annealing heuristic)
                    # Probability = exp(-delta / T)
                    # Here we use a simple heuristic: accept if random < 0.1
                    if random.random() < 0.1:
                        no_improve_steps += 1
                    else:
                        # Revert
                        current_points = prev_points
                        no_improve_steps += 1
            else:
                # Invalid, revert
                current_points = prev_points
                no_improve_steps += 1
            
            # Check for restart condition
            if no_improve_steps > max_no_improve:
                # Restart from best
                current_points = [p[:] for p in best_points]
                current_score = best_score
                no_improve_steps = 0
                # Optionally, we could break the inner loop and let the outer loop
                # handle it, but since we don't have a strict outer loop structure
                # that resets time, we just reset state here.
                # However, if we keep resetting, we might loop forever on the same
                # stuck state. 
                # Let's add a "global" no_improve counter or just break and let
                # the main while loop re-initialize?
                # The main while loop re-initializes `current_points` from `best_points`
                # at the start of *each* outer iteration.
                # So if we break here, the outer loop will restart.
                break
        
        # The outer loop will restart the inner loop with a fresh state from best_points
        # This effectively implements restarts.

    return best_points

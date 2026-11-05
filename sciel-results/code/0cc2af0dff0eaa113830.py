import math
import time
import random
from math import gcd

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return (0, 0, 0)
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _evaluate(points):
    n = len(points)
    if n != 16:
        return False, 0
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
    # Pre-compute pairs to avoid repeated indexing in inner loop? 
    # With N=16, nested loops are fine.
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue # Should not happen if distinct
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

def _generate_initial_points():
    """Generate a 4x4 grid centered at 0."""
    return [[x, y] for x in range(-1, 3) for y in range(-1, 3)]

def _perturb(points, intensity=5):
    """Create a perturbed copy of the points."""
    new_pts = [p[:] for p in points]
    for i in range(16):
        if random.random() < 0.5:
            new_pts[i][0] += random.randint(-intensity, intensity)
            new_pts[i][1] += random.randint(-intensity, intensity)
            # Clamp
            new_pts[i][0] = max(-1000, min(1000, new_pts[i][0]))
            new_pts[i][1] = max(-1000, min(1000, new_pts[i][1]))
    return new_pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial configuration: 4x4 grid
    current_points = _generate_initial_points()
    is_valid, current_score = _evaluate(current_points)
    if not is_valid:
        # Fallback to simple grid if initial is somehow invalid (shouldn't be)
        current_points = [[x, y] for x in range(4) for y in range(4)]
        is_valid, current_score = _evaluate(current_points)
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # If we already have a high score, we can be more conservative
    # But 10 is the grid score. We hope to beat it.
    
    no_improve_steps = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.7:
            # Move one point slightly
            idx = random.randint(0, 15)
            ox, oy = current_points[idx]
            # Small step
            nx = ox + random.randint(-5, 5)
            ny = oy + random.randint(-5, 5)
            if abs(nx) <= 1000 and abs(ny) <= 1000:
                current_points[idx] = [nx, ny]
        elif move_type < 0.85:
            # Move one point significantly
            idx = random.randint(0, 15)
            nx = random.randint(-100, 100)
            ny = random.randint(-100, 100)
            current_points[idx] = [nx, ny]
        else:
            # Swap two points
            i, j = random.sample(range(16), 2)
            current_points[i], current_points[j] = current_points[j], current_points[i]
        
        is_valid, new_score = _evaluate(current_points)
        
        if is_valid:
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
                no_improve_steps = 0
            elif new_score == best_score:
                no_improve_steps += 1
            else:
                # Revert if worse, unless we want to explore
                # Simple hill climbing: revert
                # We need to revert the move. 
                # It's easier to keep a copy of the previous state or revert manually.
                # Since we didn't save the previous state explicitly for all move types,
                # let's just revert by re-evaluating? No, that's slow.
                # Let's store the previous state.
                pass # This logic is flawed if we don't store prev state.
        else:
            # Invalid, revert
            pass

    # The above loop is inefficient because reverting is not handled cleanly.
    # Let's rewrite the search loop properly.
    
    return best_points

# Rewritten construct with proper state management

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Initial configuration: 4x4 grid
    # This has 10 four-point lines.
    current_points = [[x, y] for x in range(-1, 3) for y in range(-1, 3)]
    is_valid, current_score = _evaluate(current_points)
    if not is_valid:
        current_points = [[x, y] for x in range(4) for y in range(4)]
        is_valid, current_score = _evaluate(current_points)
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 100.0  # Initial temperature
    cooling_rate = 0.9999
    min_T = 0.1
    
    # Track time for cooling
    steps_since_improve = 0
    max_steps_no_improve = 5000
    
    while time.time() < deadline:
        prev_points = [p[:] for p in current_points]
        
        # Generate a move
        # We want to explore the space of integer coordinates.
        # Moves can be small local adjustments or larger jumps.
        move_type = random.random()
        
        if move_type < 0.7:
            # Move one point slightly
            idx = random.randint(0, 15)
            ox, oy = current_points[idx]
            # Scale step size with temperature? Or keep fixed?
            # Fixed small steps are good for fine-tuning.
            step = random.randint(-3, 3)
            nx = ox + step
            ny = oy + random.randint(-3, 3)
            if abs(nx) <= 1000 and abs(ny) <= 1000:
                current_points[idx] = [nx, ny]
        elif move_type < 0.9:
            # Move one point significantly
            idx = random.randint(0, 15)
            nx = random.randint(-20, 20)
            ny = random.randint(-20, 20)
            current_points[idx] = [nx, ny]
        else:
            # Swap two points (preserves set of coordinates, changes assignment)
            # This is less useful if we don't care about labels, but can help if
            # there's symmetry. However, since we just evaluate the set of points,
            # swapping is equivalent to doing nothing in terms of the set of lines,
            # UNLESS we treat points as distinguishable in a way that affects line counting?
            # No, line counting depends only on the set of points. Swapping indices
            # does not change the set of points. So this move is useless for the objective.
            # Instead, let's do a "kick" move: move two points.
            i, j = random.sample(range(16), 2)
            current_points[i][0] += random.randint(-10, 10)
            current_points[i][1] += random.randint(-10, 10)
            current_points[j][0] += random.randint(-10, 10)
            current_points[j][1] += random.randint(-10, 10)
            
            # Clamp
            for p in current_points:
                p[0] = max(-1000, min(1000, p[0]))
                p[1] = max(-1000, min(1000, p[1]))
        
        is_valid, new_score = _evaluate(current_points)
        
        if is_valid:
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
                steps_since_improve = 0
                # Optionally reset temperature or keep cooling
            elif new_score == current_score:
                # Accept
                pass
            else:
                # Accept worse moves with probability
                delta = current_score - new_score
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    pass # Accept
                else:
                    current_points = prev_points
        else:
            # Invalid configuration (5 collinear or duplicate)
            # Revert
            current_points = prev_points
            
        # Cooling
        T *= cooling_rate
        
        # If stuck, restart from best
        steps_since_improve += 1
        if steps_since_improve > max_steps_no_improve:
            current_points = [p[:] for p in best_points]
            current_score = best_score
            T = 50.0 # Reset temperature
            steps_since_improve = 0
            
    return best_points

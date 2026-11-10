"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from collections import Counter

def _canonical_line(p1, p2):
    """Return a canonical tuple (A, B, C) for the line passing through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    # Normalize by GCD
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    # Ensure a unique sign representation
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _line_membership(points):
    """
    For each pair of points, compute the canonical line and count how many
    points lie on each line. Returns (line_to_points, point_to_lines) where
    line_to_points maps line -> set of point indices, and point_to_lines maps
    point index -> set of lines.
    """
    n = len(points)
    line_to_points = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            line = _canonical_line(points[i], points[j])
            if line not in line_to_points:
                line_to_points[line] = set()
            line_to_points[line].add(i)
            line_to_points[line].add(j)
    
    # For each line, find all points that lie on it (not just the pairs that defined it)
    for line in list(line_to_points.keys()):
        A, B, C = line
        for k in range(n):
            x, y = points[k]
            if A * x + B * y + C == 0:
                line_to_points[line].add(k)
    
    # Build point_to_lines
    point_to_lines = [set() for _ in range(n)]
    for line, pts in line_to_points.items():
        for idx in pts:
            point_to_lines[idx].add(line)
    
    return line_to_points, point_to_lines

def _score_from_membership(line_to_points):
    """
    Calculate the score from line membership: number of lines with exactly 4 points.
    Returns -1 if any line has >= 5 points (invalid).
    """
    score = 0
    for pts in line_to_points.values():
        count = len(pts)
        if count >= 5:
            return -1
        if count == 4:
            score += 1
    return score

def _score(points):
    """
    Calculate the score: number of lines with exactly 4 points.
    Returns -1 if any line has >= 5 points (invalid).
    """
    line_to_points, _ = _line_membership(points)
    return _score_from_membership(line_to_points)

def _score_and_lines(points):
    """
    Return (score, line_to_points) for a set of points.
    Score is -1 if invalid.
    """
    line_to_points, _ = _line_membership(points)
    score = _score_from_membership(line_to_points)
    return score, line_to_points

def _try_move(points, idx, new_x, new_y):
    """
    Try moving point idx to (new_x, new_y). Returns (new_score, new_points) if valid,
    or (None, None) if invalid (collision or out of bounds).
    """
    if new_x < -1000 or new_x > 1000 or new_y < -1000 or new_y > 1000:
        return None, None
    
    # Check collision
    for i, p in enumerate(points):
        if i != idx and p[0] == new_x and p[1] == new_y:
            return None, None
    
    old_x, old_y = points[idx]
    points[idx][0] = new_x
    points[idx][1] = new_y
    
    score = _score(points)
    
    if score == -1:
        points[idx][0] = old_x
        points[idx][1] = old_y
        return None, None
    
    new_points = [p[:] for p in points]
    return score, new_points

def _find_3_point_lines(line_to_points):
    """Find lines with exactly 3 points from the line membership."""
    three_point_lines = []
    for line, pts in line_to_points.items():
        if len(pts) == 3:
            three_point_lines.append((line, pts))
    return three_point_lines

def _suggest_point_on_3line(points, line_to_points):
    """
    Given a line with exactly 3 points, suggest a new position for one of the
    points on that line that might create a 4-point line without creating a 5-point line.
    Returns a list of candidate (idx, new_x, new_y) tuples.
    """
    candidates = []
    for line, pts in line_to_points.items():
        if len(pts) != 3:
            continue
        A, B, C = line
        for idx in pts:
            x, y = points[idx]
            # Try moving this point slightly off the line to see if it helps
            # Actually, we want to move OTHER points onto this line, not move points off it.
            # Instead, let's find lines with 2 points and suggest adding a 3rd.
            # This is more complex. Let's just do random moves for now.
            pass
    return candidates

def _generate_initial_config():
    """Generate a diverse initial configuration."""
    configs = []
    
    # 4x4 grid
    configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # 4x4 grid with some jitter
    base = [[x, y] for x in range(4) for y in range(4)]
    jittered = [p[:] for p in base]
    for i in range(4):
        idx = random.randint(0, 15)
        jittered[idx][0] += random.randint(-1, 1)
        jittered[idx][1] += random.randint(-1, 1)
    # Ensure distinctness
    seen = set()
    valid = []
    for p in jittered:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            valid.append(p)
    while len(valid) < 16:
        x = random.randint(-2, 5)
        y = random.randint(-2, 5)
        if (x, y) not in seen:
            seen.add((x, y))
            valid.append([x, y])
    configs.append(valid)
    
    # Random points in a larger area
    seen = set()
    pts = []
    while len(pts) < 16:
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append([x, y])
    configs.append(pts)
    
    # 2x8 grid
    configs.append([[x, y] for x in range(2) for y in range(8)])
    
    # 8x2 grid
    configs.append([[x, y] for x in range(8) for y in range(2)])
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    # Try multiple initial configurations
    initial_configs = _generate_initial_config()
    
    best_points = None
    best_score = -1
    
    for config in initial_configs:
        score = _score(config)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in config]
    
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(4)]
        best_score = _score(best_points)
    
    # Simulated Annealing with better moves
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    T = 15.0
    cooling = 0.9995
    min_T = 0.5
    
    step_count = 0
    
    while time.time() < deadline:
        step_count += 1
        
        if T < min_T:
            # Reheat and restart from best
            T = 15.0
            current_points = [p[:] for p in best_points]
            current_score = best_score
            # Add some perturbation
            for _ in range(random.randint(1, 4)):
                idx = random.randint(0, 15)
                current_points[idx][0] += random.randint(-3, 3)
                current_points[idx][1] += random.randint(-3, 3)
                current_points[idx][0] = max(-1000, min(1000, current_points[idx][0]))
                current_points[idx][1] = max(-1000, min(1000, current_points[idx][1]))
            new_score = _score(current_points)
            if new_score == -1:
                current_points = [p[:] for p in best_points]
                current_score = best_score
            else:
                current_score = new_score
        
        # Choose move type with bias toward targeted moves
        move_type = random.random()
        
        if move_type < 0.3:
            # Targeted move: pick a point and try to move it to a position
            # that might create a 4-point line
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            
            # Get current line membership
            line_to_points, point_to_lines = _line_membership(current_points)
            
            # Find lines with 3 points that don't include idx
            target_lines = []
            for line, pts in line_to_points.items():
                if len(pts) == 3 and idx not in pts:
                    target_lines.append(line)
            
            if target_lines:
                # Pick a target line and try to move idx onto it
                line = random.choice(target_lines)
                A, B, C = line
                # We need to find a point on this line that is not already occupied
                # and is close to the current position
                # Try several points along the line
                found = False
                for t in range(-20, 21):
                    if A != 0:
                        # y = (-A*x - C) / B if B != 0, or x = (-B*y - C) / A if A != 0
                        if B != 0:
                            # x is free, y is determined
                            x = old_x + t
                            y_num = -A * x - C
                            if y_num % B == 0:
                                y = y_num // B
                                if -1000 <= x <= 1000 and -1000 <= y <= 1000:
                                    new_score, new_pts = _try_move(current_points, idx, x, y)
                                    if new_score is not None:
                                        delta = new_score - current_score
                                        if delta > 0:
                                            current_points = new_pts
                                            current_score = new_score
                                            if current_score > best_score:
                                                best_score = current_score
                                                best_points = [p[:] for p in current_points]
                                            found = True
                                            break
                                        elif random.random() < math.exp(delta / T):
                                            current_points = new_pts
                                            current_score = new_score
                                            found = True
                                            break
                        else:
                            # A != 0, B == 0: vertical line x = -C/A
                            x = -C // A if A != 0 else 0
                            y = old_y + t
                            if -1000 <= x <= 1000 and -1000 <= y <= 1000:
                                new_score, new_pts = _try_move(current_points, idx, x, y)
                                if new_score is not None:
                                    delta = new_score - current_score
                                    if delta > 0:
                                        current_points = new_pts
                                        current_score = new_score
                                        if current_score > best_score:
                                            best_score = current_score
                                            best_points = [p[:] for p in current_points]
                                        found = True
                                        break
                                    elif random.random() < math.exp(delta / T):
                                        current_points = new_pts
                                        current_score = new_score
                                        found = True
                                        break
                    elif B != 0:
                        # A == 0, B != 0: horizontal line y = -C/B
                        y = -C // B if B != 0 else 0
                        x = old_x + t
                        if -1000 <= x <= 1000 and -1000 <= y <= 1000:
                            new_score, new_pts = _try_move(current_points, idx, x, y)
                            if new_score is not None:
                                delta = new_score - current_score
                                if delta > 0:
                                    current_points = new_pts
                                    current_score = new_score
                                    if current_score > best_score:
                                        best_score = current_score
                                        best_points = [p[:] for p in current_points]
                                    found = True
                                    break
                                elif random.random() < math.exp(delta / T):
                                    current_points = new_pts
                                    current_score = new_score
                                    found = True
                                    break
                if found:
                    T *= cooling
                    continue
            
            # If no targeted move found, fall through to random move
            
        if move_type < 0.6:
            # Random 1-point move
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            
            # Random jump with varying range
            jump_range = random.choice([5, 10, 20, 50])
            new_x = old_x + random.randint(-jump_range, jump_range)
            new_y = old_y + random.randint(-jump_range, jump_range)
            
            new_score, new_pts = _try_move(current_points, idx, new_x, new_y)
            if new_score is not None:
                delta = new_score - current_score
                if delta > 0:
                    current_points = new_pts
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                else:
                    if random.random() < math.exp(delta / T):
                        current_points = new_pts
                        current_score = new_score
        else:
            # 2-point swap
            i1, i2 = random.sample(range(16), 2)
            p1 = current_points[i1][:]
            p2 = current_points[i2][:]
            
            current_points[i1] = p2
            current_points[i2] = p1
            new_score = _score(current_points)
            
            if new_score == -1:
                current_points[i1] = p1
                current_points[i2] = p2
                continue
            
            delta = new_score - current_score
            if delta > 0:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                if random.random() < math.exp(delta / T):
                    current_score = new_score
                else:
                    current_points[i1] = p1
                    current_points[i2] = p2
        
        T *= cooling
        
        if time.time() > deadline:
            break
    
    # Final polish: exhaustive local search on best_points
    current_points = [p[:] for p in best_points]
    current_score = best_score
    no_improve = 0
    max_no_improve = 10000
    
    while time.time() < deadline - 5.0 and no_improve < max_no_improve:
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Try all small perturbations
        found_improvement = False
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx == 0 and dy == 0:
                    continue
                new_x = old_x + dx
                new_y = old_y + dy
                if new_x < -1000 or new_x > 1000 or new_y < -1000 or new_y > 1000:
                    continue
                
                valid_move = True
                for i, p in enumerate(current_points):
                    if i != idx and p[0] == new_x and p[1] == new_y:
                        valid_move = False
                        break
                if not valid_move:
                    continue
                
                current_points[idx][0] = new_x
                current_points[idx][1] = new_y
                new_score = _score(current_points)
                
                if new_score != -1 and new_score > current_score:
                    current_score = new_score
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
                    no_improve = 0
                    found_improvement = True
                    break
                else:
                    current_points[idx][0] = old_x
                    current_points[idx][1] = old_y
            
            if found_improvement:
                break
        
        if not found_improvement:
            no_improve += 1
    
    return best_points

# EVOLVE-BLOCK-END

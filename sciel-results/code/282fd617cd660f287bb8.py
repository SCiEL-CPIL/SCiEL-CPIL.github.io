import itertools
import time
import random
from math import gcd
from collections import defaultdict

def score_points(points):
    """
    Calculate the number of lines containing exactly 4 points.
    Also returns a flag if any line has 5 or more points.
    """
    n = len(points)
    line_counts = {}
    
    # Pre-compute coordinates for speed
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    for i in range(n):
        x1, y1 = xs[i], ys[i]
        for j in range(i + 1, n):
            x2, y2 = xs[j], ys[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            
            # Canonical form
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
                
            # Line constant c: dy*x - dx*y = c
            # Using point (x1, y1)
            c = dy * x1 - dx * y1
            
            key = (dx, dy, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    exact_4 = 0
    has_five = False
    
    # C(4,2) = 6
    # C(5,2) = 10
    # C(6,2) = 15
    for count in line_counts.values():
        if count == 6:
            exact_4 += 1
        elif count >= 10:
            has_five = True
            break
            
    return exact_4, has_five

def get_line_info(points):
    """
    Returns:
    - lines_3: List of lists of indices for lines with exactly 3 points.
    - lines_4: List of lists of indices for lines with exactly 4 points.
    - all_lines: Dict mapping line_key to set of indices.
    """
    n = len(points)
    line_indices = defaultdict(set)
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    for i in range(n):
        x1, y1 = xs[i], ys[i]
        for j in range(i + 1, n):
            x2, y2 = xs[j], ys[j]
            dx = x2 - x1
            dy = y2 - y1
            
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
                
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
                
            c = dy * x1 - dx * y1
            key = (dx, dy, c)
            
            line_indices[key].add(i)
            line_indices[key].add(j)
            
    lines_3 = []
    lines_4 = []
    
    for key, idx_set in line_indices.items():
        if len(idx_set) == 3:
            lines_3.append(list(idx_set))
        elif len(idx_set) == 4:
            lines_4.append(list(idx_set))
            
    return lines_3, lines_4, line_indices

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initial state: 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score, _ = score_points(best_points)
    
    # Simulated Annealing Parameters
    T = 10.0
    cooling_rate = 0.995
    min_T = 0.01
    
    current_points = [p[:] for p in best_points]
    current_score, current_has_five = score_points(current_points)
    
    if current_has_five:
        # Should not happen for 4x4 grid, but just in case
        current_score = 0
        # Perturb until valid
        while current_has_five and time.time() < deadline:
            idx = random.randint(0, 15)
            current_points[idx][0] += random.randint(-10, 10)
            current_points[idx][1] += random.randint(-10, 10)
            if abs(current_points[idx][0]) > 1000: current_points[idx][0] = random.randint(-100, 100)
            if abs(current_points[idx][1]) > 1000: current_points[idx][1] = random.randint(-100, 100)
            current_score, current_has_five = score_points(current_points)
            if current_has_five:
                current_score = -100 # Heavy penalty in logic, but we reject invalids anyway
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        if T < min_T:
            # Restart with higher T or break if stuck
            if no_improve_count > 100:
                break
            T = 10.0
            no_improve_count = 0
            # Perturb current to escape
            for _ in range(3):
                idx = random.randint(0, 15)
                current_points[idx][0] += random.randint(-20, 20)
                current_points[idx][1] += random.randint(-20, 20)
                if abs(current_points[idx][0]) > 1000: current_points[idx][0] = random.randint(-100, 100)
                if abs(current_points[idx][1]) > 1000: current_points[idx][1] = random.randint(-100, 100)
            current_score, current_has_five = score_points(current_points)
            if current_has_five:
                # Revert to best if invalid
                current_points = [p[:] for p in best_points]
                current_score = best_score
                continue
        else:
            T *= cooling_rate
        
        # Generate Candidate Moves
        candidates = []
        
        # 1. Line Completion Moves (3 -> 4)
        lines_3, lines_4, _ = get_line_info(current_points)
        
        used_coords = set(tuple(p) for p in current_points)
        
        for line_indices in lines_3:
            # Get points
            p1 = current_points[line_indices[0]]
            p2 = current_points[line_indices[1]]
            p3 = current_points[line_indices[2]]
            
            # Determine line direction
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
                
            # Parameterize points relative to p1
            t_vals = []
            for p in [p1, p2, p3]:
                if dx != 0:
                    if (p[0] - p1[0]) % dx == 0:
                        t = (p[0] - p1[0]) // dx
                        if dy != 0:
                            if (p[1] - p1[1]) % dy == 0 and (p[1] - p1[1]) // dy == t:
                                t_vals.append(t)
                        else:
                            t_vals.append(t)
                else:
                    if dy != 0:
                        if (p[1] - p1[1]) % dy == 0:
                            t_vals.append((p[1] - p1[1]) // dy)
            
            if len(t_vals) < 3:
                continue
                
            # Look for nearby t values
            min_t = min(t_vals)
            max_t = max(t_vals)
            
            # Try extending a bit
            for t in range(min_t - 2, max_t + 3):
                if t in t_vals:
                    continue
                nx = p1[0] + t * dx
                ny = p1[1] + t * dy
                if abs(nx) <= 1000 and abs(ny) <= 1000:
                    if (nx, ny) not in used_coords:
                        # Candidate: Move some point to (nx, ny)
                        candidates.append((nx, ny))
        
        # 2. Random Moves
        for _ in range(5):
            idx = random.randint(0, 15)
            nx = current_points[idx][0] + random.randint(-10, 10)
            ny = current_points[idx][1] + random.randint(-10, 10)
            if abs(nx) <= 1000 and abs(ny) <= 1000:
                candidates.append((nx, ny))
        
        # 3. Evaluate Candidates
        best_candidate_move = None
        best_candidate_score = current_score
        best_candidate_points = None
        best_candidate_delta = 0
        
        # Deduplicate candidates
        unique_candidates = list(set(candidates))
        
        # Limit number of evaluations per step to ensure speed
        if len(unique_candidates) > 50:
            unique_candidates = random.sample(unique_candidates, 50)
        
        for nx, ny in unique_candidates:
            if (nx, ny) in used_coords:
                continue
                
            # Try moving each point to (nx, ny)
            # Optimization: Only try moving points that are "loose" or random?
            # For now, try a random subset of points or all?
            # 16 points * 50 candidates = 800 evaluations.
            # Each eval is ~120 gcds. 800 * 120 = 96k gcds.
            # This might be too slow if done every iteration.
            # Let's restrict: Only try moving points that are NOT part of a 4-line?
            # Or just pick 4 random points to move.
            
            # Heuristic: Moving a point that is on a 3-line is good.
            # Moving a point that is on a 4-line is risky (breaks 4-line).
            # Let's try moving all 16, but break early if we find a good one?
            # No, SA needs the best move.
            
            # To speed up, we only try moving points that are "candidates for movement"
            # i.e., points involved in 3-lines or random.
            # Simpler: Just try moving a random point for each candidate?
            # No, that misses the best move.
            
            # Compromise: For each candidate (nx, ny), try moving the point that is
            # "closest" to (nx, ny) in some metric? No, that's not right.
            
            # Let's just iterate all 16 points.
            for idx in range(16):
                if current_points[idx][0] == nx and current_points[idx][1] == ny:
                    continue
                    
                new_points = [p[:] for p in current_points]
                new_points[idx] = [nx, ny]
                
                # Check distinctness
                # If (nx,ny) was in used_coords, we skipped.
                # But what if moving idx to (nx,ny) collides with another point?
                # used_coords contained all points. If (nx,ny) not in used_coords,
                # it is not occupied by ANY point. So new_points is distinct.
                
                s, has_five = score_points(new_points)
                
                if has_five:
                    continue
                    
                delta = s - current_score
                
                # SA Acceptance
                if delta > 0:
                    # Always accept improvement
                    if best_candidate_move is None or s > best_candidate_score:
                        best_candidate_score = s
                        best_candidate_move = (idx, nx, ny)
                        best_candidate_points = new_points
                        best_candidate_delta = delta
                else:
                    # Accept with probability exp(delta / T)
                    if random.random() < pow(2.71828, delta / T):
                        if best_candidate_move is None or s > best_candidate_score:
                            best_candidate_score = s
                            best_candidate_move = (idx, nx, ny)
                            best_candidate_points = new_points
                            best_candidate_delta = delta
        
        # Apply Best Move
        if best_candidate_move:
            idx, nx, ny = best_candidate_move
            current_points[idx][0] = nx
            current_points[idx][1] = ny
            current_score = best_candidate_score
            no_improve_count = 0
            
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            no_improve_count += 1
            
    # Final Validation
    final = [p[:] for p in best_points]
    assert len(set(tuple(p) for p in final)) == 16
    assert all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in final)
    s, f = score_points(final)
    assert f == 0, f"Five collinear found: {f}"
    
    return final


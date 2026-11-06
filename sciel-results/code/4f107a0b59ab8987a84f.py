import itertools
import time
import random
from collections import Counter
from math import gcd

def count_exact_four_lines(points):
    """Count lines containing exactly 4 of the 16 points."""
    n = len(points)
    if n < 4:
        return 0, 0
    line_counts = {}
    # For each pair, determine the line (normalized direction + offset)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize direction
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            # Line key: direction + a value that identifies the line
            # Use cross product invariant: for line through (x1,y1) with direction (dx,dy)
            # The line is: dy*(x - x1) - dx*(y - y1) = 0
            # So key = (dx, dy, dy*x1 - dx*y1) normalized
            c = dy * x1 - dx * y1
            # Normalize the key so that (dx,dy) is in canonical form
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
                c = -c
            key = (dx, dy, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    # Each line with k points is counted C(k,2) times
    # We want lines with exactly 4 points
    four_count = 0
    five_or_more = 0
    for key, pair_count in line_counts.items():
        # pair_count = C(k,2) where k is number of points on line
        # Find k such that C(k,2) = pair_count
        # C(4,2) = 6, C(5,2) = 10
        if pair_count == 6:
            four_count += 1
        elif pair_count >= 10:
            five_or_more += 1
    return four_count, five_or_more

def get_lines_with_counts(points):
    """
    Returns a dictionary mapping line_key to count of points on that line.
    Also returns a dictionary mapping line_key to list of point indices.
    """
    n = len(points)
    line_counts = {}
    line_points = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
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
            line_counts[key] = line_counts.get(key, 0) + 1
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    # Convert sets to lists for easier access
    line_points_list = {k: list(v) for k, v in line_points.items()}
    return line_counts, line_points_list

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_points = None
    best_score = 0
    best_five_count = 0
    
    # Start with 4x4 grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    score, five = count_exact_four_lines(grid)
    best_score = score
    best_points = [p[:] for p in grid]
    best_five_count = five
    
    # Local search with line completion moves
    random.seed(42)
    
    # We will perform multiple restarts
    for restart in range(100):
        if time.time() > deadline:
            break
        
        if restart == 0:
            cur = [p[:] for p in best_points]
        else:
            # Random perturbation of best to escape local optima
            cur = [p[:] for p in best_points]
            # Perturb 3-5 points significantly
            num_perturb = random.randint(3, 5)
            for _ in range(num_perturb):
                idx = random.randint(0, 15)
                cur[idx][0] += random.randint(-20, 20)
                cur[idx][1] += random.randint(-20, 20)
                if abs(cur[idx][0]) > 1000 or abs(cur[idx][1]) > 1000:
                    cur[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
        
        cur_score, cur_five = count_exact_four_lines(cur)
        if cur_five > 0:
            # If we have 5 collinear, we must fix it first, but for now just skip if not improved
            # Actually, if we start from a valid best, perturbations might create 5s.
            # We should probably penalize 5s heavily or just restart if we get 5s.
            # Let's just continue, but don't update best if 5s exist.
            pass

        improved = True
        no_improve_count = 0
        max_no_improve = 50
        
        while improved and time.time() < deadline and no_improve_count < max_no_improve:
            improved = False
            no_improve_count += 1
            
            # 1. Generate Line Completion Moves
            # Get lines with exactly 3 points
            line_counts, line_points = get_lines_with_counts(cur)
            
            candidate_targets = set()
            
            for key, count in line_counts.items():
                # We want to find lines with exactly 3 points to try to complete to 4
                # C(3,2) = 3 pairs
                if count == 3:
                    pts_idx = line_points[key]
                    if len(pts_idx) != 3:
                        continue
                    
                    p1 = cur[pts_idx[0]]
                    p2 = cur[pts_idx[1]]
                    p3 = cur[pts_idx[2]]
                    
                    dx, dy, c = key
                    # Direction vector is (dx, dy)
                    # Base point p1
                    x1, y1 = p1
                    
                    # Find t values for the three points
                    t_vals = []
                    for p in [p1, p2, p3]:
                        if dx != 0:
                            if (p[0] - x1) % dx == 0:
                                t = (p[0] - x1) // dx
                                if dy != 0 and (p[1] - y1) % dy == 0:
                                    if (p[1] - y1) // dy == t:
                                        t_vals.append(t)
                                elif dy == 0:
                                    t_vals.append(t)
                        else:
                            # dx == 0
                            if dy != 0:
                                if (p[1] - y1) % dy == 0:
                                    t = (p[1] - y1) // dy
                                    t_vals.append(t)
                            else:
                                # Degenerate, shouldn't happen for distinct points
                                pass
                    
                    if len(t_vals) < 3:
                        continue
                        
                    min_t = min(t_vals)
                    max_t = max(t_vals)
                    
                    # Try extending the line in both directions
                    # We look for the next integer point on the line that is not occupied
                    # and is within bounds.
                    # Since dx, dy are coprime, all integer points on the line are p1 + t*(dx, dy)
                    
                    used_coords = set(tuple(p) for p in cur)
                    
                    # Check t values from min_t - 5 to max_t + 5
                    for t in range(min_t - 5, max_t + 6):
                        if t in t_vals:
                            continue
                        nx = x1 + t * dx
                        ny = y1 + t * dy
                        if abs(nx) <= 1000 and abs(ny) <= 1000:
                            if (nx, ny) not in used_coords:
                                candidate_targets.add((nx, ny))
            
            # 2. Random perturbation targets
            # Add some random nearby points as targets to allow escaping
            for _ in range(10):
                idx = random.randint(0, 15)
                nx = cur[idx][0] + random.randint(-5, 5)
                ny = cur[idx][1] + random.randint(-5, 5)
                if abs(nx) <= 1000 and abs(ny) <= 1000:
                    candidate_targets.add((nx, ny))
            
            # 3. Try moving points to these targets
            # To be efficient, we only try moving points that are "involved" or random points
            # But moving any point to a target is a valid move.
            # We iterate over targets, then over points?
            # Or iterate over points, then over targets?
            # Let's iterate over points. For each point, try moving it to each target.
            # This is 16 * len(targets).
            
            found_improvement = False
            for idx in range(16):
                if time.time() > deadline:
                    break
                if found_improvement:
                    break
                    
                orig_x, orig_y = cur[idx]
                
                for (nx, ny) in candidate_targets:
                    if nx == orig_x and ny == orig_y:
                        continue
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    
                    # Check if target is occupied by another point
                    # We need to ensure new point is distinct from all others
                    # Since we are moving point idx, we just need (nx,ny) != cur[j] for j != idx
                    # And (nx,ny) != cur[idx] (already checked)
                    
                    # Quick check: is (nx,ny) in the set of all points except cur[idx]?
                    # Constructing a set every time is expensive.
                    # Let's just create the new list and check uniqueness.
                    
                    new_pts = [p[:] for p in cur]
                    new_pts[idx] = [nx, ny]
                    
                    # Check distinctness
                    # 16 points, checking uniqueness is O(16^2) = 256 ops.
                    # 16 * 100 targets * 256 = 400k ops per iteration. Might be slow.
                    # Optimization: Pre-compute set of coordinates.
                    # But we need to remove cur[idx] and add (nx,ny).
                    
                    # Let's do a simple check first
                    valid = True
                    for j in range(16):
                        if j == idx:
                            continue
                        if new_pts[j][0] == nx and new_pts[j][1] == ny:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    s, f = count_exact_four_lines(new_pts)
                    
                    # We prefer valid configurations (f=0).
                    # If f>0, we might still accept if s is much higher?
                    # The task says "no five collinear". So f must be 0.
                    if f == 0 and s > cur_score:
                        cur = new_pts
                        cur_score = s
                        found_improvement = True
                        improved = True
                        break # Break inner loop for this idx
            
            if found_improvement:
                no_improve_count = 0
                if cur_score > best_score:
                    best_score = cur_score
                    best_points = [p[:] for p in cur]
                    best_five_count = cur_five
            else:
                # No improvement in this pass, try a larger random jump if stuck
                if no_improve_count > 10:
                    idx = random.randint(0, 15)
                    cur[idx][0] += random.randint(-10, 10)
                    cur[idx][1] += random.randint(-10, 10)
                    if abs(cur[idx][0]) > 1000: cur[idx][0] = random.randint(-100, 100)
                    if abs(cur[idx][1]) > 1000: cur[idx][1] = random.randint(-100, 100)
                    cur_score, cur_five = count_exact_four_lines(cur)
                    if cur_five > 0:
                        # Revert if invalid
                        cur[idx][0] -= random.randint(-10, 10) # This is messy, better to just break
                        # Just break the local search and restart
                        improved = False
                        break

    # Final validation
    final = [p[:] for p in best_points]
    assert len(set(tuple(p) for p in final)) == 16
    assert all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in final)
    s, f = count_exact_four_lines(final)
    assert f == 0, f"Five collinear found: {f}"
    
    return final

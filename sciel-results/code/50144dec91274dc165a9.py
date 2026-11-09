"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _count_ordinary_lines(pts):
    """Count lines containing exactly two of the points."""
    n = len(pts)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[j][0] - pts[i][0]
            dy = pts[j][1] - pts[i][1]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                key = (0, 0, pts[i][0])
            elif dy == 0:
                key = (0, 1, pts[i][1])
            else:
                if dy < 0:
                    dx, dy = -dx, -dy
                g = gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                # Normal line form: dy*x - dx*y = c
                c = dy * pts[i][0] - dx * pts[i][1]
                key = (1, dx, dy, c)
            lines[key] = lines.get(key, 0) + 1
    return sum(1 for c in lines.values() if c == 1)

def _evaluate(points):
    """Score: number of ordinary lines (lower is better)."""
    return _count_ordinary_lines(points)

def _not_all_collinear(points):
    n = len(points)
    if n < 3:
        return True
    for j in range(1, n):
        for k in range(j + 1, n):
            x0, y0 = points[0]
            x1, y1 = points[j]
            x2, y2 = points[k]
            if (x1 - x0) * (y2 - y0) != (y1 - y0) * (x2 - x0):
                return True
    return False

def _generate_hexagonal_candidates(scale=1):
    """Generate 12 points on a hexagonal lattice pattern."""
    import math
    points = []
    s = scale * 2
    
    # 6 hexagon vertices
    for i in range(6):
        angle = i * math.pi / 3
        x = round(s * math.cos(angle))
        y = round(s * math.sin(angle))
        points.append([x, y])
    
    # Center
    points.append([0, 0])
    
    # 5 more points at half-scale offset positions
    for i in range(5):
        angle = i * math.pi / 3 + math.pi / 6
        x = round(s * 0.5 * math.cos(angle))
        y = round(s * 0.5 * math.sin(angle))
        points.append([x, y])
    
    seen = set()
    unique_points = []
    for p in points:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            unique_points.append(p)
    
    while len(unique_points) < 12:
        x = len(unique_points) - 6
        y = 0
        if (x, y) not in seen:
            seen.add((x, y))
            unique_points.append([x, y])
        else:
            unique_points.append([len(unique_points), 0])
    
    return unique_points[:12]

def _generate_grid_candidates():
    """Generate several grid-based candidate configurations."""
    candidates = []
    # 3x4 grid
    grid1 = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(grid1)
    # 4x3 grid
    grid2 = [[x, y] for x in range(3) for y in range(4)]
    candidates.append(grid2)
    # 2x6 grid
    grid3 = [[x, y] for x in range(6) for y in range(2)]
    candidates.append(grid3)
    # 6x2 grid
    grid4 = [[x, y] for x in range(2) for y in range(6)]
    candidates.append(grid4)
    # 3x4 with offset (shear)
    grid5 = [[x + y//2, y] for x in range(4) for y in range(3)]
    candidates.append(grid5)
    # 4x3 with offset
    grid6 = [[x + y//2, y] for x in range(3) for y in range(4)]
    candidates.append(grid6)
    return candidates

def _generate_pencil_candidates():
    """Generate configurations based on pencil of lines through a point."""
    candidates = []
    # 11 points on lines through origin at different slopes
    for angle_offset in [0, 1, 2]:
        pts = [[0, 0]]
        for i in range(11):
            angle = i * math.pi / 11 + angle_offset * 0.1
            x = round(10 * math.cos(angle))
            y = round(10 * math.sin(angle))
            if (x, y) not in [(p[0], p[1]) for p in pts]:
                pts.append([x, y])
        # Pad if needed
        while len(pts) < 12:
            pts.append([len(pts), 0])
        candidates.append(pts[:12])
    return candidates

def _generate_parabola_candidates():
    """Generate configurations with points on a parabola plus one off-conic point.
    
    Using Sylvester-Gallai extremal structure: place 11 points on y = a*x^2 + b*x + c
    and the 12th at a position that maximizes 3+ point collinearities.
    """
    candidates = []
    rng = random.Random(123)
    
    # Try different parabolas and off-point positions
    for a in [1, 2, 3, 4, 5, -1, -2]:
        for b in [-2, -1, 0, 1, 2]:
            for c in [0, 1, -1]:
                # Choose 11 x-values that create many collinearities
                # For parabola y = a*x^2 + b*x + c, points (x_i, y_i) are collinear
                # when the secant slopes match
                x_vals = list(range(-5, 6))[:11]
                
                pts = []
                for x in x_vals:
                    y = a * x * x + b * x + c
                    pts.append([x, y])
                
                # Find best 12th point: try positions that create 3+ collinearities
                best_off = None
                best_off_score = float('inf')
                
                # Try positions near the parabola
                for ox in range(-8, 9):
                    for oy in range(-15, 16):
                        test_pts = pts + [[ox, oy]]
                        # Check duplicates
                        seen = set()
                        dup = False
                        for p in test_pts:
                            if (p[0], p[1]) in seen:
                                dup = True
                                break
                            seen.add((p[0], p[1]))
                        if dup:
                            continue
                        if not _not_all_collinear(test_pts):
                            continue
                        score = _evaluate(test_pts)
                        if score < best_off_score:
                            best_off_score = score
                            best_off = [ox, oy]
                
                if best_off is not None:
                    final_pts = pts + [best_off]
                    if len(final_pts) == 12 and _not_all_collinear(final_pts):
                        candidates.append(final_pts)
    
    # Also try with shifted/scaled x-values for more variety
    for scale in [1, 2]:
        for offset in [-3, 0, 3]:
            for a in [1, 2]:
                x_vals = [offset + i * scale for i in range(11)]
                pts = [[x, a * x * x] for x in x_vals]
                
                best_off = None
                best_off_score = float('inf')
                
                for ox in range(offset - 5, offset + 16):
                    for oy in range(-20, 40):
                        test_pts = pts + [[ox, oy]]
                        seen = set()
                        dup = False
                        for p in test_pts:
                            if (p[0], p[1]) in seen:
                                dup = True
                                break
                            seen.add((p[0], p[1]))
                        if dup:
                            continue
                        if not _not_all_collinear(test_pts):
                            continue
                        score = _evaluate(test_pts)
                        if score < best_off_score:
                            best_off_score = score
                            best_off = [ox, oy]
                
                if best_off is not None:
                    final_pts = pts + [best_off]
                    if len(final_pts) == 12 and _not_all_collinear(final_pts):
                        candidates.append(final_pts)
    
    return candidates

def construct():
    start = time.time()
    deadline = start + 135  # safety margin before 150s kill
    
    n = 12
    rng = random.Random(42)
    
    # Generate multiple candidate configurations
    candidates = []
    
    # Hexagonal lattice configurations with different scales
    for scale in [1, 2, 3, 4, 5, 6, 7, 8]:
        pts = _generate_hexagonal_candidates(scale)
        if len(pts) == 12 and _not_all_collinear(pts):
            candidates.append(pts)
    
    # Grid-based configurations
    for grid_cand in _generate_grid_candidates():
        if len(grid_cand) == 12 and _not_all_collinear(grid_cand):
            candidates.append(grid_cand)
    
    # Pencil configurations
    for pencil_cand in _generate_pencil_candidates():
        if len(pencil_cand) == 12 and _not_all_collinear(pencil_cand):
            candidates.append(pencil_cand)
    
    # Parabola-based configurations
    for para_cand in _generate_parabola_candidates():
        if len(para_cand) == 12 and _not_all_collinear(para_cand):
            candidates.append(para_cand)
    
    # Evaluate all candidates and pick the best
    best_points = None
    best_score = float('inf')
    
    for cand in candidates:
        if len(cand) != 12:
            continue
        # Check for duplicates
        seen = set()
        valid = True
        for p in cand:
            if (p[0], p[1]) in seen:
                valid = False
                break
            seen.add((p[0], p[1]))
        if not valid:
            continue
        if not _not_all_collinear(cand):
            continue
        score = _evaluate(cand)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in cand]
    
    if best_points is None:
        # Fallback: simple non-collinear configuration
        best_points = [[i, i*i] for i in range(12)]
        best_score = _evaluate(best_points)
    
    # Now perform local search: try to improve by moving points to create 3+ point lines
    # Strategy: for each point, try moving it to a position that creates a new 3-point line
    # with two existing points
    
    def _find_good_positions(points, exclude_idx, limit=300):
        """Find positions that would create 3+ point collinearities.
        
        Enhanced to explore:
        1. Points on lines through pairs (extended range)
        2. Points on lines through triples (if they exist)
        3. Midpoints and other rational combinations
        4. Positions near existing clusters
        """
        good_positions = []
        seen_pos = set()
        existing_points = [(points[i][0], points[i][1]) for i in range(n) if i != exclude_idx]
        occupied = set(existing_points)
        
        # For each pair of existing points, find positions on the line through them
        for i in range(len(existing_points)):
            for j in range(i+1, len(existing_points)):
                x1, y1 = existing_points[i]
                x2, y2 = existing_points[j]
                
                dx = x2 - x1
                dy = y2 - y1
                
                if dx == 0 and dy == 0:
                    continue
                
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx_norm = dx // g
                dy_norm = dy // g
                
                # Try multiple positions along the line (both directions, various distances)
                for t in range(-15, 16):
                    if t == 0 or t == 1:  # Skip existing point positions
                        continue
                    new_x = x1 + t * dx_norm
                    new_y = y1 + t * dy_norm
                    
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    if (new_x, new_y) in occupied or (new_x, new_y) in seen_pos:
                        continue
                    
                    seen_pos.add((new_x, new_y))
                    good_positions.append([new_x, new_y])
                
                # Also try fractional positions: midpoint, quarter points
                # Midpoint
                mx = (x1 + x2) // 2
                my = (y1 + y2) // 2
                if (x1 + x2) % 2 == 0 and (y1 + y2) % 2 == 0:
                    if (mx, my) not in occupied and (mx, my) not in seen_pos:
                        seen_pos.add((mx, my))
                        good_positions.append([mx, my])
                
                # Points at 1/3 and 2/3 along the segment (if integer)
                if dx % 3 == 0 and dy % 3 == 0:
                    for frac_t in [1/3, 2/3]:
                        fx = x1 + int(frac_t * dx)
                        fy = y1 + int(frac_t * dy)
                        if (fx, fy) not in occupied and (fx, fy) not in seen_pos:
                            seen_pos.add((fx, fy))
                            good_positions.append([fx, fy])
        
        # For triples: if 3 points are collinear, try extending that line
        # Find all lines with 3+ points
        line_points = {}
        for i in range(len(existing_points)):
            for j in range(i+1, len(existing_points)):
                x1, y1 = existing_points[i]
                x2, y2 = existing_points[j]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                if dx == 0:
                    key = (0, 0, x1)
                elif dy == 0:
                    key = (0, 1, y1)
                else:
                    if dy < 0:
                        dx, dy = -dx, -dy
                    g = gcd(abs(dx), abs(dy))
                    dx //= g
                    dy //= g
                    c = dy * x1 - dx * y1
                    key = (1, dx, dy, c)
                if key not in line_points:
                    line_points[key] = []
                line_points[key].append(i)
        
        # For lines with 3+ points, try positions beyond the endpoints
        for key, indices in line_points.items():
            if len(indices) >= 3:
                # Get the line's direction
                if key[0] == 0:
                    if key[1] == 0:  # vertical
                        x_val = key[2]
                        ys = sorted(existing_points[i][1] for i in indices)
                        for t in range(-5, 6):
                            new_y = ys[0] + t * (ys[1] - ys[0]) if len(ys) > 1 else ys[0] + t
                            if abs(new_y) <= 10**9 and (x_val, new_y) not in occupied and (x_val, new_y) not in seen_pos:
                                seen_pos.add((x_val, new_y))
                                good_positions.append([x_val, new_y])
                    else:  # horizontal
                        y_val = key[2]
                        xs = sorted(existing_points[i][0] for i in indices)
                        for t in range(-5, 6):
                            new_x = xs[0] + t * (xs[1] - xs[0]) if len(xs) > 1 else xs[0] + t
                            if abs(new_x) <= 10**9 and (new_x, y_val) not in occupied and (new_x, y_val) not in seen_pos:
                                seen_pos.add((new_x, y_val))
                                good_positions.append([new_x, y_val])
                else:
                    dx, dy, c = key[1], key[2], key[3]
                    # Line: dy*x - dx*y = c
                    # Try positions along this line
                    # Pick a reference point
                    ref_i = indices[0]
                    rx, ry = existing_points[ref_i]
                    for t in range(-10, 11):
                        new_x = rx + t * dx
                        new_y = ry + t * dy
                        if abs(new_x) <= 10**9 and abs(new_y) <= 10**9:
                            if (new_x, new_y) not in occupied and (new_x, new_y) not in seen_pos:
                                seen_pos.add((new_x, new_y))
                                good_positions.append([new_x, new_y])
        
        # Try positions that create new 3-point lines with pairs
        # For each pair, try the "reflection" of a third point
        for i in range(len(existing_points)):
            for j in range(i+1, len(existing_points)):
                x1, y1 = existing_points[i]
                x2, y2 = existing_points[j]
                # Try reflections of other points across the midpoint
                for k in range(len(existing_points)):
                    if k == i or k == j:
                        continue
                    x3, y3 = existing_points[k]
                    # Reflection of (x3,y3) across midpoint of (x1,y1)-(x2,y2)
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    ref_x = 2 * mid_x - x3
                    ref_y = 2 * mid_y - y3
                    if ref_x == int(ref_x) and ref_y == int(ref_y):
                        ref_x = int(ref_x)
                        ref_y = int(ref_y)
                        if abs(ref_x) <= 10**9 and abs(ref_y) <= 10**9:
                            if (ref_x, ref_y) not in occupied and (ref_x, ref_y) not in seen_pos:
                                seen_pos.add((ref_x, ref_y))
                                good_positions.append([ref_x, ref_y])
        
        # Random positions near the centroid
        cx = sum(p[0] for p in existing_points) // len(existing_points)
        cy = sum(p[1] for p in existing_points) // len(existing_points)
        for _ in range(150):
            new_x = cx + rng.randint(-20, 20)
            new_y = cy + rng.randint(-20, 20)
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
            if (new_x, new_y) in occupied or (new_x, new_y) in seen_pos:
                continue
            seen_pos.add((new_x, new_y))
            good_positions.append([new_x, new_y])
        
        rng.shuffle(good_positions)
        return good_positions[:limit]
    
    # Local search with multiple restarts
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Start from current best or a random perturbation
        if restart == 0:
            current_points = [p[:] for p in best_points]
            current_score = best_score
        elif restart < 25:
            # Small perturbation
            current_points = [p[:] for p in best_points]
            for _ in range(2):
                idx = rng.randrange(n)
                current_points[idx] = [rng.randint(-20, 20), rng.randint(-20, 20)]
        elif restart < 50:
            # Medium perturbation
            current_points = [p[:] for p in best_points]
            for _ in range(4):
                idx = rng.randrange(n)
                current_points[idx] = [rng.randint(-30, 30), rng.randint(-30, 30)]
        elif restart < 75:
            # Perturb 2 points to new positions on existing lines
            current_points = [p[:] for p in best_points]
            for _ in range(2):
                idx = rng.randrange(n)
                # Pick a random pair and place on their line
                i1, i2 = rng.sample(range(n), 2)
                if i1 == idx or i2 == idx:
                    continue
                x1, y1 = current_points[i1]
                x2, y2 = current_points[i2]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx_n = dx // g
                dy_n = dy // g
                t = rng.randint(-5, 5)
                if t == 0 or t == 1:
                    continue
                new_x = x1 + t * dx_n
                new_y = y1 + t * dy_n
                if abs(new_x) <= 10**9 and abs(new_y) <= 10**9:
                    current_points[idx] = [new_x, new_y]
        else:
            # Large perturbation - restart from a random candidate
            if candidates:
                base = rng.choice(candidates)
                current_points = [p[:] for p in base]
            else:
                current_points = [[rng.randint(-20, 20), rng.randint(-20, 20)] for _ in range(n)]
        
        # Ensure validity
        if not _not_all_collinear(current_points):
            continue
        
        # Check for duplicates
        seen = set()
        valid = True
        for p in current_points:
            if (p[0], p[1]) in seen:
                valid = False
                break
            seen.add((p[0], p[1]))
        if not valid:
            continue
        
        current_score = _evaluate(current_points)
        if current_score > best_score + 3:
            continue  # Skip if too far from best
        
        # Hill climbing: iteratively move points to improve score
        no_improve_count = 0
        max_no_improve = 18
        
        while no_improve_count < max_no_improve and time.time() < deadline:
            improved_this_pass = False
            
            for idx in range(n):
                if time.time() > deadline:
                    break
                
                old_x, old_y = current_points[idx]
                
                # Get candidate positions
                candidates_pos = _find_good_positions(current_points, idx)
                
                # Also add the original position
                candidates_pos.append([old_x, old_y])
                
                # Try each candidate
                for new_x, new_y in candidates_pos:
                    if new_x == old_x and new_y == old_y:
                        continue
                    
                    # Check bounds
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    
                    # Check for duplicate
                    is_dup = False
                    for i, (px, py) in enumerate(current_points):
                        if i != idx and px == new_x and py == new_y:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    
                    # Tentatively move
                    current_points[idx] = [new_x, new_y]
                    
                    if not _not_all_collinear(current_points):
                        current_points[idx] = [old_x, old_y]
                        continue
                    
                    score = _evaluate(current_points)
                    
                    if score < current_score:
                        current_score = score
                        improved_this_pass = True
                        break
                    else:
                        current_points[idx] = [old_x, old_y]
                
                if improved_this_pass:
                    break
            
            if improved_this_pass:
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Update global best if we found something better
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        
        # Early exit if we've reached the theoretical minimum
        if best_score <= 6:
            break
    
    # Final validation
    assert len(best_points) == 12
    assert all(isinstance(p, list) and len(p) == 2 for p in best_points)
    assert all(abs(p[0]) <= 10**9 and abs(p[1]) <= 10**9 for p in best_points)
    assert _not_all_collinear(best_points)
    
    return best_points
# EVOLVE-BLOCK-END

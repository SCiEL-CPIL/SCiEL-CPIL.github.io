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

def _generate_parabola_candidates():
    """Generate 12 points with 11 on a parabola y = a*x^2 + b*x + c and 1 off-point.
    
    Points on a parabola have the property that lines through pairs of points 
    on the parabola intersect the parabola at exactly one more point (if it 
    exists in our set). This creates many 3-point collinearities.
    """
    candidates = []
    
    # Try different parabola parameters
    for a in [1, 2, 3, 5, 7, 10, 15, 20]:
        for b in [-3, -2, -1, 0, 1, 2, 3]:
            for c in [-2, -1, 0, 1, 2]:
                # Choose 11 x-values that create good collinearities
                # For y = a*x^2, points at x = -5,-4,-3,-2,-1,0,1,2,3,4,5
                # have many 3-point lines
                x_vals = list(range(-5, 6))
                pts = []
                for x in x_vals:
                    y = a * x * x + b * x + c
                    pts.append([x, y])
                
                # Add 12th point - try various positions
                # The "center" of the parabola structure
                # Try the vertex and nearby points
                for off_x in [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]:
                    for off_y in [-10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10]:
                        pt = [off_x, off_y]
                        # Check not duplicate
                        if any(pt[0] == p[0] and pt[1] == p[1] for p in pts):
                            continue
                        full_pts = pts + [pt]
                        if _not_all_collinear(full_pts):
                            candidates.append(full_pts)
    
    # Also try elliptic-like configurations with rational parametrization
    # Points on y^2 = x^3 + k (elliptic curve) also have good collinearity properties
    for k in [0, 1, -1, 4, -4, 16, -16]:
        pts = []
        for x in range(-5, 6):
            rhs = x * x * x + k
            if rhs >= 0:
                y = int(math.isqrt(rhs))
                if y * y == rhs:
                    pts.append([x, y])
                    if y != 0:
                        pts.append([x, -y])
        # Take first 11 unique points
        seen = set()
        unique = []
        for p in pts:
            key = (p[0], p[1])
            if key not in seen:
                seen.add(key)
                unique.append(p)
        unique = unique[:11]
        
        while len(unique) < 11:
            unique.append([len(unique) - 5, 0])
        
        # Add 12th point
        for off_x in [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]:
            for off_y in [-10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10]:
                pt = [off_x, off_y]
                if any(pt[0] == p[0] and pt[1] == p[1] for p in unique):
                    continue
                full_pts = unique + [pt]
                if _not_all_collinear(full_pts):
                    candidates.append(full_pts)
    
    return candidates

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

def construct():
    start = time.time()
    deadline = start + 135  # safety margin before 150s kill
    
    n = 12
    rng = random.Random(42)
    
    # Generate multiple candidate configurations
    candidates = []
    
    # Parabola-based configurations (primary strategy)
    for para_cand in _generate_parabola_candidates():
        if len(para_cand) == 12:
            # Check for duplicates
            seen = set()
            valid = True
            for p in para_cand:
                if (p[0], p[1]) in seen:
                    valid = False
                    break
                seen.add((p[0], p[1]))
            if valid and _not_all_collinear(para_cand):
                candidates.append(para_cand)
    
    # Grid-based configurations as fallback
    
    # Grid-based configurations
    for grid_cand in _generate_grid_candidates():
        if len(grid_cand) == 12 and _not_all_collinear(grid_cand):
            candidates.append(grid_cand)
    
    # Pencil configurations
    for pencil_cand in _generate_pencil_candidates():
        if len(pencil_cand) == 12 and _not_all_collinear(pencil_cand):
            candidates.append(pencil_cand)
    
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
        """Find positions that would create 3+ point collinearities."""
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
        
        # Also try positions that create 3-point lines with one existing point
        # and a "virtual" direction
        for px, py in existing_points:
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = px + dx
                    new_y = py + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    if (new_x, new_y) in occupied or (new_x, new_y) in seen_pos:
                        continue
                    seen_pos.add((new_x, new_y))
                    good_positions.append([new_x, new_y])
        
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
        elif restart < 20:
            # Small perturbation
            current_points = [p[:] for p in best_points]
            for _ in range(2):
                idx = rng.randrange(n)
                current_points[idx] = [rng.randint(-20, 20), rng.randint(-20, 20)]
        elif restart < 40:
            # Medium perturbation
            current_points = [p[:] for p in best_points]
            for _ in range(4):
                idx = rng.randrange(n)
                current_points[idx] = [rng.randint(-30, 30), rng.randint(-30, 30)]
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
        if current_score > best_score + 2:
            continue  # Skip if too far from best
        
        # Hill climbing: iteratively move points to improve score
        no_improve_count = 0
        max_no_improve = 15
        
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

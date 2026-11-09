"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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

def _get_line_key(x1, y1, x2, y2):
    """Get normalized line key from two points."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    if dx == 0:
        return (0, 0, x1)
    elif dy == 0:
        return (0, 1, y1)
    else:
        if dy < 0:
            dx, dy = -dx, -dy
        g = gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        c = dy * x1 - dx * y1
        return (1, dx, dy, c)

def _compute_line_counts(points):
    """Compute count of points on each line through pairs."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _get_line_key(points[i][0], points[i][1], points[j][0], points[j][1])
            if key is not None:
                lines[key] = lines.get(key, 0) + 1
    return lines

def _count_ordinary_fast(points, line_counts):
    """Count ordinary lines using precomputed line counts."""
    return sum(1 for c in line_counts.values() if c == 1)

def construct():
    start = time.time()
    deadline = start + 135  # safety margin before 150s kill
    
    n = 12
    rng = random.Random(42)
    
    # Generate multiple candidate configurations
    candidates = []
    
    # 3x4 grid
    grid_pts = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(grid_pts)
    
    # 4x3 grid
    grid_pts2 = [[x, y] for x in range(3) for y in range(4)]
    candidates.append(grid_pts2)
    
    # 2x6 grid
    grid_pts3 = [[x, y] for x in range(6) for y in range(2)]
    candidates.append(grid_pts3)
    
    # Hexagonal pattern
    hex_pts = [
        [2, 0], [1, 1], [-1, 1], [-2, 0], [-1, -1], [1, -1],  # hexagon vertices
        [0, 0],  # center
        [3, 0], [0, 2], [-3, 0], [0, -2], [2, 1], [2, -1]  # extra points
    ]
    # Check for duplicates and ensure 12 distinct
    seen = set()
    hex_unique = []
    for p in hex_pts:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            hex_unique.append(p)
    if len(hex_unique) == 12:
        candidates.append(hex_unique)
    
    # Another structured config: points on two parallel lines with some offset
    config1 = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2]
    ]
    candidates.append(config1)
    
    # Evaluate all candidates and pick the best
    best_points = None
    best_score = float('inf')
    
    for cand in candidates:
        if len(cand) != 12:
            continue
        if not _not_all_collinear(cand):
            continue
        score = _evaluate(cand)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in cand]
    
    if best_points is None:
        best_points = [[i, i*i] for i in range(12)]
        best_score = _evaluate(best_points)
    
    # Local search with efficient evaluation
    def _find_candidate_positions(points, exclude_idx, max_candidates=50):
        """Find positions that would create 3+ point collinearities."""
        good_positions = []
        existing_points = [(points[i][0], points[i][1]) for i in range(n) if i != exclude_idx]
        occupied = set(existing_points)
        
        # For each pair of existing points, find positions on the line
        for i in range(len(existing_points)):
            for j in range(i+1, len(existing_points)):
                x1, y1 = existing_points[i]
                x2, y2 = existing_points[j]
                
                dx = x2 - x1
                dy = y2 - y1
                
                if dx == 0 and dy == 0:
                    continue
                
                # Normalize direction
                if dx < 0 or (dx == 0 and dy < 0):
                    dx, dy = -dx, -dy
                
                g = gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
                if g > 0:
                    dx_norm = dx // g
                    dy_norm = dy // g
                else:
                    dx_norm, dy_norm = dx, dy
                
                # Try positions along the line (including between the two points)
                for t in range(-2, 4):
                    new_x = x1 + t * dx_norm
                    new_y = y1 + t * dy_norm
                    
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    
                    if (new_x, new_y) not in occupied:
                        good_positions.append([new_x, new_y])
        
        # Add some random positions for diversity
        for _ in range(30):
            new_x = rng.randint(-15, 15)
            new_y = rng.randint(-15, 15)
            if (new_x, new_y) not in occupied:
                good_positions.append([new_x, new_y])
        
        # Remove duplicates
        unique_positions = []
        seen_pos = set()
        for p in good_positions:
            key = (p[0], p[1])
            if key not in seen_pos:
                seen_pos.add(key)
                unique_positions.append(p)
        
        return unique_positions[:max_candidates]
    
    # Hill climbing local search
    for restart in range(30):
        if time.time() > deadline:
            break
        
        if restart == 0:
            current_points = [p[:] for p in best_points]
            current_score = best_score
        else:
            # Perturb the best solution
            current_points = [p[:] for p in best_points]
            num_perturb = rng.randint(1, 4)
            for _ in range(num_perturb):
                idx = rng.randrange(n)
                current_points[idx] = [rng.randint(-20, 20), rng.randint(-20, 20)]
            
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
                continue
        
        # Hill climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            for idx in range(n):
                if time.time() > deadline:
                    break
                
                old_x, old_y = current_points[idx]
                
                # Get candidate positions
                candidates_pos = _find_candidate_positions(current_points, idx)
                
                # Also add the original position
                candidates_pos.append([old_x, old_y])
                
                # Shuffle for diversity
                rng.shuffle(candidates_pos)
                
                # Try each candidate
                for new_x, new_y in candidates_pos:
                    if new_x == old_x and new_y == old_y:
                        continue
                    
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
                        improved = True
                        break
                    else:
                        current_points[idx] = [old_x, old_y]
                
                if improved:
                    break
            
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        
        if best_score <= 6:
            break
    
    # Final validation
    assert len(best_points) == 12
    assert all(isinstance(p, list) and len(p) == 2 for p in best_points)
    assert all(abs(p[0]) <= 10**9 and abs(p[1]) <= 10**9 for p in best_points)
    assert _not_all_collinear(best_points)
    
    return best_points
# EVOLVE-BLOCK-END

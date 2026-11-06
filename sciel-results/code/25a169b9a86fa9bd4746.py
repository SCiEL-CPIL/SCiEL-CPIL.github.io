"""Finite-field-inspired construction: embed AG(2,4) into Z^2, then local-search refine."""
import time
import random
import math
from itertools import combinations
from fractions import Fraction

def _line_key(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx_n, dy_n = dx // g, dy // g
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n, dy_n = -dx_n, -dy_n
    a, b = dy_n, -dx_n
    c = a * p1[0] + b * p1[1]
    g2 = math.gcd(abs(a), abs(b))
    if g2 > 1:
        a, b, c = a // g2, b // g2, c // g2
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _score_lines(points):
    n = 16
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    exact_4 = 0
    has_5 = False
    for pts in line_pts.values():
        cnt = len(pts)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5 = True
    return exact_4, has_5

def _build_line_index(points):
    """Build dict: line_key -> set of indices."""
    n = 16
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    return line_pts

def _count_exact4(line_pts):
    exact_4 = 0
    has_5 = False
    for pts in line_pts.values():
        cnt = len(pts)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5 = True
    return exact_4, has_5

def _gf16_lines():
    """
    Return the 21 lines of AG(2,4) as lists of 4 field-elements (pairs).
    Field GF(16) = GF(2)[x]/(x^4 + x + 1). Represent elements as 4-bit ints 0..15.
    Points of AG(2,4) are pairs (a,b) where a,b in GF(16).
    But we only want 16 points, so we take the 4x4 grid: points (i,j) for i,j in {0,1,2,3}.
    We interpret i,j as elements of GF(16) (the subfield GF(4) is not a subfield of GF(16) since 4|16 but GF(4) not subset GF(16)... actually GF(4) IS a subfield of GF(16) because 2|4).
    
    Actually, let's use a simpler approach: define 16 points as a 4x4 grid in a 2D vector space over GF(2) with each coordinate being a 2-bit value (i.e., element of GF(4)).
    GF(4) = {0,1,2,3} with 2=alpha, 3=alpha+1, alpha^2=alpha+1.
    Points: (i,j) for i,j in GF(4).
    Lines in AG(2,4): for each direction (dx,dy) != (0,0) in GF(4)^2, and each intercept, a line has 4 points.
    Number of lines: (4^2 - 1)/(4-1) * 4 = 5 * 4 = 20 lines. Wait, AG(2,4) has (4^2-1)/(4-1) = 5 directions, each with 4 parallel lines, so 20 lines total.
    
    Actually AG(2,q) has (q^2-1)/(q-1) * q = (q+1)*q lines. For q=4: 5*4=20 lines.
    """
    # GF(4) addition and multiplication
    # Elements: 0,1,2,3 where 2=alpha, 3=alpha+1
    # alpha^2 = alpha+1 = 3, alpha^3 = alpha*3 = 3alpha = alpha*(alpha+1) = alpha^2+alpha = 3+2 = 1 (since 3+2=5 mod... wait in GF(4), 3+2 = (alpha+1)+alpha = 1)
    # Multiplication table:
    gf4_mult = {
        (0,0):0,(0,1):0,(0,2):0,(0,3):0,
        (1,0):0,(1,1):1,(1,2):2,(1,3):3,
        (2,0):0,(2,1):2,(2,2):3,(2,3):1,
        (3,0):0,(3,1):3,(3,2):1,(3,3):2,
    }
    gf4_add = {
        (0,0):0,(0,1):1,(0,2):2,(0,3):3,
        (1,0):1,(1,1):0,(1,2):3,(1,3):2,
        (2,0):2,(2,1):3,(2,2):0,(2,3):1,
        (3,0):3,(3,1):2,(3,2):1,(3,3):0,
    }
    
    def gf4_mul(a, b):
        return gf4_mult[(a, b)]
    
    def gf4_add(a, b):
        return gf4_add[(a, b)]
    
    # 16 points: (i, j) for i, j in {0,1,2,3}
    points_field = [(i, j) for i in range(4) for j in range(4)]
    
    # Find all lines: for each pair of distinct points, find the line, group by line
    # A line is determined by a point and a direction.
    # Directions: (dx, dy) where (dx,dy) != (0,0), up to scalar.
    # In GF(4)^2, nonzero vectors: 15, grouped into 5 directions (each direction has 3 nonzero scalars).
    
    lines_set = set()
    for i in range(16):
        for j in range(i+1, 16):
            p1 = points_field[i]
            p2 = points_field[j]
            dx = gf4_add(p1[0], p2[0])  # p2[0] - p1[0] = p2[0] + p1[0] in char 2
            dy = gf4_add(p1[1], p2[1])
            # Find all 4 points on this line
            line_pts = set()
            for k in range(4):
                # Point = p1 + k * (dx, dy) where k in GF(4)
                # But we need to iterate over field elements 0,1,2,3
                for t in range(4):
                    # Compute p1 + t*(dx,dy)
                    tx = gf4_mul(t, dx)
                    ty = gf4_mul(t, dy)
                    nx = gf4_add(p1[0], tx)
                    ny = gf4_add(p1[1], ty)
                    line_pts.add((nx, ny))
            # Verify all 4 points are in our 16-point set
            if len(line_pts) == 4:
                lines_set.add(frozenset(line_pts))
    
    # Convert to list of lists
    lines = [list(ls) for ls in lines_set]
    return lines, points_field

def _embed_gf4_to_int(lines, points_field, scale=1, offset=(0,0)):
    """
    Embed GF(4)^2 points into Z^2.
    Map field element i in {0,1,2,3} to integer i*scale + offset.
    This is just the 4x4 grid scaled.
    """
    pts = []
    for (i, j) in points_field:
        x = i * scale + offset[0]
        y = j * scale + offset[1]
        pts.append([x, y])
    return pts

def _local_search(points, deadline, rng):
    """
    Fast local search with incremental scoring.
    Moves:
    1. Move a point to a position that completes a 3-line to a 4-line.
    2. Random small perturbation.
    3. Swap two points' positions.
    """
    current = [p[:] for p in points]
    current_line_idx = _build_line_index(current)
    current_score, h5 = _count_exact4(current_line_idx)
    if h5:
        return current, 0
        
    best = [p[:] for p in current]
    best_score = current_score
    
    T = 1.0
    T_min = 0.01
    cooling = 0.997
    
    no_improve = 0
    max_no_improve = 2000
    steps = 0
    
    while time.time() < deadline - 8 and no_improve < max_no_improve:
        steps += 1
        move_type = rng.random()
        
        if move_type < 0.6:
            # Targeted: find 3-point lines and try to complete them
            three_lines = [k for k, v in current_line_idx.items() if len(v) == 3]
            
            if three_lines:
                target_line = rng.choice(three_lines)
                pts_on_line = current_line_idx[target_line]
                a, b, c = target_line
                
                # Direction vector
                dx_step = b
                dy_step = -a
                
                # Find candidates on this line within bounds
                existing_set = set(tuple(p) for p in current)
                candidates = []
                
                # Use one point on the line as reference
                ref_idx = list(pts_on_line)[0]
                x_ref, y_ref = current[ref_idx]
                
                # Find k range
                k_min = -2000
                k_max = 2000
                if dx_step > 0:
                    k_min = max(k_min, math.ceil((-1000 - x_ref) / dx_step))
                    k_max = min(k_max, math.floor((1000 - x_ref) / dx_step))
                elif dx_step < 0:
                    k_min = max(k_min, math.ceil((1000 - x_ref) / dx_step))
                    k_max = min(k_max, math.floor((-1000 - x_ref) / dx_step))
                
                if dy_step > 0:
                    k_min = max(k_min, math.ceil((-1000 - y_ref) / dy_step))
                    k_max = min(k_max, math.floor((1000 - y_ref) / dy_step))
                elif dy_step < 0:
                    k_min = max(k_min, math.ceil((1000 - y_ref) / dy_step))
                    k_max = min(k_max, math.floor((-1000 - y_ref) / dy_step))
                
                for k in range(max(k_min, -200), min(k_max, 200) + 1):
                    x = x_ref + k * dx_step
                    y = y_ref + k * dy_step
                    pt = (x, y)
                    if pt not in existing_set and -1000 <= x <= 1000 and -1000 <= y <= 1000:
                        candidates.append(pt)
                
                if candidates:
                    new_pt = rng.choice(candidates)
                    replace_candidates = [i for i in range(16) if i not in pts_on_line]
                    if replace_candidates:
                        idx = rng.choice(replace_candidates)
                        old_pt = current[idx][:]
                        current[idx] = list(new_pt)
                        
                        # Quick check for 5-collinear
                        new_line_idx = _build_line_index(current)
                        new_score, new_h5 = _count_exact4(new_line_idx)
                        
                        if not new_h5:
                            delta = new_score - current_score
                            if delta > 0 or rng.random() < math.exp(delta / T):
                                current_score = new_score
                                current_line_idx = new_line_idx
                                if current_score > best_score:
                                    best_score = current_score
                                    best = [p[:] for p in current]
                                    no_improve = 0
                                else:
                                    no_improve += 1
                            else:
                                current[idx] = old_pt
                                no_improve += 1
                        else:
                            current[idx] = old_pt
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        elif move_type < 0.85:
            # Random perturbation
            idx = rng.randint(0, 15)
            old_pt = current[idx][:]
            step = rng.randint(1, 15)
            new_x = old_pt[0] + rng.randint(-step, step)
            new_y = old_pt[1] + rng.randint(-step, step)
            
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            if [new_x, new_y] in current:
                continue
                
            current[idx] = [new_x, new_y]
            new_line_idx = _build_line_index(current)
            new_score, new_h5 = _count_exact4(new_line_idx)
            
            if not new_h5:
                delta = new_score - current_score
                if delta > 0 or rng.random() < math.exp(delta / T):
                    current_score = new_score
                    current_line_idx = new_line_idx
                    if current_score > best_score:
                        best_score = current_score
                        best = [p[:] for p in current]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    current[idx] = old_pt
                    no_improve += 1
            else:
                current[idx] = old_pt
                no_improve += 1
        else:
            # Swap two points
            i, j = rng.sample(range(16), 2)
            current[i], current[j] = current[j][:], current[i][:]
            
            new_line_idx = _build_line_index(current)
            new_score, new_h5 = _count_exact4(new_line_idx)
            
            if not new_h5:
                delta = new_score - current_score
                if delta > 0 or rng.random() < math.exp(delta / T):
                    current_score = new_score
                    current_line_idx = new_line_idx
                    if current_score > best_score:
                        best_score = current_score
                        best = [p[:] for p in current]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    current[i], current[j] = current[j][:], current[i][:]
                    no_improve += 1
            else:
                current[i], current[j] = current[j][:], current[i][:]
                no_improve += 1
        
        T = max(T_min, T * cooling)
        
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    rng = random.Random(12345)
    
    # Get GF(4)^2 line structure
    gf_lines, gf_points = _gf16_lines()
    
    # Try multiple embeddings: different scales and offsets
    best_points = None
    best_score = -1
    
    scales = [1, 2, 3, 4, 5, 7, 10, 13]
    offsets = [(0,0), (-1,-1), (1,1), (-2,1), (2,-1), (0,2), (2,0), (-3,-3), (3,3), (1,-2), (-1,2)]
    
    # Generate diverse initial configurations
    initial_configs = []
    
    for scale in scales:
        for ox, oy in offsets:
            pts = _embed_gf4_to_int(gf_lines, gf_points, scale=scale, offset=(ox, oy))
            # Check bounds
            if all(-1000 <= p[0] <= 1000 and -1000 <= p[1] <= 1000 for p in pts):
                s, h5 = _score_lines(pts)
                if not h5:
                    initial_configs.append((s, pts))
    
    # Sort by score descending
    initial_configs.sort(key=lambda x: -x[0])
    
    # Also add some random configurations
    for _ in range(10):
        pts = [[rng.randint(-100, 100), rng.randint(-100, 100)] for _ in range(16)]
        # Ensure distinct
        seen = set()
        valid = True
        for p in pts:
            if (p[0], p[1]) in seen:
                valid = False
                break
            seen.add((p[0], p[1]))
        if valid:
            s, h5 = _score_lines(pts)
            if not h5:
                initial_configs.append((s, pts))
    
    initial_configs.sort(key=lambda x: -x[0])
    
    # Run local search on top configurations
    num_restarts = min(12, len(initial_configs))
    for r in range(num_restarts):
        if time.time() > deadline - 25:
            break
            
        s_init, pts = initial_configs[r]
        if s_init > best_score:
            best_score = s_init
            best_points = [p[:] for p in pts]
        
        opt_pts, opt_score = _local_search(pts, deadline, rng)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts
    
    # Final verification
    if best_points is None:
        best_points = [[i, j] for i in range(4) for j in range(4)]
    
    final_score, final_h5 = _score_lines(best_points)
    if final_h5:
        # Fallback
        best_points = [[i, j] for i in range(4) for j in range(4)]
    
    return best_points

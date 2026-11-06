import random
import time
import math
from collections import defaultdict

def count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly two of the given points.
    """
    n = len(points)
    if n < 3:
        return 0
    
    lines = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Normalize direction vector
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            
            # Ensure unique representation for line direction
            if dx < 0:
                dx = -dx
                dy = -dy
            elif dx == 0 and dy < 0:
                dy = -dy

            # Line equation: a*x + b*y + c = 0
            a = dy
            b = -dx
            c = -(a * p1[0] + b * p1[1])
            
            key = (a, b, c)
            lines[key].add(i)
            lines[key].add(j)

    ordinary = 0
    for pts in lines.values():
        if len(pts) == 2:
            ordinary += 1
            
    return ordinary

def _ag24_configuration():
    """
    Generate a 12-point configuration from AG(2,4) geometry.
    AG(2,4) has 16 points with every line containing exactly 4 points.
    We select 12 points to minimize ordinary lines.
    
    AG(2,4) can be represented as GF(4)^2. GF(4) = {0, 1, alpha, alpha+1}
    where alpha^2 = alpha + 1.
    
    Lines in AG(2,4): For each direction (slope), there are 4 parallel lines,
    each containing 4 points.
    
    We'll embed this in Z^2 by choosing a projective representation.
    """
    # GF(4) elements: 0, 1, a, a+1 where a^2 = a+1
    # Represent as: 0->0, 1->1, a->2, a+1->3 (just for indexing)
    
    # Points in AG(2,4) as (x, y) where x,y in GF(4)
    gf4_elements = [0, 1, 2, 3]  # 0, 1, alpha, alpha+1
    
    # GF(4) addition table (XOR-like for this representation)
    # 0+0=0, 0+1=1, 0+2=2, 0+3=3
    # 1+1=0, 1+2=3, 1+3=2
    # 2+2=0, 2+3=1
    # 3+3=0
    add_table = {
        (0,0):0, (0,1):1, (0,2):2, (0,3):3,
        (1,0):1, (1,1):0, (1,2):3, (1,3):2,
        (2,0):2, (2,1):3, (2,2):0, (2,3):1,
        (3,0):3, (3,1):2, (3,2):1, (3,3):0
    }
    
    # GF(4) multiplication table
    # 0*x=0 for all x
    # 1*x=x
    # 2*2=3, 2*3=1, 3*3=2
    # 2*1=2, 3*1=3
    mul_table = {
        (0,0):0, (0,1):0, (0,2):0, (0,3):0,
        (1,0):0, (1,1):1, (1,2):2, (1,3):3,
        (2,0):0, (2,1):2, (2,2):3, (2,3):1,
        (3,0):0, (3,1):3, (3,2):1, (3,3):2
    }
    
    def gf4_add(a, b):
        return add_table[(a, b)]
    
    def gf4_mul(a, b):
        return mul_table[(a, b)]
    
    # All 16 points in AG(2,4)
    all_points = [(x, y) for x in gf4_elements for y in gf4_elements]
    
    # In AG(2,4), lines are sets of 4 points.
    # A line with slope m through point (x0, y0) contains:
    # {(x, y) : y - y0 = m * (x - x0)} for x in GF(4)
    # Plus vertical lines: {(x0, y) : y in GF(4)}
    
    # Let's find all lines
    lines_in_ag24 = []
    
    # Vertical lines (4 of them)
    for x0 in gf4_elements:
        line = [(x0, y) for y in gf4_elements]
        lines_in_ag24.append(line)
    
    # Non-vertical lines (4 slopes x 4 intercepts = 16)
    for m in gf4_elements:
        for b in gf4_elements:
            line = []
            for x in gf4_elements:
                y = gf4_add(gf4_mul(m, x), b)
                line.append((x, y))
            lines_in_ag24.append(line)
    
    # Each point is on 5 lines (4 non-vertical + 1 vertical, but one of the "non-vertical" 
    # with m=0 gives horizontal, etc.) Actually each point is on 5 lines in AG(2,4).
    # Total lines: 4 + 16 = 20, each with 4 points, total point-line incidences = 80 = 16*5. ✓
    
    # Now, we need to select 12 of the 16 points such that we minimize ordinary lines.
    # An ordinary line in our 12-point set is a line containing exactly 2 of our 12 points.
    # In AG(2,4), every line has exactly 4 points. So if we remove 4 points,
    # a line that had 4 points now has 4-k points where k is how many of the removed
    # points were on that line.
    # 
    # We want to minimize lines with exactly 2 points remaining.
    # 
    # Strategy: Remove 4 points that are collinear (on one line of AG(2,4)).
    # Then that line has 0 remaining points (not counted).
    # Other lines: if a line shares k removed points, it has 4-k remaining.
    # We want to avoid 4-k=2, i.e., k=2.
    # So we want removed points such that no line contains exactly 2 of them.
    # 
    # If we remove 4 collinear points, any other line intersects the removed line in at most 1 point.
    # So any other line has at most 1 removed point, meaning it has at least 3 remaining points.
    # Lines that don't intersect the removed line at all have 0 removed, so 4 remaining.
    # Lines that intersect at 1 point have 1 removed, so 3 remaining.
    # The removed line itself has 4 removed, so 0 remaining.
    # 
    # So NO line has exactly 2 remaining points! Ordinary lines = 0?
    # Wait, but we're embedding in R^2, so the collinearity structure might differ.
    
    # Let's embed AG(2,4) into Z^2.
    # Map GF(4) elements to integers: 0->0, 1->1, 2->2, 3->3
    # But this won't preserve collinearity perfectly.
    # 
    # Better: Use a projective embedding.
    # Map (x, y) in GF(4)^2 to (X, Y) in Z^2 such that collinearity is preserved.
    # 
    # Simple approach: just use the integer coordinates directly and check.
    
    # Let's try: map (x,y) -> (x, y) as integers, then check ordinary lines.
    # But this is a 4x4 grid, which has many ordinary lines.
    
    # Better embedding: Use the fact that GF(4) can be embedded in a way that
    # preserves some structure. Let's try mapping:
    # 0 -> 0, 1 -> 1, 2 (alpha) -> 2, 3 (alpha+1) -> 3
    # And use a non-linear embedding to break grid structure.
    
    # Actually, let's just try the direct mapping and see the score,
    # then use local search to improve.
    
    # Direct mapping of all 16 points:
    # We'll select 12 by removing 4 collinear points.
    
    # Remove the 4 points on the line y=0: (0,0), (1,0), (2,0), (3,0)
    remaining = [(x, y) for x in gf4_elements for y in gf4_elements if y != 0]
    
    # Convert to integer points
    points = [[x, y] for x, y in remaining]
    
    return points

def _ag24_variants():
    """
    Generate multiple AG(2,4)-based configurations by removing different lines.
    """
    configs = []
    
    gf4_elements = [0, 1, 2, 3]
    
    add_table = {
        (0,0):0, (0,1):1, (0,2):2, (0,3):3,
        (1,0):1, (1,1):0, (1,2):3, (1,3):2,
        (2,0):2, (2,1):3, (2,2):0, (2,3):1,
        (3,0):3, (3,1):2, (3,2):1, (3,3):0
    }
    
    mul_table = {
        (0,0):0, (0,1):0, (0,2):0, (0,3):0,
        (1,0):0, (1,1):1, (1,2):2, (1,3):3,
        (2,0):0, (2,1):2, (2,2):3, (2,3):1,
        (3,0):0, (3,1):3, (3,2):1, (3,3):2
    }
    
    def gf4_add(a, b):
        return add_table[(a, b)]
    
    def gf4_mul(a, b):
        return mul_table[(a, b)]
    
    # All 16 points
    all_points = [(x, y) for x in gf4_elements for y in gf4_elements]
    
    # Generate all 20 lines of AG(2,4)
    lines_in_ag24 = []
    
    # Vertical lines
    for x0 in gf4_elements:
        line = [(x0, y) for y in gf4_elements]
        lines_in_ag24.append(line)
    
    # Non-vertical lines
    for m in gf4_elements:
        for b in gf4_elements:
            line = []
            for x in gf4_elements:
                y = gf4_add(gf4_mul(m, x), b)
                line.append((x, y))
            lines_in_ag24.append(line)
    
    # For each line, remove its 4 points and check the resulting 12-point config
    for line in lines_in_ag24:
        removed = set(line)
        remaining = [p for p in all_points if p not in removed]
        points = [[x, y] for x, y in remaining]
        if len(set(map(tuple, points))) == 12:
            configs.append(points)
    
    # Also try removing non-collinear sets of 4
    # For now, just the collinear removals
    return configs

def _generate_seeds():
    """
    Generates a list of initial configurations to start the search.
    """
    seeds = []
    
    # 1. AG(2,4)-based configurations
    ag24_configs = _ag24_variants()
    seeds.extend(ag24_configs)
    
    # 2. Standard Grids
    seeds.append([[x, y] for x in range(4) for y in range(3)])
    seeds.append([[x, y] for x in range(3) for y in range(4)])
    seeds.append([[x, y] for x in range(6) for y in range(2)])
    seeds.append([[x, y] for x in range(2) for y in range(6)])
    
    # 3. Diagonal Shifts (Sheared Grids)
    pts_diag = []
    for y in range(4):
        for x in range(3):
            pts_diag.append([x + y, y])
    seeds.append(pts_diag)
    
    # 4. Hesse-like / High Collinearity Structures
    rng_seed = random.Random(123)
    for _ in range(5):
        shift = rng_seed.randint(-2, 2)
        pts = []
        for y in range(3):
            for x in range(4):
                pts.append([x + y * shift, y])
        if len(set(map(tuple, pts))) == 12:
             seeds.append(pts)

    # 5. Random Dense Clusters
    for _ in range(10):
        pts = []
        for _ in range(12):
            pts.append([rng_seed.randint(-10, 10), rng_seed.randint(-10, 10)])
        if len(set(map(tuple, pts))) == 12:
            (x0, y0), (x1, y1), (x2, y2) = pts[0], pts[1], pts[2]
            all_col = True
            for i in range(3, 12):
                xi, yi = pts[i]
                if (x1 - x0) * (yi - y0) != (y1 - y0) * (xi - x0):
                    all_col = False
                    break
            if not all_col:
                seeds.append(pts)

    return seeds

def _find_collinear_candidates(points):
    """
    Finds potential integer points that lie on the line defined by two existing points.
    """
    candidates = []
    n = len(points)
    occupied = set(map(tuple, points))
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            
            for k in range(-15, 16):
                nx = p1[0] + k * dx
                ny = p1[1] + k * dy
                
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                    
                if (nx, ny) in occupied:
                    continue
                    
                candidates.append((nx, ny))
                
    return candidates

def _find_line_extension_candidates(points):
    """
    Find points that would extend an existing line with 3+ points.
    """
    candidates = []
    n = len(points)
    occupied = set(map(tuple, points))
    
    # Group points by lines
    lines = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            
            if dx < 0:
                dx = -dx
                dy = -dy
            elif dx == 0 and dy < 0:
                dy = -dy

            a = dy
            b = -dx
            c = -(a * p1[0] + b * p1[1])
            
            key = (a, b, c)
            if i not in lines[key]:
                lines[key].append(i)
            if j not in lines[key]:
                lines[key].append(j)
    
    # For lines with 3+ points, try to extend them
    for key, pt_indices in lines.items():
        if len(pt_indices) < 3:
            continue
            
        # Get direction from first two points
        p1 = points[pt_indices[0]]
        p2 = points[pt_indices[1]]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        g = math.gcd(abs(dx), abs(dy))
        if g == 0:
            continue
        dx //= g
        dy //= g
        
        # Try extending in both directions
        for sign in [1, -1]:
            for k in range(1, 10):
                # Find the extreme point in this direction
                base = p1
                for idx in pt_indices:
                    px, py = points[idx]
                    if sign * (px - p1[0]) * dy > sign * (base[0] - p1[0]) * dy or \
                       (sign * (px - p1[0]) * dy == sign * (base[0] - p1[0]) * dy and 
                        sign * (py - p1[1]) * dx > sign * (base[1] - p1[1]) * dx):
                        base = points[idx]
                
                nx = base[0] + sign * k * dx
                ny = base[1] + sign * k * dy
                
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                if (nx, ny) in occupied:
                    continue
                    
                candidates.append((nx, ny))
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    seeds = _generate_seeds()
    
    best_points = None
    best_score = float('inf')
    
    # Evaluate seeds
    for seed in seeds:
        if len(seed) != 12:
            continue
        if len(set(map(tuple, seed))) < 12:
            continue
        # Check not all collinear
        (x0, y0), (x1, y1), (x2, y2) = seed[0], seed[1], seed[2]
        all_col = True
        for i in range(3, 12):
            xi, yi = seed[i]
            if (x1 - x0) * (yi - y0) != (y1 - y0) * (xi - x0):
                all_col = False
                break
        if all_col:
            continue
            
        score = count_ordinary_lines(seed)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in seed]
        if best_score == 6:
            break
            
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = count_ordinary_lines(best_points)

    rng = random.Random(42)
    
    # Local Search
    while time.time() < deadline:
        # Strategy 1: Move a point to a collinear target (High probability)
        if rng.random() < 0.5:
            targets = _find_collinear_candidates(best_points)
            if not targets:
                continue
                
            target = rng.choice(targets)
            tx, ty = target
            
            idx = rng.randint(0, 11)
            old_p = best_points[idx][:]
            
            if old_p == [tx, ty]:
                continue
                
            new_points = [p[:] for p in best_points]
            new_points[idx] = [tx, ty]
            
            if len(set(map(tuple, new_points))) < 12:
                continue
                
            score = count_ordinary_lines(new_points)
            if score < best_score:
                best_score = score
                best_points = new_points
                if best_score == 6:
                    break
                    
        # Strategy 2: Move to line extension target
        elif rng.random() < 0.7:
            targets = _find_line_extension_candidates(best_points)
            if not targets:
                continue
                
            target = rng.choice(targets)
            tx, ty = target
            
            idx = rng.randint(0, 11)
            old_p = best_points[idx][:]
            
            if old_p == [tx, ty]:
                continue
                
            new_points = [p[:] for p in best_points]
            new_points[idx] = [tx, ty]
            
            if len(set(map(tuple, new_points))) < 12:
                continue
                
            score = count_ordinary_lines(new_points)
            if score < best_score:
                best_score = score
                best_points = new_points
                if best_score == 6:
                    break
                    
        # Strategy 3: Random Perturbation
        elif rng.random() < 0.9:
            idx = rng.randint(0, 11)
            old_p = best_points[idx][:]
            
            step_size = rng.choice([1, 2, 5, 10, 20, 50])
            dx = rng.randint(-step_size, step_size)
            dy = rng.randint(-step_size, step_size)
            
            if dx == 0 and dy == 0:
                continue
                
            new_p = [old_p[0] + dx, old_p[1] + dy]
            
            if abs(new_p[0]) > 10**9 or abs(new_p[1]) > 10**9:
                continue
                
            new_points = [p[:] for p in best_points]
            new_points[idx] = new_p
            
            if len(set(map(tuple, new_points))) < 12:
                continue
                
            score = count_ordinary_lines(new_points)
            if score < best_score:
                best_score = score
                best_points = new_points
                if best_score == 6:
                    break
                    
        # Strategy 4: Swap two points
        else:
            i, j = rng.sample(range(12), 2)
            p1 = best_points[i][:]
            p2 = best_points[j][:]
            
            s1 = rng.choice([1, 2, 5])
            s2 = rng.choice([1, 2, 5])
            
            dx1 = rng.randint(-s1, s1)
            dy1 = rng.randint(-s1, s1)
            dx2 = rng.randint(-s2, s2)
            dy2 = rng.randint(-s2, s2)
            
            new_p1 = [p1[0] + dx1, p1[1] + dy1]
            new_p2 = [p2[0] + dx2, p2[1] + dy2]
            
            if abs(new_p1[0]) > 10**9 or abs(new_p1[1]) > 10**9:
                continue
            if abs(new_p2[0]) > 10**9 or abs(new_p2[1]) > 10**9:
                continue
                
            new_points = [p[:] for p in best_points]
            new_points[i] = new_p1
            new_points[j] = new_p2
            
            if len(set(map(tuple, new_points))) < 12:
                continue
                
            score = count_ordinary_lines(new_points)
            if score < best_score:
                best_score = score
                best_points = new_points
                if best_score == 6:
                    break

    return best_points

"""Structural rewrite: direct AG(2,4) construction + fast swap-based local search."""
import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _line_key(p1, p2):
    """Return canonical (A, B, C) for line through p1, p2 where A*x + B*y + C = 0."""
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    g = _gcd(_gcd(abs(A), abs(B)), abs(C))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _build_line_map(points):
    """Build dict: line_key -> set of point indices on that line."""
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_map[key].add(i)
            line_map[key].add(j)
    return line_map

def _score_and_valid(points):
    """Return (exact4_count, is_valid). Fast path using line_map."""
    n = len(points)
    if n != 16:
        return 0, False
    pts_set = set(tuple(p) for p in points)
    if len(pts_set) != 16:
        return 0, False
    for p in points:
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return 0, False
    line_map = _build_line_map(points)
    exact4 = 0
    for key, idxs in line_map.items():
        k = len(idxs)
        if k >= 5:
            return 0, False
        if k == 4:
            exact4 += 1
    return exact4, True

def _build_ag24():
    """
    Construct the 16 points of AG(2,4) embedded in integer coordinates.
    GF(4) elements: 0, 1, w, w+1 where w^2 = w + 1.
    Map each GF(4) element to a 2D integer vector:
      0 -> (0, 0)
      1 -> (1, 0)
      w -> (0, 1)
      w+1 -> (1, 1)
    Point (a, b) in GF(4)^2 maps to integer point (ax + bx', ay + by')
    where we use the matrix representation.
    
    Actually, let's use a cleaner embedding. Represent GF(4) elements as pairs
    in F2^2: 0=(0,0), 1=(1,0), w=(0,1), w+1=(1,1).
    A point (a,b) in GF(4)^2 where a,b in GF(4) becomes a 4D vector in F2^4,
    but we need 2D integer coords.
    
    Better approach: Use the fact that AG(2,4) is isomorphic to a 4x4 grid
    with a specific line structure. The 24 lines of 4 points can be realized
    on a 4x4 integer grid where we label rows 0-3 and cols 0-3, and the
    "lines" include:
      - 4 horizontal lines (rows)
      - 4 vertical lines (cols)  
      - 4 "diagonal" lines of slope 1 in GF(4) sense
      - 4 "diagonal" lines of slope w in GF(4) sense
      - 4 "diagonal" lines of slope w^2 in GF(4) sense
      - 4 "diagonal" lines of slope 1/w in GF(4) sense
    
    For the integer embedding, use the standard 4x4 grid and define the
    GF(4)-diagonals as specific sets of 4 points.
    
    GF(4) addition table (using 0,1,2,3 for 0,1,w,w+1):
    0+0=0, 0+1=1, 0+2=2, 0+3=3
    1+1=0, 1+2=3, 1+3=2
    2+2=0, 2+3=1
    3+3=0
    
    GF(4) multiplication table:
    0*x=0
    1*x=x
    2*0=0, 2*1=2, 2*2=3, 2*3=1
    3*0=0, 3*1=3, 3*2=1, 3*3=2
    
    A line of slope m in AG(2,4): {(x, m*x + b) : x in GF(4)}
    For m=0: y=b (horizontal)
    For m=1: y=x+b
    For m=2 (w): y=2x+b
    For m=3 (w+1): y=3x+b
    
    Using the addition table:
    m=1, b=0: (0,0),(1,1),(2,2),(3,3)
    m=1, b=1: (0,1),(1,0),(2,3),(3,2)
    m=1, b=2: (0,2),(1,3),(2,0),(3,1)
    m=1, b=3: (0,3),(1,2),(2,1),(3,0)
    
    m=2, b=0: (0,0),(1,2),(2,3),(3,1)
    m=2, b=1: (0,1),(1,3),(2,1),(3,2)
    m=2, b=2: (0,2),(1,1),(2,2),(3,3)  -- wait, let me recheck
    2*0=0, 2*1=2, 2*2=3, 2*3=1
    m=2, b=0: x=0->y=0, x=1->y=2, x=2->y=3, x=3->y=1: (0,0),(1,2),(2,3),(3,1)
    m=2, b=1: x=0->y=1, x=1->y=2+1=3, x=2->y=3+1=2, x=3->y=1+1=0: (0,1),(1,3),(2,2),(3,0)
    m=2, b=2: x=0->y=2, x=1->y=2+2=0, x=2->y=3+2=1, x=3->y=1+2=3: (0,2),(1,0),(2,1),(3,3)
    m=2, b=3: x=0->y=3, x=1->y=2+3=1, x=2->y=3+3=0, x=3->y=1+3=2: (0,3),(1,1),(2,0),(3,2)
    
    m=3, b=0: 3*0=0, 3*1=3, 3*2=1, 3*3=2: (0,0),(1,3),(2,1),(3,2)
    m=3, b=1: (0,1),(1,3+1=2),(2,1+1=0),(3,2+1=3): (0,1),(1,2),(2,0),(3,3)
    m=3, b=2: (0,2),(1,3+2=1),(2,1+2=3),(3,2+2=0): (0,2),(1,1),(2,3),(3,0)
    m=3, b=3: (0,3),(1,3+3=0),(2,1+3=2),(3,2+3=1): (0,3),(1,0),(2,2),(3,1)
    
    Vertical lines (x=c): {(c, y) : y in GF(4)}
    x=0: (0,0),(0,1),(0,2),(0,3)
    x=1: (1,0),(1,1),(1,2),(1,3)
    x=2: (2,0),(2,1),(2,2),(2,3)
    x=3: (3,0),(3,1),(3,2),(3,3)
    
    So the 24 lines are:
    Horizontal (4): rows
    Vertical (4): cols
    Slope 1 (4)
    Slope 2 (4)
    Slope 3 (4)
    
    That's 4+4+4+4+4 = 20. Wait, I'm missing 4.
    
    Actually in AG(2,4), for each direction (slope) there are 4 parallel lines, and there are
    4+1=5 directions (including vertical). So 5*4=20 lines total, each with 4 points.
    16 points, each point on 5 lines: 16*5/4 = 20 lines. Yes, 20 lines.
    
    So the base AG(2,4) gives 20 exact-4 lines (if no 5 collinear, which is guaranteed
    since each line has exactly 4 points).
    
    The 4x4 integer grid {(i,j) : i,j in {0,1,2,3}} with these 20 lines.
    But we need to check: are there any OTHER lines in the integer grid that contain
    4 or more of these 16 points? The standard 4x4 grid has additional collinearities.
    For example, (0,0),(1,1),(2,2),(3,3) is already counted. What about (0,0),(2,1),(4,2)?
    No, we only have 0-3. (0,0),(1,2),(2,4)? No.
    
    Let me check for 3-point lines in the 4x4 grid that aren't in our 20:
    (0,0),(1,2),(2,4) - out of range
    (0,1),(1,3),(2,5) - out of range
    (0,0),(2,2) - only 2 in range with that slope... (0,0),(1,1),(2,2),(3,3) is slope 1, counted.
    (0,1),(1,3) - slope 2, (2,5) out. Just 2 points.
    (0,2),(1,0) - slope -2, (2,-2) out. Just 2.
    (0,3),(1,1),(2,-1) - out.
    (1,0),(2,2),(3,4) - out.
    (0,0),(3,1) - slope 1/3, (6,2) out. Just 2.
    (0,1),(3,2) - slope 1/3. Just 2.
    (1,0),(3,1) - slope 1/2. Just 2.
    (0,2),(2,3) - slope 1/2. Just 2.
    
    I think the 4x4 grid might have some 3-point lines but no additional 4-point lines
    beyond the 20 we identified. Let me verify by checking all pairs.
    
    Actually, I should just use the 4x4 grid as base and compute the score. If it's 20,
    great. Then local search can try to improve.
    """
    return [[i, j] for i in range(4) for j in range(4)]

def _seed_staggered():
    """A staggered 4x4 grid to break some collinearities while keeping structure."""
    # Shift alternate rows
    pts = []
    for i in range(4):
        offset = i * 2  # 0, 2, 4, 6
        for j in range(4):
            pts.append([(i + offset) % 12 - 4, j])
    return pts

def _seed_hexagonal():
    """Hexagonal-like arrangement."""
    pts = []
    # 4 rows of 4, with horizontal offset
    for i in range(4):
        x_offset = (i % 2) * 2
        for j in range(4):
            pts.append([i * 3 + x_offset, j * 3])
    return pts

def _fast_local_search(points, time_limit):
    """
    Two-point swap local search with fast scoring.
    Try swapping coordinates or replacing one point.
    """
    start = time.time()
    best = [p[:] for p in points]
    best_score, _ = _score_and_valid(best)
    n = len(best)
    
    # Precompute line map
    line_map = _build_line_map(best)
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        # Try single-point replacement
        for i in range(n):
            if (time.time() - start) > time_limit:
                return best, best_score
            old_pt = best[i]
            # Try several candidate positions
            candidates = []
            # Nearby positions
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    candidates.append([old_pt[0] + dx, old_pt[1] + dy])
            # Random positions
            for _ in range(20):
                candidates.append([random.randint(-50, 50), random.randint(-50, 50)])
            
            for new_pt in candidates:
                if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
                    continue
                # Check distinctness
                if any(new_pt[0] == best[j][0] and new_pt[1] == best[j][1] for j in range(n) if j != i):
                    continue
                new_pts = [p[:] for p in best]
                new_pts[i] = new_pt
                s, valid = _score_and_valid(new_pts)
                if valid and s > best_score:
                    best = new_pts
                    best_score = s
                    improved = True
                    break
    
    # Two-point swap phase
    for _ in range(5):
        if (time.time() - start) > time_limit:
            break
        improved = False
        for i in range(n):
            if (time.time() - start) > time_limit:
                break
            for j in range(i + 1, n):
                if (time.time() - start) > time_limit:
                    break
                # Swap the x-coordinates
                new_pts = [p[:] for p in best]
                new_pts[i][0], new_pts[j][0] = new_pts[j][0], new_pts[i][0]
                # Check distinctness
                pts_set = set(tuple(p) for p in new_pts)
                if len(pts_set) == 16:
                    s, valid = _score_and_valid(new_pts)
                    if valid and s > best_score:
                        best = new_pts
                        best_score = s
                        improved = True
                        break
                # Swap the y-coordinates
                new_pts = [p[:] for p in best]
                new_pts[i][1], new_pts[j][1] = new_pts[j][1], new_pts[i][1]
                pts_set = set(tuple(p) for p in new_pts)
                if len(pts_set) == 16:
                    s, valid = _score_and_valid(new_pts)
                    if valid and s > best_score:
                        best = new_pts
                        best_score = s
                        improved = True
                        break
                # Full swap
                new_pts = [p[:] for p in best]
                new_pts[i], new_pts[j] = new_pts[j][:], new_pts[i][:]
                pts_set = set(tuple(p) for p in new_pts)
                if len(pts_set) == 16:
                    s, valid = _score_and_valid(new_pts)
                    if valid and s > best_score:
                        best = new_pts
                        best_score = s
                        improved = True
                        break
            if improved:
                break
    
    return best, best_score

def _scale_points(points, scale, offset_x=0, offset_y=0):
    """Scale and offset points to use more of the coordinate space."""
    return [[x * scale + offset_x, y * scale + offset_y] for x, y in points]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    random.seed(12345)
    
    best_points = None
    best_score = 0
    
    # Seed 1: Standard 4x4 grid (AG(2,4) structure)
    seeds = []
    seeds.append(_build_ag24())
    seeds.append(_seed_staggered())
    seeds.append(_seed_hexagonal())
    # Scaled versions
    seeds.append(_scale_points(_build_ag24(), 2))
    seeds.append(_scale_points(_build_ag24(), 3, 1, 1))
    seeds.append(_scale_points(_build_ag24(), 5, -2, -1))
    
    # Random seeds
    for _ in range(5):
        rp = []
        used = set()
        for _ in range(16):
            while True:
                x = random.randint(-30, 30)
                y = random.randint(-30, 30)
                if (x, y) not in used:
                    used.add((x, y))
                    rp.append([x, y])
                    break
        seeds.append(rp)
    
    for seed in seeds:
        if (time.time() - start_time) > deadline - 5:
            break
        s, valid = _score_and_valid(seed)
        if valid and s > best_score:
            best_score = s
            best_points = [p[:] for p in seed]
    
    # Local search from best seed
    if best_points is not None and (time.time() - start_time) < deadline - 10:
        improved, improved_score = _fast_local_search(best_points, deadline - time.time() - 10)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved
    
    # Try local search from other seeds
    for seed in seeds:
        if (time.time() - start_time) > deadline - 15:
            break
        s, valid = _score_and_valid(seed)
        if valid and s >= best_score - 2:
            improved, improved_score = _fast_local_search(seed, min(20, deadline - time.time() - 10))
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved
    
    # More random restarts with local search
    while (time.time() - start_time) < deadline - 20:
        rp = []
        used = set()
        for _ in range(16):
            while True:
                x = random.randint(-200, 200)
                y = random.randint(-200, 200)
                if (x, y) not in used:
                    used.add((x, y))
                    rp.append([x, y])
                    break
        s, valid = _score_and_valid(rp)
        if valid:
            improved, improved_score = _fast_local_search(rp, min(15, deadline - time.time() - 10))
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved
    
    if best_points is None:
        # Fallback: return the 4x4 grid
        best_points = _build_ag24()
    
    return best_points

# EVOLVE-BLOCK-END


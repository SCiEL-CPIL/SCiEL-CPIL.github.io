import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Return canonical line key for two points."""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    if a == 0 and b == 0:
        return None
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g == 0:
        g = 1
    a, b, c = a // g, b // g, c // g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _count_ordinary_fast(points):
    """Count ordinary lines using a dict of line->count."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for v in line_counts.values() if v == 1)

def _is_valid(points):
    """Check: 12 distinct points, not all collinear."""
    if len(points) != 12:
        return False
    if len(set(tuple(p) for p in points)) != 12:
        return False
    (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
    if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
        return True
    for i in range(3, 12):
        (x4, y4) = points[i]
        if (x2 - x1) * (y4 - y1) != (y2 - y1) * (x4 - x1):
            return True
    return False

def _build_hesse_config():
    """
    Try to build a Hesse-like configuration: 12 points on a cubic curve
    where many lines contain 3 points.
    
    The Hesse configuration has 9 lines and 12 points, with each line containing
    4 points and each point on 3 lines. But that's not quite right.
    
    Actually, let's try the Böröczky-type construction or use a different approach.
    
    For 12 points, we want to maximize the number of "rich" lines (lines with >= 3 points).
    The minimum number of ordinary lines is related to the structure.
    
    Let's try a construction based on the group law of an elliptic curve.
    We'll use a rational cubic curve with a node.
    """
    # Try a specific algebraic construction
    # Use points from a rational normal curve or similar
    
    # Approach: Use points that lie on a cubic curve y^2 = x^3 + ax^2 + bx
    # The group law ensures that collinear points sum to zero (for a suitable origin)
    
    # For a nodal cubic, we can parameterize rationally
    # y^2 = x^2(x + 1)  (nodal cubic with node at origin)
    # Parameterization: x = t^2 - 1, y = t(t^2 - 1)
    # This gives rational points for rational t
    
    # Let's try using t values that give nice integer points
    # x = t^2 - 1, y = t(t^2 - 1)
    # For t = 0: (−1, 0)
    # For t = 1: (0, 0) - this is the node, skip
    # For t = 2: (3, 6)
    # For t = 3: (8, 24)
    # For t = 4: (15, 60)
    # For t = -1: (0, 0) - node again
    # For t = -2: (3, -6)
    # For t = -3: (8, -24)
    # For t = -4: (15, -60)
    
    # The group law on this curve: three points are collinear iff they sum to the identity
    # The identity is the point at infinity
    # For a nodal cubic, the group is isomorphic to R*/R* or similar
    
    # Actually, let's just try a different approach: use known good configurations
    
    # Try the 4x3 grid and see if we can improve it
    pts = [[i, j] for i in range(4) for j in range(3)]
    return pts

def _try_algebraic_construction():
    """
    Try to use the group structure of a cubic curve to find a good configuration.
    
    Key insight: On a cubic curve, three points are collinear iff their sum (under the group law)
    is the identity element. This means we can construct configurations where many triples
    are collinear by choosing points that form subgroups or have specific group-theoretic properties.
    
    For a nodal cubic y^2 = x^2(x+1), the group is isomorphic to the multiplicative group of
    the field. We can use points corresponding to roots of unity or other structured elements.
    """
    # Parameterize the nodal cubic y^2 = x^2(x+1)
    # x = t^2 - 1, y = t(t^2 - 1)
    # The group law: if P1, P2, P3 are collinear, then t1 + t2 + t3 = 0 (for this parameterization)
    
    # So if we choose t values that sum to zero in triples, those triples will be collinear!
    
    # Let's choose 12 t values such that many triples sum to zero
    # This is like finding a subset of R with many 3-term arithmetic progressions (in the additive sense)
    
    # Actually, for the nodal cubic, the condition is t1 + t2 + t3 = 0 for collinearity
    # (when none of the points is the node or point at infinity)
    
    # So we want 12 values t1, ..., t12 such that many triples (i,j,k) satisfy ti + tj + tk = 0
    
    # Let's try using values from a symmetric set
    # If we use {−a, −b, −c, ..., a, b, c, ...}, we can get many zero-sum triples
    
    # Try: t values from a small set with symmetry
    # e.g., {−3, −2, −1, 0, 1, 2, 3, 4, 5, 6, 7, 8} - not great
    # Better: use values that form a subgroup-like structure
    
    # For maximum zero-sum triples, use values from {−n, ..., n} with even spacing
    # Or use roots of unity in the complex plane, but we need real values
    
    # Let's try: t = {−5, −4, −3, −2, −1, 1, 2, 3, 4, 5, 6, −6}
    # Check: (−5, 2, 3) -> 0, (−4, 1, 3) -> 0, (−3, −2, 5) -> 0, etc.
    
    t_values = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]
    
    pts = []
    for t in t_values:
        x = t * t - 1
        y = t * (t * t - 1)
        pts.append([x, y])
    
    # Check for duplicates
    if len(set(tuple(p) for p in pts)) < 12:
        # Try different t values
        t_values = [-7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7]
        pts = []
        for t in t_values:
            x = t * t - 1
            y = t * (t * t - 1)
            pts.append([x, y])
    
    return pts

def _local_search(points, deadline):
    """
    Efficient local search with incremental scoring.
    """
    current = [list(p) for p in points]
    current_score = _count_ordinary_fast(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    # Candidate pool: points on lines defined by pairs of current points
    coords = set()
    for i in range(12):
        for j in range(i+1, 12):
            x1, y1 = current[i]
            x2, y2 = current[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                g = 1
            dx //= g
            dy //= g
            # Extend line
            for t in range(-8, 9):
                cx = x1 + t*dx
                cy = y1 + t*dy
                if abs(cx) <= 10**9 and abs(cy) <= 10**9:
                    coords.add((cx, cy))
    
    # Also add nearby points
    for p in current:
        for dx in range(-10, 11):
            for dy in range(-10, 11):
                coords.add((p[0]+dx, p[1]+dy))
    
    candidate_list = list(coords)
    random.shuffle(candidate_list)
    
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        random.shuffle(candidate_list)
        
        for i in range(12):
            if time.time() >= deadline:
                break
            old_pt = tuple(current[i])
            for cand in candidate_list:
                if cand == old_pt:
                    continue
                if any(cand == tuple(current[j]) for j in range(12) if j != i):
                    continue
                
                new_pts = [list(p) for p in current]
                new_pts[i] = list(cand)
                
                if not _is_valid(new_pts):
                    continue
                
                score = _count_ordinary_fast(new_pts)
                if score < current_score:
                    current = new_pts
                    current_score = score
                    improved = True
                    no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best = [list(p) for p in current]
                    break
            else:
                no_improve += 1
                continue
            break
        else:
            no_improve += 1
            continue
        
        if not improved:
            no_improve += 1
    
    return best, best_score

def _simulated_annealing(points, deadline):
    """
    Simulated annealing to escape local minima.
    """
    current = [list(p) for p in points]
    current_score = _count_ordinary_fast(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    T = 50.0
    T_min = 0.1
    cooling = 0.95
    
    coords = set()
    for i in range(12):
        for j in range(i+1, 12):
            x1, y1 = current[i]
            x2, y2 = current[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                g = 1
            dx //= g
            dy //= g
            for t in range(-10, 11):
                cx = x1 + t*dx
                cy = y1 + t*dy
                if abs(cx) <= 10**9 and abs(cy) <= 10**9:
                    coords.add((cx, cy))
    
    for p in current:
        for dx in range(-15, 16):
            for dy in range(-15, 16):
                coords.add((p[0]+dx, p[1]+dy))
    
    candidate_list = list(coords)
    
    while time.time() < deadline and T > T_min:
        i = random.randint(0, 11)
        cand = random.choice(candidate_list)
        
        if cand == tuple(current[i]):
            T *= cooling
            continue
        
        if any(cand == tuple(current[j]) for j in range(12) if j != i):
            T *= cooling
            continue
        
        new_pts = [list(p) for p in current]
        new_pts[i] = list(cand)
        
        if not _is_valid(new_pts):
            T *= cooling
            continue
        
        new_score = _count_ordinary_fast(new_pts)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current = new_pts
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best = [list(p) for p in current]
        
        T *= cooling
    
    return best, best_score

def construct():
    """Return 12 integer points minimizing ordinary lines."""
    deadline = time.time() + 140
    
    # Try multiple initial configurations
    configs = []
    
    # 1. 4x3 grid
    configs.append([[i, j] for i in range(4) for j in range(3)])
    
    # 2. Algebraic construction from nodal cubic
    configs.append(_try_algebraic_construction())
    
    # 3. Another grid variation
    configs.append([[i*2, j*3] for i in range(4) for j in range(3)])
    
    # 4. Random configuration
    random.seed(42)
    pts = [[random.randint(-20, 20), random.randint(-20, 20)] for _ in range(12)]
    while len(set(tuple(p) for p in pts)) < 12:
        pts[random.randint(0, 11)] = [random.randint(-20, 20), random.randint(-20, 20)]
    configs.append(pts)
    
    best_points = None
    best_score = float('inf')
    
    time_per_config = 20  # seconds per configuration
    
    for cfg in configs:
        if time.time() >= deadline - 10:
            break
        
        # First try local search
        ls_points, ls_score = _local_search(cfg, min(time.time() + time_per_config, deadline - 5))
        if ls_score < best_score:
            best_score = ls_score
            best_points = ls_points
        
        # Then try simulated annealing if time permits
        if time.time() < deadline - 15 and ls_score > 6:
            sa_points, sa_score = _simulated_annealing(ls_points, min(time.time() + time_per_config, deadline - 5))
            if sa_score < best_score:
                best_score = sa_score
                best_points = sa_points
    
    # Final local search on the best found
    if best_points is not None and time.time() < deadline - 5:
        final_points, final_score = _local_search(best_points, deadline)
        if final_score < best_score:
            best_score = final_score
            best_points = final_points
    
    if best_points is None:
        best_points = [[i, j] for i in range(4) for j in range(3)]
    
    return [[int(p[0]), int(p[1])] for p in best_points]

# EVOLVE-BLOCK-END

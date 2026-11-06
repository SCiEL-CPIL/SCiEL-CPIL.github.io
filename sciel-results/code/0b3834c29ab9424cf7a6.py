<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations

def construct():
    # Greedy construction: maximize number of 3-collinear subsets while
    # enforcing the constraint that no line contains 4 or more points.
    # This creates a "clumpy" configuration that reduces the largest
    # general-position subset (no three collinear).
    
    random.seed(42)
    
    # Start with a few random points
    points = []
    
    # First, add some random points to get started
    for _ in range(5):
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        while [x, y] in points:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
        points.append([x, y])
    
    # Helper: check if adding new_point creates a 4-collinear line
    def is_valid_addition(points, new_point):
        nx, ny = new_point
        for i, j in combinations(range(len(points)), 2):
            px1, py1 = points[i]
            px2, py2 = points[j]
            # Check if new_point is collinear with points[i] and points[j]
            # Cross product: (px2 - px1)*(ny - py1) - (py2 - py1)*(nx - px1) == 0
            cross = (px2 - px1) * (ny - py1) - (py2 - py1) * (nx - px1)
            if cross == 0:
                # new_point is on the line through points[i] and points[j]
                # Check if there's already a third point on this line
                for k in range(len(points)):
                    if k == i or k == j:
                        continue
                    px3, py3 = points[k]
                    cross3 = (px2 - px1) * (py3 - py1) - (py2 - py1) * (px3 - px1)
                    if cross3 == 0:
                        # We'd have 4 points on this line
                        return False
        return True
    
    # Helper: count how many 3-collinear lines are created by adding new_point
    def count_new_3collinear(points, new_point):
        nx, ny = new_point
        count = 0
        for i, j in combinations(range(len(points)), 2):
            px1, py1 = points[i]
            px2, py2 = points[j]
            cross = (px2 - px1) * (ny - py1) - (py2 - py1) * (nx - px1)
            if cross == 0:
                count += 1
        return count
    
    # Greedily add points
    while len(points) < 20:
        best_candidate = None
        best_score = -1
        
        # Try random candidates
        for _ in range(500):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            candidate = [x, y]
            if candidate in points:
                continue
            if not is_valid_addition(points, candidate):
                continue
            score = count_new_3collinear(points, candidate)
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        if best_candidate is not None:
            points.append(best_candidate)
        else:
            # Fallback: just add a valid point
            while True:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                candidate = [x, y]
                if candidate not in points and is_valid_addition(points, candidate):
                    points.append(candidate)
                    break
    
    return points
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations

def construct():
    """
    Construct 20 integer points with |x|,|y| <= 100, no four collinear,
    minimizing the largest subset with no three collinear.
    
    Strategy: Use a structured grid-based approach to create many 3-point
    collinear triples, then optimize via local search.
    """
    random.seed(42)
    
    def get_line_key(p1, p2):
        """Get normalized line representation for two points."""
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None
        if dx == 0:
            return (1, 0, x1)  # vertical line x = x1
        if dy == 0:
            return (0, 1, y1)  # horizontal line y = y1
        # Normalize: make dx positive, and gcd(dx,dy) divides both
        g = abs(__import__('math').gcd(dx, dy))
        dx_n = dx // g
        dy_n = dy // g
        if dx_n < 0:
            dx_n = -dx_n
            dy_n = -dy_n
        # Line: dy_n * x - dx_n * y = c
        c = dy_n * x1 - dx_n * y1
        return (dx_n, dy_n, c)
    
    def count_3collinear(points):
        """Count number of 3-point collinear subsets."""
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                key = get_line_key(points[i], points[j])
                if key is None:
                    continue
                for k in range(j+1, n):
                    if get_line_key(points[i], points[k]) == key:
                        count += 1
        return count
    
    def check_no_4collinear(points):
        """Check that no line contains 4 or more points."""
        from collections import defaultdict
        line_points = defaultdict(set)
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                key = get_line_key(points[i], points[j])
                if key is not None:
                    line_points[key].add(i)
        for key, idxs in line_points.items():
            if len(idxs) >= 4:
                return False
        return True
    
    def compute_score(points):
        """
        Compute the largest subset with no three collinear.
        This is an approximation using greedy + local search.
        """
        n = len(points)
        # Precompute which triples are collinear
        collinear = set()
        for i in range(n):
            for j in range(i+1, n):
                key = get_line_key(points[i], points[j])
                if key is None:
                    continue
                for k in range(j+1, n):
                    if get_line_key(points[i], points[k]) == key:
                        collinear.add((i, j, k))
        
        # Greedy: start with all points, remove points that are in collinear triples
        # We want to find the largest subset with NO three collinear
        # This is the complement of finding a minimum vertex cover of collinear triples
        
        # Use a simple greedy approach
        remaining = set(range(n))
        # Try to find a large independent set in the 3-uniform hypergraph
        # Greedy: repeatedly pick a point that is in the fewest collinear triples with other remaining points
        while remaining:
            # Find point with minimum degree in collinear triples
            min_deg = float('inf')
            min_point = None
            for p in remaining:
                deg = 0
                for q in remaining:
                    if q <= p:
                        continue
                    for r in remaining:
                        if r <= q:
                            continue
                        if (p, q, r) in collinear or (q, p, r) in collinear or (r, p, q) in collinear or \
                           (p, r, q) in collinear or (q, r, p) in collinear or (r, q, p) in collinear:
                            deg += 1
                if deg < min_deg:
                    min_deg = deg
                    min_point = p
                if min_deg == 0:
                    break
            # Remove min_point from consideration (it's "safe" to include in the no-3-collinear set)
            # Actually, we want to find the LARGEST subset with no 3 collinear
            # This is NP-hard in general, so we use a heuristic
            remaining.discard(min_point)
        
        # The answer is n - len(remaining) if we removed all "bad" points
        # But this greedy isn't quite right. Let's use a different approach.
        
        # Better approach: try to build the largest subset with no 3 collinear
        # Start with empty set, add points one by one if they don't create a 3-collinear
        best_subset = set()
        # Try multiple orderings
        for trial in range(20):
            order = list(range(n))
            random.shuffle(order)
            current = set()
            for p in order:
                # Check if adding p creates any 3-collinear with two in current
                valid = True
                for q in current:
                    for r in current:
                        if q == r:
                            continue
                        if (p, q, r) in collinear or (q, p, r) in collinear or (r, p, q) in collinear:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    current.add(p)
            if len(current) > len(best_subset):
                best_subset = current
        
        return len(best_subset)
    
    # Initial construction: use a structured pattern
    # Place points on a grid with some perturbation to create 3-collinear triples
    points = []
    
    # Create a base grid and add points that form 3-collinear patterns
    # Use lines with 3 points each
    
    # Define some 3-point lines
    lines = [
        [(-50, -50), (0, 0), (50, 50)],      # diagonal
        [(-50, 0), (0, 10), (50, 20)],       # slight slope
        [(-40, -40), (0, -10), (40, 20)],    # another slope
        [(-30, 30), (0, 0), (30, -30)],      # anti-diagonal
        [(-60, 0), (0, 30), (60, 60)],       # another
        [(-20, -60), (0, -30), (20, 0)],     # steep
        [(-70, 10), (0, 40), (70, 70)],      # 
        [(-10, -70), (0, -40), (10, -10)],   #
        [(-80, 20), (0, 50), (80, 80)],      #
        [(-15, -55), (0, -25), (15, 5)],     #
    ]
    
    # Add points from these lines, ensuring no 4-collinear
    for line in lines:
        for p in line:
            if p not in points and check_no_4collinear(points + [p]):
                points.append(p)
    
    # Fill remaining points randomly while maintaining constraints
    while len(points) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        p = [x, y]
        if p not in points and check_no_4collinear(points + [p]):
            points.append(p)
    
    # Local search: try to improve by swapping points
    best_points = points[:]
    best_score = compute_score(best_points)
    
    for iteration in range(500):
        # Try swapping two points
        i = random.randint(0, 19)
        j = random.randint(0, 19)
        if i == j:
            continue
        
        new_points = best_points[:]
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        new_points[i] = [x, y]
        
        if new_points[i] not in [p for k, p in enumerate(new_points) if k != i] and \
           check_no_4collinear(new_points):
            score = compute_score(new_points)
            if score < best_score:
                best_points = new_points
                best_score = score
    
    return best_points
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

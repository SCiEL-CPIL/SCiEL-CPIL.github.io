import time
import itertools

def construct():
    deadline = time.time() + 115.0
    
    # Hardcoded configuration: 6 horizontal lines of 3 points + 2 isolated points
    # This guarantees no four collinear and gives score 14 (each line contributes at most 2, plus 2 isolated)
    base_pts = [
        [10, 0], [11, 0], [12, 0],
        [-12, 2], [-11, 2], [-10, 2],
        [10, 4], [11, 4], [12, 4],
        [-12, 6], [-11, 6], [-10, 6],
        [10, 8], [11, 8], [12, 8],
        [-12, 10], [-11, 10], [-10, 10],
        [50, 50], [-50, -50]
    ]
    
    def is_valid(points):
        if len(points) != 20:
            return False
        if len(set(map(tuple, points))) != 20:
            return False
        for p in points:
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
        for comb in itertools.combinations(points, 4):
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = comb
            d1 = (x2 - x1) * (y3 - y1)
            d2 = (x3 - x1) * (y2 - y1)
            if d1 != d2:
                continue
            d3 = (x2 - x1) * (y4 - y1)
            d4 = (x4 - x1) * (y2 - y1)
            if d3 != d4:
                continue
            return False
        return True
    
    if not is_valid(base_pts):
        return [[x, x * x] for x in range(-10, 10)]
    
    # Try to improve via local search: swap a point on a line with a new point
    # to potentially break up a line of 3 into two separate points that don't form
    # a new line of 3, potentially allowing a smaller independence number.
    
    # Actually, let's think about this more carefully. The current config has score 14.
    # To get below 14, we need a configuration where the largest subset with no 3 collinear is smaller.
    
    # Let's try a different structure: 5 lines of 4 points? No, that violates no-four-collinear.
    # We need exactly no four collinear. So max line size is 3.
    
    # What if we use more lines? 20 points, max 3 per line.
    # If all 20 points are on lines of 3, that's at least ceil(20/3) = 7 lines.
    # But lines can share points? No, if two lines share 2 points, they're the same line.
    # So lines are disjoint in points (each point is on at most one line of 3, or could be on multiple lines of 2, etc.)
    
    # Actually, a point can be on multiple lines. E.g., a point can be on a line of 3 AND another line of 2.
    # The constraint is: no 4 points on the same line.
    
    # Let's try a random search with local improvement.
    
    import random
    random.seed(42)
    
    def count_collinear3(points):
        """Count number of collinear triples."""
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    if (x2-x1)*(y3-y1) == (x3-x1)*(y2-y1):
                        count += 1
        return count
    
    def largest_no3_collinear(points):
        """Find the largest subset with no 3 collinear using branch and bound."""
        n = len(points)
        # Precompute: for each pair, which other points are collinear with them
        collinear = [[set() for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                for k in range(n):
                    if k == i or k == j:
                        x3, y3 = points[k]
                        if (x2-x1)*(y3-y1) == (x3-x1)*(y2-y1):
                            collinear[i][j].add(k)
                            collinear[j][i].add(k)
        
        # Branch and bound
        best = [0]
        used = [False] * n
        
        def backtrack(start, selected, size):
            if size > best[0]:
                best[0] = size
            # Upper bound: size + (n - start)
            if size + (n - start) <= best[0]:
                return
            for i in range(start, n):
                if not used[i]:
                    # Check if adding i creates a collinear triple
                    valid = True
                    for j in selected:
                        if i in collinear[i][j]:
                            valid = False
                            break
                    if valid:
                        selected.append(i)
                        backtrack(i+1, selected, size+1)
                        selected.pop()
        
        backtrack(0, [], 0)
        return best[0]
    
    # Verify base config score
    score = largest_no3_collinear(base_pts)
    
    if time.time() > deadline:
        return base_pts
    
    # Local search: try swapping one point
    best_pts = base_pts[:]
    best_score = score
    
    # Try random perturbations
    for _ in range(200):
        if time.time() > deadline:
            break
        cand = best_pts[:]
        idx = random.randint(0, 19)
        # Generate a random new point
        while True:
            nx = random.randint(-100, 100)
            ny = random.randint(-100, 100)
            if not any(cand[i][0] == nx and cand[i][1] == ny for i in range(20)):
                break
        cand[idx] = [nx, ny]
        
        if not is_valid(cand):
            continue
        
        s = largest_no3_collinear(cand)
        if s < best_score:
            best_pts = cand
            best_score = s
    
    return best_pts

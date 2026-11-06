"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Strategy: Use a structured set of points that creates many collinear triples
    # while ensuring no 4 points are collinear.
    # 
    # Approach: Start with a grid and carefully select points to maximize
    # the number of collinear triples. We'll use points from a small grid
    # region and verify constraints.
    
    # Candidate: Use points arranged to create many 3-collinear sets
    # A good structure is to use points on multiple short lines (length 3)
    # that intersect, creating a dense collinearity hypergraph.
    
    # Let's try a construction based on a 5x4 grid subset
    # Points chosen to have many collinear triples
    
    # Grid points: x in {0,1,2,3,4}, y in {0,1,2,3}
    # But we need 20 points and no 4 collinear.
    # In a 5x4 grid, rows have 5 points (4+ collinear issue), columns have 4.
    # So we can't use full rows or columns.
    
    # Better: Use a more creative construction.
    # Let's place points to form many triangles of collinearity.
    
    # Construction: Use points from a few arithmetic progressions
    # Line 1: (0,0), (1,1), (2,2) - 3 points
    # Line 2: (0,1), (1,2), (2,3) - 3 points  
    # Line 3: (0,2), (1,3), (2,4) - 3 points
    # Line 4: (3,0), (4,1), (5,2) - 3 points
    # Line 5: (3,1), (4,2), (5,3) - 3 points
    # Line 6: (3,2), (4,3), (5,4) - 3 points
    # That's 18 points. Add 2 more carefully.
    
    # Let me try a different approach: use a known good configuration
    # or do a quick local search.
    
    # Initial candidate: mix of grid points
    # Use x in [0..9], y in [0..9] subset
    
    # Let's build a set with many collinear triples
    # Use points that lie on multiple lines of 3
    
    # Construction using modular arithmetic for collinearity
    # Points (i, i^2 mod p) for prime p, but we need integer coords <= 100
    
    # Simpler: hand-craft a dense configuration
    # 20 points with many 3-collinear subsets
    
    # Let's use a 4x5 arrangement but break 4-collinearities
    # by shifting some points
    
    base_points = []
    # Row 0: y=0, x = 0,2,4,6,8 (5 points, but 4+ collinear!)
    # Need to avoid 4 on a line.
    
    # Use a different structure: points on a circle-like pattern with collinear triples
    
    # Let me try: 20 points where we create as many collinear triples as possible
    # while keeping max collinear = 3
    
    # Grid-based: x in {0,1,2,3,4,5}, y in {0,1,2,3,4,5}
    # Select 20 from 36, avoiding 4-collinear
    
    # Quick heuristic: start with all grid points, remove to break 4-collinearities
    # then check score
    
    import random
    random.seed(42)
    
    # Generate candidate: 20 points from a 10x10 grid
    # Use local search to maximize collinear triples
    
    def count_3collinear(points):
        """Count number of collinear triples in the set."""
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    # Check collinearity: (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1)
                    if (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1):
                        count += 1
        return count
    
    def has_4collinear(points):
        """Check if any 4 points are collinear."""
        n = len(points)
        # Group points by line
        lines = {}
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                # Normalize line representation
                dx = x2 - x1
                dy = y2 - y1
                # Find gcd to normalize
                g = abs(dx)
                if g == 0:
                    g = abs(dy)
                else:
                    g2 = abs(dy)
                    while g2:
                        g, g2 = g2, g % g2
                if g == 0:
                    continue
                dx_n = dx // g
                dy_n = dy // g
                # Normalize sign
                if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                    dx_n = -dx_n
                    dy_n = -dy_n
                # Line: point (x1,y1) with direction (dx_n, dy_n)
                # Normalize by using the smallest point on the line
                # For simplicity, use (dx_n, dy_n, x1*dy_n - y1*dx_n) as key
                # But this might not be unique. Let's use a different approach.
                pass
        
        # Simpler: for each pair, count how many points are on that line
        from collections import defaultdict
        line_points = defaultdict(set)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                g = abs(dx)
                if g == 0:
                    g = abs(dy)
                else:
                    g2 = abs(dy)
                    while g2:
                        g, g2 = g2, g % g2
                if g == 0:
                    continue
                dx_n = dx // g
                dy_n = dy // g
                if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                    dx_n = -dx_n
                    dy_n = -dy_n
                # Canonical point: the one with smaller x, then smaller y
                if (x1, y1) < (x2, y2):
                    px, py = x1, y1
                else:
                    px, py = x2, y2
                key = (dx_n, dy_n, px * dy_n - py * dx_n)
                line_points[key].add(i)
                line_points[key].add(j)
        
        for pts in line_points.values():
            if len(pts) >= 4:
                return True
        return False
    
    def largest_no3collinear(points):
        """Find the largest subset with no three collinear using branch and bound."""
        n = len(points)
        if n <= 2:
            return n
        
        # Precompute collinear triples
        collinear = set()
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    if (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1):
                        collinear.add((i, j, k))
        
        # Branch and bound
        best = 0
        
        def backtrack(remaining, current_size):
            nonlocal best
            if not remaining:
                if current_size > best:
                    best = current_size
                return
            if current_size + len(remaining) <= best:
                return
            
            # Try adding the first point
            first = remaining[0]
            rest = remaining[1:]
            
            # Check if first can be added (no collinear triple with two in current)
            # We need to track which points are in current set
            # This is getting complex. Let's use a simpler approach.
            pass
        
        # Simpler: use greedy + local search for the maximum independent set
        # in the 3-uniform hypergraph
        
        # For now, use a heuristic: try to find large subset with no 3 collinear
        # by random sampling and checking
        
        import random
        random.seed(123)
        
        best_size = 0
        best_subset = None
        
        # Greedy: start with all points, remove points involved in collinear triples
        # until no collinear triples remain
        
        # Try multiple random orderings
        for trial in range(50):
            random.seed(trial)
            indices = list(range(n))
            random.shuffle(indices)
            
            current = set()
            for idx in indices:
                # Check if adding idx creates a collinear triple
                can_add = True
                for i, j in itertools.combinations(sorted(current), 2):
                    if (idx, i, j) in collinear or (i, j, idx) in collinear or \
                       (i, idx, j) in collinear or (i, j, idx) in collinear or \
                       (j, i, idx) in collinear or (j, idx, i) in collinear:
                        # Check all permutations
                        triple = tuple(sorted((idx, i, j)))
                        if triple in collinear:
                            can_add = False
                            break
                if can_add:
                    current.add(idx)
            
            if len(current) > best_size:
                best_size = len(current)
                best_subset = current
        
        return best_size
    
    # Now do local search on point configurations
    # Start with a random set of 20 points from a grid
    
    grid_size = 15
    all_candidates = [[x, y] for x in range(-5, grid_size-5) for y in range(-5, grid_size-5)]
    
    best_score = float('inf')
    best_config = None
    
    # Initial: try a structured configuration
    # Use points that create many collinear triples
    
    # Configuration 1: Points on a few lines of 3
    config1 = [
        [0, 0], [1, 1], [2, 2],      # line y=x
        [0, 1], [1, 2], [2, 3],      # line y=x+1
        [0, 2], [1, 3], [2, 4],      # line y=x+2
        [3, 0], [4, 1], [5, 2],      # line y=x-3
        [3, 1], [4, 2], [5, 3],      # line y=x-2
        [3, 2], [4, 3], [5, 4],      # line y=x-1
        [6, 0], [7, 1], [8, 2],      # line y=x-6
        [0, 5], [1, 6], [2, 7],      # line y=x+5
    ]
    # That's 21 points, remove one to get 20
    config1 = config1[:20]
    
    if not has_4collinear(config1):
        score = largest_no3collinear(config1)
        if score < best_score:
            best_score = score
            best_config = [p[:] for p in config1]
    
    # Try local search: swap points to increase collinear triples
    if time.time() < deadline:
        current = [p[:] for p in best_config] if best_config else config1[:20]
        
        for iteration in range(200):
            if time.time() > deadline - 20:
                break
            
            # Try swapping one point with another from the grid
            improved = False
            i = random.randrange(20)
            new_pt = random.choice(all_candidates)
            
            if new_pt == current[i]:
                continue
            
            old_pt = current[i][:]
            current[i] = new_pt
            
            if has_4collinear(current):
                current[i] = old_pt
                continue
            
            new_score = largest_no3collinear(current)
            if new_score < best_score:
                best_score = new_score
                best_config = [p[:] for p in current]
                improved = True
            else:
                current[i] = old_pt
        
        if not improved and best_config is None:
            best_config = config1[:20]
    
    return best_config if best_config else config1[:20]
# EVOLVE-BLOCK-END


import time
import random
import sys

def _check_no_four_collinear(points):
    """Check if no four points are collinear. Returns True if valid."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
            if count >= 4:
                return False
    return True

def _max_independent_set(points):
    """
    Exact branch and bound for the largest subset with no three collinear.
    Optimized with bitmask and precomputed conflicts.
    """
    n = len(points)
    if n == 0:
        return 0
    
    # Precompute collinear triples
    # conflict_mask[i] is a bitmask of points j such that there exists k 
    # where i, j, k are collinear? No, that's not quite right for 3-uniform.
    
    # Better: Precompute for each pair (i, j), the set of k such that i, j, k collinear.
    # Let's use a simpler representation for the backtracking:
    # A set S is valid if it contains no collinear triple.
    
    # Precompute: for each point i, and each other point j, 
    # does adding i to a set containing j conflict?
    # Actually, the standard approach for "Max Subset with No 3 Collinear" is:
    # Build the set incrementally.
    # State: current_set (list of indices), candidates (list of indices).
    # Pruning: len(current_set) + len(candidates) <= best.
    
    # To speed up the "can I add point p?" check:
    # If I have current_set S, and I want to add p, I need to check if there exist
    # i, j in S such that i, j, p are collinear.
    # Let's precompute for each point p, a list of pairs (i, j) such that i, j, p are collinear.
    # Then, when adding p, I just check if any of those pairs is fully contained in S.
    
    collinear_pairs_for_point = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    # i, j, k are collinear
                    collinear_pairs_for_point[k].add((i, j))
                    collinear_pairs_for_point[i].add((j, k))
                    collinear_pairs_for_point[j].add((i, k))
                    
    # Actually, for the check "can I add p to S", I need to know if there is any pair (u, v) in S
    # such that u, v, p are collinear.
    # Let's precompute for each p, the set of pairs (u, v) with u < v such that u, v, p are collinear.
    pairs_for_p = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    pairs_for_p[k].add((i, j))
                    pairs_for_p[i].add((j, k))
                    pairs_for_p[j].add((i, k))
    
    best = 0
    
    def backtrack(current_set, candidates):
        nonlocal best
        if len(current_set) + len(candidates) <= best:
            return
            
        if not candidates:
            if len(current_set) > best:
                best = len(current_set)
            return

        # Heuristic: Pick the candidate with the most conflicts?
        # Or just pick the first one.
        # Let's pick the one that is "most constrained" (forms most collinear triples with current_set?)
        # Actually, picking the one that eliminates the most candidates is good.
        
        # For simplicity and speed in Python, let's just pick the first candidate.
        # n=20 is small.
        
        p_idx = candidates[0]
        rest_candidates = candidates[1:]
        
        # Branch 1: Include p_idx
        # Check if p_idx can be added to current_set
        valid = True
        for pair in pairs_for_p[p_idx]:
            u, v = pair
            if u in current_set and v in current_set:
                valid = False
                break
                
        if valid:
            new_set = current_set + [p_idx]
            # Filter candidates: remove any k that cannot be added to new_set
            new_candidates = []
            for k in rest_candidates:
                # Check if k can be added to new_set
                can_add = True
                for pair in pairs_for_p[k]:
                    u, v = pair
                    if u in new_set and v in new_set:
                        can_add = False
                        break
                if can_add:
                    new_candidates.append(k)
            backtrack(new_set, new_candidates)
        
        # Branch 2: Exclude p_idx
        backtrack(current_set, rest_candidates)

    backtrack([], list(range(n)))
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Strategy: Constructive Greedy
    # We want to place 20 points such that no 4 are collinear, but we maximize the number of collinear triples.
    # A collinear triple "blocks" the pair of points on that line from being in a 3-free set with the third point.
    # If we can make every pair of points part of at least one collinear triple, the score will be low.
    
    # Start with a small seed set that has good properties.
    # A regular polygon or grid fragment?
    # Let's try to build from scratch.
    
    best_points = None
    best_score = float('inf')
    
    def generate_greedy():
        pts = []
        # Start with 3 non-collinear points
        pts.append([0, 0])
        pts.append([1, 0])
        pts.append([0, 1])
        
        while len(pts) < 20:
            # Identify all lines defined by pairs of existing points
            # For each line, count how many points are on it.
            # We want to place a new point on a line that currently has exactly 2 points.
            # This creates a triple.
            # We must avoid placing a point on a line that already has 3 points (would make 4).
            # We must also avoid placing a point that creates a 4-collinear with other existing points not on the same line?
            # No, if we place on a line with 2 points, we create a triple. 
            # We need to check if this new point creates a 4-collinear with any OTHER line.
            
            candidate_lines = []
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    # Normalize direction? Not strictly needed for counting, but good for grouping.
                    # Let's just count points on this specific line segment extended.
                    count = 2
                    for k in range(len(pts)):
                        if k == i or k == j:
                            continue
                        x3, y3 = pts[k]
                        if dx * (y3 - y1) == dy * (x3 - x1):
                            count += 1
                            
                    if count == 2:
                        # This line has exactly 2 points. We can place a 3rd.
                        # Where? We need to find an integer point on this line that is not already used.
                        # And check if placing it there creates any 4-collinear with other lines.
                        candidate_lines.append((i, j, dx, dy, x1, y1))
            
            if not candidate_lines:
                # No lines with exactly 2 points? This might happen if we are stuck.
                # In that case, we might need to break the symmetry or add a point that doesn't form a triple?
                # Or maybe we should have chosen differently earlier.
                # For now, pick a random valid point to unstick.
                while True:
                    x = random.randint(-50, 50)
                    y = random.randint(-50, 50)
                    p = [x, y]
                    if p not in pts:
                        # Check validity
                        temp_pts = pts + [p]
                        if _check_no_four_collinear(temp_pts):
                            pts.append(p)
                            break
                continue
            
            # Pick a random candidate line to place the next point
            # Prioritize lines where we can place multiple options?
            random.shuffle(candidate_lines)
            placed = False
            for i, j, dx, dy, x1, y1 in candidate_lines:
                # Find integer points on this line
                # Equation: dy * (x - x1) - dx * (y - y1) = 0
                # dy * x - dx * y = dy * x1 - dx * y1
                # We need integer solutions (x, y).
                # GCD(dy, dx) must divide (dy * x1 - dx * y1).
                # Since (x1, y1) and (x2, y2) are integers, this is always true.
                
                # Let's just search for a point in a reasonable range
                # We want to find a point (x, y) such that it is on the line, 
                # not in pts, and doesn't create 4-collinear.
                
                # Try points near the existing ones? Or spread out?
                # Let's try a few random points on the line.
                # Parametric form: x = x1 + t * dx/g, y = y1 + t * dy/g where g = gcd(dx, dy)
                # To keep it simple, let's just iterate t from -10 to 10 and check.
                
                g = abs(dx)
                if g == 0:
                    g = 1
                # Actually, just try a range of t values
                found = False
                for t in range(-20, 21):
                    if t == 0: continue # Avoid existing points? Not necessarily, but dx,dy are from i,j so t=0 is not on line unless i=j which is false.
                    # Wait, the line is defined by i and j.
                    # Let's use a simpler method: just pick a random integer point in the box and check if it's on the line.
                    pass
                
                # More robust: 
                # Generate a few candidate points on the line
                # We know (x1, y1) and (x2, y2) are on the line.
                # Vector v = (dx, dy).
                # Points are (x1 + t*dx, y1 + t*dy) for integer t? No, only if dx, dy are coprime.
                # Let's just sample random integer points in a box and check collinearity with i, j.
                
                for _ in range(50):
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    p = [x, y]
                    if p in pts:
                        continue
                    # Check if p is on line ij
                    if dx * (y - y1) != dy * (x - x1):
                        continue
                    # Check if adding p creates 4-collinear
                    temp_pts = pts + [p]
                    if _check_no_four_collinear(temp_pts):
                        pts.append(p)
                        placed = True
                        break
                if placed:
                    break
            
            if not placed:
                # Fallback: add a random valid point
                while True:
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    p = [x, y]
                    if p not in pts:
                        temp_pts = pts + [p]
                        if _check_no_four_collinear(temp_pts):
                            pts.append(p)
                            break
        
        return pts

    # Run multiple greedy constructions
    for _ in range(20):
        if time.time() > deadline:
            break
        pts = generate_greedy()
        if len(pts) != 20:
            continue
        score = _max_independent_set(pts)
        if score < best_score:
            best_score = score
            best_points = pts[:]
            
    # If we have time, do some local search on the best found
    if best_points and time.time() < deadline - 5:
        current = best_points[:]
        current_score = best_score
        T = 1.0
        while time.time() < deadline - 2:
            idx = random.randint(0, 19)
            old_p = current[idx]
            new_p = [random.randint(-100, 100), random.randint(-100, 100)]
            if new_p == old_p or new_p in current:
                continue
            current[idx] = new_p
            if not _check_no_four_collinear(current):
                current[idx] = old_p
                continue
            new_score = _max_independent_set(current)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_points = current[:]
            else:
                current[idx] = old_p
            T *= 0.99

    if best_points is None:
        # Fallback: return a known valid configuration
        # Simple grid 5x4? No, 4 collinear.
        # Parabola: (i, i^2) for i in -5..4? 
        # (0,0), (1,1), (2,4), (3,9), (4,16), (-1,1), (-2,4)...
        # Check 4 collinear on parabola? No 3 are collinear on y=x^2.
        # So score is 20.
        pts = []
        for i in range(-10, 10):
            pts.append([i, i*i])
            if len(pts) == 20:
                break
        # Verify
        if not _check_no_four_collinear(pts):
             # Should not happen for parabola
             pts = [[0,0], [1,0], [0,1], [1,1], [2,0], [2,1], [3,0], [3,1], [4,0], [4,1],
                    [0,2], [1,2], [2,2], [3,2], [4,2], [0,3], [1,3], [2,3], [3,3], [4,3]]
             # This grid has 4 collinear. 
             # Let's just return the parabola.
             pts = [[i, i*i] for i in range(20)]
        best_points = pts

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_max_independent_set(pts)}")
    print(f"Valid: {_check_no_four_collinear(pts)}")


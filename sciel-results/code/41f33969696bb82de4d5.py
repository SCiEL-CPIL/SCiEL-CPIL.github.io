import time
import random
import itertools
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def is_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])
    
    def line_key(p1, p2):
        """Compute a canonical key for the line through p1 and p2."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0, p1[0], p1[1])
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        c = dy * p1[0] - dx * p1[1]
        return (dx, dy, c)
    
    def count_triples_and_check_four(points):
        n = len(points)
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                line_counts[key] = line_counts.get(key, 0) + 1
        
        triples = 0
        has_four = False
        for key, count in line_counts.items():
            if count >= 4:
                has_four = True
            if count >= 3:
                triples += count * (count - 1) * (count - 2) // 6
        
        return triples, has_four
    
    def get_score(points):
        triples, has_four = count_triples_and_check_four(points)
        if has_four:
            return -1000000
        return triples
    
    def find_lines_with_three(points):
        n = len(points)
        line_to_points = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                if key not in line_to_points:
                    line_to_points[key] = set()
                line_to_points[key].add(i)
                line_to_points[key].add(j)
        
        lines_three = []
        for key, pts in line_to_points.items():
            if len(pts) == 3:
                lines_three.append((key, pts))
        return lines_three
    
    def max_subset_no_three_collinear(points):
        """Exact branch and bound for largest subset with no three collinear."""
        n = len(points)
        if n <= 2:
            return n
        
        # Build conflict graph: edge between i,j if there exists k such that i,j,k collinear
        # Actually we need: a subset is valid if no three points in it are collinear
        # So we need to find max subset where no line contains 3+ of the chosen points
        
        # Find all lines with their point sets
        line_to_points = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                if key not in line_to_points:
                    line_to_points[key] = set()
                line_to_points[key].add(i)
                line_to_points[key].add(j)
        
        # Only lines with 3+ points matter
        bad_lines = [pts for pts in line_to_points.values() if len(pts) >= 3]
        
        if not bad_lines:
            return n
        
        # Branch and bound: we want max subset S such that for every line L, |S ∩ L| <= 2
        # This is equivalent to: pick points, and for each bad line, we can pick at most 2 from it
        
        # Use a simple recursive approach with memoization on remaining points
        # But n=20 is too large for full memoization. Use branch and bound.
        
        best = [0]
        
        def branch(remaining, chosen_count, chosen_set):
            if not remaining:
                if chosen_count > best[0]:
                    best[0] = chosen_count
                return
            
            # Upper bound: chosen_count + len(remaining)
            if chosen_count + len(remaining) <= best[0]:
                return
            
            # Check if adding all remaining would violate constraints
            # For each bad line, count how many are in chosen_set + remaining
            # If any line would have 3+ total, we can't add all
            
            # Try branching on the first point in remaining
            first = remaining[0]
            rest = remaining[1:]
            
            # Option 1: don't include first
            branch(rest, chosen_count, chosen_set)
            
            # Option 2: include first (if it doesn't create a violation)
            new_chosen = chosen_set | {first}
            valid = True
            for line_pts in bad_lines:
                count = sum(1 for p in new_chosen if p in line_pts)
                if count > 2:
                    valid = False
                    break
            if valid:
                branch(rest, chosen_count + 1, new_chosen)
        
        branch(list(range(n)), 0, frozenset())
        return best[0]
    
    def construct_deterministic():
        """
        Use a structured approach: pick x-coordinates and y-coordinates from
        sets that minimize collinear triples.
        
        Strategy: Use points (x_i, f(x_i)) where f is a function that avoids
        three points being collinear. A quadratic function y = ax^2 + bx + c
        has the property that no three points on the parabola are collinear
        (for distinct x values). But we need 20 points with |x|,|y| <= 100.
        
        Actually, for a parabola y = x^2, three points (x1, x1^2), (x2, x2^2), (x3, x3^2)
        are collinear iff x1=x2=x3. So any 3 distinct x values give non-collinear points.
        But we need 20 points, and x^2 grows fast. For x in range(-5, 6), y = x^2 gives
        y in [0, 25]. That's only 11 points.
        
        Better: use multiple curves or a grid with perturbation.
        
        Let's try: take 20 points on a circle or ellipse, which has no three collinear.
        But we need integer coordinates.
        
        Alternative: Use the fact that for any set of points on a strictly convex curve
        like y = x^2, no three are collinear. But we're limited to |y| <= 100.
        
        For y = x^2: x in [-10, 10] gives y in [0, 100]. That's 21 points!
        Points: (-10, 100), (-9, 81), ..., (0, 0), ..., (9, 81), (10, 100)
        
        But wait - are any three of these collinear? For a parabola, no three distinct
        points are collinear. So this gives score 20 (the entire set has no three collinear).
        
        But the task says "no four collinear" as a constraint, and we want to MINIMIZE
        the largest subset with no three collinear. So we WANT many collinear triples!
        
        We need to MAXIMIZE collinear triples while keeping no four collinear.
        """
        # We want to pack points onto lines to create many triples, but no line has 4+.
        # Each line with exactly 3 points contributes C(3,3) = 1 triple.
        # We want to maximize the number of such lines.
        
        # Start with a grid and perturb to break 4-collinear while keeping 3-collinear.
        pts = []
        # Use a 4x5 grid with perturbations
        for x in range(4):
            for y in range(5):
                pts.append([x * 20 + random.randint(-2, 2), y * 20 + random.randint(-2, 2)])
        
        # Ensure no duplicates and within bounds
        seen = set()
        unique_pts = []
        for p in pts:
            p[0] = max(-100, min(100, p[0]))
            p[1] = max(-100, min(100, p[1]))
            if tuple(p) not in seen:
                seen.add(tuple(p))
                unique_pts.append(p)
        
        # If we lost points, add new ones
        while len(unique_pts) < 20:
            p = [random.randint(-100, 100), random.randint(-100, 100)]
            if tuple(p) not in seen:
                seen.add(tuple(p))
                unique_pts.append(p)
        
        return unique_pts[:20]
    
    def local_search(points, time_limit):
        current_pts = [p[:] for p in points]
        current_score = get_score(current_pts)
        best_pts = [p[:] for p in current_pts]
        best_score = current_score
        
        no_improve_count = 0
        
        while time.time() < time_limit:
            improved = False
            
            # Try moving each point to a nearby position
            for i in range(20):
                if time.time() > time_limit:
                    break
                
                old_p = current_pts[i]
                best_new_p = None
                best_new_score = current_score
                
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx == 0 and dy == 0:
                            continue
                        new_p = [old_p[0] + dx, old_p[1] + dy]
                        if -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                            if new_p not in current_pts:
                                test_pts = current_pts[:]
                                test_pts[i] = new_p
                                new_score = get_score(test_pts)
                                if new_score > best_new_score:
                                    best_new_score = new_score
                                    best_new_p = new_p
                
                if best_new_p is not None:
                    current_pts[i] = best_new_p
                    current_score = best_new_score
                    improved = True
                    no_improve_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_pts = [p[:] for p in current_pts]
                    break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count >= 5:
                    # Random jump
                    idx = random.randint(0, 19)
                    new_p = [random.randint(-100, 100), random.randint(-100, 100)]
                    if new_p not in current_pts:
                        test_pts = current_pts[:]
                        test_pts[idx] = new_p
                        new_score = get_score(test_pts)
                        if new_score > -1000000:
                            current_pts = test_pts
                            current_score = new_score
                            no_improve_count = 0
                else:
                    break
        
        return best_pts
    
    # Run local search from multiple initial configurations
    best_overall = None
    best_overall_score = -1
    
    # Try several different initial configurations
    for attempt in range(5):
        if time.time() > deadline - 10:
            break
        
        init_pts = construct_deterministic()
        
        # Run local search
        search_time = min(20, deadline - 10 - time.time())
        if search_time < 2:
            break
        
        current_best = local_search(init_pts, time.time() + search_time)
        score = get_score(current_best)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall = current_best
    
    return best_overall

import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    # Helper: Check if 4 points are collinear
    def are_4_collinear(pts):
        """pts is a tuple of 4 distinct points (x, y)."""
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts
        # Check if p1, p2, p3 are collinear
        if (x2 - x1) * (y3 - y1) != (x3 - x1) * (y2 - y1):
            return False
        # Check if p1, p2, p4 are collinear
        if (x2 - x1) * (y4 - y1) != (x4 - x1) * (y2 - y1):
            return False
        return True
    
    def has_4_collinear(points):
        """Check if any 4 points in the list are collinear."""
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        if are_4_collinear((points[i], points[j], points[k], points[l])):
                            return True
        return False
    
    def calc_score(points):
        """
        Calculate the largest subset of points with no three collinear.
        Uses a greedy/branch-and-bound approach for small n=20.
        Since we know no 4 are collinear, we just need to avoid 3.
        
        Actually, for n=20, an exact branch and bound might be too slow if not optimized.
        However, the problem states "exact branch and bound" for the evaluator.
        For our internal calculation to guide search, we can use a heuristic or a simpler check.
        But wait, the score is defined by the evaluator. We just need to return points.
        The evaluator will compute the score.
        
        However, to minimize the score, we need to *construct* points such that the max no-3-collinear subset is small.
        This is equivalent to maximizing the number of collinear triples (or lines with >=3 points), subject to no 4 collinear.
        
        Let's think about the structure.
        If we have a set of lines, each with exactly 3 points, and these lines are disjoint (share at most 1 point? No, if they share 2 points, those 2 points define a line, so the lines must be the same or share at most 1 point).
        Actually, any two distinct lines intersect in at most 1 point.
        So we can pack the 20 points into lines of size 3.
        Max number of disjoint lines of size 3 is floor(20/3) = 6.
        6 lines * 3 points = 18 points.
        The remaining 2 points can be anywhere, as long as they don't form a line of 3 with existing points or each other and a third.
        
        If we have 6 lines of 3, the "no 3 collinear" subset can pick at most 2 points from each line.
        So the size would be at most 6*2 + 2 = 14?
        Wait, the subset must have *no three collinear*.
        If we pick 2 points from each of the 6 lines, are any of those 12 points collinear?
        The points on different lines might form new collinearities.
        This is a complex combinatorial geometry problem.
        
        Let's try to construct a configuration with many 3-point lines.
        A known construction is using a grid or specific patterns.
        The "no 4 collinear" constraint is strict.
        
        Let's try a random search with local improvement.
        Objective: Minimize the size of the largest subset with no 3 collinear.
        Equivalent: Maximize the "collinearity density".
        
        Heuristic score for search:
        Count the number of collinear triples.
        Let T be the set of all collinear triples in the configuration.
        We want to maximize |T|.
        Note: If 4 points are collinear, it contains 4 triples. But 4 collinear is forbidden.
        So we only count triples that are *exactly* 3 (i.e., the 4th point on that line is not in the set).
        Since no 4 are collinear, every collinear triple is "exactly 3".
        
        So, maximize the number of collinear triples.
        """
        n = len(points)
        triples = 0
        # Precompute lines?
        # For each pair, find the line, count how many points are on it.
        # If count == 3, add 1 to triples.
        # If count == 4, invalid (but we check validity separately).
        
        line_counts = defaultdict(int)
        # Map line representation to count
        # Line representation: (A, B, C) normalized.
        # For points (x1,y1), (x2,y2), (x3,y3), (x4,y4)
        # Normalized line: (A, B, C) where Ax + By + C = 0
        # A = y1 - y2, B = x2 - x1, C = x1*y2 - x2*y1
        # Normalize by GCD and sign.
        
        def get_line(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            # Normalize
            from math import gcd
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g > 0:
                A //= g
                B //= g
                C //= g
            # Sign convention: first non-zero positive
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            return (A, B, C)
        
        # Count points on each line
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_counts[line] += 1
        
        # A line with k points contributes C(k,3) triples.
        # Since no 4 collinear, k is at most 3.
        # So if k==3, it contributes 1.
        # If k==2, 0.
        # If k==1, 0.
        
        for line, count in line_counts.items():
            if count == 3:
                triples += 1
            elif count > 3:
                # Should not happen if valid
                pass
                
        return triples

    def is_valid(points):
        """Check distinctness, bounds, and no 4 collinear."""
        if len(points) != 20:
            return False
        if len(set(tuple(p) for p in points)) != 20:
            return False
        for p in points:
            if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
                return False
        
        # Check no 4 collinear
        # Use line counting
        n = len(points)
        line_counts = defaultdict(int)
        
        def get_line(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            from math import gcd
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g > 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            return (A, B, C)
        
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_counts[line] += 1
        
        for line, count in line_counts.items():
            if count >= 4:
                return False
        return True

    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]
    
    # Initial configuration: Random
    current_points = []
    while len(current_points) < 20:
        p = random_point()
        if p not in current_points:
            current_points.append(p)
    
    # Ensure initial validity? Random points might have 4 collinear.
    # Let's try to fix it or restart.
    if not is_valid(current_points):
        # Try a few times to find a valid random start
        for _ in range(100):
            current_points = []
            while len(current_points) < 20:
                p = random_point()
                if p not in current_points:
                    current_points.append(p)
            if is_valid(current_points):
                break
    
    if not is_valid(current_points):
        # Fallback to parabola if random fails (unlikely)
        return [[x, x * x] for x in range(-10, 10)]
        
    current_score = calc_score(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # Local Search
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_p = current_points[idx][:]
        
        # Generate a new point
        # Try to move it to a position that creates more collinear triples
        # Simple: Random restart of that point
        new_p = random_point()
        
        # Try to improve by placing it on a line with 2 existing points
        # Find pairs of points in current_points (excluding idx)
        # and check if the line segment or extension has a good spot?
        # Or just check if placing new_p creates a triple.
        
        # Heuristic: Try to form a triple with two existing points.
        improved = False
        for i in range(20):
            if i == idx:
                continue
            for j in range(i + 1, 20):
                if j == idx:
                    continue
                p1 = current_points[i]
                p2 = current_points[j]
                # Line through p1, p2
                # Check if we can find a point on this line within bounds
                # that is not already in current_points (except p1, p2)
                # and is distinct from old_p
                # Actually, we just need to pick a point on the line.
                # To avoid 4 collinear, we must ensure no other point is on this line.
                # So this line must currently have exactly 2 points (p1, p2) in the set.
                
                # Check count of points on line p1-p2
                line = get_line(p1, p2)
                # Count
                cnt = 0
                for k in range(20):
                    if k == i or k == j:
                        continue
                    if get_line(p1, current_points[k]) == line:
                        cnt += 1
                if cnt == 0:
                    # Line has only p1 and p2.
                    # We can place a new point on this line.
                    # Generate a point on the line.
                    # p1 + t * (p2 - p1)
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    # Try small integer steps
                    found = False
                    for t in range(-10, 11):
                        if t == 0 or t == 1:
                            continue
                        nx = p1[0] + t * dx
                        ny = p1[1] + t * dy
                        if -100 <= nx <= 100 and -100 <= ny <= 100:
                            new_candidate = [nx, ny]
                            if new_candidate not in current_points:
                                # Check if this creates 4 collinear?
                                # We know line has 2 points. Adding one makes 3.
                                # So no 4 collinear on this line.
                                # But could it create 4 collinear on another line?
                                # The new point is on line L.
                                # It might form a line with p1 and some other point? No, p1 is on L.
                                # It might form a line with p2 and some other point? No.
                                # It might form a line with two other points not on L?
                                # Yes.
                                # So we must check validity of the whole set.
                                test_points = current_points[:]
                                test_points[idx] = new_candidate
                                if is_valid(test_points):
                                    test_score = calc_score(test_points)
                                    if test_score > current_score:
                                        current_points = test_points
                                        current_score = test_score
                                        if current_score > best_score:
                                            best_score = current_score
                                            best_points = [p[:] for p in current_points]
                                        improved = True
                                        break
                    if improved:
                        break
            if improved:
                break
        
        if not improved:
            # Try random move
            new_p = random_point()
            if new_p not in current_points:
                test_points = current_points[:]
                test_points[idx] = new_p
                if is_valid(test_points):
                    test_score = calc_score(test_points)
                    if test_score > current_score:
                        current_points = test_points
                        current_score = test_score
                        if current_score > best_score:
                            best_score = current_score
                            best_points = [p[:] for p in current_points]
                    else:
                        # Accept with some probability? No, we want to maximize.
                        # Maybe accept if equal to escape local optima?
                        if test_score == current_score and random.random() < 0.1:
                            current_points = test_points
                            current_score = test_score
    
    # Final check
    if is_valid(best_points):
        return best_points
    else:
        # Fallback
        return [[x, x * x] for x in range(-10, 10)]

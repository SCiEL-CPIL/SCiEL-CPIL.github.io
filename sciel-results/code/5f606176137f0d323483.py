import random
import time
import itertools
from typing import List, Tuple

def construct():
    """
    Returns 20 distinct integer points [x, y] with |x|, |y| <= 100
    such that no four are collinear.
    Aims to minimize the largest subset with no three collinear
    by maximizing the number of 3-collinear triples.
    """
    N = 20
    BOUND = 100
    
    # Helper: Check if 4 points are collinear
    def are_collinear(p1, p2, p3, p4):
        # Cross product of (p2-p1) and (p3-p1)
        # (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
        def cross(p1, p2, p3):
            return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        
        # Check if any 4 form a line.
        # Since we only have 4 points, we just need to check if they are all on the same line.
        # This is true if cross(p1,p2,p3) == 0 and cross(p1,p2,p4) == 0.
        # But what if p1, p2, p4 are collinear and p3 is not? Then not all 4 are collinear.
        # The condition "no four collinear" means we cannot have a set of 4 points that are mutually collinear.
        # So for any 4 points, if they are all on a line, it's invalid.
        
        # Efficient check: 
        # Sort by x, then y.
        # If any 3 are collinear, check if the 4th is also on that line.
        # Actually, simpler: 
        # If cross(p1,p2,p3)==0, check if p4 is on line p1-p2.
        # If cross(p1,p2,p4)==0, check if p3 is on line p1-p2.
        # If cross(p1,p3,p4)==0, check if p2 is on line p1-p3.
        # If none of the first 3 crosses are 0, they are not all collinear.
        
        c123 = cross(p1, p2, p3)
        if c123 == 0:
            if cross(p1, p2, p4) == 0:
                return True
            # If p1,p2,p3 collinear, but p4 not, then not all 4 are collinear.
            # But wait, could p1,p2,p4 be collinear? 
            # If p1,p2,p3 collinear and p1,p2,p4 collinear, then p3,p4 are on same line as p1,p2.
            # So checking one pair that is collinear is enough if we verify the 4th.
            # However, it's possible that p1,p2,p3 are NOT collinear, but p1,p2,p4 are?
            # If p1,p2,p3 not collinear, they don't form a line of 3. 
            # For 4 to be collinear, ANY 3 must be collinear.
            # So if c123 != 0, check c124.
            # If c124 == 0, check if p3 is on line p1-p2.
            
        c124 = cross(p1, p2, p4)
        if c124 == 0:
            if cross(p1, p2, p3) == 0:
                return True
            
        c134 = cross(p1, p3, p4)
        if c134 == 0:
            if cross(p1, p3, p2) == 0:
                return True
                
        # If we get here, no 3 of them are collinear, or the 3 that are collinear don't include the 4th?
        # Actually, if any 3 are collinear, say p1,p2,p3, and p4 is NOT on that line, then the 4 points are NOT all collinear.
        # So the condition for "4 collinear" is that there exists a line containing all 4.
        # This is equivalent to: cross(p1,p2,p3)==0 AND cross(p1,p2,p4)==0.
        # OR cross(p1,p2,p4)==0 AND cross(p1,p3,p4)==0? No.
        # If p1,p2,p3 collinear and p1,p2,p4 collinear, then all 4 are on line p1-p2.
        # So:
        if cross(p1, p2, p3) == 0 and cross(p1, p2, p4) == 0:
            return True
        if cross(p1, p3, p2) == 0 and cross(p1, p3, p4) == 0:
            return True
        if cross(p1, p4, p2) == 0 and cross(p1, p4, p3) == 0:
            return True
        if cross(p2, p3, p1) == 0 and cross(p2, p3, p4) == 0:
            return True
        if cross(p2, p4, p1) == 0 and cross(p2, p4, p3) == 0:
            return True
        if cross(p3, p4, p1) == 0 and cross(p3, p4, p2) == 0:
            return True
            
        return False

    def has_four_collinear(points):
        for p1, p2, p3, p4 in itertools.combinations(points, 4):
            if are_collinear(p1, p2, p3, p4):
                return True
        return False

    def count_three_collinear(points):
        count = 0
        for p1, p2, p3 in itertools.combinations(points, 3):
            if are_collinear(p1, p2, p3, p1): # Reusing logic, p4=p1 is redundant, just check 3
                 # Simpler:
                 if (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]) == 0:
                     count += 1
        return count

    def is_valid(points):
        # Check distinct
        if len(set(map(tuple, points))) != len(points):
            return False
        # Check bounds
        for p in points:
            if abs(p[0]) > BOUND or abs(p[1]) > BOUND:
                return False
        # Check no 4 collinear
        if has_four_collinear(points):
            return False
        return True

    # Initialization: Random points, fix if invalid
    def init_points():
        while True:
            pts = [[random.randint(-BOUND, BOUND), random.randint(-BOUND, BOUND)] for _ in range(N)]
            if is_valid(pts):
                return pts

    # Local Search
    best_points = init_points()
    best_score = count_three_collinear(best_points)
    
    start_time = time.time()
    time_limit = 110 # Leave buffer
    
    while time.time() - start_time < time_limit:
        improved = False
        # Shuffle points to avoid bias in order
        indices = list(range(N))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() - start_time > time_limit:
                break
            x, y = best_points[i]
            # Try moving point i to nearby positions
            # Range of movement: small step to stay local
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if abs(nx) > BOUND or abs(ny) > BOUND:
                        continue
                    
                    new_points = best_points.copy()
                    new_points[i] = [nx, ny]
                    
                    if is_valid(new_points):
                        new_score = count_three_collinear(new_points)
                        if new_score > best_score:
                            best_points = new_points
                            best_score = new_score
                            improved = True
                            # Update x, y for next iterations of this point? 
                            # No, we break out or continue with new state.
                            # Since we updated best_points, the loop continues.
                            # But we are inside a nested loop.
                            # Let's just accept and break inner loops?
                            # Or continue searching for even better?
                            # Greedy: accept first improvement.
                            break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Restart with new random init if stuck? Or just break.
            # Let's try a few restarts if stuck.
            # For simplicity, break.
            break

    return best_points

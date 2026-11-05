import random
from itertools import combinations

def construct():
    # Fixed seed for reproducibility
    random.seed(42)
    
    # Candidate pool: small grid to encourage collinearities
    # We need to ensure we can find 20 points.
    # A 10x10 grid has 100 points.
    pool = [[x, y] for x in range(-10, 10) for y in range(-10, 10)]
    
    points = []
    
    # Helper to check if adding p creates 4 collinear
    def is_valid_4(p):
        # Check all lines defined by pairs of existing points + p
        # Or simpler: check all triples in points+[p]
        # If any triple is collinear, check if a 4th point exists on that line.
        # This is O(N^3) per check, N=20. 20^3 = 8000. 20 steps * 400 candidates * 8000 = 6.4M ops. Fast.
        
        new_points = points + [p]
        for i in range(len(new_points)):
            for j in range(i+1, len(new_points)):
                for k in range(j+1, len(new_points)):
                    p1, p2, p3 = new_points[i], new_points[j], new_points[k]
                    # Check if collinear
                    if (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]) == 0):
                        # They are collinear. Check if any other point in new_points is also on this line.
                        # Line equation: A x + B y + C = 0
                        x1, y1 = p1
                        x2, y2 = p2
                        # A = y1 - y2
                        # B = x2 - x1
                        # C = x1*y2 - x2*y1
                        A = y1 - y2
                        B = x2 - x1
                        C = x1*y2 - x2*y1
                        count = 0
                        for q in new_points:
                            if A*q[0] + B*q[1] + C == 0:
                                count += 1
                        if count >= 4:
                            return False
        return True

    # Helper to count triples (lines with exactly 3 points)
    def count_triples(pts):
        triples = 0
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    if (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]) == 0):
                        # Check if exactly 3
                        A = p1[1] - p2[1]
                        B = p2[0] - p1[0]
                        C = p1[0]*p2[1] - p2[0]*p1[1]
                        count = 0
                        for q in pts:
                            if A*q[0] + B*q[1] + C == 0:
                                count += 1
                        if count == 3:
                            triples += 1
        return triples

    # Greedy construction
    for _ in range(20):
        best_p = None
        best_score = -1
        candidates = [p for p in pool if p not in points]
        random.shuffle(candidates) # Break ties randomly
        
        for p in candidates:
            if is_valid_4(p):
                # We want to maximize triples
                # But calculating count_triples for every candidate is expensive.
                # Optimization: only check if p lies on a line with 2 existing points.
                # Let's just pick the first valid one? No, that's random.
                # Let's pick the one that maximizes "new triples".
                
                # Quick check: how many existing pairs form a line with p?
                new_triples = 0
                for i in range(len(points)):
                    for j in range(i+1, len(points)):
                        p1, p2 = points[i], points[j]
                        # Check if p, p1, p2 collinear
                        if (p[0]*(p1[1]-p2[1]) + p1[0]*(p2[1]-p[1]) + p2[0]*(p[1]-p1[1]) == 0):
                            # Check if this line has exactly 3 points in points+[p]
                            # It has p, p1, p2. Does it have a 4th in points?
                            A = p1[1] - p2[1]
                            B = p2[0] - p1[0]
                            C = p1[0]*p2[1] - p2[0]*p1[1]
                            count = 0
                            for q in points:
                                if A*q[0] + B*q[1] + C == 0:
                                    count += 1
                            # If count < 2, then adding p makes count 3. (Since p is new)
                            # If count == 2, then p1, p2 are the only ones. Adding p makes 3.
                            # If count > 2, then there are already 3, adding p makes 4 (invalid, already checked)
                            # So if valid, and count==2, we form a new triple.
                            if count == 2:
                                new_triples += 1
                
                if new_triples > best_score:
                    best_score = new_triples
                    best_p = p
                    # Optional: if best_score is high, break? No, keep looking for max.
        
        if best_p is None:
            # Fallback: pick random valid point
            for p in candidates:
                if is_valid_4(p):
                    best_p = p
                    break
            if best_p is None:
                # Should not happen with large pool
                best_p = candidates[0]
        
        points.append(best_p)
        
    return points

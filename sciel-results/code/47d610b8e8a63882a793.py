import random
import itertools

def construct():
    # Initial configuration: 4x4 grid
    # Points: (0,0) to (3,3)
    points = [[x, y] for x in range(4) for y in range(4)]
    
    def is_valid(pts):
        # Check distinct
        if len(set(map(tuple, pts))) != 16:
            return False
        # Check bounds
        for p in pts:
            if abs(p[0]) > 1000 or abs(p[1]) > 1000:
                return False
        # Check no 5 collinear
        # We can do this in the score function or here.
        # For speed, let's do it in score.
        return True

    def score(pts):
        if not is_valid(pts):
            return -1000000
        
        # Check no 5 collinear
        # If any line has > 4 points, return -1000000
        # We can combine this with counting 4-point lines.
        
        lines = set()
        count_4 = 0
        
        # To avoid double counting, we can normalize line equations.
        # Or just count unique lines that have exactly 4 points.
        
        # Efficient way:
        # For each pair, get line.
        # Use a dict to count points on each line.
        line_counts = {}
        
        for i in range(16):
            for j in range(i+1, 16):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                
                # Line equation: A x + B y = C
                # A = y2 - y1
                # B = x1 - x2
                # C = A x1 + B y1
                A = y2 - y1
                B = x1 - x2
                C = A * x1 + B * y1
                
                # Normalize A, B, C to ensure same line has same key
                # Find GCD of A, B, C
                def gcd(a, b):
                    a, b = abs(a), abs(b)
                    while b:
                        a, b = b, a % b
                    return a
                    
                g = gcd(gcd(A, B), C)
                if g == 0:
                    # This shouldn't happen for distinct points
                    continue
                A_n = A // g
                B_n = B // g
                C_n = C // g
                
                # Ensure canonical sign (e.g., first non-zero is positive)
                if A_n < 0 or (A_n == 0 and B_n < 0) or (A_n == 0 and B_n == 0 and C_n < 0):
                    A_n = -A_n
                    B_n = -B_n
                    C_n = -C_n
                    
                key = (A_n, B_n, C_n)
                
                if key not in line_counts:
                    line_counts[key] = 0
                line_counts[key] += 1
        
        # Now line_counts has the number of pairs for each line.
        # If a line has k points, it has k*(k-1)/2 pairs.
        # We want lines with k=4.
        # k=4 -> 6 pairs.
        # k=5 -> 10 pairs.
        # k=6 -> 15 pairs.
        
        # We need to be careful: multiple pairs from the same line map to the same key.
        # So line_counts[key] is the number of pairs, not points.
        
        final_score = 0
        for key, pair_count in line_counts.items():
            # Find k such that k*(k-1)/2 = pair_count
            # pair_count can be 1 (k=2), 3 (k=3), 6 (k=4), 10 (k=5), 15 (k=6)...
            # We only care about k=4.
            # But we must ensure no k>=5.
            
            # Solve k^2 - k - 2*pair_count = 0
            # k = (1 + sqrt(1 + 8*pair_count)) / 2
            import math
            disc = 1 + 8 * pair_count
            sqrt_disc = int(math.sqrt(disc))
            if sqrt_disc * sqrt_disc != disc:
                # This shouldn't happen if points are integers and we counted correctly
                # But floating point issues? No, integers.
                # Actually, pair_count must be a triangular number.
                continue
            
            k = (1 + sqrt_disc) // 2
            if k * (k-1) // 2 != pair_count:
                continue
                
            if k > 4:
                return -1000000 # Invalid: 5 or more collinear
            elif k == 4:
                final_score += 1
                
        return final_score

    best_points = [p[:] for p in points]
    best_score = score(points)
    
    # Hill Climbing
    # We will try to move one point at a time.
    # To be efficient, we can pre-calculate which lines are "full" or "nearly full".
    # But with 16 points, full recalculation is fast enough.
    
    num_iterations = 5000
    for _ in range(num_iterations):
        # Pick a random point
        idx = random.randint(0, 15)
        old_point = points[idx][:]
        
        # Pick a new position
        # Search in a local neighborhood
        # If we are far from the grid, we might want to search wider.
        # But let's start with a small step.
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        new_point = [old_point[0] + dx, old_point[1] + dy]
        
        # Check bounds
        if abs(new_point[0]) > 1000 or abs(new_point[1]) > 1000:
            continue
            
        # Check distinctness
        is_distinct = True
        for j, p in enumerate(points):
            if j != idx and p == new_point:
                is_distinct = False
                break
        if not is_distinct:
            continue
            
        # Try the move
        points[idx] = new_point
        new_score = score(points)
        
        if new_score > best_score:
            best_score = new_score
            best_points = [p[:] for p in points]
        else:
            # Revert
            points[idx] = old_point
            
    return best_points

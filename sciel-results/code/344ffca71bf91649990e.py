import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Initial configuration: 3x4 grid
    # Points: (x,y) for x in 0..3, y in 0..2
    points = [[x, y] for x in range(4) for y in range(3)]
    
    def count_ordinary(pts):
        """Count lines containing exactly 2 points."""
        n = len(pts)
        if n < 2:
            return 0
        
        # Map from line (A, B, C) to count of points on it
        # Line: Ax + By + C = 0
        # Normalize: A, B, C are integers, gcd(A,B,C)=1, A>=0 (if A=0, B>0)
        line_counts = {}
        
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                
                # Calculate A, B, C for line through (x1,y1) and (x2,y2)
                # A = y2 - y1
                # B = x1 - x2
                # C = -(A*x1 + B*y1)
                A = y2 - y1
                B = x1 - x2
                C = -(A * x1 + B * y1)
                
                # Normalize
                # Find gcd of A, B, C
                import math
                g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                
                # Ensure consistent sign
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A = -A
                    B = -B
                    C = -C
                
                key = (A, B, C)
                line_counts[key] = line_counts.get(key, 0) + 1
        
        ordinary_count = 0
        for count in line_counts.values():
            if count == 2:
                ordinary_count += 1
        
        return ordinary_count

    best_points = [p[:] for p in points]
    best_score = count_ordinary(best_points)
    
    # Local Search
    # Moves:
    # 1. Snap to line: Move point i to an integer point on a line defined by j, k
    # 2. Small perturbation
    
    def get_line_candidates(pts, i):
        """Generate candidate positions for point i based on lines from other points."""
        candidates = set()
        n = len(pts)
        # Bounding box for search
        # Use a reasonable range, e.g., based on current points
        min_x = min(p[0] for p in pts) - 5
        max_x = max(p[0] for p in pts) + 5
        min_y = min(p[1] for p in pts) - 5
        max_y = max(p[1] for p in pts) + 5
        
        # Lines defined by pairs (j, k) where j, k != i
        for j in range(n):
            if j == i: continue
            for k in range(j + 1, n):
                if k == i: continue
                x1, y1 = pts[j]
                x2, y2 = pts[k]
                
                A = y2 - y1
                B = x1 - x2
                C = -(A * x1 + B * y1)
                
                # Find integer points (x,y) on Ax + By + C = 0
                # within bounding box
                # To avoid too many, we only check points that are "close" or specific steps
                # Actually, we can just check a small range around existing points or intersections
                # But for 12 points, checking all integer points in a 20x20 box is 400 points.
                # 11*10/2 = 55 lines. 55 * 400 = 22000 checks. Fast enough.
                
                # Optimization: Only consider points that are integer coordinates.
                # If A == 0: By + C = 0 => y = -C/B. If integer, x is free.
                # If B == 0: Ax + C = 0 => x = -C/A. If integer, y is free.
                # Else: y = (-C - Ax) / B.
                
                # We will just iterate x in range and check if y is integer.
                # But range is 10..-10 relative? Let's use absolute box.
                
                # To keep it fast, we only look for points that are not too far from current point i
                # Or just scan the box.
                
                for x in range(min_x, max_x + 1):
                    # A*x + B*y + C = 0 => B*y = -A*x - C
                    num = -A * x - C
                    if B == 0:
                        continue # Vertical line, handled separately or skip if x matches
                    if num % B == 0:
                        y = num // B
                        if min_y <= y <= max_y:
                            # Check if this point is already in pts (excluding i)
                            # We want to move i TO this point.
                            # The point must be distinct from other points?
                            # Yes, points must be distinct.
                            is_dup = False
                            for m in range(n):
                                if m == i: continue
                                if pts[m][0] == x and pts[m][1] == y:
                                    is_dup = True
                                    break
                            if not is_dup:
                                candidates.add((x, y))
        
        return candidates

    def local_search(pts, max_iters=10000):
        current_points = [p[:] for p in pts]
        current_score = count_ordinary(current_points)
        n = len(current_points)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            improved = False
            i = random.randint(0, n - 1)
            
            # Generate candidates
            candidates = get_line_candidates(current_points, i)
            
            # Also add small perturbations
            x, y = current_points[i]
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0: continue
                    nx, ny = x + dx, y + dy
                    # Check distinctness
                    is_dup = False
                    for m in range(n):
                        if m == i: continue
                        if current_points[m][0] == nx and current_points[m][1] == ny:
                            is_dup = True
                            break
                    if not is_dup:
                        candidates.add((nx, ny))
            
            if not candidates:
                continue
                
            # Pick a random candidate
            nx, ny = random.choice(list(candidates))
            
            # Apply move
            old_x, old_y = current_points[i]
            current_points[i] = [nx, ny]
            
            new_score = count_ordinary(current_points)
            
            # Accept if better, or equal with some probability (simulated annealing style)
            # For simplicity, accept if better or equal (to explore)
            if new_score <= current_score:
                current_score = new_score
                improved = True
            else:
                # Revert
                current_points[i] = [old_x, old_y]
            
            if improved:
                # Update best
                if current_score < best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
                if best_score == 6:
                    return best_points # Reached theoretical minimum
            
        return best_points

    # Run multiple restarts or just one long search?
    # Given 150s, one long search with random restarts if stuck.
    
    # Initial evaluation
    best_points = [p[:] for p in points]
    best_score = count_ordinary(best_points)
    
    if best_score == 6:
        return best_points

    # Local search
    best_points = local_search(points)
    
    if best_score == 6:
        return best_points

    # If not 6, try a few restarts with different initializations
    # e.g. random points in a small grid
    for restart in range(10):
        if time.time() > deadline:
            break
            
        # Random initialization in a small box
        random_points = [[random.randint(-5, 5), random.randint(-5, 5)] for _ in range(12)]
        # Ensure distinct
        seen = set()
        valid_points = []
        for p in random_points:
            if tuple(p) not in seen:
                seen.add(tuple(p))
                valid_points.append(p)
        
        # Fill if not 12
        while len(valid_points) < 12:
            x, y = random.randint(-10, 10), random.randint(-10, 10)
            if (x, y) not in seen:
                seen.add((x, y))
                valid_points.append([x, y])
        
        # Not all collinear check (random is likely fine)
        
        candidate_best = local_search(valid_points, max_iters=5000)
        candidate_score = count_ordinary(candidate_best)
        
        if candidate_score < best_score:
            best_score = candidate_score
            best_points = candidate_best
            
        if best_score == 6:
            break

    return best_points

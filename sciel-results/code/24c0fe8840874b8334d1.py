import time
import random
import itertools
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def get_line(p1, p2):
        # Returns (a, b, c) for ax + by + c = 0
        x1, y1 = p1
        x2, y2 = p2
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        # Normalize
        from math import gcd
        g = gcd(gcd(abs(a), abs(b)), abs(c))
        if g != 0:
            a //= g
            b //= g
            c //= g
        # Ensure unique representation
        if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
            a, b, c = -a, -b, -c
        return (a, b, c)

    def count_ordinary(points):
        lines = {}
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                l = get_line(points[i], points[j])
                if l not in lines:
                    lines[l] = []
                lines[l].append((i, j))
        
        ordinary = 0
        for l, pairs in lines.items():
            if len(pairs) == 1:
                ordinary += 1
        return ordinary

    def not_all_collinear(points):
        if len(points) < 3:
            return True
        p0, p1, p2 = points[0], points[1], points[2]
        return (p1[0]-p0[0])*(p2[1]-p0[1]) != (p2[0]-p0[0])*(p1[1]-p0[1])

    # Initial 3x4 grid
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_score = count_ordinary(best_points)
    
    # Local search
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 11)
        old_point = current_points[idx]
        
        # Try moving to a nearby point
        moved = False
        for _ in range(20):
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            new_point = [old_point[0] + dx, old_point[1] + dy]
            
            # Check bounds and distinctness
            if abs(new_point[0]) > 10**9 or abs(new_point[1]) > 10**9:
                continue
            if new_point in current_points:
                continue
            
            # Swap and check
            current_points[idx] = new_point
            if not not_all_collinear(current_points):
                current_points[idx] = old_point
                continue
            
            new_score = count_ordinary(current_points)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
                moved = True
                break
            else:
                current_points[idx] = old_point
        
        if not moved:
            # Try a larger jump
            new_point = [random.randint(-10, 10), random.randint(-10, 10)]
            if new_point not in current_points and abs(new_point[0]) <= 10**9 and abs(new_point[1]) <= 10**9:
                current_points[idx] = new_point
                if not_all_collinear(current_points):
                    new_score = count_ordinary(current_points)
                    if new_score < current_score:
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_points = [p[:] for p in current_points]
                    else:
                        current_points[idx] = old_point

    return best_points

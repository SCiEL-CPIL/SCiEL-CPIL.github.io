import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Known 5-point set with integer coordinates and integer pairwise distances
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    def is_square(n):
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    def dist_sq(p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    def check_rational_dist(p1, p2):
        return is_square(dist_sq(p1, p2))

    def check_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]) == 0

    def check_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        d1 = x1*x1 + y1*y1
        d2 = x2*x2 + y2*y2
        d3 = x3*x3 + y3*y3
        d4 = x4*x4 + y4*y4
        
        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1]) -
                    m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0]) +
                    m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]))
        
        m1 = [[x2, y2, 1], [x3, y3, 1], [x4, y4, 1]]
        m2 = [[d2, y2, 1], [d3, y3, 1], [d4, y4, 1]]
        m3 = [[d2, x2, 1], [d3, x3, 1], [d4, x4, 1]]
        m4 = [[d2, x2, y2], [d3, x3, y3], [d4, x4, y4]]
        
        return d1*det3(m1) - x1*det3(m2) + y1*det3(m3) - det3(m4) == 0

    def is_valid_set(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if not check_rational_dist(points[i], points[j]):
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_collinear(points[i], points[j], points[k]):
                        return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if check_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    def try_add_point(points, new_point):
        for p in points:
            if not check_rational_dist(p, new_point):
                return False
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if check_collinear(points[i], points[j], new_point):
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_concyclic(points[i], points[j], points[k], new_point):
                        return False
        return True

    # Generate candidates
    candidates = []
    for x in range(-100, 101):
        for y in range(-100, 101):
            candidates.append((x, y))
    
    best_found = list(best_points)
    
    # Try multiple random orders
    for attempt in range(20):
        if time.time() > deadline:
            break
            
        # Shuffle candidates
        shuffled = list(candidates)
        random.shuffle(shuffled)
        
        current_set = list(best_points)
        
        # Greedy add
        added = True
        while added and time.time() < deadline:
            added = False
            for c in shuffled:
                if c in current_set:
                    continue
                if try_add_point(current_set, c):
                    current_set.append(c)
                    added = True
                    break
        
        if len(current_set) > len(best_found):
            best_found = current_set
            if len(best_found) >= 8: # Found a great set
                break
                
    # Final validation
    if not is_valid_set(best_found):
        best_found = best_points
    
    D = 1
    points_out = []
    for x, y in best_found:
        points_out.append([[[x, 1], [y, 1]]])
        
    return {'D': D, 'points': points_out}

if __name__ == '__main__':
    res = construct()
    print(res)

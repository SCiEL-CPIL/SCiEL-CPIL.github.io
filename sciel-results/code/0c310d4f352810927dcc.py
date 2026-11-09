import time
import random
import math

def is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def is_integer_dist(p1, p2):
    return is_square(dist_sq(p1, p2))

def are_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

def det4(M):
    return (M[0][0]*(M[1][1]*(M[2][2]*M[3][3]-M[2][3]*M[3][2]) - M[1][2]*(M[2][1]*M[3][3]-M[2][3]*M[3][1]) + M[1][3]*(M[2][1]*M[3][2]-M[2][2]*M[3][1])) -
            M[0][1]*(M[1][0]*(M[2][2]*M[3][3]-M[2][3]*M[3][2]) - M[1][2]*(M[2][0]*M[3][3]-M[2][3]*M[3][0]) + M[1][3]*(M[2][0]*M[3][2]-M[2][2]*M[3][0])) +
            M[0][2]*(M[1][0]*(M[2][1]*M[3][3]-M[2][3]*M[3][1]) - M[1][1]*(M[2][0]*M[3][3]-M[2][3]*M[3][0]) + M[1][3]*(M[2][0]*M[3][1]-M[2][1]*M[3][0])) -
            M[0][3]*(M[1][0]*(M[2][1]*M[3][2]-M[2][2]*M[3][1]) - M[1][1]*(M[2][0]*M[3][2]-M[2][2]*M[3][0]) + M[1][2]*(M[2][0]*M[3][1]-M[2][1]*M[3][0])))

def are_concyclic(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    M = [[x1*x1+y1*y1, x1, y1, 1],
         [x2*x2+y2*y2, x2, y2, 1],
         [x3*x3+y3*y3, x3, y3, 1],
         [x4*x4+y4*y4, x4, y4, 1]]
         
    return det4(M) == 0

def can_add_point(P, S):
    for p in S:
        if not is_integer_dist(P, p):
            return False
            
    n = len(S)
    for i in range(n):
        for j in range(i+1, n):
            if are_collinear(P, S[i], S[j]):
                return False
                
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if are_concyclic(P, S[i], S[j], S[k]):
                    return False
                    
    return True

def construct():
    start_time = time.time()
    time_limit = 140.0
    
    initial_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_points = initial_points
    best_size = len(best_points)
    
    # Fallback if initial is bad (unlikely)
    if not all(is_integer_dist(p1, p2) for i, p1 in enumerate(best_points) for p2 in best_points[i+1:]):
        best_points = [(0,0), (3,4), (4,0)]
        best_size = 3

    seeds = [
        [(0,0), (3,4), (4,0)],
        [(0,0), (5,12), (0,13)],
        [(0,0), (8,15), (0,17)],
        [(0,0), (9,12), (12,9)] # Dist: 15, 15, sqrt(9+9)=3sqrt(2) No.
        # Use valid seeds only
    ]
    # Filter valid seeds
    valid_seeds = []
    for s in seeds:
        ok = True
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                if not is_integer_dist(s[i], s[j]):
                    ok = False
                    break
        if ok:
            valid_seeds.append(s)
            
    if not valid_seeds:
        valid_seeds = [[(0,0), (3,4), (4,0)]]

    search_mode = 0 # 0: extend best, 1: random build
    
    while time.time() - start_time < time_limit:
        if search_mode == 0:
            # Extend best_points
            cx = sum(p[0] for p in best_points) // len(best_points)
            cy = sum(p[1] for p in best_points) // len(best_points)
            
            # Increase box size as we get larger sets?
            # For 5 points, box 100. For 6, maybe 200.
            box = 50 + best_size * 20
            
            found = False
            for _ in range(200):
                if time.time() - start_time > time_limit: break
                rx = random.randint(cx - box, cx + box)
                ry = random.randint(cy - box, cy + box)
                P = (rx, ry)
                
                if P in best_points:
                    continue
                    
                if can_add_point(P, best_points):
                    best_points.append(P)
                    best_size += 1
                    found = True
                    break
            
            if found:
                # If we grew, maybe stay in mode 0 to try to grow more
                pass
            else:
                # If we didn't grow, switch to mode 1 to find new structures
                search_mode = 1
        else:
            # Random build
            seed = random.choice(valid_seeds)
            current_set = seed[:]
            
            # Try to grow
            box = 50 + len(current_set) * 20
            for _ in range(100):
                if time.time() - start_time > time_limit: break
                if len(current_set) >= best_size:
                    break
                    
                rx = random.randint(-box, box)
                ry = random.randint(-box, box)
                P = (rx, ry)
                
                if P in current_set:
                    continue
                    
                if can_add_point(P, current_set):
                    current_set.append(P)
                    if len(current_set) > best_size:
                        best_points = current_set
                        best_size = len(current_set)
                        search_mode = 0
                        break

    result_points = []
    for x, y in best_points:
        result_points.append([[x, 1], [y, 1]])
        
    return {'D': 1, 'points': result_points}

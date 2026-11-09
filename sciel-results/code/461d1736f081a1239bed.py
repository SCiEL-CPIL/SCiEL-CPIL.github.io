import random
import math

def construct():
    points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    def is_valid_candidate(cand, existing_points):
        cx, cy = cand
        for px, py in existing_points:
            dx = cx - px
            dy = cy - py
            d2 = dx*dx + dy*dy
            root = math.isqrt(d2)
            if root * root != d2:
                return False
        
        n = len(existing_points)
        for i in range(n):
            for j in range(i+1, n):
                p1 = existing_points[i]
                p2 = existing_points[j]
                v1x = p2[0] - p1[0]
                v1y = p2[1] - p1[1]
                v2x = cx - p1[0]
                v2y = cy - p1[1]
                if v1x * v2y - v1y * v2x == 0:
                    return False
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1 = existing_points[i]
                    p2 = existing_points[j]
                    p3 = existing_points[k]
                    a = cx*cx + cy*cy
                    b = p1[0]*p1[0] + p1[1]*p1[1]
                    c = p2[0]*p2[0] + p2[1]*p2[1]
                    d = p3[0]*p3[0] + p3[1]*p3[1]
                    mat = [
                        [a, cx, cy, 1],
                        [b, p1[0], p1[1], 1],
                        [c, p2[0], p2[1], 1],
                        [d, p3[0], p3[1], 1]
                    ]
                    # det4
                    m = mat
                    det = (
                        m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
                        - m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
                        + m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
                        - m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
                    )
                    if det == 0:
                        return False
        return True

    best_points = points
    best_count = len(points)
    
    # Grid search
    for x in range(-150, 151):
        for y in range(-150, 151):
            if (x, y) in points:
                continue
            if is_valid_candidate((x, y), points):
                best_points = points + [(x, y)]
                best_count = 6
                break
        if best_count == 6:
            break
            
    # Random search if grid failed
    if best_count < 6:
        for _ in range(100000):
            x = random.randint(-500, 500)
            y = random.randint(-500, 500)
            if (x, y) in points:
                continue
            if is_valid_candidate((x, y), points):
                best_points = points + [(x, y)]
                best_count = 6
                break
                
    D = 1
    out_points = []
    for x, y in best_points:
        out_points.append([[[x, 1], [y, 1]]])
        
    return {'D': D, 'points': out_points}

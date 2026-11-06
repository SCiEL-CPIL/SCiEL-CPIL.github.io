import time
import random

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    
    line_counts = {}
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return abs(a)
            
            g = gcd(gcd(A, B), C)
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            if key not in line_counts:
                line_counts[key] = 1
            else:
                line_counts[key] += 1
            
    ordinary = 0
    for key, pair_count in line_counts.items():
        if pair_count == 1:
            ordinary += 1
            
    return ordinary

def construct():
    start_time = time.time()
    timeout = 115
    
    best_points = None
    best_count = float('inf')
    
    # Pappus configuration scaled by 12
    pappus = [
        [0, 0], [12, 0], [24, 0],
        [0, 12], [36, 12], [12, 12],
        [9, 3], [12, -12], [8, 8]
    ]
    
    # Try optimizing Pappus + 3 random points
    for restart in range(100):
        if time.time() - start_time > timeout:
            break
            
        points = pappus[:]
        for _ in range(3):
            while True:
                x = random.randint(-50, 50)
                y = random.randint(-50, 50)
                if [x, y] not in points:
                    points.append([x, y])
                    break
        
        # Check not all collinear
        collinear = True
        if len(points) >= 3:
            (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
            if (x2-x1)*(y3-y1) != (y2-y1)*(x3-x1):
                collinear = False
        
        if collinear:
            continue
            
        c = count_ordinary_lines(points)
        if c < best_count:
            best_count = c
            best_points = points[:]
            if best_count == 6:
                return best_points
                
        # Local search
        improved = True
        while improved and time.time() - start_time < timeout:
            improved = False
            for i in range(12):
                if time.time() - start_time > timeout:
                    break
                old_x, old_y = points[i]
                for _ in range(10):
                    new_x = old_x + random.randint(-10, 10)
                    new_y = old_y + random.randint(-10, 10)
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    if [new_x, new_y] in points:
                        continue
                    
                    points[i] = [new_x, new_y]
                    
                    collinear = True
                    if len(points) >= 3:
                        (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
                        if (x2-x1)*(y3-y1) != (y2-y1)*(x3-x1):
                            collinear = False
                    
                    if collinear:
                        points[i] = [old_x, old_y]
                        continue
                        
                    c = count_ordinary_lines(points)
                    if c < best_count:
                        best_count = c
                        best_points = points[:]
                        improved = True
                        if best_count == 6:
                            return best_points
                    else:
                        points[i] = [old_x, old_y]
                        
    # If still not found, try random restarts
    while time.time() - start_time < timeout:
        points = []
        for _ in range(12):
            while True:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if [x, y] not in points:
                    points.append([x, y])
                    break
        
        collinear = True
        if len(points) >= 3:
            (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
            if (x2-x1)*(y3-y1) != (y2-y1)*(x3-x1):
                collinear = False
        
        if collinear:
            continue
            
        c = count_ordinary_lines(points)
        if c < best_count:
            best_count = c
            best_points = points[:]
            if best_count == 6:
                return best_points
                
        improved = True
        while improved and time.time() - start_time < timeout:
            improved = False
            for i in range(12):
                if time.time() - start_time > timeout:
                    break
                old_x, old_y = points[i]
                for _ in range(10):
                    new_x = old_x + random.randint(-10, 10)
                    new_y = old_y + random.randint(-10, 10)
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    if [new_x, new_y] in points:
                        continue
                    
                    points[i] = [new_x, new_y]
                    
                    collinear = True
                    if len(points) >= 3:
                        (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
                        if (x2-x1)*(y3-y1) != (y2-y1)*(x3-x1):
                            collinear = False
                    
                    if collinear:
                        points[i] = [old_x, old_y]
                        continue
                        
                    c = count_ordinary_lines(points)
                    if c < best_count:
                        best_count = c
                        best_points = points[:]
                        improved = True
                        if best_count == 6:
                            return best_points
                    else:
                        points[i] = [old_x, old_y]
                        
    return best_points

import time
import random
import math
from collections import defaultdict

def count_exact_four_lines(points):
    n = len(points)
    if n < 4:
        return 0, 0
    
    line_counts = defaultdict(int)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = x1*y2 - x2*y1
            
            if A == 0 and B == 0:
                continue
                
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            else:
                continue
                
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            line_counts[key] += 1
            
    count_4 = 0
    count_5_plus = 0
    for count in line_counts.values():
        if count == 4:
            count_4 += 1
        elif count >= 5:
            count_5_plus += 1
            
    return count_4, count_5_plus

def score(points):
    c4, c5 = count_exact_four_lines(points)
    if c5 > 0:
        return -1000000
    return c4

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = score(best_points)
    
    current_points = best_points[:]
    current_score = best_score
    
    T = 20.0
    cooling = 0.99995
    
    while time.time() < deadline:
        i = random.randint(0, 15)
        x, y = current_points[i]
        
        if random.random() < 0.3:
            new_x = random.randint(-1000, 1000)
            new_y = random.randint(-1000, 1000)
        else:
            new_x = x + random.randint(-10, 10)
            new_y = y + random.randint(-10, 10)
            
        if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
            continue
            
        new_points = current_points[:]
        new_points[i] = [new_x, new_y]
        
        if len(set(map(tuple, new_points))) < 16:
            continue
            
        new_score = score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = new_points[:]
        else:
            delta = new_score - current_score
            if delta < 0:
                p = math.exp(delta / T)
                if random.random() < p:
                    current_points = new_points
                    current_score = new_score
        
        T *= cooling
        if T < 0.1:
            T = 0.1
            
    return best_points

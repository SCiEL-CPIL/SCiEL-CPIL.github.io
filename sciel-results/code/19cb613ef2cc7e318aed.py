import random
import math

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    
    lines = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = -(A * x1 + B * y1)
            
            # Normalize
            if A == 0 and B == 0:
                continue
            
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return abs(a)
            
            g = gcd(gcd(A, B), C)
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
            
            if key not in lines:
                lines[key] = set()
            
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary = 0
    for key, pts_in_line in lines.items():
        if len(pts_in_line) == 2:
            ordinary += 1
            
    return ordinary

def construct():
    # Use a random search with simulated annealing
    best_points = None
    best_count = float('inf')
    
    # Try multiple restarts
    for restart in range(20):
        # Initialize with 3x4 grid or random
        if restart % 2 == 0:
            points = []
            for r in range(3):
                for c in range(4):
                    points.append([c, r])
        else:
            points = []
            while len(points) < 12:
                x = random.randint(0, 10)
                y = random.randint(0, 10)
                if [x, y] not in points:
                    points.append([x, y])
        
        current_count = count_ordinary_lines(points)
        if current_count < best_count:
            best_count = current_count
            best_points = [p[:] for p in points]
        
        # Simulated Annealing
        T = 10.0
        for i in range(5000):
            T *= 0.99
            idx = random.randint(0, 11)
            old_x, old_y = points[idx]
            
            # Random move
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
                
            new_x = old_x + dx
            new_y = old_y + dy
            
            if new_x < 0 or new_x > 20 or new_y < 0 or new_y > 20:
                continue
                
            if [new_x, new_y] in points:
                continue
                
            points[idx] = [new_x, new_y]
            new_count = count_ordinary_lines(points)
            
            delta = new_count - current_count
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_count = new_count
                if new_count < best_count:
                    best_count = new_count
                    best_points = [p[:] for p in points]
            else:
                points[idx] = [old_x, old_y]
                
        if best_count <= 6:
            break
            
    # Scale up to satisfy bounds if needed (though small coords are fine)
    # The task allows |x|,|y| <= 10^9, so small integers are valid.
    return best_points

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import math
import time
from collections import defaultdict

# EVOLVE-BLOCK-START

def _evaluate(points):
    """Return the number of 4-point lines, or -1 if invalid."""
    n = len(points)
    if n != 16:
        return -1
    
    # Check bounds and distinctness
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return -1
        if (x, y) in seen:
            return -1
        seen.add((x, y))
        
    # Count lines
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            g = math.gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            a = dy
            b = -dx
            c = a * x1 + b * y1
            # Normalize (a, b, c)
            g2 = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            count = line_counts.get(key, 0) + 1
            line_counts[key] = count
            # Early exit if 5 collinear
            if count >= 5:
                return -1

    four_count = 0
    for count in line_counts.values():
        if count == 4:
            four_count += 1
    return four_count

def _perturb(points, rng):
    """Perturb points with various moves."""
    new_points = [p[:] for p in points]
    move_type = rng.random()
    
    if move_type < 0.5:
        # Standard random jump for 1-2 points
        num_moves = rng.choice([1, 2])
        for _ in range(num_moves):
            idx = rng.randint(0, 15)
            new_points[idx][0] += rng.randint(-20, 20)
            new_points[idx][1] += rng.randint(-20, 20)
    elif move_type < 0.8:
        # Translate a small cluster
        center_idx = rng.randint(0, 15)
        cx, cy = new_points[center_idx]
        neighbors = [center_idx]
        for i in range(16):
            if i == center_idx: continue
            dx = abs(new_points[i][0] - cx)
            dy = abs(new_points[i][1] - cy)
            if dx <= 2 and dy <= 2:
                neighbors.append(i)
        
        dx = rng.randint(-10, 10)
        dy = rng.randint(-10, 10)
        for idx in neighbors:
            new_points[idx][0] += dx
            new_points[idx][1] += dy
    else:
        # Swap coordinates of two points
        i, j = rng.sample(range(16), 2)
        new_points[i][0], new_points[j][0] = new_points[j][0], new_points[i][0]
        new_points[i][1], new_points[j][1] = new_points[j][1], new_points[i][1]
        
    return new_points

def construct():
    """Find 16 integer points maximizing 4-point lines with no five collinear."""
    deadline = time.time() + 110
    
    # Initial configurations
    init_grid = [[x, y] for x in range(4) for y in range(4)]
    best_points = [p[:] for p in init_grid]
    best_score = _evaluate(best_points)
    
    rng = random.Random(42)
    
    # Multi-start Simulated Annealing
    while time.time() < deadline:
        # Start from best or random
        if rng.random() < 0.7 or best_score <= 0:
            current_points = [p[:] for p in best_points]
            current_score = best_score
        else:
            current_points = [[rng.randint(-20, 20), rng.randint(-20, 20)] for _ in range(16)]
            current_score = _evaluate(current_points)
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        
        # Simulated Annealing
        temperature = 30.0
        cooling = 0.995
        min_temperature = 0.5
        max_iter = 5000
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if temperature < min_temperature:
                break
                
            new_points = _perturb(current_points, rng)
            new_score = _evaluate(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            elif new_score >= 0:
                delta = current_score - new_score
                if rng.random() < math.exp(-delta / max(temperature, 0.01)):
                    current_points = new_points
                    current_score = new_score
            
            temperature *= cooling

    return best_points

# EVOLVE-BLOCK-END

"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import math
import time
import numpy as np
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
    # We use a dictionary to count how many points are on each line.
    # A line is defined by (a, b, c) where ax + by = c.
    # To avoid floating point issues, we use integer coefficients.
    # For points (x1, y1) and (x2, y2):
    # dx = x2 - x1, dy = y2 - y1
    # g = gcd(dx, dy)
    # dx /= g, dy /= g
    # Line equation: dy * (x - x1) - dx * (y - y1) = 0
    # dy * x - dx * y = dy * x1 - dx * y1
    # a = dy, b = -dx, c = a*x1 + b*y1
    # Normalize (a, b, c) by gcd(a, b, c) and sign.
    
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
            
            # Canonicalize direction
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            
            a = dy
            b = -dx
            c = a * x1 + b * y1
            
            # Normalize line equation
            g2 = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
                
            key = (a, b, c)
            
            if key in line_counts:
                line_counts[key] += 1
                # Early exit if 5 collinear
                if line_counts[key] >= 5:
                    return -1
            else:
                line_counts[key] = 1
                
    four_count = 0
    for count in line_counts.values():
        if count == 4:
            four_count += 1
            
    return four_count

def _generate_grid_variations(rng, base_grid):
    """Generate variations of a base grid."""
    variations = []
    # 1. Original
    variations.append([p[:] for p in base_grid])
    
    # 2. Random shifts
    for _ in range(10):
        shifted = [p[:] for p in base_grid]
        for p in shifted:
            p[0] += rng.randint(-2, 2)
            p[1] += rng.randint(-2, 2)
        variations.append(shifted)
        
    # 3. Random point swaps
    for _ in range(10):
        swapped = [p[:] for p in base_grid]
        i, j = rng.sample(range(16), 2)
        swapped[i], swapped[j] = swapped[j], swapped[i]
        variations.append(swapped)
        
    return variations

def _perturb(points, rng):
    """Perturb points with various moves."""
    new_points = [p[:] for p in points]
    move_type = rng.random()
    
    if move_type < 0.6:
        # Standard random jump for 1-2 points
        num_moves = rng.choice([1, 2])
        for _ in range(num_moves):
            idx = rng.randint(0, 15)
            # Bias towards small moves
            if rng.random() < 0.7:
                new_points[idx][0] += rng.randint(-5, 5)
                new_points[idx][1] += rng.randint(-5, 5)
            else:
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
        
        dx = rng.randint(-5, 5)
        dy = rng.randint(-5, 5)
        for idx in neighbors:
            new_points[idx][0] += dx
            new_points[idx][1] += dy
    else:
        # Swap coordinates of two points
        i, j = rng.sample(range(16), 2)
        new_points[i][0], new_points[j][0] = new_points[j][0], new_points[i][0]
        new_points[i][1], new_points[j][1] = new_points[j][1], new_points[i][1]
        
    return new_points

def _local_search(points, rng, deadline):
    """Run local search on a specific configuration."""
    current_points = [p[:] for p in points]
    current_score = _evaluate(current_points)
    if current_score < 0:
        return points, 0 # Return original if invalid
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    temperature = 10.0
    cooling = 0.99
    min_temperature = 0.1
    
    while time.time() < deadline:
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
            # Reset temperature on improvement
            temperature = min(temperature + 2.0, 20.0)
        elif new_score >= 0:
            delta = current_score - new_score
            # Accept worse solution with probability based on temperature
            if rng.random() < math.exp(-delta / max(temperature, 0.01)):
                current_points = new_points
                current_score = new_score
                
        temperature *= cooling
        
    return best_points, best_score

def construct():
    """Find 16 integer points maximizing 4-point lines with no five collinear."""
    deadline = time.time() + 110
    
    # Initial configurations
    init_grid = [[x, y] for x in range(4) for y in range(4)]
    
    # Generate a diverse set of initial candidates
    candidates = []
    
    # 1. The 4x4 grid
    candidates.append([p[:] for p in init_grid])
    
    # 2. Variations of the grid
    rng_init = random.Random(123)
    grid_vars = _generate_grid_variations(rng_init, init_grid)
    candidates.extend(grid_vars)
    
    # 3. Random configurations
    for _ in range(20):
        rand_points = [[rng_init.randint(-10, 10), rng_init.randint(-10, 10)] for _ in range(16)]
        candidates.append(rand_points)
        
    # 4. Specific "affine-like" patterns
    # Try to construct a pattern based on $AG(2,4)$ mapping
    # Points: (i, j) for i,j in 0..3
    # But we already have the grid.
    # Let's try a "twisted" grid
    twisted = []
    for x in range(4):
        for y in range(4):
            # Shift y based on x to break some symmetries but keep others
            new_y = (y + x * x) % 4 
            # Scale to spread out
            twisted.append([x * 10, new_y * 10])
    candidates.append(twisted)
    
    # Evaluate all candidates
    best_points = None
    best_score = -1
    valid_candidates = []
    
    for c in candidates:
        score = _evaluate(c)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in c]
        if score >= 0:
            valid_candidates.append(c)
            
    # If no valid candidates, return best found (or grid)
    if not valid_candidates:
        return best_points if best_points else init_grid

    # Select top 5 candidates for local search
    scored_candidates = []
    for c in valid_candidates:
        s = _evaluate(c)
        scored_candidates.append((s, c))
        
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = [c for s, c in scored_candidates[:5]]
    
    # Run local search on top candidates
    rng = random.Random(42)
    
    for i, cand in enumerate(top_candidates):
        if time.time() > deadline - 10:
            break
            
        # Allocate time to each local search
        # If 5 candidates, give each ~1/5 of remaining time
        time_per_search = (deadline - time.time()) / (len(top_candidates) - i)
        search_deadline = time.time() + time_per_search
        
        # Ensure at least 5 seconds per search
        search_deadline = max(search_deadline, time.time() + 5)
        
        improved_points, improved_score = _local_search(cand, rng, search_deadline)
        
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_points
            
    # Final check
    if best_score < 0:
        # Fallback to grid if everything failed
        return init_grid
        
    return best_points

# EVOLVE-BLOCK-END

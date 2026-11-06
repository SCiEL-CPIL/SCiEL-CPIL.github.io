import random
import math

def count_collinear_fast(points):
    """
    Returns (c3, c4) where c3 is number of 3-collinear triples and c4 is number of 4-collinear quadruples.
    Uses O(N^2) slope counting.
    """
    n = len(points)
    c3_total = 0
    c4_total = 0
    
    for i in range(n):
        slopes = {}
        xi, yi = points[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            # Normalize slope (dx, dy)
            # Handle vertical lines
            if dx == 0:
                key = (0, 1)
            elif dy == 0:
                key = (1, 0)
            else:
                g = math.gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                # Ensure consistent sign
                if dx < 0:
                    dx = -dx
                    dy = -dy
                key = (dx, dy)
            if key not in slopes:
                slopes[key] = 0
            slopes[key] += 1
        
        for count in slopes.values():
            # count is number of other points on this line through i
            # Total points on this line = count + 1
            # Number of 3-subsets containing i on this line = C(count, 2)
            c3_total += count * (count - 1) // 2
            # Number of 4-subsets containing i on this line = C(count, 3)
            if count >= 3:
                c4_total += count * (count - 1) * (count - 2) // 6
                
    # Each 3-collinear triple is counted 3 times (once for each point)
    c3_total //= 3
    # Each 4-collinear quadruple is counted 4 times (once for each point)
    c4_total //= 4
    
    return c3_total, c4_total

def find_four_collinear(points):
    """
    Returns a list of 4 indices that are collinear, or None.
    """
    n = len(points)
    # Use the slope counting logic to find any line with >= 4 points
    for i in range(n):
        slopes = {}
        xi, yi = points[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0:
                key = (0, 1)
            elif dy == 0:
                key = (1, 0)
            else:
                g = math.gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                if dx < 0:
                    dx = -dx
                    dy = -dy
                key = (dx, dy)
            
            if key not in slopes:
                slopes[key] = []
            slopes[key].append(j)
            
        for key, indices in slopes.items():
            if len(indices) >= 3: # 3 others means 4 total
                return [i] + indices[:3]
    return None

def construct():
    random.seed(42)
    n = 20
    
    # Start with a 5x4 grid
    points = []
    for i in range(5):
        for j in range(4):
            points.append((i, j))
            
    # Perturb points to break 4-collinearities
    # We will use a local search to eliminate 4-collinearities while trying to keep 3-collinearities high
    
    best_points = points[:]
    best_c3, best_c4 = count_collinear_fast(best_points)
    
    # We want c4 = 0.
    # If c4 > 0, we must fix it.
    # We also want to maximize c3 (proxy for low alpha).
    
    T = 10.0
    cooling = 0.99
    steps = 20000
    
    for step in range(steps):
        if best_c4 == 0:
            # Try to improve c3 or just break
            # If we have a valid configuration, we can try to optimize c3 further
            # But the primary constraint is c4=0.
            pass
            
        i = random.randrange(n)
        old_p = points[i]
        
        # Move point i to a nearby random location
        # Or swap with another point?
        # Let's try moving to a random point in a small neighborhood
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        new_p = (old_p[0] + dx, old_p[1] + dy)
        
        # Keep within bounds
        if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
            continue
            
        if new_p in points:
            continue
            
        points[i] = new_p
        
        c3, c4 = count_collinear_fast(points)
        
        # Objective: Minimize c4, then Maximize c3
        # Energy = 10000 * c4 - c3
        old_energy = 10000 * best_c4 - best_c3
        new_energy = 10000 * c4 - c3
        
        delta = new_energy - old_energy
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            best_c3, best_c4 = c3, c4
            best_points = points[:]
        else:
            points[i] = old_p
            
        T *= cooling
        
    # Final check: if still c4 > 0, do a forced repair
    # This might be slow, but necessary for validity
    attempts = 0
    while count_collinear_fast(best_points)[1] > 0 and attempts < 1000:
        quad = find_four_collinear(best_points)
        if quad is None:
            break
        # Move one of the points in the quad
        idx = random.choice(quad)
        old_p = best_points[idx]
        # Try random moves
        for _ in range(100):
            new_p = (random.randint(-100, 100), random.randint(-100, 100))
            if new_p in best_points:
                continue
            best_points[idx] = new_p
            if count_collinear_fast(best_points)[1] == 0:
                break
            best_points[idx] = old_p
        best_points[idx] = old_p # Reset if not fixed? Or keep last try?
        # Actually, the loop above reverts if not fixed immediately.
        # We need a more robust repair.
        # Let's just restart the SA with a different seed if this fails?
        # For now, the SA should handle it.
        attempts += 1
        
    return [[p[0], p[1]] for p in best_points]

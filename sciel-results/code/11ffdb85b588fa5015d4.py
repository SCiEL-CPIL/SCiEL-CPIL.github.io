import random
import math
import time

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Points are [x, y]
    # We will use tuples (x, y) internally for hashing/sets
    
    def normalize_line(x1, y1, x2, y2):
        # Line equation: A(x - x1) + B(y - y1) = 0 => Ax + By + C = 0
        A = y1 - y2
        B = x2 - x1
        C = -(A * x1 + B * y1)
        
        # Normalize to avoid duplicates like (1, 1, 1) and (2, 2, 2)
        # Find GCD of A, B, C
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return abs(a)
        
        g = gcd(gcd(A, B), C)
        if g != 0:
            A //= g
            B //= g
            C //= g
        
        # Canonical sign: first non-zero of A, B, C should be positive
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
            
        return (A, B, C)

    def evaluate(points):
        # points: list of (x, y)
        # Check distinctness
        if len(set(points)) != 16:
            return -1000000000
        
        # Check bounds
        for x, y in points:
            if abs(x) > 1000 or abs(y) > 1000:
                return -1000000000
        
        line_map = {}
        n = len(points)
        
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                key = normalize_line(x1, y1, x2, y2)
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
        
        score = 0
        for line_set in line_map.values():
            size = len(line_set)
            if size == 4:
                score += 1
            elif size >= 5:
                score -= 1000000000  # Invalid
        
        return score

    def random_point():
        return (random.randint(-1000, 1000), random.randint(-1000, 1000))

    def generate_initial():
        # Start with 4x4 grid as a strong seed, then randomize slightly or fully
        # 4x4 grid gives 10.
        # Let's try a mix of grid and random restarts.
        mode = random.random()
        if mode < 0.5:
            # 4x4 grid
            return [(x, y) for x in range(4) for y in range(4)]
        else:
            # Random
            return [random_point() for _ in range(16)]

    best_score = -1
    best_points = None
    
    # Simulated Annealing
    # We will run several restarts
    max_restarts = 20
    steps_per_restart = 20000
    
    current_time = time.time()
    elapsed = current_time - start_time
    
    # Estimate time per step
    # Evaluation is ~120 pairs * set ops. Very fast.
    # Let's just run as many as fit in the time.
    
    restart = 0
    while elapsed < time_limit * 0.9 and restart < max_restarts:
        points = generate_initial()
        current_score = evaluate(points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = points[:]
        
        T = 1000.0
        T_min = 1.0
        cooling = 0.99
        
        # Local search loop
        local_steps = 0
        while local_steps < steps_per_restart:
            # Check time
            if time.time() - start_time > time_limit * 0.95:
                break
            
            # Pick a point to move
            idx = random.randint(0, 15)
            old_x, old_y = points[idx]
            
            # Random perturbation
            # Sometimes small step, sometimes large
            if random.random() < 0.5:
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
            else:
                dx = random.randint(-100, 100)
                dy = random.randint(-100, 100)
            
            new_x = old_x + dx
            new_y = old_y + dy
            
            # Clip to bounds
            if new_x < -1000: new_x = -1000
            if new_x > 1000: new_x = 1000
            if new_y < -1000: new_y = -1000
            if new_y > 1000: new_y = 1000
            
            # If no change, skip
            if new_x == old_x and new_y == old_y:
                continue
                
            points[idx] = (new_x, new_y)
            new_score = evaluate(points)
            
            delta = new_score - current_score
            current_score = new_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                # Accept
                pass
            else:
                # Reject
                points[idx] = (old_x, old_y)
                current_score = evaluate(points) # Re-eval or keep old? We have old score.
                # Actually, we can just revert and keep current_score as old.
                # But we modified points. So revert.
                # The score variable `current_score` was updated to new_score.
                # So if we reject, we must revert points and score.
                points[idx] = (old_x, old_y)
                current_score = current_score - delta # Back to old score
            
            if current_score > best_score:
                best_score = current_score
                best_points = points[:]
            
            T *= cooling
            if T < T_min:
                T = T_min
            local_steps += 1
        
        restart += 1
        elapsed = time.time() - start_time

    # Return best valid points
    if best_points is None:
        # Fallback to 4x4 grid
        return [[x, y] for x in range(4) for y in range(4)]
    
    # Ensure return format is list of lists [x, y]
    return [[x, y] for x, y in best_points]

# EVOLVE-BLOCK-START
# The function construct is defined above.
# To satisfy the "keep function names" constraint, we just need construct to be present.
# The code above defines it.
# EVOLVE-BLOCK-END

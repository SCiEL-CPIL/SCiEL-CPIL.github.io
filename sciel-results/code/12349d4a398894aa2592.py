import random
import math
import time

def construct():
    # Grid bounds
    BOUND = 100
    N_POINTS = 20
    
    # Pre-generate grid points for fast random selection
    # To save memory/time, we can just generate random x,y in range
    # But a list allows uniform sampling from a pool.
    # Pool size 201x201 = 40401. Small enough.
    grid_points = [(x, y) for x in range(-BOUND, BOUND + 1) for y in range(-BOUND, BOUND + 1)]
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        
        # Normalize (A, B, C)
        # Find gcd of A, B, C
        from math import gcd
        # gcd for 3 numbers
        def gcd3(a, b, c):
            return gcd(gcd(abs(a), abs(b)), abs(c))
            
        g = gcd3(A, B, C)
        if g != 0:
            A //= g
            B //= g
            C //= g
        
        # Ensure unique representation
        # A > 0 or (A == 0 and B > 0) or (A == 0 and B == 0 and C > 0)
        if A < 0:
            A = -A
            B = -B
            C = -C
        elif A == 0:
            if B < 0:
                B = -B
                C = -C
            elif B == 0 and C < 0:
                C = -C
                
        return (A, B, C)

    def evaluate(points):
        """
        Returns (is_valid, num_triples)
        """
        line_counts = {}
        n = len(points)
        for i in range(n):
            p1 = points[i]
            for j in range(i + 1, n):
                p2 = points[j]
                key = get_line_key(p1, p2)
                line_counts[key] = line_counts.get(key, 0) + 1
        
        is_valid = True
        num_triples = 0
        
        # Check counts
        # If a line has k points, count is k*(k-1)/2
        # k=2 -> 1
        # k=3 -> 3
        # k=4 -> 6
        # k=5 -> 10
        # We need to ensure no k >= 4.
        # So if any count >= 6, invalid.
        # If count == 3, it's a triple.
        
        for count in line_counts.values():
            if count >= 6:
                is_valid = False
                break
            if count == 3:
                num_triples += 1
        
        return is_valid, num_triples

    def random_valid_points():
        points = [random.choice(grid_points) for _ in range(N_POINTS)]
        # Remove duplicates
        # Use a set to ensure distinctness
        point_set = set(points)
        while len(point_set) < N_POINTS:
            point_set.add(random.choice(grid_points))
        points = list(point_set)
        
        # Ensure no 4 collinear
        # Repair loop
        max_attempts = 1000
        for _ in range(max_attempts):
            is_valid, _ = evaluate(points)
            if is_valid:
                return points
            
            # Find a point to swap
            # Pick a random point
            idx = random.randint(0, N_POINTS - 1)
            # Pick a new random point
            new_p = random.choice(grid_points)
            # Ensure new_p is not already in points (except the one being replaced)
            # Simple retry if duplicate
            while new_p in points and new_p != points[idx]:
                new_p = random.choice(grid_points)
            
            points[idx] = new_p
            
        # If failed to find valid, return a known valid config or raise
        # Fallback: Greedy construction
        return greedy_construct()

    def greedy_construct():
        points = []
        # Sort grid points to have some structure? Or just random.
        # Use random order
        candidates = grid_points[:]
        random.shuffle(candidates)
        
        for p in candidates:
            if len(points) == N_POINTS:
                break
            # Check if adding p is valid
            test_points = points + [p]
            is_valid, _ = evaluate(test_points)
            if is_valid:
                points.append(p)
        
        if len(points) < N_POINTS:
            # Should not happen, but fallback to random fill
            while len(points) < N_POINTS:
                p = random.choice(grid_points)
                if p not in points:
                    points.append(p)
                    # Re-validate? Greedy might break. 
                    # For robustness, just return whatever we have if it's valid, else random.
                    # But evaluate says valid.
        
        return points

    # Simulated Annealing
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    
    # Initial solution
    current_points = random_valid_points()
    current_valid, current_triples = evaluate(current_points)
    
    best_points = current_points[:]
    best_triples = current_triples
    
    T = 10.0
    T_min = 0.1
    cooling = 0.9999
    
    while time.time() - start_time < time_limit:
        # Mutation
        idx = random.randint(0, N_POINTS - 1)
        new_p = random.choice(grid_points)
        
        # Ensure distinctness
        if new_p in current_points and new_p != current_points[idx]:
            continue
            
        new_points = current_points[:]
        new_points[idx] = new_p
        
        new_valid, new_triples = evaluate(new_points)
        
        if not new_valid:
            # Reject invalid
            T *= cooling
            continue
            
        delta = new_triples - current_triples
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_points = new_points
            current_triples = new_triples
            
            if new_triples > best_triples:
                best_points = new_points[:]
                best_triples = new_triples
        
        T *= cooling
        if T < T_min:
            T = T_min

    return [[x, y] for x, y in best_points]

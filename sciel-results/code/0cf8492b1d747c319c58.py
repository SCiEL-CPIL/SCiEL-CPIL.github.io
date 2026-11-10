import time
import random
import itertools

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    def check_no_four_collinear(points):
        n = len(points)
        # Precompute all pairs and their lines? 
        # For N=20, O(N^4) is 116280 checks. 
        # Each check is 2 cross products. 
        # This is fast enough to do frequently? 
        # 116k * 2 ops = 232k ops. 
        # In Python, this might take ~0.01-0.05s. 
        # We can do this ~2000 times in 110s. 
        # That's enough for local search.
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                        # Collinear if cross product of (p2-p1) and (p3-p1) is 0 
                        # AND cross product of (p2-p1) and (p4-p1) is 0
                        d1x = p2[0] - p1[0]
                        d1y = p2[1] - p1[1]
                        
                        d2x = p3[0] - p1[0]
                        d2y = p3[1] - p1[1]
                        
                        cross1 = d1x * d2y - d1y * d2x
                        if cross1 != 0:
                            continue
                        
                        d3x = p4[0] - p1[0]
                        d3y = p4[1] - p1[1]
                        
                        cross2 = d1x * d3y - d1y * d3x
                        if cross2 == 0:
                            return False
        return True

    def estimate_max_3_free(points):
        # Greedy lower bound: 
        # Sort points by some heuristic? 
        # Just try to build a set with no 3 collinear.
        # This is an independent set problem on the hypergraph of collinear triples.
        # Greedy: 
        # 1. Identify all collinear triples.
        # 2. Use a simple greedy algorithm: 
        #    Start with all points. 
        #    While there is a collinear triple in the set, remove one point from it.
        #    This gives a valid 3-free set. The size is an upper bound? No, it's a valid set, so S >= size.
        #    Actually, we want to MINIMIZE S. 
        #    If we can find a small 3-free set, that doesn't help. 
        #    We want to know if there EXISTS a large 3-free set. 
        #    If the greedy removal process leaves a small set, it suggests S is small? 
        #    No, greedy removal is not optimal.
        
        # Better heuristic: 
        # Randomly permute points and greedily add to a candidate set if no collinear triple is formed.
        # Do this multiple times and take the maximum size found. This approximates S from below.
        # If the max found is small, S is likely small.
        
        best_size = 0
        n = len(points)
        indices = list(range(n))
        
        # Precompute collinearity for triples? 
        # For N=20, 1140 triples. 
        # We can precompute a set of collinear triples.
        collinear_triples = set()
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    d1x = p2[0] - p1[0]
                    d1y = p2[1] - p1[1]
                    d2x = p3[0] - p1[0]
                    d2y = p3[1] - p1[1]
                    if d1x * d2y - d1y * d2x == 0:
                        collinear_triples.add((i, j, k))
        
        # Greedy estimation
        for _ in range(5): # 5 restarts for estimation
            random.shuffle(indices)
            current_set = set()
            for idx in indices:
                # Check if adding idx creates a collinear triple with any pair in current_set
                valid = True
                for i, j in itertools.combinations(current_set, 2):
                    if (min(idx, i, j), min(max(idx, i, j), max(i, j)), max(idx, i, j)) in collinear_triples:
                        # Normalize tuple to check
                        t = tuple(sorted((idx, i, j)))
                        if t in collinear_triples:
                            valid = False
                            break
                if valid:
                    current_set.add(idx)
            if len(current_set) > best_size:
                best_size = len(current_set)
                
        return best_size

    def get_initial_config():
        # Start with a random configuration in a box
        # Or a structured one?
        # Let's try a "grid-like" start but perturbed to avoid 4-collinearities.
        # Actually, random is safer to start, then optimize.
        # But random has very few collinearities, so S is high (close to 20).
        # We want to INCREASE collinearities (to reduce S).
        # So we should start with a configuration that HAS many collinearities, 
        # then perturb to break 4-collinearities while keeping 3-collinearities.
        
        # Idea: Use points on a few lines.
        # Line 1: y=0, points (0,0), (1,0), (2,0) -> 3 collinear.
        # If we put 4 points on y=0, it's invalid.
        # So we can have at most 3 points on any line.
        
        # Construction: 
        # Take a 4x5 grid. 
        # Perturb points slightly.
        # This might be hard to control.
        
        # Alternative: 
        # Use the fact that we want to minimize S.
        # A known good configuration for small N is often derived from finite fields or specific polynomials.
        # For N=20, let's just use random restarts and simulated annealing.
        # To help the SA, we can bias moves to create collinearities?
        # No, just minimize the estimated S.
        
        # Let's start with a random set.
        points = []
        used = set()
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                used.add((x, y))
                points.append([x, y])
        return points

    best_points = None
    best_score = 21  # Max possible is 20
    
    # Run multiple restarts
    while time.time() - start_time < time_limit:
        # Initialize
        points = get_initial_config()
        
        # Ensure no 4 collinear (random likely has none, but check)
        if not check_no_four_collinear(points):
            # Perturb until valid
            for _ in range(100):
                idx = random.randint(0, 19)
                points[idx][0] += random.randint(-5, 5)
                points[idx][1] += random.randint(-5, 5)
                # Clamp
                points[idx][0] = max(-100, min(100, points[idx][0]))
                points[idx][1] = max(-100, min(100, points[idx][1]))
                if check_no_four_collinear(points):
                    break
        
        if not check_no_four_collinear(points):
            continue
            
        current_score = estimate_max_3_free(points)
        
        if current_score < best_score:
            best_score = current_score
            best_points = [p[:] for p in points]
            if best_score <= 7: # Good enough?
                break
                
        # Simulated Annealing
        T = 1.0
        cooling = 0.99
        for step in range(500):
            if time.time() - start_time > time_limit:
                break
                
            # Pick a random point
            idx = random.randint(0, 19)
            old_x, old_y = points[idx]
            
            # Random move
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
            new_x = old_x + dx
            new_y = old_y + dy
            
            # Bounds check
            if abs(new_x) > 100 or abs(new_y) > 100:
                continue
            
            # Distinctness check
            if (new_x, new_y) in [(p[0], p[1]) for i, p in enumerate(points) if i != idx]:
                continue
                
            points[idx] = [new_x, new_y]
            
            # Validity check (No 4 collinear)
            if not check_no_four_collinear(points):
                points[idx] = [old_x, old_y]
                continue
            
            new_score = estimate_max_3_free(points)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = [p[:] for p in points]
            else:
                points[idx] = [old_x, old_y]
                
            T *= cooling
            
    return best_points

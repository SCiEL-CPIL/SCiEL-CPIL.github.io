import random
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 110  # Safety margin
    
    N = 20
    GRID = 100
    
    # Helper: Check if 3 points are collinear
    def is_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])
    
    # Helper: Count 3-lines and check for 4-collinear
    # Returns (num_3_lines, has_4_collinear)
    def evaluate_lines(points):
        num_3 = 0
        has_4 = False
        # Use a set of normalized line equations to avoid double counting
        # Or just iterate pairs and count? 
        # Iterating pairs O(N^2) and counting is O(N) -> O(N^3) which is 8000 ops. Fast.
        
        lines = {}
        for i in range(N):
            for j in range(i+1, N):
                p1 = points[i]
                p2 = points[j]
                # Normalize line (A, B, C) for Ax + By + C = 0
                A = p2[1] - p1[1]
                B = p1[0] - p2[0]
                C = p2[0]*p1[1] - p1[0]*p2[1]
                # Normalize
                g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
                if g > 0:
                    A //= g
                    B //= g
                    C //= g
                # Ensure canonical sign
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A = -A
                    B = -B
                    C = -C
                key = (A, B, C)
                if key not in lines:
                    lines[key] = 2 # at least 2
                else:
                    lines[key] += 1
        
        for count in lines.values():
            if count == 3:
                num_3 += 1
            elif count > 3:
                has_4 = True
                
        return num_3, has_4

    # Helper: Exact Max 3-Free Subset (Branch and Bound)
    def max_3_free(points):
        N = len(points)
        # Precompute collinearity matrix
        coll = [[False]*N for _ in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    if is_collinear(points[i], points[j], points[k]):
                        coll[i][j] = coll[i][k] = coll[j][i] = coll[j][k] = coll[k][i] = coll[k][j] = True
        
        best = 0
        # Greedy initial bound
        # Sort by degree?
        
        def dfs(idx, current_set, current_size):
            nonlocal best
            if idx == N:
                if current_size > best:
                    best = current_size
                return
            
            # Pruning: if current_size + (N - idx) <= best, return
            if current_size + (N - idx) <= best:
                return
                
            # Option 1: Exclude idx
            dfs(idx + 1, current_set, current_size)
            
            # Option 2: Include idx
            # Check if idx collinear with any in current_set
            valid = True
            for j in current_set:
                if coll[idx][j]:
                    valid = False
                    break
            if valid:
                current_set.append(idx)
                dfs(idx + 1, current_set, current_size + 1)
                current_set.pop()
                
        dfs(0, [], 0)
        return best

    # Simulated Annealing
    def random_point():
        return [random.randint(-GRID, GRID), random.randint(-GRID, GRID)]
    
    def random_config():
        pts = []
        while len(pts) < N:
            p = random_point()
            if p not in pts:
                pts.append(p)
        return pts

    best_config = None
    best_score = 20 # Max possible
    best_num_3 = -1
    
    current_config = random_config()
    current_num_3, current_has_4 = evaluate_lines(current_config)
    if not current_has_4:
        best_config = current_config.copy()
        best_num_3 = current_num_3
    
    T = 1.0
    cooling = 0.9995
    min_T = 0.01
    
    steps = 0
    while time.time() - start_time < time_limit:
        # Pick a point to move
        i = random.randint(0, N-1)
        old_p = current_config[i]
        new_p = random_point()
        
        if new_p in current_config:
            continue
            
        current_config[i] = new_p
        new_num_3, new_has_4 = evaluate_lines(current_config)
        
        # Calculate delta
        # We want to maximize num_3 and minimize has_4
        # Score = num_3 - 1000*has_4
        old_score = current_num_3 - (1000 if current_has_4 else 0)
        new_score = new_num_3 - (1000 if new_has_4 else 0)
        
        delta = new_score - old_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_num_3 = new_num_3
            current_has_4 = new_has_4
            
            if not new_has_4:
                if new_num_3 > best_num_3:
                    best_num_3 = new_num_3
                    best_config = current_config.copy()
        else:
            current_config[i] = old_p
            
        T *= cooling
        steps += 1
        
        # Periodic check for exact score on best_config if it's been a while
        # To save time, we only run the expensive exact solver at the end or very rarely
        if steps % 10000 == 0 and best_config is not None:
             # Just a light check? No, exact solver is too slow for frequent checks.
             pass

    # Final Phase: Verify the best candidate found by SA
    if best_config is None:
        # Fallback: Return a known valid config (parabola)
        return [[x, x * x] for x in range(-10, 10)]
        
    # Run exact max 3-free on the best candidate
    # It might be slow, but we have ~10s left.
    # To be safe, if it takes too long, we return a heuristic estimate?
    # The task asks for the config, and the evaluator calculates the score.
    # So we just need to return the config. The evaluator will do the exact B&B.
    # So we don't need to run the exact B&B ourselves unless we want to compare multiple candidates.
    # Since we only have one "best" from SA, we return it.
    
    # However, SA maximizes 3-lines. Does maximizing 3-lines guarantee minimizing max 3-free?
    # Not strictly, but it's a strong proxy.
    # To be more robust, we can try a few restarts or local improvements.
    
    # Let's try to improve the best_config further with a local search that uses the exact score?
    # No, exact score is too slow.
    
    # Let's just return the best_config found by SA.
    return best_config

# EVOLVE-BLOCK-END

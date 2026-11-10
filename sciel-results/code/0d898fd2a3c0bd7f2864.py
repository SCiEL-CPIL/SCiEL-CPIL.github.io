# EVOLVE-BLOCK-START
def construct():
    import random
    import time
    import math
    from itertools import combinations

    n = 16
    time_limit = 115.0
    start_time = time.time()
    
    # Fast scoring function
    # Returns (score, is_valid)
    # We represent lines by their normalized coefficients (A, B, C)
    # A*x + B*y + C = 0
    # Normalized: gcd(A,B,C)=1, A>0 or A==0 and B>0
    
    def get_score(pts):
        lines = {}
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i+1, n):
                x2, y2 = pts[j]
                A = y2 - y1
                B = x1 - x2
                C = x2*y1 - x1*y2
                g = math.gcd(A, math.gcd(B, C))
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
                lines[key] = lines.get(key, 0) + 1
        
        score = 0
        is_valid = True
        for count in lines.values():
            # k points on line -> count = k*(k-1)/2
            # k=2 -> 1, k=3 -> 3, k=4 -> 6, k=5 -> 10
            if count == 6:
                score += 1
            elif count == 10:
                is_valid = False
            # count > 10 means k>=6, also invalid
            elif count > 10:
                is_valid = False
            # count < 1 is impossible for distinct points
            # count 1, 3 are fine (2 or 3 points)
            # count 4, 5, 7, 8, 9 are impossible for integer k?
            # k=2:1, k=3:3, k=4:6, k=5:10.
            # So 4,5,7,8,9 should not occur.
            # If they do, it's an error in logic or non-integer geometry? No, integer points.
            # So we can ignore them or treat as invalid?
            # Actually, if points are distinct integers, the count MUST be a triangular number.
            # So if count is not 1, 3, 6, 10, 15..., something is wrong.
            # But we only care about 4 (score) and 5 (invalid).
            
        return score, is_valid

    def is_valid_move(pts, new_pt, idx):
        """
        Check if replacing pts[idx] with new_pt results in a valid configuration (no 5 collinear).
        Optimization: Only check lines involving the moved point.
        """
        # This is tricky because moving a point changes lines involving it AND lines that might have had 5 points but now have 4?
        # No, if a line had 5 points and we move one away, it becomes 4. That's a fix.
        # If a line had 4 points and we move one to it, it becomes 5. That's a break.
        # So we must check all lines involving the moved point in the NEW configuration.
        
        # To be safe and simple, we can just call get_score on the whole set.
        # With n=16, get_score is O(n^2) = 120 operations. Very fast.
        # So we can just use get_score for validation.
        
        # However, the directive says "prunes branches... during construction".
        # In local search, "pruning" means rejecting moves that create 5-collinear.
        
        pass

    def try_construct():
        """
        Greedy construction with local search refinement.
        """
        # Step 1: Initialize with a random set of 16 points?
        # Or start with a known good structure like a grid and perturb?
        # The directive says "from scratch".
        # Let's start with 0 points and add them one by one.
        
        points = []
        
        # To speed up, we can pre-select a few "anchor" points to define lines.
        # But random is safer for diversity.
        
        # Let's try a hybrid: Start with a small random set, then add points greedily.
        
        # Initial few points: random
        for _ in range(4):
            points.append([random.randint(-50, 50), random.randint(-50, 50)])
        
        # Ensure distinct
        # (Simplification: assume random distinct, check later)
        
        # Add remaining 12 points
        for _ in range(12):
            best_candidate = None
            best_score = -1
            # Generate candidates
            # 1. Points on lines defined by pairs of existing points
            # 2. Random points
            
            candidates = set()
            
            # Candidates from lines
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    g = math.gcd(dx, dy)
                    if g == 0: continue
                    step_x = dx // g
                    step_y = dy // g
                    # Try a few steps from x1
                    for t in range(-10, 11):
                        nx = x1 + step_x * t
                        ny = y1 + step_y * t
                        if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                            candidates.add((nx, ny))
            
            # Add some random candidates
            for _ in range(50):
                candidates.add((random.randint(-1000, 1000), random.randint(-1000, 1000)))
            
            # Evaluate candidates
            for cand in candidates:
                # Check if cand is already in points
                if cand in points:
                    continue
                
                # Temporarily add
                new_points = points + [list(cand)]
                
                # Check validity and score
                score, is_valid = get_score(new_points)
                
                if is_valid:
                    if score > best_score:
                        best_score = score
                        best_candidate = list(cand)
            
            if best_candidate:
                points.append(best_candidate)
            else:
                # Fallback: random point that is valid
                while True:
                    cand = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
                    if cand not in points:
                        new_points = points + [cand]
                        score, is_valid = get_score(new_points)
                        if is_valid:
                            points.append(cand)
                            break
        
        # Step 2: Local Search Refinement
        current_points = [p[:] for p in points]
        current_score, _ = get_score(current_points)
        best_points = [p[:] for p in current_points]
        best_score = current_score
        
        # Hill Climbing
        # Time budget for this instance
        local_time_limit = 5.0
        local_start = time.time()
        
        while time.time() - local_start < local_time_limit:
            # Pick a random point to move
            idx = random.randint(0, n-1)
            old_pt = current_points[idx]
            
            # Generate new position
            # Try to move it to a line with 2 other points to form a 4-line?
            # Or just random small move?
            # Let's try moving it to a position on a line defined by 2 other points.
            
            others = [i for i in range(n) if i != idx]
            i, j = random.sample(others, 2)
            x1, y1 = current_points[i]
            x2, y2 = current_points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(dx, dy)
            
            new_pt = None
            if g != 0:
                step_x = dx // g
                step_y = dy // g
                # Try a few positions on this line
                for t in range(-20, 21):
                    nx = x1 + step_x * t
                    ny = y1 + step_y * t
                    if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                        if [nx, ny] != old_pt:
                            # Check if distinct from other points
                            is_distinct = True
                            for k in range(n):
                                if k == idx: continue
                                if current_points[k] == [nx, ny]:
                                    is_distinct = False
                                    break
                            if is_distinct:
                                new_pt = [nx, ny]
                                break
            
            if new_pt is None:
                # Small random move
                new_pt = [old_pt[0] + random.randint(-10, 10), 
                          old_pt[1] + random.randint(-10, 10)]
                new_pt[0] = max(-1000, min(1000, new_pt[0]))
                new_pt[1] = max(-1000, min(1000, new_pt[1]))
                
                # Ensure distinct
                if any(current_points[k] == new_pt for k in range(n) if k != idx):
                    continue

            current_points[idx] = new_pt
            new_score, is_valid = get_score(current_points)
            
            if is_valid and new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                # Revert
                current_points[idx] = old_pt
                
        return best_points, best_score

    # Main Loop: Run multiple instances of the constructive search
    global_best_points = None
    global_best_score = -1
    
    while time.time() - start_time < time_limit:
        pts, score = try_construct()
        if score > global_best_score:
            global_best_score = score
            global_best_points = pts
            
        # If we hit a high score, we might stop early?
        # The real max is 15. Integer max is likely lower.
        # If we get 10+, that's good.
        
        # Add a bit of randomization to the start of try_construct?
        # It's already random.
        
    return global_best_points

# EVOLVE-BLOCK-END

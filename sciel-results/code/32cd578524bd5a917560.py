import random
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer for safety
    
    # Initial configuration: 4x4 grid
    # This has 8 lines of 4 (rows/cols) + 2 diagonals = 10 lines of 4.
    # No 5 collinear.
    def initial_grid():
        return [[x, y] for x in range(4) for y in range(4)]
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        if A == 0 and B == 0:
            return None
        g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
        if g != 0:
            A //= g
            B //= g
            C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def count_lines_for_point(points, idx):
        """Return list of line keys formed by points[idx] and all other points."""
        keys = []
        pi = points[idx]
        for j in range(len(points)):
            if i == j: continue
            key = get_line_key(pi, points[j])
            if key:
                keys.append(key)
        return keys

    # State variables
    points = initial_grid()
    n = len(points)
    
    # line_counts: key -> count of points on that line
    line_counts = defaultdict(int)
    # We need to initialize line_counts
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key:
                line_counts[key] += 1
                
    # Calculate initial score
    # Score = number of lines with count 4 - penalty * number of lines with count >= 5
    # We want to maximize score, but also ensure validity (no count >= 5)
    def calc_score(line_counts):
        n4 = 0
        n5plus = 0
        for count in line_counts.values():
            if count == 4:
                n4 += 1
            elif count >= 5:
                n5plus += 1
        # Heavy penalty for invalidity to steer away
        return n4 - 100 * n5plus

    current_score = calc_score(line_counts)
    best_score = current_score
    best_points = [p[:] for p in points]
    
    # If initial is valid (n5plus=0), best_score is just n4
    # Check validity
    def is_valid(line_counts):
        for count in line_counts.values():
            if count >= 5:
                return False
        return True

    if is_valid(line_counts):
        best_points = [p[:] for p in points]
        best_score = current_score

    T = 10.0
    cooling = 0.995
    steps = 0
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, n - 1)
        old_p = points[idx][:]
        
        # Determine move range
        # Start small, increase if stuck? 
        # For now, random small move
        move_range = 5
        dx = random.randint(-move_range, move_range)
        dy = random.randint(-move_range, move_range)
        
        if dx == 0 and dy == 0:
            continue
            
        new_p = [old_p[0] + dx, old_p[1] + dy]
        
        # Bounds check
        if not (-1000 <= new_p[0] <= 1000 and -1000 <= new_p[1] <= 1000):
            continue
            
        # Distinctness check
        if any(p == new_p for j, p in enumerate(points) if j != idx):
            continue

        # Perform move
        points[idx] = new_p
        
        # Update line counts
        # Remove old point's contributions
        old_keys = count_lines_for_point(points, idx) # Wait, points[idx] is new_p now.
        # We need old keys based on old_p.
        # Let's recompute old keys before changing points[idx]?
        # Yes, better to compute old_keys and new_keys.
        
        # Undo move temporarily for key calculation? Or just compute.
        # Let's revert points[idx] to old_p to get old keys?
        # No, we can just calculate keys for old_p and new_p against other points.
        
        # Revert points[idx] to old_p for a moment to get old_keys?
        # Actually, count_lines_for_point uses points list.
        # Let's just calculate keys manually.
        
        def get_keys_for_point(p, points_list, exclude_idx):
            keys = []
            for j in range(n):
                if j == exclude_idx: continue
                key = get_line_key(p, points_list[j])
                if key:
                    keys.append(key)
            return keys
            
        # We already changed points[idx] to new_p.
        # So points list has new_p at idx.
        # old_p is stored.
        # other_points = points[0:idx] + points[idx+1:]
        
        other_points = [points[j] for j in range(n) if j != idx]
        
        old_keys = []
        for p in other_points:
            key = get_line_key(old_p, p)
            if key:
                old_keys.append(key)
                
        new_keys = []
        for p in other_points:
            key = get_line_key(new_p, p)
            if key:
                new_keys.append(key)
                
        # Update line_counts
        # Decrement old keys
        for key in old_keys:
            line_counts[key] -= 1
            if line_counts[key] == 0:
                del line_counts[key]
                
        # Increment new keys
        for key in new_keys:
            line_counts[key] += 1
            
        new_score = calc_score(line_counts)
        
        # Acceptance
        delta = new_score - current_score
        if delta >= 0 or random.random() < math.exp(delta / T):
            # Accept
            current_score = new_score
            # Check if valid and better
            if is_valid(line_counts):
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in points]
        else:
            # Reject: Revert points and line_counts
            points[idx] = old_p
            
            # Revert line_counts
            # Decrement new keys
            for key in new_keys:
                line_counts[key] -= 1
                if line_counts[key] == 0:
                    del line_counts[key]
                
            # Increment old keys
            for key in old_keys:
                line_counts[key] += 1

        T *= cooling
        if T < 0.01:
            T = 0.01
            # Maybe restart with higher T if stuck?
            # Simple restart: if T is very low, boost it occasionally
            if random.random() < 0.01:
                T = 5.0

    # Final check
    # Ensure best_points is valid
    # Re-evaluate best_points to be sure
    final_points = best_points
    # Quick validation
    lc = defaultdict(int)
    for i in range(n):
        for j in range(i+1, n):
            key = get_line_key(final_points[i], final_points[j])
            if key:
                lc[key] += 1
    valid = True
    for c in lc.values():
        if c >= 5:
            valid = False
            break
    if not valid:
        # Fallback to initial grid
        return initial_grid()
        
    return final_points


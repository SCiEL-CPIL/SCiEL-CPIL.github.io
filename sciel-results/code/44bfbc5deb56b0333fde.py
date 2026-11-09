import time
import sys
import random
from collections import defaultdict

def _count_ordinary_lines(points):
    """
    Count the number of lines containing exactly 2 points.
    points: list of (x, y) tuples
    """
    n = len(points)
    if n < 2:
        return 0
    
    lines = defaultdict(list)
    
    # Map line to its points
    # Line represented by (A, B, C) for Ax + By + C = 0
    # Normalize (A, B, C) such that gcd(A,B,C)=1 and A>0 or (A==0, B>0) or (A==0, B==0, C>0)
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        
        # Normalize
        if A < 0:
            A, B, C = -A, -B, -C
        elif A == 0 and B < 0:
            A, B, C = -A, -B, -C
        elif A == 0 and B == 0 and C < 0:
            A, B, C = -A, -B, -C
            
        # Divide by gcd
        if A == 0 and B == 0:
            # This shouldn't happen for distinct points
            return (0, 0, 0)
            
        g = abs(A)
        if B != 0:
            g = __import__('math').gcd(g, abs(B))
        if C != 0:
            g = __import__('math').gcd(g, abs(C))
            
        if g > 0:
            A //= g
            B //= g
            C //= g
            
        return (A, B, C)

    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            lines[key].append((i, j))
            
    ordinary_count = 0
    for key, pairs in lines.items():
        # A line is ordinary if it contains exactly 2 points.
        # The number of pairs is k*(k-1)/2 where k is number of points.
        # If len(pairs) == 1, then k=2, so it's ordinary.
        # If len(pairs) == 3, then k=3, so it's not ordinary.
        # If len(pairs) == 6, then k=4, so it's not ordinary.
        # Generally, if len(pairs) is a triangular number T_k = k(k-1)/2, 
        # and k > 2, it's not ordinary.
        # If len(pairs) is not a triangular number, something is wrong (shouldn't happen).
        # If len(pairs) == 1, k=2, ordinary.
        
        if len(pairs) == 1:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Initial configuration: 11 on a line, 1 off
    # This has 11 ordinary lines.
    # Let's try to improve it.
    
    # Random seed for reproducibility in search, but we want to explore.
    random.seed(42)
    
    # We will use a local search.
    # State: list of 12 points [x, y]
    # We start with a random configuration or a structured one.
    
    # A good starting point might be a small grid.
    # Let's try a 3x4 grid in a small box.
    # Points: (0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2), (3,2)
    # This is a valid set of 12 distinct integer points.
    
    current_points = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2]
    ]
    
    current_score = _count_ordinary_lines(current_points)
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # Local search
    # We will try to move each point to a nearby integer coordinate.
    # We limit the coordinate range to avoid huge numbers initially, then expand if needed?
    # Actually, the bound is 10^9, but small integers are usually sufficient for these combinatorial configurations.
    
    step = 0
    while time.time() < deadline:
        step += 1
        # Pick a random point
        idx = random.randint(0, 11)
        x, y = current_points[idx]
        
        # Try a random move
        # Move within a small radius
        radius = random.randint(1, 5)
        dx = random.randint(-radius, radius)
        dy = random.randint(-radius, radius)
        
        new_x = x + dx
        new_y = y + dy
        
        # Check bounds (loose check, we can scale later if needed, but small is better for collision avoidance)
        # Actually, we just need distinct points.
        
        # Check if new point is distinct from others
        is_distinct = True
        for j in range(12):
            if j == idx:
                continue
            if current_points[j][0] == new_x and current_points[j][1] == new_y:
                is_distinct = False
                break
        
        if not is_distinct:
            continue
            
        # Calculate new score
        # Temporarily replace
        old_point = current_points[idx]
        current_points[idx] = [new_x, new_y]
        
        new_score = _count_ordinary_lines(current_points)
        
        # Accept if better, or with some probability if same (to explore)
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        elif new_score == current_score:
            # Accept with 50% probability to allow wandering
            if random.random() < 0.5:
                current_score = new_score
            else:
                # Revert
                current_points[idx] = old_point
        else:
            # Revert
            current_points[idx] = old_point
            
    return best_points

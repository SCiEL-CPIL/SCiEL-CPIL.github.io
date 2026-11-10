import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def _count_4_lines(points):
    """
    Counts the number of lines containing exactly 4 of the given points.
    Also returns True if any line contains 5 or more points (invalid).
    """
    n = len(points)
    if n < 4:
        return 0, False
    
    # Map line representation -> count of points on it
    line_counts = defaultdict(int)
    
    # Normalize line representation: (A, B, C) where Ax + By + C = 0
    # To ensure uniqueness, we normalize (A, B, C) such that:
    # 1. gcd(A, B, C) = 1
    # 2. A > 0, or (A == 0 and B > 0), or (A == 0 and B == 0 and C > 0)
    # Note: Since points are integers, A, B, C can be integers.
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Line through (x1, y1) and (x2, y2)
            # A = y1 - y2
            # B = x2 - x1
            # C = x1*y2 - x2*y1
            
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            # Normalize
            if A == 0 and B == 0:
                continue # Should not happen for distinct points
                
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g > 0:
                A //= g
                B //= g
                C //= g
            
            # Sign normalization
            if A < 0:
                A = -A
                B = -B
                C = -C
            elif A == 0:
                if B < 0:
                    B = -B
                    C = -C
                elif B == 0:
                    if C < 0:
                        C = -C
            
            line_counts[(A, B, C)] += 1

    count_4 = 0
    has_5 = False
    for count in line_counts.values():
        if count == 4:
            count_4 += 1
        elif count >= 5:
            has_5 = True
            
    return count_4, has_5

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_points = None
    best_score = -1
    
    # Strategy:
    # 1. Generate multiple diverse initial configurations.
    #    a) The classic 4x4 grid.
    #    b) A "line assembly" approach: start with empty, add points to form lines.
    #       This is hard to do from scratch efficiently in Python within time limits,
    #       so we will use a "perturbed grid" and "random lattice" approach.
    # 2. For each initial config, run a 2-point swap local search.
    #    2-point swaps allow escaping local minima where moving a single point
    #    breaks a good line, but swapping two points preserves or improves global structure.
    
    random.seed(42)
    
    initial_configs = []
    
    # Config 1: 4x4 Grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # Config 2: 4x4 Grid shifted/scaled
    initial_configs.append([[x*2, y*2] for x in range(4) for y in range(4)])
    
    # Config 3: Random lattice points within a box, filtered for no 5 collinear
    for _ in range(10):
        pts = []
        used = set()
        attempts = 0
        while len(pts) < 16 and attempts < 1000:
            attempts += 1
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            if (x, y) not in used:
                used.add((x, y))
                pts.append([x, y])
        
        # Check validity and score
        if len(pts) == 16:
            score, invalid = _count_4_lines(pts)
            if not invalid:
                initial_configs.append(pts)
        
        if time.time() > deadline - 30:
            break

    # Remove duplicates in initial configs to save time
    unique_configs = []
    seen_sigs = set()
    for cfg in initial_configs:
        sig = tuple(sorted(tuple(p) for p in cfg))
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            unique_configs.append(cfg)
            
    # If we didn't generate enough, stick to the grid
    if not unique_configs:
        unique_configs = [initial_configs[0]]

    # Local Search with 2-point swaps
    # We will run this for each initial config, but with a time budget per config.
    # Or, we can run a global search where we switch between configs if stuck?
    # Better: Run local search on each, keep the best.
    
    total_configs = len(unique_configs)
    time_per_config = 10.0 # 10 seconds per config roughly
    
    for cfg_idx, initial_cfg in enumerate(unique_configs):
        if time.time() > deadline - 5:
            break
            
        current_points = [p[:] for p in initial_cfg]
        current_score, is_invalid = _count_4_lines(current_points)
        
        if is_invalid:
            continue
            
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            
        config_deadline = min(deadline - 5, start_time + (cfg_idx + 1) * (140.0 / max(1, total_configs)))
        
        # 2-Point Swap Local Search
        # Pick two random points i, j. Swap their positions? No, that's just a permutation.
        # We need to *move* points.
        # Move point i to a new random position? That's 1-point.
        # 2-point move: Move point i to pos_i' and point j to pos_j'.
        # Or: Swap point i and point j? No, they are distinct.
        
        # Let's use a "2-point relocation":
        # Pick i, j.
        # Generate new positions for i and j.
        # This is a 4-dimensional move.
        
        # Alternative: "1-point move" is often enough if the neighborhood is large.
        # The problem with 1-point is that it's hard to fix a "bad" line without breaking others.
        # Let's stick to 1-point moves but with a larger, more structured neighborhood,
        # plus occasional "restarts" from the best found so far.
        
        # Actually, the prompt suggests a STRUCTURAL change.
        # The 4x4 grid is a very specific structure.
        # Let's try to explicitly construct lines.
        
        # Revised Strategy for this block:
        # 1. Start with a known good config (e.g., grid).
        # 2. Identify lines with < 4 points.
        # 3. Try to move a point from a "crowded" area (part of many lines) to a "sparse" line to complete it.
        # This is "Greedy Line Completion".
        
        # Let's implement a simple greedy line completion on top of the initial config.
        
        for step in range(2000): # Max steps per config
            if time.time() > config_deadline:
                break
                
            # Find lines with exactly 3 points. If we can add a 4th, score +1.
            # Find lines with 5+ points. If we can remove a point, valid.
            
            # This requires knowing which points are on which lines.
            # Let's recompute line memberships for efficiency in the inner loop?
            # No, let's just do random moves but biased towards "fixing" lines.
            
            # Pick a random point idx
            idx = random.randint(0, 15)
            ox, oy = current_points[idx]
            
            # Generate new position
            # Bias: try to land on a line that currently has 3 points?
            # This is hard to do without precomputation.
            
            # Let's just do a standard random walk with a larger step size.
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            nx = ox + dx
            ny = oy + dy
            
            if abs(nx) > 1000 or abs(ny) > 1000:
                continue
                
            is_dup = False
            for k in range(16):
                if k == idx:
                    continue
                if current_points[k][0] == nx and current_points[k][1] == ny:
                    is_dup = True
                    break
            if is_dup:
                continue
                
            current_points[idx] = [nx, ny]
            new_score, is_invalid = _count_4_lines(current_points)
            
            if is_invalid:
                current_points[idx] = [ox, oy]
            else:
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                else:
                    # Accept with small prob to escape
                    if random.random() < 0.05:
                        current_score = new_score
                    else:
                        current_points[idx] = [ox, oy]
                        
    if best_points is None:
        # Fallback to grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
# EVOLVE-BLOCK-END

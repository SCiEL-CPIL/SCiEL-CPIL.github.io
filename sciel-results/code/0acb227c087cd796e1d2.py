import time
import random
import math

DEADLINE = 115.0

def construct():
    start_time = time.time()
    random.seed(42)
    
    def are_collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1])
    
    def has_four_collinear(points):
        n = len(points)
        for i in range(n):
            slopes = {}
            xi, yi = points[i]
            for j in range(n):
                if i == j:
                    continue
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0 and dy == 0:
                    continue
                if dx == 0:
                    slope = (0, 1)
                elif dy == 0:
                    slope = (1, 0)
                else:
                    g = math.gcd(abs(dx), abs(dy))
                    dx, dy = dx // g, dy // g
                    if dx < 0:
                        dx, dy = -dx, -dy
                    slope = (dx, dy)
                if slope in slopes:
                    slopes[slope] += 1
                    if slopes[slope] >= 3:
                        return True
                else:
                    slopes[slope] = 1
        return False

    def count_triples(points):
        """Count collinear triples."""
        count = 0
        n = len(points)
        for i in range(n):
            slopes = {}
            xi, yi = points[i]
            for j in range(n):
                if i == j:
                    continue
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0 and dy == 0:
                    continue
                if dx == 0:
                    slope = (0, 1)
                elif dy == 0:
                    slope = (1, 0)
                else:
                    g = math.gcd(abs(dx), abs(dy))
                    dx, dy = dx // g, dy // g
                    if dx < 0:
                        dx, dy = -dx, -dy
                    slope = (dx, dy)
                if slope in slopes:
                    slopes[slope] += 1
                else:
                    slopes[slope] = 1
            for k in slopes.values():
                if k >= 2:
                    count += k * (k - 1) // 2
        return count

    def exact_max_subset_size(points):
        """
        Calculate the largest subset with no 3 collinear points.
        Uses DP over bitmask for N=20.
        """
        n = len(points)
        # Precompute pair masks for each vertex v
        # pair_masks[v] is a list of masks (1<<u | 1<<w) for u < w collinear with v
        pair_masks = [[] for _ in range(n)]
        
        for i in range(n):
            # Group other points by slope from i
            slope_to_points = {}
            xi, yi = points[i]
            for j in range(n):
                if i == j:
                    continue
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0 and dy == 0:
                    continue
                if dx == 0:
                    slope = (0, 1)
                elif dy == 0:
                    slope = (1, 0)
                else:
                    g = math.gcd(abs(dx), abs(dy))
                    dx, dy = dx // g, dy // g
                    if dx < 0:
                        dx, dy = -dx, -dy
                    slope = (dx, dy)
                
                if slope not in slope_to_points:
                    slope_to_points[slope] = []
                slope_to_points[slope].append(j)
            
            # For each slope, generate all pairs
            for pts_on_line in slope_to_points.values():
                k = len(pts_on_line)
                if k >= 2:
                    for a in range(k):
                        for b in range(a+1, k):
                            u = pts_on_line[a]
                            w = pts_on_line[b]
                            mask = (1 << u) | (1 << w)
                            pair_masks[i].append(mask)
        
        # DP array
        # valid[S] is True if S has no 3 collinear points
        # We only need to check the last added bit's conflicts
        size = 1 << n
        valid = [False] * size
        valid[0] = True
        
        max_size = 0
        
        for S in range(1, size):
            # Find lowest set bit
            # v = (S & -S).bit_length() - 1
            lowbit = S & -S
            v = lowbit.bit_length() - 1
            S_prev = S ^ lowbit
            
            if not valid[S_prev]:
                valid[S] = False
                continue
            
            # Check if S_prev contains any pair collinear with v
            is_conflict = False
            for pm in pair_masks[v]:
                if (S_prev & pm) == pm:
                    is_conflict = True
                    break
            
            if not is_conflict:
                valid[S] = True
                # Update max size
                # popcount
                cnt = bin(S).count('1')
                if cnt > max_size:
                    max_size = cnt
            else:
                valid[S] = False
                
        return max_size

    def generate_initial():
        """Generate a line-packed initial configuration."""
        pts = []
        used = set()
        # Start with 5 random points
        while len(pts) < 5:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                used.add((x, y))
                pts.append((x, y))
        
        # Add remaining 15 by placing on lines defined by existing pairs
        for _ in range(15):
            if time.time() - start_time > DEADLINE:
                break
            placed = False
            for _ in range(50): # Try 50 times to place
                # Pick two existing points
                i = random.randint(0, len(pts)-1)
                j = random.randint(0, len(pts)-1)
                if i == j:
                    continue
                p1 = pts[i]
                p2 = pts[j]
                
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                if dx == 0 and dy == 0:
                    continue
                
                # Find integer points on the line
                # Line: x = x1 + t*dx, y = y1 + t*dy
                # We want t such that x, y are integers and within bounds
                # Since x1, y1, dx, dy are integers, any integer t works.
                # But we want points distinct from existing ones.
                
                # Try random t
                # Estimate range of t
                # |x1 + t*dx| <= 100 => -100 - x1 <= t*dx <= 100 - x1
                # If dx != 0:
                #   t_min = ceil((-100 - x1) / dx) if dx > 0 else floor((-100 - x1) / dx)
                #   t_max = floor((100 - x1) / dx) if dx > 0 else ceil((100 - x1) / dx)
                
                t_min = -100
                t_max = 100
                if dx != 0:
                    # Solve for t bounds
                    # x1 + t*dx in [-100, 100]
                    # t*dx in [-100-x1, 100-x1]
                    if dx > 0:
                        t_min = math.ceil((-100 - p1[0]) / dx)
                        t_max = math.floor((100 - p1[0]) / dx)
                    else:
                        t_min = math.ceil((100 - p1[0]) / dx)
                        t_max = math.floor((-100 - p1[0]) / dx)
                
                if dy != 0:
                    # y1 + t*dy in [-100, 100]
                    if dy > 0:
                        t_min_y = math.ceil((-100 - p1[1]) / dy)
                        t_max_y = math.floor((100 - p1[1]) / dy)
                    else:
                        t_min_y = math.ceil((100 - p1[1]) / dy)
                        t_max_y = math.floor((-100 - p1[1]) / dy)
                    
                    t_min = max(t_min, t_min_y)
                    t_max = min(t_max, t_max_y)
                
                if t_min > t_max:
                    continue
                
                t = random.randint(t_min, t_max)
                nx = p1[0] + t * dx
                ny = p1[1] + t * dy
                
                if -100 <= nx <= 100 and -100 <= ny <= 100:
                    if (nx, ny) not in used:
                        used.add((nx, ny))
                        pts.append((nx, ny))
                        placed = True
                        break
            
            if not placed:
                # Fallback: random point
                while True:
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    if (x, y) not in used:
                        used.add((x, y))
                        pts.append((x, y))
                        break
                        
        return pts

    def local_search_line_pack(pts):
        """
        Improve configuration by moving points to lie on lines defined by pairs.
        """
        current_pts = pts[:]
        current_triples = count_triples(current_pts)
        
        T = 10.0
        cooling = 0.99
        
        steps = 0
        while time.time() - start_time < DEADLINE:
            steps += 1
            if steps % 1000 == 0 and T < 0.1:
                break
            
            idx = random.randint(0, 19)
            old_p = current_pts[idx]
            
            # Strategy: Pick a line defined by two other points, and move idx to a point on that line
            # Or pick a random move
            if random.random() < 0.8:
                # Structural move
                # Pick two other points
                candidates = [j for j in range(20) if j != idx]
                j1 = random.choice(candidates)
                j2 = random.choice(candidates)
                if j1 == j2:
                    continue
                
                p1 = current_pts[j1]
                p2 = current_pts[j2]
                
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                if dx == 0 and dy == 0:
                    continue
                
                # Find a good t
                t_min = -100
                t_max = 100
                if dx != 0:
                    if dx > 0:
                        t_min = math.ceil((-100 - p1[0]) / dx)
                        t_max = math.floor((100 - p1[0]) / dx)
                    else:
                        t_min = math.ceil((100 - p1[0]) / dx)
                        t_max = math.floor((-100 - p1[0]) / dx)
                
                if dy != 0:
                    if dy > 0:
                        t_min_y = math.ceil((-100 - p1[1]) / dy)
                        t_max_y = math.floor((100 - p1[1]) / dy)
                    else:
                        t_min_y = math.ceil((100 - p1[1]) / dy)
                        t_max_y = math.floor((-100 - p1[1]) / dy)
                    
                    t_min = max(t_min, t_min_y)
                    t_max = min(t_max, t_max_y)
                
                if t_min > t_max:
                    continue
                
                # Try a few random t's near the current position or random
                best_new_p = None
                best_new_triples = -1
                for _ in range(5):
                    t = random.randint(t_min, t_max)
                    nx = p1[0] + t * dx
                    ny = p1[1] + t * dy
                    if -100 <= nx <= 100 and -100 <= ny <= 100:
                        new_p = (nx, ny)
                        if new_p == old_p:
                            continue
                        # Check if new_p is already occupied by another point
                        occupied = False
                        for k in range(20):
                            if k != idx and current_pts[k] == new_p:
                                occupied = True
                                break
                        if occupied:
                            continue
                        
                        # Test validity (no 4 collinear)
                        test_pts = current_pts[:]
                        test_pts[idx] = new_p
                        if has_four_collinear(test_pts):
                            continue
                        
                        new_triples = count_triples(test_pts)
                        if new_triples > best_new_triples:
                            best_new_triples = new_triples
                            best_new_p = new_p
                
                if best_new_p is not None:
                    new_p = best_new_p
                    # Acceptance criterion
                    delta = best_new_triples - current_triples
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_pts[idx] = new_p
                        current_triples = best_new_triples
                else:
                    # No valid move found in this structural attempt, skip
                    continue
            else:
                # Random move
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                new_p = (x, y)
                if new_p == old_p:
                    continue
                occupied = False
                for k in range(20):
                    if k != idx and current_pts[k] == new_p:
                        occupied = True
                        break
                if occupied:
                    continue
                
                test_pts = current_pts[:]
                test_pts[idx] = new_p
                if has_four_collinear(test_pts):
                    continue
                
                new_triples = count_triples(test_pts)
                delta = new_triples - current_triples
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_pts = test_pts
                    current_triples = new_triples
            
            T *= cooling
            if T < 0.1:
                T = 0.1
        
        return current_pts, current_triples

    best_points = None
    best_score = 21
    
    # Generate and optimize multiple initial configurations
    for trial in range(5):
        if time.time() - start_time > DEADLINE:
            break
        
        init_pts = generate_initial()
        
        # Check initial validity
        if has_four_collinear(init_pts):
            # Try to fix or restart
            continue
            
        optimized_pts, triples = local_search_line_pack(init_pts)
        
        if time.time() - start_time > DEADLINE:
            break
            
        # Calculate exact score
        score = exact_max_subset_size(optimized_pts)
        
        if score < best_score:
            best_score = score
            best_points = optimized_pts[:]
            
        # If we found a very good score, we might stop early or keep searching
        # Theoretical minimum for 20 points is likely around 8-10?
        # If best_score <= 8, we can stop.
        if best_score <= 8:
            break

    if best_points is None:
        # Fallback
        while True:
            pts = []
            used = set()
            while len(pts) < 20:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in used:
                    used.add((x, y))
                    pts.append((x, y))
            if not has_four_collinear(pts):
                best_points = pts
                best_score = exact_max_subset_size(pts)
                break

    return [[p[0], p[1]] for p in best_points]

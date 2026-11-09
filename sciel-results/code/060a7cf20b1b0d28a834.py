import random
import math
import time
from itertools import combinations

_START_TIME = None
_DEADLINE = 145.0

def _time_left():
    if _START_TIME is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START_TIME)

def _is_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

def _build_collinearity(points):
    """Build collinearity[i][j] = bitmask of k collinear with i and j (k != i, j)."""
    n = len(points)
    coll = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if k != i and k != j and _is_collinear(points[i], points[j], points[k]):
                    coll[i][j] |= (1 << k)
                    coll[j][i] |= (1 << k)
    return coll

def _count_collinear_triples(points):
    """Count number of collinear triples (as a proxy for 3-free subset size)."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _is_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def _max_3_free_subset(points):
    """Exact branch and bound for the largest subset with no three collinear."""
    n = len(points)
    if n <= 2:
        return n
    
    coll = _build_collinearity(points)
    best = [0]
    
    # Order points by degree (number of collinear partners) descending
    # Points with more collinearities are "harder" to place in a 3-free set
    degrees = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            deg = coll[i][j].bit_count()
            degrees[i] += deg
            degrees[j] += deg
    
    order = sorted(range(n), key=lambda x: -degrees[x])
    # Remap: new index -> old index
    remap = [0] * n
    for new_idx, old_idx in enumerate(order):
        remap[new_idx] = old_idx
    
    # Rebuild collinearity in new order
    coll2 = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            oi = remap[i]
            oj = remap[j]
            mask = coll[oi][oj]
            # Remap bits
            new_mask = 0
            for k in range(n):
                if mask & (1 << k):
                    # Find new index of point k
                    for ni in range(n):
                        if remap[ni] == k:
                            new_mask |= (1 << ni)
                            break
            coll2[i][j] = new_mask
            coll2[j][i] = new_mask
    
    def backtrack(i, subset_mask, size):
        if size > best[0]:
            best[0] = size
        if i == n:
            return
        
        if size + (n - i) <= best[0]:
            return
        
        # Try include first (since we ordered by difficulty)
        can_add = True
        temp_mask = subset_mask
        while temp_mask:
            a_bit = temp_mask & (-temp_mask)
            a = a_bit.bit_length() - 1
            if coll2[a][i] & subset_mask:
                can_add = False
                break
            temp_mask ^= a_bit
        
        if can_add:
            backtrack(i + 1, subset_mask | (1 << i), size + 1)
        
        backtrack(i + 1, subset_mask, size)
    
    backtrack(0, 0, 0)
    return best[0]

def _has_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            count = 2
            for k in range(n):
                if k != i and k != j:
                    if _is_collinear(points[i], points[j], points[k]):
                        count += 1
                        if count >= 4:
                            return True
    return False

def _structured_initial():
    """Create a structured configuration with many collinear triples.
    
    Use a near-projective-plane construction: place points on a set of lines
    such that many triples are collinear but no 4 are.
    
    Strategy: Use a 5x4 grid with row offsets, then perturb to create
    additional diagonal collinearities while keeping no 4-collinear.
    """
    random.seed(42)
    
    # Start with a grid that has some structure
    points = []
    # 4 rows, 5 columns each, with row-dependent x-offset
    for row in range(4):
        y_base = (row - 1.5) * 30
        for col in range(5):
            x = (col - 2) * 25 + row * 4
            y = y_base + row * 2
            points.append((int(x), int(y)))
    
    # Ensure no duplicates
    seen = set()
    unique = []
    for p in points:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    points = unique
    while len(points) < 20:
        p = (random.randint(-100, 100), random.randint(-100, 100))
        if p not in seen:
            seen.add(p)
            points.append(p)
    
    # Fix any 4-collinear violations
    for _ in range(1000):
        if not _has_four_collinear(points):
            break
        idx = random.randint(0, 19)
        ox, oy = points[idx]
        for _ in range(50):
            nx = ox + random.randint(-8, 8)
            ny = oy + random.randint(-8, 8)
            if -100 <= nx <= 100 and -100 <= ny <= 100 and (nx, ny) not in seen:
                points[idx] = (nx, ny)
                seen.discard((ox, oy))
                seen.add((nx, ny))
                if not _has_four_collinear(points):
                    break
        else:
            points[idx] = (ox, oy)
            seen.add((ox, oy))
    
    return points

def _local_search(points, time_budget):
    """Local search with simulated annealing to minimize the 3-free subset size."""
    random.seed(12345)
    
    current_points = list(points)
    current_score = _max_3_free_subset(current_points)
    best_score = current_score
    best_points = list(current_points)
    
    start = time.time()
    temperature = 8.0
    cooling = 0.997
    min_temp = 0.05
    
    while temperature > min_temp:
        if time.time() - start > time_budget or _time_left() < 15.0:
            break
        
        idx = random.randint(0, 19)
        old_point = current_points[idx]
        
        # Try multiple candidate moves
        accepted = False
        for _ in range(20):
            # Random nearby move
            dx = random.randint(-12, 12)
            dy = random.randint(-12, 12)
            if dx == 0 and dy == 0:
                continue
            nx = old_point[0] + dx
            ny = old_point[1] + dy
            if nx < -100 or nx > 100 or ny < -100 or ny > 100:
                continue
            new_point = (nx, ny)
            if new_point in current_points:
                continue
            
            current_points[idx] = new_point
            
            if _has_four_collinear(current_points):
                current_points[idx] = old_point
                continue
            
            # Use triple count as fast proxy first
            new_triples = _count_collinear_triples(current_points)
            old_triples = _count_collinear_triples(current_points[:idx] + [old_point] + current_points[idx+1:])
            
            # If triple count improved significantly, verify with exact B&B
            if new_triples > old_triples:
                new_score = _max_3_free_subset(current_points)
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_points = list(current_points)
                    accepted = True
                    break
                elif new_score == current_score:
                    if random.random() < 0.15:
                        accepted = True
                        break
                else:
                    delta = new_score - current_score
                    if random.random() < math.exp(-delta / temperature):
                        current_score = new_score
                        accepted = True
                        break
            else:
                # Triple count didn't improve; might still be worth checking
                # Only check exact B&B occasionally to save time
                if random.random() < 0.3:
                    new_score = _max_3_free_subset(current_points)
                    if new_score < current_score:
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_points = list(current_points)
                        accepted = True
                        break
                    elif new_score == current_score:
                        if random.random() < 0.1:
                            accepted = True
                            break
        
        if not accepted:
            current_points[idx] = old_point
        
        temperature *= cooling
    
    return best_points, best_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Generate structured initial configuration
    points = _structured_initial()
    
    # Run local search with most of the time budget
    time_budget = 100.0
    best_points, best_score = _local_search(points, time_budget)
    
    # If we have time, try a few more restarts from different initial configs
    if _time_left() > 30.0:
        for seed in [777, 999, 1234]:
            if _time_left() < 15.0:
                break
            random.seed(seed)
            # Generate a different structured config
            pts2 = []
            for row in range(4):
                y_base = (row - 1.5) * random.randint(20, 40)
                for col in range(5):
                    x = (col - 2) * random.randint(15, 30) + row * random.randint(2, 8)
                    y = y_base + row * random.randint(1, 5)
                    pts2.append((int(x), int(y)))
            seen = set()
            unique2 = []
            for p in pts2:
                if p not in seen:
                    seen.add(p)
                    unique2.append(p)
            pts2 = unique2
            while len(pts2) < 20:
                p = (random.randint(-100, 100), random.randint(-100, 100))
                if p not in seen:
                    seen.add(p)
                    pts2.append(p)
            
            for _ in range(500):
                if not _has_four_collinear(pts2):
                    break
                idx = random.randint(0, 19)
                ox, oy = pts2[idx]
                for _ in range(30):
                    nx = ox + random.randint(-8, 8)
                    ny = oy + random.randint(-8, 8)
                    if -100 <= nx <= 100 and -100 <= ny <= 100 and (nx, ny) not in seen:
                        pts2[idx] = (nx, ny)
                        seen.discard((ox, oy))
                        seen.add((nx, ny))
                        if not _has_four_collinear(pts2):
                            break
                else:
                    pts2[idx] = (ox, oy)
                    seen.add((ox, oy))
            
            # Short local search from this config
            remaining = min(_time_left() - 15.0, 20.0)
            if remaining > 5.0:
                bp2, bs2 = _local_search(pts2, remaining)
                if bs2 < best_score:
                    best_score = bs2
                    best_points = bp2
    
    return [list(p) for p in best_points]

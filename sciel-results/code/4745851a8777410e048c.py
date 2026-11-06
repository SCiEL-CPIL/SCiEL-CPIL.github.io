import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Return canonical line key for two points."""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    if a == 0 and b == 0:
        return None
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g == 0:
        g = 1
    a, b, c = a // g, b // g, c // g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _count_ordinary(points):
    """Count ordinary lines."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for v in line_counts.values() if v == 1)

def _is_valid(points):
    """Check: 12 distinct points, not all collinear."""
    if len(points) != 12:
        return False
    if len(set(tuple(p) for p in points)) != 12:
        return False
    # Check not all collinear
    (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
    if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
        return True
    for i in range(3, 12):
        (x4, y4) = points[i]
        if (x2 - x1) * (y4 - y1) != (y2 - y1) * (x4 - x1):
            return True
    return False

def _try_structured_configs():
    """Try several structured configurations and return the best."""
    candidates = []
    
    # 1. 3x4 grid
    candidates.append([[i, j] for i in range(4) for j in range(3)])
    
    # 2. 4x3 grid (transposed)
    candidates.append([[i, j] for i in range(3) for j in range(4)])
    
    # 3. Hesse-like: 3x3 grid plus 3 extra points on existing lines
    h9 = [[i, j] for i in range(3) for j in range(3)]
    candidates.append(h9 + [[3, 0], [3, 1], [3, 2]])
    candidates.append(h9 + [[0, 3], [1, 3], [2, 3]])
    candidates.append(h9 + [[3, 0], [2, 3], [0, 3]])
    candidates.append(h9 + [[3, 3], [4, 0], [0, 4]])
    candidates.append(h9 + [[3, 0], [0, 3], [3, 3]])
    
    # 4. Specific known good configurations for 12 points
    # Try to find a config with 6 ordinary lines
    # One candidate: 
    # (0,0), (1,0), (2,0), (3,0)
    # (0,1), (1,1), (2,1), (3,1)
    # (0,2), (1,2), (2,2), (3,2) -> This is 3x4 grid.
    
    # Let's try a configuration based on the Pappus line structure extended.
    # Pappus: 9 points, 3 lines of 3.
    
    best_pts = None
    best_score = float('inf')
    
    for pts in candidates:
        if not _is_valid(pts):
            continue
        score = _count_ordinary(pts)
        if score < best_score:
            best_score = score
            best_pts = [list(p) for p in pts]
    
    return best_pts, best_score

def _simulated_annealing(points, deadline):
    """
    Simulated Annealing with line-aware moves.
    """
    current = [tuple(p) for p in points]
    n = len(current)
    
    # Precompute line counts for current state
    def get_line_counts(pts):
        lc = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = _line_key(pts[i], pts[j])
                if key is not None:
                    lc[key] = lc.get(key, 0) + 1
        return lc
    
    line_counts = get_line_counts(current)
    current_cost = sum(1 for v in line_counts.values() if v == 1)
    
    best_points = [list(p) for p in current]
    best_cost = current_cost
    
    # Parameters
    start_temp = 10.0
    end_temp = 0.01
    # Estimate number of steps we can afford
    # Each step involves O(n^2) work for line count update if done naively, or O(n) if incremental.
    # Incremental update is O(n) per move.
    # We have ~120s. Let's aim for ~100k steps.
    # But we need to check time.
    
    start_time = time.time()
    steps = 0
    max_steps = 200000 # Upper bound, will stop by time
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        # Cooling
        progress = steps / max_steps
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.5:
            # Random perturbation
            i = random.randint(0, n - 1)
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            new_pt = (current[i][0] + dx, current[i][1] + dy)
            if any(new_pt == current[j] for j in range(n) if j != i):
                continue
            new_points = current[:]
            new_points[i] = new_pt
        elif move_type < 0.8:
            # Line-aware move: Align point i with points j and k
            i = random.randint(0, n - 1)
            # Pick two other points
            others = [j for j in range(n) if j != i]
            j, k = random.sample(others, 2)
            
            # Find integer points on line j-k
            x1, y1 = current[j]
            x2, y2 = current[k]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            
            # Try a few positions on this line
            found = False
            for t in range(-3, 4):
                # Point at x1 + t*dx, y1 + t*dy
                # Ensure it's not j or k
                # t=0 is x1, t=1 is x2 (if dx,dy are unit steps)
                # Actually dx,dy are unit steps.
                # So t=0 is P_j, t=1 is P_k.
                # Try t = -2, -1, 2, 3
                if t in [0, 1]:
                    continue
                cand = (x1 + t * dx, y1 + t * dy)
                if any(cand == current[other] for other in range(n) if other != i):
                    continue
                new_points = current[:]
                new_points[i] = cand
                found = True
                break
            if not found:
                continue
        else:
            # Swap move? Or just skip
            continue
            
        # Calculate new cost
        # Efficient update:
        # Remove lines through i, add lines through i in new state.
        
        # This incremental update is complex to get right quickly.
        # Given time constraints, we might just recalculate if n is small?
        # n=12, pairs=66. Recalculating is fast.
        new_cost = _count_ordinary(new_points)
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = new_points
            current_cost = new_cost
            line_counts = get_line_counts(current) # Keep in sync
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = [list(p) for p in current]
                
    return best_points, best_cost

def construct():
    deadline = time.time() + 135  # Safety margin
    
    # Step 1: Try structured configurations
    best_points, best_score = _try_structured_configs()
    
    # Step 2: Simulated Annealing
    if time.time() < deadline - 10:
        sa_points, sa_score = _simulated_annealing(best_points, deadline - 10)
        if sa_score < best_score:
            best_score = sa_score
            best_points = sa_points
            
    # Step 3: Final validation
    if not _is_valid(best_points):
        # Fallback
        best_points = [[i, j] for i in range(4) for j in range(3)]
        best_score = _count_ordinary(best_points)

    return [[int(p[0]), int(p[1])] for p in best_points]

# EVOLVE-BLOCK-END

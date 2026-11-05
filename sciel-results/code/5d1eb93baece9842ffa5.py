"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from itertools import combinations
from math import gcd

_start_time = time.time()
_DEADLINE = 110  # seconds, safety margin for 120s kill

def _count_four_lines(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    count = 0
    seen = set()
    for i in range(n):
        for j in range(i+1, n):
            p1 = pts[i]
            p2 = pts[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            g = gcd(abs(dx), abs(dy))
            if g > 1:
                dx //= g
                dy //= g
            c = dy * p1[0] - dx * p1[1]
            key = (dx, dy, c)
            if key in seen:
                continue
            seen.add(key)
            diffs_x = pts[:, 0] - p1[0]
            diffs_y = pts[:, 1] - p1[1]
            on_line = (diffs_x * dy == diffs_y * dx)
            cnt = int(np.sum(on_line))
            if cnt == 4:
                count += 1
    return count

def _is_valid(points):
    """Check validity: distinct, bounds, no five collinear."""
    if len(points) != 16:
        return False
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    if len(set(tuple(p) for p in points)) != 16:
        return False
    pts = np.array(points, dtype=np.int64)
    n = 16
    for i in range(n):
        for j in range(i+1, n):
            p1 = pts[i]
            p2 = pts[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            g = gcd(abs(dx), abs(dy))
            if g > 1:
                dx //= g
                dy //= g
            diffs_x = pts[:, 0] - p1[0]
            diffs_y = pts[:, 1] - p1[1]
            on_line = (diffs_x * dy == diffs_y * dx)
            cnt = int(np.sum(on_line))
            if cnt >= 5:
                return False
    return True

def _check_no_five_with_point(points, moved_idx):
    """Check no five collinear, only checking lines that include the moved point."""
    pts = np.array(points, dtype=np.int64)
    n = 16
    for j in range(n):
        if j == moved_idx:
            continue
        p1 = pts[moved_idx]
        p2 = pts[j]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx == 0 and dy == 0:
            return False  # duplicate
        if dx < 0 or (dx == 0 and dy < 0):
            dx, dy = -dx, -dy
        g = gcd(abs(dx), abs(dy))
        if g > 1:
            dx //= g
            dy //= g
        diffs_x = pts[:, 0] - p1[0]
        diffs_y = pts[:, 1] - p1[1]
        on_line = (diffs_x * dy == diffs_y * dx)
        cnt = int(np.sum(on_line))
        if cnt >= 5:
            return False
    return True

def _try_move(points, idx, dx, dy):
    """Try moving point idx by (dx, dy). Return new points or None if invalid."""
    new_pts = [list(p) for p in points]
    new_pts[idx] = [new_pts[idx][0] + dx, new_pts[idx][1] + dy]
    x, y = new_pts[idx]
    if abs(x) > 1000 or abs(y) > 1000:
        return None
    # Check distinct
    for j in range(16):
        if j != idx and new_pts[j] == new_pts[idx]:
            return None
    # Check no five collinear involving the moved point
    if not _check_no_five_with_point(new_pts, idx):
        return None
    return new_pts

def _initial_grid():
    return [[x, y] for x in range(4) for y in range(4)]

def _simulated_annealing(points, score, deadline=None):
    """Simulated annealing to find better configurations."""
    current = [list(p) for p in points]
    current_score = score
    best = [list(p) for p in current]
    best_score = current_score
    
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    while time.time() < deadline and T > T_min:
        idx = random.randint(0, 15)
        # Larger steps to explore more
        step_size = max(3, int(10 * T / 5.0))
        dx = random.randint(-step_size, step_size)
        dy = random.randint(-step_size, step_size)
        if dx == 0 and dy == 0:
            T *= cooling_rate
            continue
            
        new_pts = _try_move(current, idx, dx, dy)
        if new_pts is None:
            T *= cooling_rate
            continue
            
        new_score = _count_four_lines(new_pts)
        
        # Accept if better, or with probability based on temperature if worse
        delta = new_score - current_score
        if delta > 0:
            current = new_pts
            current_score = new_score
            if current_score > best_score:
                best = [list(p) for p in current]
                best_score = current_score
        else:
            # Metropolis criterion
            if T > 0.01:
                p = math.exp(delta / T)
                if random.random() < p:
                    current = new_pts
                    current_score = new_score
        
        T *= cooling_rate
        
        # Occasionally do larger perturbation to escape local minima
        if random.random() < 0.02:
            for _ in range(3):
                idx2 = random.randint(0, 15)
                dx2 = random.randint(-30, 30)
                dy2 = random.randint(-30, 30)
                new_pts2 = _try_move(current, idx2, dx2, dy2)
                if new_pts2 is not None:
                    ns = _count_four_lines(new_pts2)
                    if ns > current_score:
                        current = new_pts2
                        current_score = ns
                        if current_score > best_score:
                            best = [list(p) for p in current]
                            best_score = current_score
                        break
                        
    return best, best_score

def _local_search(points, score, max_iter=2000, deadline=None):
    """Local search: try small moves and accept improvements only."""
    current = [list(p) for p in points]
    current_score = score
    best = [list(p) for p in current]
    best_score = current_score
    no_improve = 0
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        idx = random.randint(0, 15)
        improved = False
        for _ in range(15):
            dx = random.randint(-4, 4)
            dy = random.randint(-4, 4)
            if dx == 0 and dy == 0:
                continue
            new_pts = _try_move(current, idx, dx, dy)
            if new_pts is None:
                continue
            new_score = _count_four_lines(new_pts)
            if new_score > current_score:
                current = new_pts
                current_score = new_score
                if current_score > best_score:
                    best = [list(p) for p in current]
                    best_score = current_score
                no_improve = 0
                improved = True
                break
        if not improved:
            no_improve += 1
            if no_improve > 100:
                break
    return best, best_score

def construct():
    global _start_time
    random.seed(42)
    
    # Start with 4x4 grid
    best = _initial_grid()
    best_score = _count_four_lines(best)
    
    deadline = _start_time + _DEADLINE
    
    # Phase 1: Simulated annealing from grid
    if time.time() < deadline - 60:
        best, best_score = _simulated_annealing(best, best_score, deadline=deadline - 60)
    
    # Phase 2: Local search to polish
    if time.time() < deadline - 30:
        best, best_score = _local_search(best, best_score, max_iter=2000, deadline=deadline - 30)
    
    # Phase 3: More simulated annealing with different random seed
    if time.time() < deadline - 20:
        random.seed(123)
        best2, score2 = _simulated_annealing(best, best_score, deadline=deadline - 20)
        if score2 > best_score:
            best, best_score = best2, score2
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 5:
        best, best_score = _local_search(best, best_score, max_iter=1000, deadline=deadline - 5)
    
    # Try a few different initial configurations
    for trial_seed in [7, 13, 21]:
        if time.time() > deadline - 15:
            break
        random.seed(trial_seed)
        # Create a perturbed grid
        pts = [[x, y] for x in range(4) for y in range(4)]
        for _ in range(random.randint(2, 4)):
            idx = random.randint(0, 15)
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            pts[idx] = [pts[idx][0] + dx, pts[idx][1] + dy]
        if not _is_valid(pts):
            continue
        score = _count_four_lines(pts)
        if time.time() < deadline - 10:
            pts, score = _simulated_annealing(pts, score, deadline=deadline - 10)
            if time.time() < deadline - 5:
                pts, score = _local_search(pts, score, max_iter=500, deadline=deadline - 5)
        if score > best_score:
            best, best_score = pts, score
    
    return [[int(x), int(y)] for x, y in best]

# EVOLVE-BLOCK-END

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

def _get_line_signature(p1, p2):
    """Get canonical line signature (dx, dy, c)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    g = gcd(abs(dx), abs(dy))
    if g > 1:
        dx //= g
        dy //= g
    c = dy * p1[0] - dx * p1[1]
    return (dx, dy, c)

def _count_four_lines_fast(points):
    """Count lines containing exactly 4 of the given points using line signatures."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    count = 0
    seen = set()
    for i in range(n):
        for j in range(i+1, n):
            p1 = pts[i]
            p2 = pts[j]
            sig = _get_line_signature(p1, p2)
            if sig in seen:
                continue
            seen.add(sig)
            dx, dy, c = sig
            diffs_x = pts[:, 0] - p1[0]
            diffs_y = pts[:, 1] - p1[1]
            on_line = (diffs_x * dy == diffs_y * dx)
            cnt = int(np.sum(on_line))
            if cnt == 4:
                count += 1
    return count

def _is_valid_fast(points):
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
    seen = set()
    for i in range(n):
        for j in range(i+1, n):
            p1 = pts[i]
            p2 = pts[j]
            sig = _get_line_signature(p1, p2)
            if sig in seen:
                continue
            seen.add(sig)
            dx, dy, c = sig
            diffs_x = pts[:, 0] - p1[0]
            diffs_y = pts[:, 1] - p1[1]
            on_line = (diffs_x * dy == diffs_y * dx)
            cnt = int(np.sum(on_line))
            if cnt >= 5:
                return False
    return True

def _try_move_fast(points, idx, dx, dy):
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
    # Check no five collinear (fast vectorized)
    pts = np.array(new_pts, dtype=np.int64)
    n = 16
    seen = set()
    for i in range(n):
        for j in range(i+1, n):
            p1 = pts[i]
            p2 = pts[j]
            sig = _get_line_signature(p1, p2)
            if sig in seen:
                continue
            seen.add(sig)
            ddx, ddy, c = sig
            diffs_x = pts[:, 0] - p1[0]
            diffs_y = pts[:, 1] - p1[1]
            on_line = (diffs_x * ddy == diffs_y * ddx)
            cnt = int(np.sum(on_line))
            if cnt >= 5:
                return None
    return new_pts

def _initial_grid():
    return [[x, y] for x in range(4) for y in range(4)]

def _simulated_annealing(points, score, max_iter=5000, deadline=None):
    """Simulated annealing to escape local optima."""
    current = [list(p) for p in points]
    current_score = score
    best = [list(p) for p in current]
    best_score = current_score
    
    T = 5.0
    cooling = 0.999
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        idx = random.randint(0, 15)
        dx = random.randint(-10, 10)
        dy = random.randint(-10, 10)
        if dx == 0 and dy == 0:
            continue
            
        new_pts = _try_move_fast(current, idx, dx, dy)
        if new_pts is None:
            continue
            
        new_score = _count_four_lines_fast(new_pts)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current = new_pts
            current_score = new_score
            if current_score > best_score:
                best = [list(p) for p in current]
                best_score = current_score
                
        T *= cooling
        if T < 0.1:
            T = 0.1
            
    return best, best_score

def _generate_random_valid_points():
    """Generate random valid 16-point configuration."""
    for _ in range(1000):
        pts = [[random.randint(-50, 50), random.randint(-50, 50)] for _ in range(16)]
        if _is_valid_fast(pts):
            return pts
    return None

def _systematic_local_search(points, score, deadline=None):
    """Systematic local search: try all single-point moves in a neighborhood."""
    current = [list(p) for p in points]
    current_score = score
    best = [list(p) for p in current]
    best_score = current_score
    no_improve = 0
    max_no_improve = 50
    
    for _ in range(200):  # Max rounds
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Try all points and all moves in a small neighborhood
        for idx in range(16):
            if deadline is not None and time.time() > deadline:
                break
            cx, cy = current[idx]
            # Try moves in a 5x5 neighborhood
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    # Check if position is occupied
                    occupied = False
                    for j in range(16):
                        if j != idx and current[j] == [nx, ny]:
                            occupied = True
                            break
                    if occupied:
                        continue
                    # Try the move
                    new_pts = [list(p) for p in current]
                    new_pts[idx] = [nx, ny]
                    if not _is_valid_fast(new_pts):
                        continue
                    new_score = _count_four_lines_fast(new_pts)
                    if new_score > current_score:
                        current = new_pts
                        current_score = new_score
                        if current_score > best_score:
                            best = [list(p) for p in current]
                            best_score = current_score
                        improved = True
                        break
                else:
                    continue
                break
        if not improved:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
        else:
            no_improve = 0
    
    return best, best_score

def construct():
    global _start_time
    random.seed(42)
    
    # Start with 4x4 grid
    best = _initial_grid()
    best_score = _count_four_lines_fast(best)
    
    deadline = _start_time + _DEADLINE
    
    # Phase 1: Systematic local search from grid
    best, best_score = _systematic_local_search(best, best_score, deadline=deadline)
    
    # Phase 2: Simulated Annealing to escape local optima
    if time.time() < deadline - 30:
        best, best_score = _simulated_annealing(best, best_score, max_iter=5000, deadline=deadline)
    
    # Phase 3: More systematic search after SA
    if time.time() < deadline - 20:
        best, best_score = _systematic_local_search(best, best_score, deadline=deadline)
        
    # Phase 4: Try different initial configurations
    for trial in range(10):
        if time.time() > deadline - 10:
            break
        # Create a perturbed grid
        pts = [[x, y] for x in range(4) for y in range(4)]
        # Randomly perturb a few points
        for _ in range(random.randint(1, 3)):
            idx = random.randint(0, 15)
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            pts[idx] = [pts[idx][0] + dx, pts[idx][1] + dy]
        # Check validity
        if not _is_valid_fast(pts):
            continue
        score = _count_four_lines_fast(pts)
        if score > best_score:
            best, best_score = _systematic_local_search(pts, score, deadline=deadline)
            if time.time() < deadline - 15:
                best, best_score = _simulated_annealing(best, best_score, max_iter=2000, deadline=deadline)
            if time.time() < deadline - 10:
                best, best_score = _systematic_local_search(best, best_score, deadline=deadline)
    
    # Phase 5: Final polish with systematic search
    if time.time() < deadline - 5:
        best, best_score = _systematic_local_search(best, best_score, deadline=deadline)
    
    return [[int(x), int(y)] for x, y in best]

# EVOLVE-BLOCK-END

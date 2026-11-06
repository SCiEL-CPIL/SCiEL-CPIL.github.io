"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import Counter

DEADLINE = 85.0  # safety margin before 90s kill

def _score(points):
    """Compute the maximum multiplicity of a single positive doubled area."""
    n = len(points)
    if n < 3:
        return 0
    # Compute all pairwise differences
    # For points i < j, vector = points[j] - points[i]
    # For triple (i, j, k), doubled area = |det(points[j]-points[i], points[k]-points[i])|
    # We can compute this as: for each pair (i,j) with i<j, and for each k>j,
    # det = (px[j]-px[i])*(py[k]-py[i]) - (py[j]-py[i])*(px[k]-px[i])
    
    px = np.array([p[0] for p in points])
    py = np.array([p[1] for p in points])
    
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ij = px[j] - px[i]
            dy_ij = py[j] - py[i]
            if dx_ij == 0 and dy_ij == 0:
                continue
            for k in range(j + 1, n):
                dx_ik = px[k] - px[i]
                dy_ik = py[k] - py[i]
                det = dx_ij * dy_ik - dy_ij * dx_ik
                if det != 0:
                    areas.append(abs(det))
    
    if not areas:
        return 0
    counts = Counter(areas)
    return max(counts.values())

def _score_fast(points):
    """Faster scoring using numpy vectorization where possible."""
    n = len(points)
    if n < 3:
        return 0
    px = np.array([p[0] for p in points], dtype=np.int64)
    py = np.array([p[1] for p in points], dtype=np.int64)
    
    # Build all triples efficiently
    areas = []
    for i in range(n - 2):
        dx_j = px[i+1:] - px[i]
        dy_j = py[i+1:] - py[i]
        for j_idx in range(1, n - i - 1):
            j = i + j_idx
            dx_ij = px[j] - px[i]
            dy_ij = py[j] - py[i]
            if dx_ij == 0 and dy_ij == 0:
                continue
            dx_k = px[j+1:] - px[i]
            dy_k = py[j+1:] - py[i]
            dets = dx_ij * dy_k - dy_ij * dx_k
            dets = dets[dets != 0]
            if len(dets) > 0:
                areas.extend(dets.tolist())
    
    if not areas:
        return 0
    counts = Counter(areas)
    return max(counts.values())

def _score_counter(points):
    """Scoring that returns both score and the counter for analysis."""
    n = len(points)
    if n < 3:
        return 0, Counter()
    px = np.array([p[0] for p in points], dtype=np.int64)
    py = np.array([p[1] for p in points], dtype=np.int64)
    
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ij = px[j] - px[i]
            dy_ij = py[j] - py[i]
            if dx_ij == 0 and dy_ij == 0:
                continue
            for k in range(j + 1, n):
                dx_ik = px[k] - px[i]
                dy_ik = py[k] - py[i]
                det = dx_ij * dy_ik - dy_ij * dx_ik
                if det != 0:
                    areas.append(abs(det))
    
    if not areas:
        return 0, Counter()
    counts = Counter(areas)
    return max(counts.values()), counts

def _random_points(n=30, lo=-30, hi=30):
    """Generate n distinct random integer points in [lo, hi]^2."""
    pts = set()
    while len(pts) < n:
        x = random.randint(lo, hi)
        y = random.randint(lo, hi)
        pts.add((x, y))
    return list(pts)

def _neighborhood_move(points, radius=3):
    """Generate a random neighbor by moving one point slightly."""
    pts = list(points)
    idx = random.randint(0, len(pts) - 1)
    dx = random.randint(-radius, radius)
    dy = random.randint(-radius, radius)
    new_x = pts[idx][0] + dx
    new_y = pts[idx][1] + dy
    if -30 <= new_x <= 30 and -30 <= new_y <= 30:
        new_pt = (new_x, new_y)
        # Check distinctness
        rest = pts[:idx] + pts[idx+1:]
        if new_pt not in rest:
            pts[idx] = new_pt
            return pts
    return points

def _multi_move(points, num_moves=2, radius=3):
    """Generate a random neighbor by moving multiple points."""
    pts = list(points)
    for _ in range(num_moves):
        idx = random.randint(0, len(pts) - 1)
        dx = random.randint(-radius, radius)
        dy = random.randint(-radius, radius)
        new_x = pts[idx][0] + dx
        new_y = pts[idx][1] + dy
        if -30 <= new_x <= 30 and -30 <= new_y <= 30:
            new_pt = (new_x, new_y)
            rest = pts[:idx] + pts[idx+1:]
            if new_pt not in rest:
                pts[idx] = new_pt
    return pts

def _swap_points(points):
    """Generate a neighbor by swapping two points with nearby alternatives."""
    pts = list(points)
    i, j = random.sample(range(len(pts)), 2)
    dx1 = random.randint(-2, 2)
    dy1 = random.randint(-2, 2)
    dx2 = random.randint(-2, 2)
    dy2 = random.randint(-2, 2)
    new_i = (pts[i][0] + dx1, pts[i][1] + dy1)
    new_j = (pts[j][0] + dx2, pts[j][1] + dy2)
    if (-30 <= new_i[0] <= 30 and -30 <= new_i[1] <= 30 and
        -30 <= new_j[0] <= 30 and -30 <= new_j[1] <= 30):
        rest = pts[:i] + pts[i+1:j] + pts[j+1:]
        if new_i not in rest and new_j not in rest and new_i != new_j:
            pts[i] = new_i
            pts[j] = new_j
            return pts
    return points

def _local_search(points, max_iter=5000, radius=3, timeout=None):
    """Local search with multiple move types."""
    best = list(points)
    best_score = _score_fast(best)
    
    start = time.time()
    accepted = 0
    for it in range(max_iter):
        if timeout and (time.time() - start) > timeout:
            break
        # Choose move type
        r = random.random()
        if r < 0.5:
            candidate = _neighborhood_move(best, radius)
        elif r < 0.8:
            candidate = _multi_move(best, num_moves=2, radius=radius)
        else:
            candidate = _swap_points(best)
        
        if candidate == best:
            continue
        
        cand_score = _score_fast(candidate)
        if cand_score > best_score:
            best = candidate
            best_score = cand_score
            accepted += 1
    
    return best, best_score

def _structured_grid_search(points, timeout=None):
    """Try to improve by adjusting points on a structured grid."""
    best = list(points)
    best_score = _score_fast(best)
    
    start = time.time()
    for _ in range(200):
        if timeout and (time.time() - start) > timeout:
            break
        # Pick a random point and try all nearby positions
        idx = random.randint(0, len(best) - 1)
        ox, oy = best[idx]
        best_improved = False
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ox + dx, oy + dy
                if -30 <= nx <= 30 and -30 <= ny <= 30:
                    new_pt = (nx, ny)
                    rest = best[:idx] + best[idx+1:]
                    if new_pt not in rest:
                        candidate = best[:idx] + [new_pt] + best[idx+1:]
                        cand_score = _score_fast(candidate)
                        if cand_score > best_score:
                            best = candidate
                            best_score = cand_score
                            best_improved = True
                            break
            if best_improved:
                break
    
    return best, best_score

def construct():
    """Main entry point: search for 30 points maximizing equal-area multiplicity."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    random.seed(42)
    np.random.seed(42)
    
    # Initial diverse starting points
    best_points = None
    best_score = 0
    
    # Multiple restarts
    num_restarts = 8
    restarts_done = 0
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Grid-like initial
            pts = [(x, y) for x in range(6) for y in range(5)]
            # Scale to fit in [-30,30]
            pts = [(x*10-25, y*10-20) for x in range(6) for y in range(5)]
            pts = [(min(30, max(-30, x)), min(30, max(-30, y))) for x, y in pts]
            # Ensure distinct
            seen = set()
            unique_pts = []
            for p in pts:
                if p not in seen:
                    seen.add(p)
                    unique_pts.append(p)
            while len(unique_pts) < 30:
                x = random.randint(-30, 30)
                y = random.randint(-30, 30)
                if (x, y) not in seen:
                    seen.add((x, y))
                    unique_pts.append((x, y))
            pts = unique_pts[:30]
        elif restart == 1:
            # Sparse grid
            pts = []
            seen = set()
            for x in range(-30, 31, 6):
                for y in range(-30, 31, 6):
                    if (x, y) not in seen and len(pts) < 30:
                        seen.add((x, y))
                        pts.append((x, y))
            while len(pts) < 30:
                x = random.randint(-30, 30)
                y = random.randint(-30, 30)
                if (x, y) not in seen:
                    seen.add((x, y))
                    pts.append((x, y))
        else:
            # Random
            pts = _random_points(30)
        
        # Phase 1: Quick local search
        timeout1 = min(5.0, max(1.0, deadline - time.time() - 5))
        pts, score = _local_search(pts, max_iter=3000, radius=3, timeout=timeout1)
        
        if score > best_score:
            best_score = score
            best_points = pts
        
        # Phase 2: Structured improvement
        if time.time() < deadline - 5:
            timeout2 = min(3.0, max(0.5, deadline - time.time() - 5))
            pts, score = _structured_grid_search(pts, timeout=timeout2)
            if score > best_score:
                best_score = score
                best_points = pts
        
        # Phase 3: More local search
        if time.time() < deadline - 5:
            timeout3 = min(4.0, max(0.5, deadline - time.time() - 5))
            pts, score = _local_search(pts, max_iter=5000, radius=4, timeout=timeout3)
            if score > best_score:
                best_score = score
                best_points = pts
        
        restarts_done += 1
    
    # Final polish with remaining time
    if best_points and time.time() < deadline - 2:
        timeout_final = max(0.5, deadline - time.time() - 1)
        final_pts, final_score = _local_search(best_points, max_iter=10000, radius=3, timeout=timeout_final)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts
    
    if best_points is None:
        # Fallback
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_points = [(x*10-25, y*10-20) for x, y in best_points]
        best_points = [(min(30, max(-30, x)), min(30, max(-30, y))) for x, y in best_points]
        seen = set()
        unique_pts = []
        for p in best_points:
            if p not in seen:
                seen.add(p)
                unique_pts.append(p)
        while len(unique_pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in seen:
                seen.add((x, y))
                unique_pts.append((x, y))
        best_points = unique_pts[:30]
    
    # Ensure exactly 30 distinct points in range
    result = []
    seen = set()
    for p in best_points:
        if p not in seen and -30 <= p[0] <= 30 and -30 <= p[1] <= 30:
            seen.add(p)
            result.append(p)
    while len(result) < 30:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        if (x, y) not in seen:
            seen.add((x, y))
            result.append((x, y))
    
    return result[:30]
# EVOLVE-BLOCK-END

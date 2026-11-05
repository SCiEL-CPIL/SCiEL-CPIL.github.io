import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (90s limit)
    
    N = 30
    RANGE = 30
    
    def compute_all_areas(points):
        """Compute all non-zero doubled areas for all triples using vectorized ops."""
        pts = np.array(points, dtype=np.int64)
        areas = []
        for i in range(N - 2):
            for j in range(i + 1, N - 1):
                for k in range(j + 1, N):
                    d = abs((pts[j][0] - pts[i][0]) * (pts[k][1] - pts[i][1]) - 
                            (pts[k][0] - pts[i][0]) * (pts[j][1] - pts[i][1]))
                    if d > 0:
                        areas.append(d)
        return Counter(areas)
    
    def compute_max_multiplicity(areas_counter):
        return max(areas_counter.values()) if areas_counter else 0
    
    def gen_grid(w, h, offset_x=0, offset_y=0):
        pts = []
        for i in range(w):
            for j in range(h):
                pts.append((i + offset_x, j + offset_y))
        return pts
    
    # Initial configurations
    inits = []
    inits.append(gen_grid(6, 5, -2, -2))
    inits.append(gen_grid(5, 6, -2, -2))
    inits.append(gen_grid(10, 3, -4, -1))
    inits.append(gen_grid(3, 10, -1, -4))
    inits.append(gen_grid(15, 2, -6, 0))
    inits.append(gen_grid(2, 15, 0, -6))
    
    # Add random initializations
    random.seed(42)
    for _ in range(8):
        pts = []
        while len(pts) < N:
            p = (random.randint(-RANGE, RANGE), random.randint(-RANGE, RANGE))
            if p not in pts:
                pts.append(p)
        inits.append(pts)
    
    best_overall_score = 0
    best_points = inits[0]
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
        
        current_points = list(init_pts)
        if len(set(current_points)) != N:
            continue
        valid = True
        for p in current_points:
            if p[0] < -RANGE or p[0] > RANGE or p[1] < -RANGE or p[1] > RANGE:
                valid = False
                break
        if not valid:
            continue
        
        # Compute initial areas
        areas = compute_all_areas(current_points)
        curr_max = compute_max_multiplicity(areas)
        
        if curr_max > best_overall_score:
            best_overall_score = curr_max
            best_points = list(current_points)
        
        # Local search with vectorized delta computation
        no_improve = 0
        max_no_improve = 8000
        
        # Precompute indices for faster access
        other_indices = list(range(N))
        
        while time.time() < deadline - 2.0 and no_improve < max_no_improve:
            idx = random.randint(0, N - 1)
            old_pt = current_points[idx]
            
            # Generate candidate moves
            moves = []
            # Local moves (small perturbations)
            for dx in [-3, -2, -1, 0, 1, 2, 3]:
                for dy in [-3, -2, -1, 0, 1, 2, 3]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                    if -RANGE <= new_pt[0] <= RANGE and -RANGE <= new_pt[1] <= RANGE:
                        if new_pt not in current_points:
                            moves.append(new_pt)
            
            # Global moves
            for _ in range(5):
                new_pt = (random.randint(-RANGE, RANGE), random.randint(-RANGE, RANGE))
                if new_pt not in current_points and new_pt != old_pt:
                    moves.append(new_pt)
            
            if not moves:
                no_improve += 1
                continue
            
            random.shuffle(moves)
            new_pt = moves[0]
            
            # Apply move temporarily
            current_points[idx] = new_pt
            
            # Compute delta areas: remove old contributions, add new
            # Triangles involving idx: (idx, j, k) for all j < k, j,k != idx
            # Remove old areas
            for j in range(N):
                if j == idx:
                    continue
                for k in range(j + 1, N):
                    if k == idx:
                        continue
                    x1, y1 = old_pt
                    x2, y2 = current_points[j]
                    x3, y3 = current_points[k]
                    d_old = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                    if d_old > 0:
                        areas[d_old] -= 1
                        if areas[d_old] <= 0:
                            del areas[d_old]
            
            # Add new areas
            for j in range(N):
                if j == idx:
                    continue
                for k in range(j + 1, N):
                    if k == idx:
                        continue
                    x1, y1 = new_pt
                    x2, y2 = current_points[j]
                    x3, y3 = current_points[k]
                    d_new = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                    if d_new > 0:
                        areas[d_new] += 1
            
            # Evaluate
            new_max = compute_max_multiplicity(areas)
            
            # Acceptance criteria
            if new_max > curr_max:
                curr_max = new_max
                no_improve = 0
                if curr_max > best_overall_score:
                    best_overall_score = curr_max
                    best_points = list(current_points)
            elif new_max == curr_max:
                # Accept with small probability to explore
                if random.random() < 0.05:
                    no_improve += 1
                else:
                    # Revert
                    current_points[idx] = old_pt
                    # Revert areas
                    for j in range(N):
                        if j == idx:
                            continue
                        for k in range(j + 1, N):
                            if k == idx:
                                continue
                            x1, y1 = new_pt
                            x2, y2 = current_points[j]
                            x3, y3 = current_points[k]
                            d_new = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                            if d_new > 0:
                                areas[d_new] -= 1
                                if areas[d_new] <= 0:
                                    del areas[d_new]
                            x1, y1 = old_pt
                            x2, y2 = current_points[j]
                            x3, y3 = current_points[k]
                            d_old = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                            if d_old > 0:
                                areas[d_old] += 1
                    no_improve += 1
            else:
                # Revert
                current_points[idx] = old_pt
                for j in range(N):
                    if j == idx:
                        continue
                    for k in range(j + 1, N):
                        if k == idx:
                            continue
                        x1, y1 = new_pt
                        x2, y2 = current_points[j]
                        x3, y3 = current_points[k]
                        d_new = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                        if d_new > 0:
                            areas[d_new] -= 1
                            if areas[d_new] <= 0:
                                del areas[d_new]
                        x1, y1 = old_pt
                        x2, y2 = current_points[j]
                        x3, y3 = current_points[k]
                        d_old = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                        if d_old > 0:
                            areas[d_old] += 1
                no_improve += 1
    
    # Final validation
    if len(best_points) != N or len(set(best_points)) != N:
        return gen_grid(6, 5, -2, -2)
    for p in best_points:
        if p[0] < -30 or p[0] > 30 or p[1] < -30 or p[1] > 30:
            return gen_grid(6, 5, -2, -2)
    
    return best_points

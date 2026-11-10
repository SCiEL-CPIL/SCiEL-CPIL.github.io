import time
import random
import math
from collections import defaultdict
from itertools import combinations

def _normalize_line(A, B, C):
    if A == 0 and B == 0:
        return (0, 0, 1) if C != 0 else (0, 0, 0)
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
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
    return (A, B, C)

def _line_from_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    return _normalize_line(A, B, C)

def _count_4_lines(points):
    """
    Returns (count_4, is_valid, line_map)
    line_map: dict mapping line_key -> set of point_indices on that line
    """
    n = len(points)
    line_map = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_from_points(points[i], points[j])
            line_map[key].add(i)
            line_map[key].add(j)
            
    count_4 = 0
    is_valid = True
    for key, indices in line_map.items():
        cnt = len(indices)
        if cnt == 4:
            count_4 += 1
        elif cnt >= 5:
            is_valid = False
    return count_4, is_valid, line_map

def _local_improve(selection, pool, deadline):
    """Try all single-point swaps to improve the score."""
    n = len(pool)
    current_points = [pool[i] for i in selection]
    current_score, current_valid, _ = _count_4_lines(current_points)
    
    if not current_valid:
        return selection, current_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_swap = None
        best_swap_score = current_score
        
        for out_pos in range(16):
            if time.time() > deadline:
                break
            out_pool_idx = selection[out_pos]
            
            for in_pool_idx in range(n):
                if in_pool_idx == out_pool_idx:
                    continue
                if in_pool_idx in selection:
                    continue
                
                new_selection = selection[:]
                new_selection[out_pos] = in_pool_idx
                
                if len(set(new_selection)) != 16:
                    continue
                
                new_points = [pool[i] for i in new_selection]
                new_score, new_valid, _ = _count_4_lines(new_points)
                
                if not new_valid:
                    continue
                
                if new_score > best_swap_score:
                    best_swap_score = new_score
                    best_swap = (out_pos, in_pool_idx)
        
        if best_swap is not None:
            out_pos, in_pool_idx = best_swap
            selection[out_pos] = in_pool_idx
            current_score = best_swap_score
            improved = True
    
    return selection, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Larger pool: 21x21 grid from -10 to 10
    pool = []
    for x in range(-10, 11):
        for y in range(-10, 11):
            pool.append([x, y])
    
    pool_size = len(pool)
    
    random.seed(42)
    
    best_points = None
    best_score = 0
    
    # Multiple restarts with different strategies
    restart_count = 0
    max_restarts = 8
    
    while time.time() < deadline and restart_count < max_restarts:
        restart_count += 1
        
        # Different initial configurations for each restart
        if restart_count == 1:
            # Start with 4x4 grid
            initial_indices = []
            for x in range(0, 4):
                for y in range(0, 4):
                    idx = (x + 10) * 21 + (y + 10)
                    initial_indices.append(idx)
        elif restart_count == 2:
            # Start with a different 4x4 grid offset
            initial_indices = []
            for x in range(-3, 1):
                for y in range(-3, 1):
                    idx = (x + 10) * 21 + (y + 10)
                    initial_indices.append(idx)
        elif restart_count == 3:
            # Random 16 points
            initial_indices = random.sample(range(pool_size), 16)
        elif restart_count == 4:
            # Points on a 5x5 grid minus 9 random points, plus 1 random
            initial_indices = []
            for x in range(-2, 3):
                for y in range(-2, 3):
                    initial_indices.append((x + 10) * 21 + (y + 10))
            # Remove 9 random points and add 1 random from outside
            random.shuffle(initial_indices)
            initial_indices = initial_indices[:16]
        elif restart_count == 5:
            # Diagonal-ish pattern
            initial_indices = []
            for i in range(4):
                for j in range(4):
                    x = i - 2
                    y = j - 2
                    idx = (x + 10) * 21 + (y + 10)
                    initial_indices.append(idx)
        elif restart_count == 6:
            # Random but biased toward center
            initial_indices = []
            attempts = 0
            while len(initial_indices) < 16 and attempts < 1000:
                x = random.randint(-5, 5)
                y = random.randint(-5, 5)
                idx = (x + 10) * 21 + (y + 10)
                if idx not in initial_indices:
                    initial_indices.append(idx)
                attempts += 1
            if len(initial_indices) < 16:
                remaining = random.sample([i for i in range(pool_size) if i not in initial_indices], 16 - len(initial_indices))
                initial_indices.extend(remaining)
        elif restart_count == 7:
            # Two 3x3 grids
            initial_indices = []
            for x in range(-3, 0):
                for y in range(-3, 0):
                    initial_indices.append((x + 10) * 21 + (y + 10))
            for x in range(1, 4):
                for y in range(1, 4):
                    initial_indices.append((x + 10) * 21 + (y + 10))
        else:
            # Pure random
            initial_indices = random.sample(range(pool_size), 16)
        
        current_selection = initial_indices[:]
        current_points = [pool[i] for i in current_selection]
        current_score, current_valid, _ = _count_4_lines(current_points)
        
        if not current_valid:
            # Try to fix by local improvement
            current_selection, current_score = _local_improve(current_selection, pool, deadline)
            if current_score <= best_score:
                continue
        
        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        
        steps = 0
        stuck_counter = 0
        while time.time() < deadline:
            steps += 1
            
            # Try multiple swap candidates and pick the best
            best_delta = -999
            best_out_pos = -1
            best_in_pool_idx = -1
            
            num_candidates = 5
            for _ in range(num_candidates):
                out_idx_pos = random.randint(0, 15)
                out_pool_idx = current_selection[out_idx_pos]
                
                in_pool_idx = random.randint(0, pool_size - 1)
                while in_pool_idx in current_selection:
                    in_pool_idx = random.randint(0, pool_size - 1)
                
                new_selection = current_selection[:]
                new_selection[out_idx_pos] = in_pool_idx
                
                if len(set(new_selection)) != 16:
                    continue
                
                new_points = [pool[i] for i in new_selection]
                new_score, new_valid, _ = _count_4_lines(new_points)
                
                if not new_valid:
                    continue
                
                delta = new_score - current_score
                
                if delta > best_delta:
                    best_delta = delta
                    best_out_pos = out_idx_pos
                    best_in_pool_idx = in_pool_idx
            
            if best_out_pos == -1:
                # No valid swap found, try local improvement
                current_selection, current_score = _local_improve(current_selection, pool, deadline)
                if current_score > best_score:
                    best_score = current_score
                    best_points = [pool[i][:] for i in current_selection]
                stuck_counter += 1
                if stuck_counter > 100:
                    break
                continue
            
            stuck_counter = 0
            new_selection = current_selection[:]
            new_selection[best_out_pos] = best_in_pool_idx
            new_points = [pool[i] for i in new_selection]
            new_score, new_valid, _ = _count_4_lines(new_points)
            
            if not new_valid:
                continue
            
            delta = new_score - current_score
            
            if delta > 0:
                current_selection = new_selection
                current_points = new_points
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [pool[i][:] for i in new_selection]
            else:
                if random.random() < math.exp(delta / T):
                    current_selection = new_selection
                    current_points = new_points
                    current_score = new_score
            
            T *= cooling
            if T < T_min:
                T = T_min
            
            # Reheat periodically
            if steps % 300 == 0:
                T = max(T, 0.5)
        
        # Final local improvement on the best found in this restart
        current_selection, final_score = _local_improve(current_selection, pool, deadline)
        if final_score > best_score:
            best_score = final_score
            best_points = [pool[i][:] for i in current_selection]
        
        if time.time() > deadline - 20:
            break
    
    if best_points is None:
        # Fallback: return 4x4 grid
        result = []
        for x in range(0, 4):
            for y in range(0, 4):
                result.append([x, y])
        return result
    
    return best_points

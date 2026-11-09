import numpy as np
import time
import random
import heapq

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    total_points = q * q
    
    # Field ops
    def add_e(a, b):
        a_a, a_b = a // 7, a % 7
        b_a, b_b = b // 7, b % 7
        return ((a_a + b_a) % p) * 7 + ((a_b + b_b) % p)
        
    def mul_e(a, b):
        a_a, a_b = a // 7, a % 7
        b_a, b_b = b // 7, b % 7
        return ((a_a * b_a + a_b * b_b * 3) % p) * 7 + ((a_a * b_b + a_b * b_a) % p)

    # Precompute lines
    lines = [[None for _ in range(q)] for _ in range(q + 1)]
    point_line_param = [[0] * (q + 1) for _ in range(total_points)]
    
    for c in range(q):
        pts = [c * q + y for y in range(q)]
        lines[49][c] = pts
        for y in range(q):
            point_line_param[c * q + y][49] = c
            
    for m in range(q):
        for c in range(q):
            pts = []
            for x in range(q):
                y = add_e(mul_e(x, m), c)
                pts.append(x * q + y)
                point_line_param[x * q + y][m] = c
            lines[m][c] = pts

    # State
    C = np.zeros(total_points, dtype=bool)
    line_counts = np.zeros((q + 1, q), dtype=int)
    valid_count = np.full(total_points, q + 1, dtype=int)
    # single_point_on_line[d][c] = point index if count==1, else -1
    single_point = np.full((q + 1, q), -1, dtype=int)
    
    best_C = C.copy()
    best_size = 0
    
    deadline = time.time() + 590
    
    # Greedy with priority queue
    # Cost of adding p: sum over d of (number of points that lose d)
    # If count==0: 48 points lose d. Cost += 48.
    # If count==1: 1 point loses d. Cost += 1.
    # If count>=2: 0 points lose d. Cost += 0.
    
    # We want to add points with low cost.
    # However, adding a point changes the cost of others.
    # So we need to recompute costs or update them.
    # Updating costs is complex.
    # Let's just use a random greedy with multiple restarts, but smarter selection.
    # Pick a random subset of candidate points, compute their current cost, pick min.
    
    while time.time() < deadline:
        # Reset
        C[:] = False
        line_counts[:] = 0
        valid_count[:] = q + 1
        single_point[:] = -1
        
        added_any = True
        while added_any and time.time() < deadline:
            added_any = False
            # Candidate points: those not in C
            candidates = np.where(~C)[0]
            if len(candidates) == 0:
                break
            
            # Randomly sample a subset to evaluate
            sample_size = min(100, len(candidates))
            sample = random.sample(list(candidates), sample_size)
            
            best_p = -1
            min_cost = float('inf')
            
            for p in sample:
                cost = 0
                valid = True
                for d in range(q + 1):
                    c = point_line_param[p][d]
                    count = line_counts[d][c]
                    if count == 0:
                        cost += 48
                        # Check if any point on line has valid_count == 1
                        # If so, adding p makes it 0. Invalid.
                        for y in lines[d][c]:
                            if y != p and valid_count[y] == 1:
                                valid = False
                                break
                        if not valid: break
                    elif count == 1:
                        cost += 1
                        q_pt = single_point[d][c]
                        if valid_count[q_pt] == 1:
                            valid = False
                            break
                    else:
                        pass
                        
                if valid and cost < min_cost:
                    min_cost = cost
                    best_p = p
                    
            if best_p == -1:
                # No valid point in sample? Try full scan if sample small?
                # Or just break
                break
                
            # Commit best_p
            p = best_p
            C[p] = True
            for d in range(q + 1):
                c = point_line_param[p][d]
                count = line_counts[d][c]
                
                if count == 0:
                    valid_count[p] += 1
                    for y in lines[d][c]:
                        if y != p:
                            valid_count[y] -= 1
                elif count == 1:
                    q_pt = single_point[d][c]
                    valid_count[q_pt] -= 1
                else:
                    pass
                    
                line_counts[d][c] += 1
                if line_counts[d][c] == 1:
                    single_point[d][c] = p
                elif line_counts[d][c] == 2:
                    single_point[d][c] = -1
                    
            added_any = True
            current_size = np.sum(C)
            if current_size > best_size:
                best_size = current_size
                best_C = C.copy()
                
    N = ~best_C
    indices = np.where(N)[0]
    points = []
    for idx in indices:
        x = idx // q
        y = idx % q
        xa, xb = x // 7, x % 7
        ya, yb = y // 7, y % 7
        points.append([[xa, xb], [ya, yb]])
        
    return np.array(points, dtype=np.int64)

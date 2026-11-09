import time
import numpy as np
from collections import defaultdict
import random

def construct():
    t0 = time.time()
    DEADLINE = 55.0  # Safety margin before 60s limit
    
    def elapsed():
        return time.time() - t0

    def compute_max_area_count(pts):
        """
        Compute the maximum multiplicity of any non-zero doubled area.
        pts: list of 30 (x, y) tuples.
        """
        n = len(pts)
        if n < 3:
            return 0
        
        P = np.array(pts, dtype=np.int64)
        area_counts = defaultdict(int)
        
        # Loop over i, vectorize j, k
        # i from 0 to n-3
        for i in range(n - 2):
            pi = P[i]
            # Points from i+1 to end
            sub = P[i+1:]
            m = len(sub)
            if m < 2:
                continue
            
            # Vectors from pi to all points in sub
            dx = sub[:, 0] - pi[0]
            dy = sub[:, 1] - pi[1]
            
            # We need pairs (u, v) with u < v in range(m)
            # u from 0 to m-2, v from u+1 to m-1
            for u in range(m - 1):
                du = dx[u]
                dv = dy[u]
                # v indices: u+1 to m-1
                dv_vec = dy[u+1:]
                du_vec = dx[u+1:]
                
                # Determinant = | du * dv_vec - dv * du_vec |
                dets = np.abs(du * dv_vec - dv * du_vec)
                
                # Filter zeros
                non_zero = dets[dets > 0]
                if len(non_zero) > 0:
                    # Count occurrences
                    vals, counts = np.unique(non_zero, return_counts=True)
                    for val, cnt in zip(vals, counts):
                        area_counts[int(val)] += int(cnt)
                        
        if not area_counts:
            return 0
        return max(area_counts.values())

    def score_of(pts):
        return compute_max_area_count(pts)

    # 1. Initialize with strong structured candidates
    
    # Candidate 1: Two lines, 15 points each
    # y=0, x=0..14
    # y=1, x=0..14
    cand1 = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    
    # Candidate 2: Grid 6x5
    cand2 = [(x, y) for x in range(6) for y in range(5)]
    
    # Candidate 3: Three lines, 10 points each
    # y=0, x=0..9
    # y=1, x=0..9
    # y=2, x=0..9
    cand3 = [(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + [(x, 2) for x in range(10)]
    
    # Candidate 4: Random
    random.seed(42)
    cand4 = []
    used = set()
    while len(cand4) < 30:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        if (x, y) not in used:
            used.add((x, y))
            cand4.append((x, y))
            
    candidates = [cand1, cand2, cand3, cand4]
    
    best_points = None
    best_score = 0
    
    for cand in candidates:
        s = score_of(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
            
    # 2. Local Search
    current_points = list(best_points)
    current_score = best_score
    
    # To allow exploration, we might accept equal scores or slightly worse scores
    # But strictly improving is safer for "maximize largest multiplicity".
    # Let's stick to strictly improving, but use random restarts if stuck.
    
    random.seed(42)
    step_size = 2
    
    # Track if we've made progress recently
    last_improvement_iter = 0
    iter_count = 0
    
    while elapsed() < DEADLINE:
        iter_count += 1
        
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        # Generate a new position
        # Mix of small steps and occasional larger jumps
        if random.random() < 0.05:
            # Large jump
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
        else:
            # Small step
            new_x = old_x + random.randint(-step_size, step_size)
            new_y = old_y + random.randint(-step_size, step_size)
            
        # Ensure bounds
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
            
        # Check distinctness
        is_valid = True
        for j in range(30):
            if j == idx:
                continue
            if current_points[j] == (new_x, new_y):
                is_valid = False
                break
        
        if not is_valid:
            continue
            
        # Try the move
        current_points[idx] = (new_x, new_y)
        
        new_score = score_of(current_points)
        
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = list(current_points)
                last_improvement_iter = iter_count
                # Decay step size on improvement to refine
                if step_size > 1 and random.random() < 0.5:
                    step_size = max(1, step_size - 1)
        else:
            # Revert
            current_points[idx] = (old_x, old_y)
            
        # If stuck for a long time, restart from best or random
        if iter_count - last_improvement_iter > 5000:
            # Restart from best points
            current_points = list(best_points)
            current_score = best_score
            last_improvement_iter = iter_count
            # Increase step size to explore more
            step_size = random.randint(3, 10)
            
        # Occasionally adjust step size
        if random.random() < 0.01:
            step_size = random.randint(1, 5)

    return best_points

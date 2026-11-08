import math
import random
import time
import numpy as np

def score_set(U):
    if not U: return 0
    max_u = max(U)
    if max_u == 0: return 1.0
    
    U_arr = np.array(U)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    
    size_sum = len(np.unique(sums))
    size_diff = len(np.unique(diffs))
    
    if size_sum == 0: return 1.0
    
    ratio = size_diff / size_sum
    denom = math.log(2 * max_u + 1)
    if denom == 0: return 1.0
    
    return 1 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin (evaluator kills at 150s)
    
    best_score = 0
    best_set = [0, 1, 3]
    
    # Initialize with diverse candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 2, 5],
        [0, 1, 4, 6],
        [0, 1, 2, 4, 7],
        [0, 1, 3, 4, 7],
        [0, 1, 2, 5, 6],
        [0, 1, 3, 9],
        [0, 1, 4, 10],
        [0, 1, 2, 4, 8, 12],
        [0, 1, 3, 7, 12],
        [0, 1, 2, 5, 10],
        [0, 1, 3, 6, 10, 15],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 3, 6, 12, 24],
        [0, 1, 2, 3, 5, 8, 13],
        [0, 1, 4, 9, 16],
        [0, 1, 2, 5, 10, 17],
        [0, 1, 3, 7, 14, 21],
    ]
    
    for c in candidates:
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
    
    # Local Search with expanded neighborhood
    def get_neighbors(U):
        neighbors = []
        n = len(U)
        max_val = max(U) if U else 10
        seen = set()
        
        # 1. Move an element (larger deltas)
        for i in range(n):
            for delta in [-3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]:
                new_val = U[i] + delta
                if new_val >= 0:
                    new_U = U[:]
                    new_U[i] = new_val
                    if new_val not in new_U[:i] + new_U[i+1:]:
                        key = tuple(sorted(new_U))
                        if key not in seen:
                            seen.add(key)
                            neighbors.append(sorted(new_U))
        
        # 2. Add an element
        if n < 15:
            # Try values near existing elements and some random larger values
            for val in list(set(range(0, max_val + 10) - set(U))) + list(set(random.sample(range(max_val, max_val + 30), min(10, max(1, max_val + 30 - max_val)))) if max_val + 30 > max_val else []):
                if val not in U and val >= 0:
                    new_U = U + [val]
                    key = tuple(sorted(new_U))
                    if key not in seen:
                        seen.add(key)
                        neighbors.append(sorted(new_U))
        
        # 3. Remove an element
        if n > 2:
            for i in range(n):
                if U[i] == 0: continue
                new_U = U[:i] + U[i+1:]
                key = tuple(sorted(new_U))
                if key not in seen:
                    seen.add(key)
                    neighbors.append(sorted(new_U))
        
        return neighbors
    
    # Simulated Annealing with periodic restarts
    current_set = best_set[:]
    current_score = best_score
    T = 0.15
    cooling = 0.9995
    
    steps = 0
    no_improve_steps = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        neighbors = get_neighbors(current_set)
        if not neighbors:
            break
        next_set = random.choice(neighbors)
        next_score = score_set(next_set)
        
        delta = next_score - current_score
        if delta > 0 or (T > 0.001 and random.random() < math.exp(delta / T)):
            current_set = next_set
            current_score = next_score
            if current_score > best_score:
                best_score = current_score
                best_set = current_set[:]
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1
        
        T *= cooling
        steps += 1
        
        # Restart if stuck
        if no_improve_steps > max_no_improve or steps % 300 == 0:
             # Generate diverse random sets
             for _ in range(5):
                 n = random.randint(3, 12)
                 # Mix of small and large values
                 if random.random() < 0.5:
                     vals = random.sample(range(0, 25), min(n, 25))
                 else:
                     vals = random.sample(range(0, 60), min(n, 60))
                 if 0 not in vals:
                     vals[0] = 0
                 r_set = sorted(vals)
                 r_score = score_set(r_set)
                 if r_score > current_score:
                     current_set = r_set
                     current_score = r_score
                     if r_score > best_score:
                         best_score = r_score
                         best_set = r_set[:]
                         no_improve_steps = 0
             # Reset temperature occasionally
             if steps % 600 == 0:
                 T = max(0.01, T * 2)
             no_improve_steps = 0
    
    # Final polish pass: tighter local search around best
    if time.time() < deadline - 10:
        current_set = best_set[:]
        current_score = best_score
        T = 0.01
        cooling = 0.999
        polish_steps = 0
        polish_no_improve = 0
        polish_max_no_improve = 50
        
        while time.time() < deadline - 5 and polish_no_improve < polish_max_no_improve:
            neighbors = get_neighbors(current_set)
            if not neighbors:
                break
            next_set = random.choice(neighbors)
            next_score = score_set(next_set)
            
            delta = next_score - current_score
            if delta > 0 or (T > 0.0001 and random.random() < math.exp(delta / T)):
                current_set = next_set
                current_score = next_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set[:]
                    polish_no_improve = 0
                else:
                    polish_no_improve += 1
            else:
                polish_no_improve += 1
            
            T *= cooling
            polish_steps += 1
    
    return best_set

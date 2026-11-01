import math
import time
import random
import numpy as np

def compute_score(u):
    """Compute the score for a given set U."""
    if not u:
        return 0.0
    u_arr = np.array(u, dtype=np.int64)
    max_u = int(u_arr.max())
    m = 2 * max_u + 1
    if m <= 1:
        return 1.0
    
    # Sumset and Diffset
    # For small N, direct calculation is fast
    sumset = set()
    diffset = set()
    for i in range(len(u_arr)):
        for j in range(len(u_arr)):
            sumset.add(u_arr[i] + u_arr[j])
            diffset.add(u_arr[i] - u_arr[j])
            
    r = len(diffset) / len(sumset)
    return 1 + math.log(r) / math.log(m)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_set = [0]
    best_score = 0.0
    
    # Known optimal or near-optimal Golomb rulers
    # N, Length, Marks
    golomb_rulers = [
        [0, 1],
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 4, 10, 12, 17, 25, 34],
        [0, 1, 4, 10, 12, 17, 25, 34, 44],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 68],
    ]
    
    # Evaluate initial candidates
    for u in golomb_rulers:
        if time.time() > deadline:
            break
        score = compute_score(u)
        if score > best_score:
            best_score = score
            best_set = list(u)
            
    # Local search on the best set
    current_set = list(best_set)
    current_score = best_score
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        improved = False
        no_improve += 1
        
        # Try adding elements
        if len(current_set) < 15:
            max_cur = max(current_set)
            # Try a range of new values
            for new_val in range(1, max_cur * 2 + 10):
                if new_val in current_set:
                    continue
                new_set = sorted(current_set + [new_val])
                score = compute_score(new_set)
                if score > current_score + 1e-10:
                    current_set = new_set
                    current_score = score
                    if score > best_score:
                        best_score = score
                        best_set = list(current_set)
                    improved = True
                    no_improve = 0
                    break
        
        # Try removing elements
        if len(current_set) > 3 and not improved:
            for i in range(len(current_set) - 1, 0, -1):
                new_set = current_set[:i] + current_set[i+1:]
                score = compute_score(new_set)
                if score > current_score + 1e-10:
                    current_set = new_set
                    current_score = score
                    if score > best_score:
                        best_score = score
                        best_set = list(current_set)
                    improved = True
                    no_improve = 0
                    break
        
        # Try shifting elements
        if not improved:
            for i in range(1, len(current_set)):
                orig = current_set[i]
                for delta in [-2, -1, 1, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 8]:
                    new_val = orig + delta
                    if new_val < 1 or new_val in current_set:
                        continue
                    new_set = sorted([x if j != i else new_val for j, x in enumerate(current_set)])
                    score = compute_score(new_set)
                    if score > current_score + 1e-10:
                        current_set = new_set
                        current_score = score
                        if score > best_score:
                            best_score = score
                            best_set = list(current_set)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
                    
    return best_set

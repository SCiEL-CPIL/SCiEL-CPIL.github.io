import numpy as np
import math
import random

def calculate_score(U):
    if not U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    
    U_arr = np.array(U)
    sums = np.add.outer(U_arr, U_arr)
    diffs = np.subtract.outer(U_arr, U_arr)
    
    size_sum = len(np.unique(sums))
    size_diff = len(np.unique(diffs))
    
    if size_sum == 0:
        return 0
        
    ratio = size_diff / size_sum
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0
        
    val = 1 + (math.log(ratio) / denom)
    return val

def construct():
    best_score = 0
    best_set = [0, 1, 3]
    
    # Search geometric progressions
    for b in range(2, 20):
        for n in range(3, 30):
            U = [0] + [b**i for i in range(1, n)]
            # Check if max U is reasonable
            if U[-1] > 10**9:
                break
            score = calculate_score(U)
            if score > best_score:
                best_score = score
                best_set = U
                
    # Search random sets
    for _ in range(500):
        n = random.randint(3, 50)
        max_val = random.randint(10, 1000)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        score = calculate_score(U)
        if score > best_score:
            best_score = score
            best_set = U
            
    # Search "greedy" sets: pick next element to maximize score
    # This is complex, so we skip for now.
    
    return best_set

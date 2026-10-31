import itertools
import math

def calculate_score(U):
    if not U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
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
    best_set = [0, 1, 4, 6]
    
    # Search for sets of size 4 to 7
    # Golomb rulers/Sidon sets are candidates
    # We search up to 50 to find small max_u sets
    max_val = 50
    
    for n in range(4, 8):
        # Choose n-1 elements from 1..max_val, add 0
        for combo in itertools.combinations(range(1, max_val + 1), n - 1):
            U = [0] + list(combo)
            score = calculate_score(U)
            if score > best_score:
                best_score = score
                best_set = U
                
    return best_set

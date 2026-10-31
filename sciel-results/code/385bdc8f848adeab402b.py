import math
import time

def construct():
    """
    Constructs a finite set U of nonnegative integers containing 0
    to maximize the sumset/difference-set exponent lower bound.
    """
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 29],
        [0, 1, 4, 10, 18, 23, 29, 36] # Approximate for n=8
    ]
    
    best_score = 0
    best_set = [0]
    
    for U in candidates:
        if 0 not in U:
            continue
            
        # Calculate |U+U|
        sums = set()
        n = len(U)
        for i in range(n):
            for j in range(i, n):
                sums.add(U[i] + U[j])
        
        # Calculate |U-U|
        diffs = set()
        for i in range(n):
            for j in range(n):
                diffs.add(U[i] - U[j])
        
        max_u = max(U)
        
        # Avoid division by zero or log issues
        if len(sums) == 0 or max_u < 0:
            continue
            
        ratio = len(diffs) / len(sums)
        if ratio <= 1:
            score = 1.0
        else:
            score = 1 + math.log(ratio) / math.log(2 * max_u + 1)
            
        if score > best_score:
            best_score = score
            best_set = U
            
    return best_set

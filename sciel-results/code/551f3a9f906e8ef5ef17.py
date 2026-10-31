import math
import time

def construct():
    # Known optimal Golomb rulers (smallest span for n marks)
    # n: [marks]
    candidates = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 25],
        8: [0, 1, 4, 10, 12, 17, 25, 29],
        9: [0, 1, 4, 10, 12, 17, 25, 29, 34],
        10: [0, 1, 4, 10, 12, 17, 25, 29, 34, 41],
        11: [0, 1, 4, 10, 12, 17, 25, 29, 34, 41, 45], # Approx, need to verify
        12: [0, 1, 4, 10, 12, 17, 25, 29, 34, 41, 45, 51] # Approx
    }
    
    def compute_score(U):
        if not U or len(U) < 2:
            return 0.0
        if 0 not in U:
            return 0.0
        max_u = max(U)
        if max_u == 0:
            return 0.0
        
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
                
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return 0.0
            
        base = 2 * max_u + 1
        if base <= 1:
            return 0.0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
            
        score = 1 + math.log(ratio) / math.log(base)
        return score

    best_score = 0.0
    best_set = [0]
    
    for n, u in candidates.items():
        s = compute_score(u)
        if s > best_score:
            best_score = s
            best_set = list(u)
            
    return best_set

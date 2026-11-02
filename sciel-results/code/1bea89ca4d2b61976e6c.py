import math

def score_of(U):
    if not U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num_sums = len(sums)
    num_diffs = len(diffs)
    if num_sums == 0:
        return 0.0
    ratio = num_diffs / num_sums
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0.0
    return 1 + math.log(ratio) / denom

def construct():
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 25] # k=7, length 25
    ]
    
    best_set = candidates[0]
    best_score = score_of(best_set)
    
    for U in candidates:
        s = score_of(U)
        if s > best_score:
            best_score = s
            best_set = U
            
    return best_set

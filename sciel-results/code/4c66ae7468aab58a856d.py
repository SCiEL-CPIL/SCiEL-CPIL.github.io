import math
import itertools

def _score(U):
    if not U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sum_set = set()
    diff_set = set()
    for i in U:
        for j in U:
            sum_set.add(i + j)
            diff_set.add(i - j)
    len_sum = len(sum_set)
    len_diff = len(diff_set)
    if len_sum == 0:
        return 0.0
    ratio = len_diff / len_sum
    if ratio <= 1:
        return 1.0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1.0 + math.log(ratio) / denom

def construct():
    # Test specific families
    best_score = 1.0
    best_set = [0, 1, 3]
    
    # Family 1: 2^k - 1
    for k in range(1, 10):
        U = [0]
        for i in range(1, k+1):
            U.append(2**i - 1)
        s = _score(U)
        if s > best_score:
            best_score = s
            best_set = U
            
    # Family 2: 3^k
    for k in range(1, 10):
        U = [0]
        for i in range(1, k+1):
            U.append(3**i)
        s = _score(U)
        if s > best_score:
            best_score = s
            best_set = U
            
    # Family 3: Small random search around [0, 1, 3, 7]
    # Just a few fixed candidates
    candidates = [
        [0, 1, 3, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 7, 15, 31],
        [0, 2, 3, 5],
        [0, 1, 2, 5],
        [0, 1, 4, 6],
    ]
    for U in candidates:
        s = _score(U)
        if s > best_score:
            best_score = s
            best_set = U
            
    return best_set

import math

def calc_score(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 0
    return 1 + math.log(num) / den

def construct():
    # Candidate 1: Seed
    u1 = [0, 1, 3]
    # Candidate 2: 2^k - 1 for k=4
    u2 = [0, 1, 3, 7]
    # Candidate 3: 2^k - 1 for k=5
    u3 = [0, 1, 3, 7, 15]
    # Candidate 4: 2^k - 1 for k=6
    u4 = [0, 1, 3, 7, 15, 31]
    
    best = u1
    best_score = calc_score(u1)
    
    for u in [u2, u3, u4]:
        s = calc_score(u)
        if s > best_score:
            best_score = s
            best = u
            
    return best

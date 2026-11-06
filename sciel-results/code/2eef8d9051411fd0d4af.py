import time
from itertools import combinations

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Inverses of 1..10 mod 101
    invs = [0] * 11
    for i in range(1, 11):
        invs[i] = pow(i, -1, 101)
    
    # For each residue, find the minimum subset size needed
    # Since subsets can be reused across rows, we independently minimize each row
    min_size = [11] * 101  # 11 = not found
    min_combo = [None] * 101
    
    for k in range(1, 11):
        if time.time() > deadline:
            break
        for combo in combinations(range(1, 11), k):
            s = 0
            for x in combo:
                s += invs[x]
            s %= 101
            if k < min_size[s]:
                min_size[s] = k
                min_combo[s] = list(combo)
    
    # Construct result
    result = []
    for r in range(101):
        if min_combo[r] is not None:
            result.append(min_combo[r])
        else:
            # Fallback: should not happen, but find any valid subset
            found = False
            for mask in range(1, 1 << 10):
                terms = [x for x in range(1, 11) if (mask >> (x-1)) & 1]
                s = sum(invs[x] for x in terms) % 101
                if s == r:
                    result.append(terms)
                    found = True
                    break
            if not found:
                result.append([1])
    
    return result

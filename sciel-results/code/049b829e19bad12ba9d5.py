import itertools
import math
import time

def score_set(U):
    if not U:
        return 0
    if 0 not in U:
        return 0
    U = sorted(set(U))
    if U[0] != 0:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sums = set()
    diffs = set()
    for i in range(len(U)):
        for j in range(len(U)):
            sums.add(U[i] + U[j])
            diffs.add(U[i] - U[j])
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0
    val = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    # Search for small sets
    # Size 4
    for x in range(1, 20):
        for y in range(x+1, 20):
            for z in range(y+1, 20):
                U = [0, x, y, z]
                s = score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U
                if time.time() > deadline:
                    return best_set
                    
    # Size 5
    for x in range(1, 15):
        for y in range(x+1, 15):
            for z in range(y+1, 15):
                for w in range(z+1, 30):
                    U = [0, x, y, z, w]
                    s = score_set(U)
                    if s > best_score:
                        best_score = s
                        best_set = U
                    if time.time() > deadline:
                        return best_set
                        
    # Size 6 (limited)
    for x in range(1, 10):
        for y in range(x+1, 10):
            for z in range(y+1, 10):
                for w in range(z+1, 15):
                    for v in range(w+1, 20):
                        for u in range(v+1, 30):
                            U = [0, x, y, z, w, v, u]
                            s = score_set(U)
                            if s > best_score:
                                best_score = s
                                best_set = U
                            if time.time() > deadline:
                                return best_set
                                
    return best_set

import random
import time

N = 256

def construct():
    best_set = []
    best_size = N + 1
    deadline = time.time() + 55.0
    
    while time.time() < deadline:
        S = random_greedy_maximal()
        S = try_reduce(S, deadline)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S
            
    return best_set

def random_greedy_maximal():
    order = list(range(1, N + 1))
    random.shuffle(order)
    S = []
    sums = set()
    for x in order:
        if can_add(x, S, sums):
            S.append(x)
            sums.add(2 * x)
            for a in S[:-1]: # Exclude x itself
                sums.add(x + a)
    return S

def can_add(x, S, sums):
    if 2 * x in sums:
        return False
    for a in S:
        if x + a in sums:
            return False
    return True

def is_maximal(S):
    sums = get_sums(S)
    S_set = set(S)
    for x in range(1, N + 1):
        if x not in S_set:
            if can_add(x, S, sums):
                return False
    return True

def get_sums(S):
    sums = set()
    for i, a in enumerate(S):
        sums.add(2 * a)
        for b in S[i+1:]:
            sums.add(a + b)
    return sums

def try_reduce(S, deadline):
    S = list(S)
    while time.time() < deadline:
        improved = False
        candidates = list(S)
        random.shuffle(candidates)
        for y in candidates:
            S_new = [x for x in S if x != y]
            if is_maximal(S_new):
                S = S_new
                improved = True
                break
        if not improved:
            break
    return S

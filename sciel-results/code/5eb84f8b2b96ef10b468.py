import random
import time

N = 256

def is_sidon(S):
    """Check if S is a Sidon set."""
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            b = S[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def is_maximal(S, N):
    """Check if S is inclusion-maximal in [1,N]."""
    S_set = set(S)
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            sums.add(a + S[j])
    
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if x can be added
        new_sums = set(sums)
        valid = True
        for a in S:
            s = x + a
            if s in new_sums:
                valid = False
                break
            new_sums.add(s)
        if valid:
            s = 2 * x
            if s in new_sums:
                valid = False
            else:
                new_sums.add(s)
        if valid:
            return False
    return True

def greedy_sidon(order, N):
    """Greedy Sidon set construction given an order of elements."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def try_reduce(S, N):
    """Try to reduce the maximal Sidon set by removing elements."""
    S = list(S)
    improved = True
    while improved:
        improved = False
        for i in range(len(S)):
            candidate = S[:i] + S[i+1:]
            if len(candidate) == 0:
                continue
            if is_sidon(candidate) and is_maximal(candidate, N):
                S = candidate
                improved = True
                break
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    best = None
    best_size = N + 1
    
    # First, try the standard ascending greedy as a baseline
    order = list(range(1, N + 1))
    S = greedy_sidon(order, N)
    S = try_reduce(S, N)
    if len(S) < best_size:
        best = S
        best_size = len(S)
    
    # Try multiple random orderings
    random.seed(42)
    for trial in range(200):
        if time.time() > deadline:
            break
        
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S = greedy_sidon(order, N)
        
        # Try to reduce
        if time.time() < deadline - 5:
            S = try_reduce(S, N)
        
        if len(S) < best_size:
            best = S
            best_size = len(S)
    
    if best is None:
        # Fallback to ascending greedy
        order = list(range(1, N + 1))
        best = greedy_sidon(order, N)
    
    return best

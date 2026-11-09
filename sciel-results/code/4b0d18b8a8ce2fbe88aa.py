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
    
    # Compute blocked set efficiently
    # y is blocked if there exists a in S such that (y+a) in sums, or 2y in sums
    blocked = set()
    for s in sums:
        for a in S:
            y = s - a
            if 1 <= y <= N:
                blocked.add(y)
        if s % 2 == 0:
            y = s // 2
            if 1 <= y <= N:
                blocked.add(y)
                
    # Check if all y not in S are blocked
    for y in range(1, N + 1):
        if y in S_set:
            continue
        if y not in blocked:
            return False
    return True

def greedy_sidon(order, N):
    """Greedy Sidon set construction given an order of elements."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        # Check internal consistency: no duplicate sums within added
        if len(set(added)) != len(added):
            continue
        if not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def try_reduce(S, N, deadline=None):
    """Try to reduce the maximal Sidon set by removing elements."""
    S = list(S)
    improved = True
    while improved:
        improved = False
        indices = list(range(len(S)))
        random.shuffle(indices)
        
        for i in indices:
            if deadline and time.time() > deadline:
                return S
            
            candidate = S[:i] + S[i+1:]
            if len(candidate) == 0:
                continue
            
            if is_maximal(candidate, N):
                S = candidate
                improved = True
                break
    
    # Try swaps to potentially reduce further
    improved = True
    while improved:
        improved = False
        S_set = set(S)
        indices = list(range(len(S)))
        random.shuffle(indices)
        
        for i in indices:
            if deadline and time.time() > deadline:
                return S
            
            x = S[i]
            # Try replacing x with a random element
            for _ in range(10):
                if deadline and time.time() > deadline:
                    return S
                y = random.randint(1, N)
                if y in S_set:
                    continue
                
                candidate = [s for s in S if s != x] + [y]
                if not is_sidon(candidate):
                    continue
                if is_maximal(candidate, N):
                    S = candidate
                    S_set = set(S)
                    improved = True
                    break
            if improved:
                break
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    best = None
    best_size = N + 1
    
    # First, try the standard ascending greedy as a baseline
    order = list(range(1, N + 1))
    S = greedy_sidon(order, N)
    S = try_reduce(S, N, deadline)
    if len(S) < best_size:
        best = S
        best_size = len(S)
    
    # Try multiple random orderings
    random.seed(42)
    for trial in range(1000):
        if time.time() > deadline:
            break
        
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S = greedy_sidon(order, N)
        
        # Try to reduce
        S = try_reduce(S, N, deadline)
        
        if len(S) < best_size:
            best = S
            best_size = len(S)
    
    if best is None:
        # Fallback to ascending greedy
        order = list(range(1, N + 1))
        best = greedy_sidon(order, N)
    
    return best

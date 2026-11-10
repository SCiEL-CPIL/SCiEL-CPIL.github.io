import itertools
import time
import random

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(POINTS)
# Map point to index
POINT_INDEX = {p: i for i, p in enumerate(POINTS)}

# Precompute squared distances
# dist2[i][j] is squared distance between POINTS[i] and POINTS[j]
dist2 = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_POINTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) ** 2 + (y1 - y2) ** 2
        dist2[i][j] = d
        dist2[j][i] = d

def is_isosceles(i, j, k):
    """Check if points i, j, k form an isosceles triangle."""
    d1 = dist2[i][j]
    d2 = dist2[i][k]
    d3 = dist2[j][k]
    return d1 == d2 or d1 == d3 or d2 == d3

def can_add(p_idx, S):
    """Check if point p_idx can be added to set S (list of indices)."""
    for i in range(len(S)):
        a = S[i]
        for j in range(i + 1, len(S)):
            b = S[j]
            if is_isosceles(p_idx, a, b):
                return False
    return True

def random_greedy(seed):
    """Construct a maximal no-isosceles set using randomized greedy."""
    random.seed(seed)
    S = []
    candidates = list(range(NUM_POINTS))
    random.shuffle(candidates)
    
    for p in candidates:
        if can_add(p, S):
            S.append(p)
            
    # The result is maximal by construction of standard greedy
    return S

def is_maximal(S):
    """Check if set S is maximal."""
    S_set = set(S)
    for p in range(NUM_POINTS):
        if p not in S_set:
            if can_add(p, S):
                return False
    return True

def minimize_maximal(S):
    """Try to minimize a maximal set by removing points."""
    S = list(S)
    changed = True
    while changed:
        changed = False
        # Try removing each point
        for i in range(len(S)):
            p = S[i]
            S_try = S[:i] + S[i+1:]
            if is_maximal(S_try):
                S = S_try
                changed = True
                break # Restart loop as set changed
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin
    
    best_set = None
    best_size = NUM_POINTS + 1
    
    seed = 42
    while time.time() < deadline:
        # Generate a random maximal set
        S = random_greedy(seed)
        # Minimize it
        S_min = minimize_maximal(S)
        
        if len(S_min) < best_size:
            best_size = len(S_min)
            best_set = S_min
            
        # If we find a very small set, we can be more aggressive or stop?
        # For now, just keep searching.
        
        seed += 1
        
        # Optional: If best_size is very small, maybe try local search around it?
        # But random restarts seem effective.

    if best_set is None:
        # Fallback to deterministic greedy if no time left or error
        S = []
        for p in range(NUM_POINTS):
            if can_add(p, S):
                S.append(p)
        return [POINTS[i] for i in S]
        
    return [POINTS[i] for i in best_set]

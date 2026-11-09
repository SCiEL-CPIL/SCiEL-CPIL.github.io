import time
import random
import math

def _score(S):
    """Count distinct d>0 such that a, a+d, a+2d all in S."""
    s = set(S)
    n = len(S)
    diffs = set()
    # For each pair (a, c) with a < c, check if midpoint is in S
    # d = (c - a) / 2, need c - a even
    arr = sorted(S)
    for i in range(n):
        a = arr[i]
        for j in range(i + 1, n):
            c = arr[j]
            diff = c - a
            if diff % 2 == 0:
                b = a + diff // 2
                if b in s:
                    d = diff // 2
                    if d > 0:
                        diffs.add(d)
    return len(diffs)

def _score_from_set(S):
    """Score for a set of integers."""
    return _score(S)

def _random_individual():
    """Generate a random 32-element subset of 1..512."""
    return set(random.sample(range(1, 513), 32))

def _hill_climb(S, max_iter=2000, time_limit=12.0):
    """Local search: swap elements to improve score."""
    start = time.time()
    S = set(S)
    best_score = _score(S)
    best_S = set(S)
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        # Try random swap: remove one element, add another
        improved = False
        elems = list(S)
        random.shuffle(elems)
        
        for _ in range(20):  # try 20 random swaps per iteration
            if time.time() - start > time_limit:
                break
            
            a = random.choice(elems)
            # Try to replace a with a new element
            new_val = random.randint(1, 512)
            if new_val in S:
                continue
            
            S.remove(a)
            S.add(new_val)
            sc = _score(S)
            
            if sc > best_score:
                best_score = sc
                best_S = set(S)
                improved = True
            else:
                S.remove(new_val)
                S.add(a)
        
        if not improved and it > 100:
            # Stuck, try a bigger move: remove 2, add 2
            if time.time() - start > time_limit:
                break
            elems = list(S)
            a1, a2 = random.sample(elems, 2)
            S.discard(a1)
            S.discard(a2)
            nv1 = random.randint(1, 512)
            nv2 = random.randint(1, 512)
            if nv1 not in S and nv2 not in S and nv1 != nv2:
                S.add(nv1)
                S.add(nv2)
                sc = _score(S)
                if sc > best_score:
                    best_score = sc
                    best_S = set(S)
                else:
                    S.discard(nv1)
                    S.discard(nv2)
                    S.add(a1)
                    S.add(a2)
    
    return best_S, best_score

def _grid_initial():
    """Create a structured initial set using grid-like patterns."""
    candidates = []
    
    # Pattern 1: 4x8 grid
    for M in [40, 50, 60, 80, 100, 120, 150, 160]:
        S = set()
        for i in range(4):
            for j in range(8):
                v = i * M + j + 1
                if 1 <= v <= 512:
                    S.add(v)
        if len(S) == 32:
            candidates.append(S)
    
    # Pattern 2: 8x4 grid
    for M in [40, 50, 60, 80, 100, 120, 150, 160]:
        S = set()
        for i in range(8):
            for j in range(4):
                v = i * M + j + 1
                if 1 <= v <= 512:
                    S.add(v)
        if len(S) == 32:
            candidates.append(S)
    
    # Pattern 3: Arithmetic progressions
    for a in range(1, 50):
        for d in range(1, 20):
            S = set(a + i * d for i in range(32))
            if all(1 <= x <= 512 for x in S) and len(S) == 32:
                candidates.append(S)
    
    # Pattern 4: Two interleaved APs
    for a1 in range(1, 30):
        for d1 in range(2, 15):
            for a2 in range(1, 30):
                for d2 in range(2, 15):
                    S = set()
                    for i in range(16):
                        S.add(a1 + i * d1)
                        S.add(a2 + i * d2)
                    if len(S) == 32 and all(1 <= x <= 512 for x in S):
                        candidates.append(S)
    
    return candidates

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    best_S = None
    best_score = 0
    
    # Phase 1: Try structured initial candidates
    candidates = _grid_initial()
    random.shuffle(candidates)
    
    for S in candidates:
        if time.time() > deadline - 20.0:
            break
        sc = _score(S)
        if sc > best_score:
            best_score = sc
            best_S = set(S)
    
    # Phase 2: Multi-restart hill climbing
    n_restarts = 15
    for r in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Start from best found or random
        if r == 0 and best_S:
            start_S = set(best_S)
        elif r < 5 and best_S:
            # Perturb best
            start_S = set(best_S)
            for _ in range(random.randint(1, 5)):
                elem = random.choice(list(start_S))
                start_S.discard(elem)
                start_S.add(random.randint(1, 512))
                while len(start_S) < 32:
                    start_S.add(random.randint(1, 512))
                start_S = set(random.sample(list(start_S), 32))
        else:
            start_S = _random_individual()
        
        S, sc = _hill_climb(start_S, max_iter=1500, time_limit=8.0)
        if sc > best_score:
            best_score = sc
            best_S = set(S)
    
    # Phase 3: Final polish with more iterations on best
    if best_S and time.time() < deadline - 10.0:
        S, sc = _hill_climb(best_S, max_iter=3000, time_limit=10.0)
        if sc > best_score:
            best_score = sc
            best_S = set(S)
    
    if best_S is None:
        best_S = set(range(1, 33))
    
    return sorted(best_S)

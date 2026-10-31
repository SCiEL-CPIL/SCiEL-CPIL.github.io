import time
import math
import random
from itertools import combinations

def _score(U):
    """Compute the score for set U."""
    U = sorted(set(U))
    if not U or U[0] != 0:
        return -float('inf')
    max_u = max(U)
    if max_u == 0:
        return 1.0
    # Compute U+U and U-U
    U_set = set(U)
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return -float('inf')
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def _fast_score(U):
    """Faster score computation using sorted list."""
    U = sorted(set(U))
    if not U or U[0] != 0:
        return -float('inf')
    max_u = max(U)
    if max_u == 0:
        return 1.0
    n = len(U)
    sumset = set()
    diffset = set()
    for i in range(n):
        for j in range(n):
            sumset.add(U[i] + U[j])
            diffset.add(U[i] - U[j])
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return -float('inf')
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def _local_search(U, max_iterations=5000, time_limit=100.0):
    """Local search to improve the set."""
    start_time = time.time()
    U = sorted(set(U))
    current_score = _score(U)
    best_U = list(U)
    best_score = current_score
    
    # Determine search space
    max_u = max(U)
    search_max = max_u * 2 + 10
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try various moves
        moved = False
        
        # Move 1: Remove an element (not 0)
        if len(U) > 1:
            idx = random.randint(1, len(U) - 1)
            new_U = U[:idx] + U[idx+1:]
            new_score = _score(new_U)
            if new_score > current_score + 1e-12:
                U = new_U
                current_score = new_score
                moved = True
        
        # Move 2: Add a new element
        if not moved:
            candidates = []
            # Try values near existing elements and at geometric positions
            for x in U:
                candidates.extend([x-1, x+1, x*2, x+max_u//10])
            candidates.extend([max_u + 1, max_u + max_u//10, max_u * 2])
            candidates = [x for x in candidates if x >= 0 and x != max_u and x not in U]
            if candidates:
                new_val = random.choice(candidates)
                new_U = sorted(U + [new_val])
                new_score = _score(new_U)
                if new_score > current_score + 1e-12:
                    U = new_U
                    current_score = new_score
                    moved = True
        
        # Move 3: Replace an element
        if not moved and len(U) > 1:
            idx = random.randint(1, len(U) - 1)
            old_val = U[idx]
            candidates = [old_val - 1, old_val + 1, old_val * 2, old_val // 2 if old_val > 1 else 1,
                          max_u + 1, random.randint(0, max_u * 2)]
            candidates = [x for x in candidates if x >= 0 and x not in U and x != old_val]
            if candidates:
                new_val = random.choice(candidates)
                new_U = U[:idx] + [new_val] + U[idx+1:]
                new_U = sorted(set(new_U))
                new_score = _score(new_U)
                if new_score > current_score + 1e-12:
                    U = new_U
                    current_score = new_score
                    moved = True
        
        if current_score > best_score:
            best_score = current_score
            best_U = list(U)
        
        if not moved:
            # Perturb: try a random restart-ish move
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                U = U[:idx] + U[idx+1:]
                current_score = _score(U)
            elif len(U) == 2:
                U = [0, random.randint(1, max_u * 2)]
                current_score = _score(U)
    
    return best_U, best_score

def construct():
    deadline = time.time() + 140  # Safety margin
    
    # Start with geometric set
    best_U = [0, 1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049, 177147, 531441]
    best_score = _score(best_U)
    
    # Try different initial structures
    initial_sets = [
        [0, 1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049, 177147, 531441],
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191],
        [0, 2, 6, 18, 54, 162, 486, 1458, 4374, 13122, 39366, 118098],
    ]
    
    for init in initial_sets:
        if time.time() > deadline:
            break
        s = _score(init)
        if s > best_score:
            best_score = s
            best_U = list(init)
    
    # Local search from best
    if time.time() < deadline:
        U, score = _local_search(best_U, max_iterations=10000, time_limit=min(60, deadline - time.time()))
        if score > best_score:
            best_score = score
            best_U = U
    
    # Try more local searches from different starts
    remaining = deadline - time.time()
    if remaining > 10:
        for init in initial_sets:
            if time.time() > deadline - 5:
                break
            U, score = _local_search(init, max_iterations=3000, time_limit=min(20, deadline - time.time() - 5))
            if score > best_score:
                best_score = score
                best_U = U
    
    return sorted(set(best_U))

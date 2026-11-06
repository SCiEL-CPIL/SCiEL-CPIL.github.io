import time
import random
from itertools import combinations

def _can_add(x, S):
    """
    Check if x can be added to set S without violating the condition.
    S is a list of valid integers.
    Condition: No a in S_new divides sum of any 3 distinct OTHERS in S_new.
    """
    n = len(S)
    
    # 1. x does not divide sum of any 3 distinct elements in S
    if n >= 3:
        # Iterate combinations of S
        for b, c, d in combinations(S, 3):
            if (b + c + d) % x == 0:
                return False
            
    # 2. No a in S divides x + b + c for distinct b, c in S\{a}
    # We need to check for each a in S, if there exist distinct b, c in S\{a}
    # such that (x + b + c) % a == 0
    
    # Optimization: Sort S ascending? Small a are more likely to divide.
    # But S is a list, sorting takes time. 
    # Given n <= 48, we can just iterate.
    
    for a in S:
        # Find distinct b, c in S such that b != a, c != a, b != c
        # and (x + b + c) % a == 0
        
        # To avoid checking pairs where one is a
        # We can iterate i < j in S
        for i in range(n):
            bi = S[i]
            if bi == a:
                continue
            for j in range(i + 1, n):
                bj = S[j]
                if bj == a:
                    continue
                # bi and bj are distinct and neither is a
                if (x + bi + bj) % a == 0:
                    return False
                    
    return True

def _is_valid(selected):
    """Check if selected is a valid set."""
    n = len(selected)
    for idx_a, a in enumerate(selected):
        others = [v for i, v in enumerate(selected) if i != idx_a]
        if len(others) < 3:
            continue
        for b, c, d in combinations(others, 3):
            if (b + c + d) % a == 0:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best = []
    best_len = 0
    
    def _update_best(s):
        nonlocal best, best_len
        if len(s) > best_len and _is_valid(s):
            best = list(s)
            best_len = len(s)
            
    all_nums = list(range(1, 193))
    rng = random.Random(42)
    
    # Phase 1: Deterministic Greedy (Descending)
    order = sorted(all_nums, reverse=True)
    current = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(x, current):
            current.append(x)
    _update_best(current)
    
    # Phase 2: Randomized Search with Local Improvement
    # We will run multiple iterations.
    # Each iteration:
    # 1. Generate a random order (biased towards larger numbers).
    # 2. Greedy select.
    # 3. Local search: Try to add, then try swaps.
    
    while time.time() < deadline:
        # Generate Order
        # Bias: value/192 + random noise
        # Higher value -> higher key -> sorted descending
        order = sorted(all_nums, key=lambda x: (x / 192.0) + rng.random() * 0.5, reverse=True)
        
        # Greedy Select
        current = []
        for x in order:
            if time.time() > deadline:
                break
            if _can_add(x, current):
                current.append(x)
                
        # Local Search
        current_set = set(current)
        pool = [x for x in all_nums if x not in current_set]
        
        # 1. Try to add any available number
        added = True
        while added and time.time() < deadline:
            added = False
            for x in pool:
                if _can_add(x, current):
                    current.append(x)
                    current_set.add(x)
                    pool.remove(x)
                    added = True
                    break
            if added:
                continue
                
        # 2. Try swaps to escape local minima
        # We allow a limited number of swaps per iteration to prevent getting stuck
        max_swaps = 50
        for _ in range(max_swaps):
            if time.time() > deadline:
                break
            if not pool or not current:
                break
                
            # Random swap
            y_idx = rng.randrange(len(current))
            y = current[y_idx]
            x = rng.choice(pool)
            
            # Check if swap is valid
            # Remove y, add x
            candidate = current[:y_idx] + current[y_idx+1:]
            if _can_add(x, candidate):
                candidate.append(x)
                # Accept swap
                current = candidate
                current_set.remove(y)
                current_set.add(x)
                pool.remove(x)
                pool.append(y)
                
                # Try to add again after swap
                added = True
                while added and time.time() < deadline:
                    added = False
                    for z in pool:
                        if _can_add(z, current):
                            current.append(z)
                            current_set.add(z)
                            pool.remove(z)
                            added = True
                            break
                            
        _update_best(current)
        
    return best

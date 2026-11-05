import random
import time
from itertools import combinations

def _check_validity(selected):
    """Check if the selected set satisfies the constraint."""
    s = set(selected)
    for a in selected:
        others = [x for x in selected if x != a]
        for tri in combinations(others, 3):
            if sum(tri) % a == 0:
                return False
    return True

def _can_add(selected, x):
    """Check if x can be added to selected without violating constraints."""
    s = set(selected)
    # Check: x must not divide sum of any 3 distinct others in selected
    others = [v for v in selected if v != x]
    for tri in combinations(others, 3):
        if sum(tri) % x == 0:
            return False
    # Check: no existing a in selected should divide sum of any 3 distinct others including x
    for a in selected:
        # The 3 others must be distinct from a; they can include x
        # We need to check all triples from (selected \ {a}) that include x
        # Actually: for each a, check if a divides sum of any 3 distinct elements from (selected ∪ {x}) \ {a}
        pool = [v for v in selected if v != a]
        # We need triples from pool that may or may not include x
        # But we only need to check triples that include x (triples without x were already valid)
        for pair in combinations(pool, 2):
            tri_sum = pair[0] + pair[1] + x
            if tri_sum % a == 0:
                return False
    return True

def _greedy_build(order, selected):
    """Try to add elements in given order."""
    for x in order:
        if x in selected:
            continue
        if _can_add(selected, x):
            selected.add(x)
    return selected

def _local_search(selected, max_iters=200):
    """Try swap and add operations to improve the set."""
    s = set(selected)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Try adding a new element
        for x in range(192, 0, -1):
            if x not in s and _can_add(s, x):
                s.add(x)
                improved = True
                break
        if improved:
            continue
        # Try swap: remove one, add another
        for a in list(s):
            s.remove(a)
            for x in range(192, 0, -1):
                if x not in s and _can_add(s, x):
                    s.add(x)
                    improved = True
                    break
            if improved:
                break
            s.add(a)  # restore
    return s

def construct():
    start = time.time()
    deadline = start + 75  # 75s safety margin before 90s kill
    
    best = set()
    best_size = 0
    
    # Initial deterministic greedy (descending)
    s = set()
    for x in range(192, 0, -1):
        if _can_add(s, x):
            s.add(x)
    best = set(s)
    best_size = len(s)
    
    # Multi-restart randomized greedy + local search
    rng = random.Random(42)
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        # Randomized ordering
        candidates = list(range(1, 193))
        rng.shuffle(candidates)
        
        s = set()
        # Greedy build with random order
        for x in candidates:
            if _can_add(s, x):
                s.add(x)
        
        # Local search
        s = _local_search(s, max_iters=50)
        
        if len(s) > best_size:
            best = set(s)
            best_size = len(s)
        
        # Also try biased random: prefer larger numbers
        if attempt % 3 == 0:
            candidates = list(range(192, 0, -1))
            # Randomly perturb
            for i in range(len(candidates) - 1, 0, -1):
                j = rng.randint(0, i)
                candidates[i], candidates[j] = candidates[j], candidates[i]
            s = set()
            for x in candidates:
                if _can_add(s, x):
                    s.add(x)
            s = _local_search(s, max_iters=50)
            if len(s) > best_size:
                best = set(s)
                best_size = len(s)
    
    return list(best)

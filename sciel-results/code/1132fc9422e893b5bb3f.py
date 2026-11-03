import time
import math
import random
from itertools import combinations

def _score(U):
    """Compute the score for a given set U (as a sorted list)."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0  # degenerate case
    
    # Compute U+U
    sum_set = set()
    for a in U_set:
        for b in U_set:
            sum_set.add(a + b)
    
    # Compute U-U
    diff_set = set()
    for a in U_set:
        for b in U_set:
            diff_set.add(a - b)
    
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 1:
        return 1.0
    
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
    
    return 1.0 + log_ratio / log_denom

def _geometric_score(b, m):
    """Compute score for U = {floor(b^k) for k=0..m}."""
    if b <= 1 or m < 1:
        return 0.0
    U = []
    for k in range(m + 1):
        val = int(math.floor(b ** k))
        if val not in U:
            U.append(val)
    U.sort()
    if len(U) < 2:
        return 0.0
    return _score(U)

def _ternary_search_1d(f, a, b, iterations=50):
    """Ternary search for maximum of f on [a, b]."""
    for _ in range(iterations):
        m1 = a + (b - a) / 3
        m2 = b - (b - a) / 3
        if f(m1) < f(m2):
            a = m1
        else:
            b = m2
    return (a + b) / 2

def _optimize_geometric(deadline):
    """Optimize geometric progression base b and length m."""
    best_score = 0.0
    best_U = [0]
    
    # Try different ranges of b and m
    for m in range(3, 20):
        if time.time() > deadline:
            break
        # Search for optimal b for this m
        # b should be such that b^m is reasonable
        # Try b in [1.5, 5.0]
        best_b = _ternary_search_1d(lambda x: _geometric_score(x, m), 1.5, 5.0, iterations=30)
        s = _geometric_score(best_b, m)
        if s > best_score:
            best_score = s
            U = []
            for k in range(m + 1):
                val = int(math.floor(best_b ** k))
                if val not in U:
                    U.append(val)
            U.sort()
            best_U = U
    
    return best_U, best_score

def _local_search(U, deadline, max_iterations=10000):
    """Local search: try adding/removing/replacing elements."""
    best_score = _score(U)
    best_set = U[:]
    current = U[:]
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Try a random modification
        move = random.random()
        
        if move < 0.3:
            # Add a new element
            max_val = max(current) if current else 0
            new_val = random.randint(1, max_val * 2 + 10)
            if new_val not in current:
                candidate = current + [new_val]
                candidate.sort()
                s = _score(candidate)
                if s > best_score:
                    best_score = s
                    best_set = candidate
                    current = candidate
        elif move < 0.5:
            # Remove an element (keep 0)
            if len(current) > 2:
                idx = random.randint(1, len(current) - 1)
                candidate = current[:idx] + current[idx+1:]
                s = _score(candidate)
                if s > best_score:
                    best_score = s
                    best_set = candidate
                    current = candidate
        else:
            # Replace an element
            if len(current) > 2:
                idx = random.randint(1, len(current) - 1)
                old_val = current[idx]
                new_val = random.randint(1, max(current) * 2 + 10)
                if new_val not in current:
                    candidate = current[:idx] + [new_val] + current[idx+1:]
                    candidate.sort()
                    s = _score(candidate)
                    if s > best_score:
                        best_score = s
                        best_set = candidate
                        current = candidate
    
    return best_set, best_score

def _search_small_sets(deadline):
    """Search over small sets of nonnegative integers containing 0."""
    best_score = 0.0
    best_set = [0]
    
    max_val = 15
    max_size = 6
    
    candidates = list(range(1, max_val + 1))
    
    for size in range(2, max_size + 1):
        if time.time() > deadline:
            break
        for combo in combinations(candidates, size):
            if time.time() > deadline:
                break
            U = [0] + list(combo)
            U.sort()
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    best_set = [0]
    best_score = 0.0
    
    # Phase 1: Small exhaustive search
    s_set, s_score = _search_small_sets(deadline)
    if s_score > best_score:
        best_score = s_score
        best_set = s_set
    
    # Phase 2: Geometric progression optimization
    g_set, g_score = _optimize_geometric(deadline)
    if g_score > best_score:
        best_score = g_score
        best_set = g_set
    
    # Phase 3: Local search starting from best found
    if time.time() < deadline - 10:
        ls_set, ls_score = _local_search(best_set, deadline, max_iterations=20000)
        if ls_score > best_score:
            best_score = ls_score
            best_set = ls_set
    
    # Phase 4: Try some known good structures
    for n in range(3, 15):
        if time.time() > deadline:
            break
        U = [2**i for i in range(n)]
        s = _score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Try arithmetic progressions
    for a in range(1, 10):
        for n in range(3, 15):
            if time.time() > deadline:
                break
            U = [i * a for i in range(n)]
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    return best_set

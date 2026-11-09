import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def _precompute_conflicts(n):
    """
    Precompute all conflict triples for each element.
    For 1/a = 1/b + 1/c with distinct a,b,c:
    (b-a)(c-a) = a^2
    """
    conflict_triples = {i: [] for i in range(1, n+1)}
    
    for a in range(1, n+1):
        a2 = a * a
        for d in range(1, int(math.isqrt(a2)) + 1):
            if a2 % d != 0:
                continue
            d1 = d
            d2 = a2 // d
            if d1 == d2:
                continue
            b = a + d1
            c = a + d2
            if b > n or c > n:
                continue
            if b == c:
                continue
            conflict_triples[a].append((b, c))
            conflict_triples[b].append((a, c))
            conflict_triples[c].append((a, b))
    
    return conflict_triples

def _can_add(s_set, e, conflict_triples):
    """Check if adding element e to s_set creates a conflict."""
    for (a, b) in conflict_triples[e]:
        if a in s_set and b in s_set:
            return False
    return True

def _check_valid(s_set, conflict_triples):
    """Check if the entire set is valid (no conflict triples)."""
    for a in s_set:
        for (b, c) in conflict_triples[a]:
            if b in s_set and c in s_set:
                return False
    return True

def _greedy_build(n, conflict_triples, order, deadline):
    """Greedy construction following a given order."""
    s = set()
    for e in order:
        if time.time() > deadline:
            break
        if _can_add(s, e, conflict_triples):
            s.add(e)
    return s

def _local_search_1swap(s_set, n, conflict_triples, deadline, max_iter=5000):
    """1-swap local search: remove one element, add another."""
    current = set(s_set)
    best = set(current)
    best_size = len(current)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        not_in = list(set(range(1, n+1)) - current)
        if not not_in:
            break
        
        random.shuffle(not_in)
        added = False
        for e in not_in:
            if _can_add(current, e, conflict_triples):
                current.add(e)
                added = True
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(current)
                break
        
        if added:
            continue
        
        x = random.choice(list(current))
        temp = set(current)
        temp.discard(x)
        
        not_in_temp = list(set(range(1, n+1)) - temp)
        random.shuffle(not_in_temp)
        swapped = False
        for y in not_in_temp:
            if _can_add(temp, y, conflict_triples):
                temp.add(y)
                swapped = True
                if len(temp) > best_size:
                    best = set(temp)
                    best_size = len(temp)
                current = temp
                break
    
    return best

def _local_search_2swap(s_set, n, conflict_triples, deadline, max_iter=2000):
    """2-swap local search: remove two elements, add two others."""
    current = set(s_set)
    best = set(current)
    best_size = len(current)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        not_in = list(set(range(1, n+1)) - current)
        if not_in:
            random.shuffle(not_in)
            for e in not_in:
                if _can_add(current, e, conflict_triples):
                    current.add(e)
                    if len(current) > best_size:
                        best = set(current)
                        best_size = len(current)
                    break
            continue
        
        if len(current) < 2:
            break
        
        x1, x2 = random.sample(list(current), 2)
        temp = set(current)
        temp.discard(x1)
        temp.discard(x2)
        
        not_in_temp = list(set(range(1, n+1)) - temp)
        random.shuffle(not_in_temp)
        
        added_count = 0
        for y in not_in_temp:
            if added_count >= 2:
                break
            if _can_add(temp, y, conflict_triples):
                temp.add(y)
                added_count += 1
        
        if added_count == 2 and len(temp) > best_size:
            best = set(temp)
            best_size = len(temp)
            current = temp
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit
    
    n = 180
    random.seed(42)
    
    conflict_triples = _precompute_conflicts(n)
    
    best_set = set()
    best_size = 0
    
    # Multiple greedy orderings
    orders = [
        list(range(1, n+1)),
        list(range(n, 0, -1)),
        list(range(1, n+1, 2)) + list(range(2, n+1, 2)),
        list(range(2, n+1, 2)) + list(range(1, n+1, 2)),
    ]
    
    # Add random orderings
    for _ in range(10):
        order = list(range(1, n+1))
        random.shuffle(order)
        orders.append(order)
    
    for order in orders:
        if time.time() > deadline:
            break
        s = _greedy_build(n, conflict_triples, order, deadline)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # Local search
    best_set = _local_search_1swap(best_set, n, conflict_triples, deadline, max_iter=3000)
    best_size = len(best_set)
    
    best_set = _local_search_2swap(best_set, n, conflict_triples, deadline, max_iter=1000)
    best_size = len(best_set)
    
    # More restarts
    for restart in range(10):
        if time.time() > deadline:
            break
        
        random.seed(10000 + restart * 7)
        order = list(range(1, n+1))
        random.shuffle(order)
        s = _greedy_build(n, conflict_triples, order, deadline)
        s = _local_search_1swap(s, n, conflict_triples, deadline, max_iter=1000)
        
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # Final polish
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        not_in = list(set(range(1, n+1)) - best_set)
        random.shuffle(not_in)
        for e in not_in:
            if _can_add(best_set, e, conflict_triples):
                best_set.add(e)
                improved = True
                break
    
    # Final validation
    if not _check_valid(best_set, conflict_triples):
        best_set = set(range(91, 181))
    
    if not best_set:
        best_set = {1}
    
    return list(best_set)

# EVOLVE-BLOCK-END

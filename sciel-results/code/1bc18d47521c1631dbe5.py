"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

def _build_conflict_graph(n_max=256):
    squares = set()
    k = 2
    while k * k <= 2 * n_max:
        squares.add(k * k)
        k += 1
    
    conflict = [set() for _ in range(n_max + 1)]
    for i in range(1, n_max + 1):
        for j in range(i, n_max + 1):
            if (i + j) in squares:
                conflict[i].add(j)
                conflict[j].add(i)
    return conflict

def _greedy_is(conflict, order):
    in_set = set()
    for v in order:
        if not any(u in conflict[v] for u in in_set):
            in_set.add(v)
    return in_set

def _local_search(conflict, initial_set, n_iter=200000):
    current = set(initial_set)
    
    for _ in range(n_iter):
        # Try to add a random vertex not in current
        candidates = [v for v in range(1, 257) if v not in current]
        if candidates:
            v = random.choice(candidates)
            conflicting = [u for u in current if u in conflict[v]]
            if not conflicting:
                current.add(v)
            elif len(conflicting) == 1:
                u = conflicting[0]
                current.discard(u)
                current.add(v)
    
    return current

def construct():
    conflict = _build_conflict_graph(256)
    
    best_set = set()
    best_size = 0
    
    for trial in range(300):
        order = list(range(1, 257))
        random.shuffle(order)
        s = _greedy_is(conflict, order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
    
    for _ in range(20):
        improved = _local_search(conflict, best_set, n_iter=100000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
        else:
            break
    
    return list(best_set)
# EVOLVE-BLOCK-END

"""
Snake-in-the-box in Q_9 using a MODEL-AND-OPTIMIZE approach:
1. Fast greedy initialization to get a baseline snake.
2. Local search / optimization to extend the snake by trying to insert new vertices
   or extend the ends, and by repairing/optimizing the path structure.
3. Backtracking with time limit to find the longest possible induced path.

The key insight: the problem is to find the longest induced path in Q_9.
- Seed is 104 edges (105 vertices).
- We use a combination of greedy extension, local improvement, and randomized search.
"""
import time
import random
import numpy as np
from collections import deque

N = 9
FULL_MASK = (1 << N) - 1
TIME_LIMIT = 145  # seconds, safety margin before 150s kill

# Precompute adjacency as a numpy array for fast neighbor lookups
# adj[v] is a list of neighbors of v in Q_9
_ADJ = None

def _get_adj():
    """Lazily precompute adjacency list for Q_9."""
    global _ADJ
    if _ADJ is None:
        _ADJ = [[] for _ in range(1 << N)]
        for v in range(1 << N):
            for b in range(N):
                _ADJ[v].append(v ^ (1 << b))
    return _ADJ

def _popcount(x):
    """Number of set bits in x."""
    return bin(x).count('1')

def _hamming(x, y):
    """Hamming distance between x and y."""
    return _popcount(x ^ y)

def _is_induced_path(path):
    """Check if path is a valid induced path (snake)."""
    n = len(path)
    if n < 2:
        return True
    # Check consecutive differ by exactly 1 bit
    for i in range(n - 1):
        if _hamming(path[i], path[i+1]) != 1:
            return False
    # Check non-consecutive differ by at least 2 bits
    for i in range(n):
        for j in range(i+2, n):
            if _hamming(path[i], path[j]) == 1:
                return False
    return True

def _greedy_snake(start=0, bit_order=None, time_budget=5.0):
    """
    Greedy snake: always extend with the lowest bit flip that keeps the path induced.
    Returns the list of vertices.
    """
    start_time = time.time()
    path = [start]
    used = {start}
    adj = _get_adj()
    
    while time.time() - start_time < time_budget:
        cur = path[-1]
        extended = False
        if bit_order is None:
            bits = range(N)
        else:
            bits = bit_order
        
        for b in bits:
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check if v is adjacent to any non-consecutive vertex
            ok = True
            for p in path[:-1]:
                if _hamming(v, p) == 1:
                    ok = False
                    break
            if ok:
                path.append(v)
                used.add(v)
                extended = True
                break
        if not extended:
            break
    
    return path

def _try_extend_end(path, end='tail', used=None):
    """Try to extend the snake at the specified end."""
    if used is None:
        used = set(path)
    
    cur_idx = len(path) - 1 if end == 'tail' else 0
    cur = path[cur_idx]
    
    # Try all possible next vertices
    for b in range(N):
        v = cur ^ (1 << b)
        if v in used:
            continue
        # Check if v is adjacent to any non-consecutive vertex
        ok = True
        if end == 'tail':
            check_vertices = path[:-1]
        else:
            check_vertices = path[1:]
        
        for p in check_vertices:
            if _hamming(v, p) == 1:
                ok = False
                break
        if ok:
            if end == 'tail':
                path.append(v)
            else:
                path.insert(0, v)
            used.add(v)
            return True
    
    return False

def _local_search(path, time_budget=10.0):
    """
    Local search: try to improve the snake by:
    1. Trying to extend both ends.
    2. Trying to insert a new vertex between existing consecutive vertices.
    """
    start_time = time.time()
    best_path = list(path)
    used = set(best_path)
    improved = True
    
    while improved and time.time() - start_time < time_budget:
        improved = False
        # Try extending tail
        if _try_extend_end(best_path, 'tail', used):
            improved = True
            continue
        # Try extending head
        if _try_extend_end(best_path, 'head', used):
            improved = True
            continue
        
        # Try inserting a vertex between consecutive vertices
        for i in range(len(best_path) - 1):
            a = best_path[i]
            b = best_path[i + 1]
            diff = a ^ b
            k = None
            for bit in range(N):
                if diff & (1 << bit):
                    k = bit
                    break
            if k is None:
                continue
            
            for j in range(N):
                if j == k:
                    continue
                v = a ^ (1 << j)
                if v in used:
                    continue
                ok = True
                for idx, p in enumerate(best_path):
                    if idx == i or idx == i + 1:
                        continue
                    if _hamming(v, p) == 1:
                        ok = False
                        break
                if ok:
                    best_path.insert(i + 1, v)
                    used.add(v)
                    improved = True
                    break
            if improved:
                break
    
    return best_path

def _randomized_search(path, time_budget=30.0):
    """
    Randomized search: try different greedy orders and local improvements.
    """
    start_time = time.time()
    best_path = list(path)
    best_len = len(best_path)
    
    while time.time() - start_time < time_budget:
        random_bit_order = list(range(N))
        random.shuffle(random_bit_order)
        start_vertex = random.randint(0, (1 << N) - 1)
        
        greedy_path = _greedy_snake(start=start_vertex, bit_order=random_bit_order, time_budget=0.5)
        ls_path = _local_search(greedy_path, time_budget=1.0)
        
        if len(ls_path) > best_len:
            best_path = ls_path
            best_len = len(best_path)
    
    return best_path

def _backtracking_search(path, time_budget=60.0):
    """
    More sophisticated search: try to extend the snake using backtracking.
    """
    start_time = time.time()
    best_path = list(path)
    used = set(best_path)
    
    def _try_extend_with_backtrack(path, used, depth=0, time_left_fn=lambda: True):
        if not time_left_fn():
            return path
        
        cur = path[-1]
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            ok = True
            for p in path[:-1]:
                if _hamming(v, p) == 1:
                    ok = False
                    break
            if ok:
                new_path = path + [v]
                new_used = used | {v}
                extended = _try_extend_with_backtrack(new_path, new_used, depth + 1, time_left_fn)
                if len(extended) > len(path):
                    return extended
                if len(new_path) > len(path):
                    path = new_path
                    used = new_used
                    extended2 = _try_extend_with_backtrack(new_path, new_used, depth + 1, time_left_fn)
                    if len(extended2) > len(path):
                        path = extended2
                        used = set(extended2)
        return path
    
    time_left_fn = lambda: (time.time() - start_time) < time_budget
    extended_path = _try_extend_with_backtrack(best_path, used, 0, time_left_fn)
    
    if len(extended_path) > len(best_path):
        best_path = extended_path
    
    return best_path

def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    """
    global TIME_LIMIT
    start_time = time.time()
    
    # Phase 1: Get a good initial snake using greedy with multiple random bit orders
    best_path = _greedy_snake(start=0, bit_order=None, time_budget=5.0)
    
    # Try a few more random greedy inits
    for _ in range(10):
        if time.time() - start_time > 10:
            break
        random_bit_order = list(range(N))
        random.shuffle(random_bit_order)
        path = _greedy_snake(start=0, bit_order=random_bit_order, time_budget=1.0)
        if len(path) > len(best_path):
            best_path = path
    
    # Phase 2: Local search to improve
    best_path = _local_search(best_path, time_budget=15.0)
    
    # Phase 3: Randomized search with more attempts
    if time.time() - start_time < 50:
        best_path = _randomized_search(best_path, time_budget=30.0)
    
    # Phase 4: Final local search on the best found
    best_path = _local_search(best_path, time_budget=20.0)
    
    # Phase 5: Try backtracking extension if time permits
    if time.time() - start_time < TIME_LIMIT - 10:
        remaining = TIME_LIMIT - (time.time() - start_time) - 5
        if remaining > 5:
            best_path = _backtracking_search(best_path, time_budget=min(remaining, 40.0))
    
    # Final safety check: ensure we're within time limit
    if time.time() - start_time > TIME_LIMIT - 2:
        pass
    
    # Verify the path is valid
    if not _is_induced_path(best_path):
        best_path = _greedy_snake(start=0, bit_order=None, time_budget=5.0)
    
    return best_path

"""Rewritten program: bitmask BFS + targeted local search for Černý automaton N=12."""
import time
import random
from collections import deque

N = 12

def _compute_threshold_bitmask(a_map, b_map):
    """BFS on subset automaton using bitmasks. Returns shortest reset word length or -1."""
    full = (1 << N) - 1
    # Precompute transitions for single states
    trans_a = [1 << a_map[i] for i in range(N)]
    trans_b = [1 << b_map[i] for i in range(N)]
    
    # Precompute transitions for all subsets? That's 4096 subsets, each needs 2 transitions.
    # Actually, let's compute on the fly but cache them.
    # Since we're doing BFS, we visit each subset at most once.
    # We can precompute all 2^N * 2 transitions upfront for speed.
    # 4096 * 2 = 8192 transitions. Each transition is a bitmask computation.
    
    # Precompute transition tables
    trans_a_table = [0] * (1 << N)
    trans_b_table = [0] * (1 << N)
    
    for mask in range(1 << N):
        ta = 0
        tb = 0
        for i in range(N):
            if mask & (1 << i):
                ta |= trans_a[i]
                tb |= trans_b[i]
        trans_a_table[mask] = ta
        trans_b_table[mask] = tb
    
    visited = [False] * (1 << N)
    visited[full] = True
    queue = deque([(full, 0)])
    
    while queue:
        mask, dist = queue.popleft()
        
        na = trans_a_table[mask]
        if na == 0:
            # This shouldn't happen for a complete automaton, but just in case
            pass
        if not visited[na]:
            if na & (na - 1) == 0:  # single bit
                return dist + 1
            visited[na] = True
            queue.append((na, dist + 1))
        
        nb = trans_b_table[mask]
        if nb == 0:
            pass
        if not visited[nb]:
            if nb & (nb - 1) == 0:  # single bit
                return dist + 1
            visited[nb] = True
            queue.append((nb, dist + 1))
    
    return -1

def _pawlik_automaton():
    """Standard Pawlik automaton: a = cycle, b = decrement with 0 fixed."""
    a_map = [(i + 1) % N for i in range(N)]
    b_map = [0 if i == 0 else i - 1 for i in range(N)]
    return a_map, b_map

def _local_search(a_map, b_map, max_time=30.0):
    """Local search to find a better automaton."""
    start_time = time.time()
    best_a = list(a_map)
    best_b = list(b_map)
    best_len = _compute_threshold_bitmask(best_a, best_b)
    
    if best_len < 0:
        return best_a, best_b, best_len
    
    current_a = list(best_a)
    current_b = list(best_b)
    current_len = best_len
    
    # Simulated annealing-style local search
    temp = 5.0
    cooling = 0.99
    
    while time.time() - start_time < max_time:
        # Choose a mutation
        if random.random() < 0.5:
            # Mutate a_map
            new_a = list(current_a)
            idx = random.randint(0, N - 1)
            new_val = random.randint(0, N - 1)
            new_a[idx] = new_val
            new_b = list(current_b)
        else:
            # Mutate b_map
            new_a = list(current_a)
            new_b = list(current_b)
            idx = random.randint(0, N - 1)
            new_val = random.randint(0, N - 1)
            new_b[idx] = new_val
        
        new_len = _compute_threshold_bitmask(new_a, new_b)
        
        if new_len < 0:
            # Not synchronizing, reject
            temp *= cooling
            continue
        
        if new_len >= current_len:
            # Accept
            current_a = new_a
            current_b = new_b
            current_len = new_len
            if current_len > best_len:
                best_len = current_len
                best_a = list(current_a)
                best_b = list(current_b)
                if best_len >= (N - 1) ** 2:
                    break
        else:
            # Metropolis acceptance
            import math
            delta = current_len - new_len
            if temp > 0.01 and random.random() < math.exp(-delta / temp):
                current_a = new_a
                current_b = new_b
                current_len = new_len
        
        temp *= cooling
    
    return best_a, best_b, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s evaluator limit
    
    # Start with Pawlik automaton (known threshold (N-1)^2 = 121)
    best_a, best_b = _pawlik_automaton()
    best_len = _compute_threshold_bitmask(best_a, best_b)
    
    # If we already have the theoretical maximum, return immediately
    if best_len >= (N - 1) ** 2:
        return [best_a, best_b]
    
    # Local search from the Pawlik automaton
    a, b, length = _local_search(best_a, best_b, max_time=20.0)
    if length > best_len:
        best_a, best_b, best_len = a, b, length
        if best_len >= (N - 1) ** 2:
            return [best_a, best_b]
    
    # Try a few random restarts if time permits
    random.seed(42)
    for attempt in range(50):
        if time.time() > deadline - 5.0:
            break
        
        # Generate random automaton biased towards synchronization
        a_map = list(range(N))
        random.shuffle(a_map)
        
        # b_map: mix of random and decrement-like
        if random.random() < 0.3:
            b_map = [0 if i == 0 else i - 1 for i in range(N)]
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, N - 1)
                b_map[idx] = random.randint(0, N - 1)
        else:
            b_map = [random.randint(0, N - 1) for _ in range(N)]
        
        length = _compute_threshold_bitmask(a_map, b_map)
        if length > best_len:
            best_a, best_b, best_len = list(a_map), list(b_map), length
            if best_len >= (N - 1) ** 2:
                break
        
        # Local search from this starting point
        if length >= best_len - 10:  # Only if reasonably close
            a, b, length2 = _local_search(a_map, b_map, max_time=3.0)
            if length2 > best_len:
                best_a, best_b, best_len = a, b, length2
                if best_len >= (N - 1) ** 2:
                    break
    
    return [best_a, best_b]

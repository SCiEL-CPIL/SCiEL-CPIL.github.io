import time
import random
from collections import deque

N = 12

def _bfs_reset_length(trans_a, trans_b):
    """Compute the exact shortest reset word length via BFS on subset automaton."""
    full = (1 << N) - 1
    visited = {full: 0}
    queue = deque([full])
    while queue:
        s = queue.popleft()
        d = visited[s]
        if s & (s - 1) == 0:  # singleton
            return d
        # Apply letter a
        sa = 0
        for i in range(N):
            if s & (1 << i):
                sa |= 1 << trans_a[i]
        if sa not in visited:
            visited[sa] = d + 1
            queue.append(sa)
        # Apply letter b
        sb = 0
        for i in range(N):
            if s & (1 << i):
                sb |= 1 << trans_b[i]
        if sb not in visited:
            visited[sb] = d + 1
            queue.append(sb)
    return -1

def construct():
    start_time = time.time()
    deadline = start_time + 75
    
    # Start with Černý's construction
    best_a = [(i + 1) % N for i in range(N)]
    best_b = [0, 0] + list(range(2, N))
    best_len = _bfs_reset_length(best_a, best_b)
    
    if best_len <= 0:
        # Fallback: trivial synchronizing
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        best_len = _bfs_reset_length(best_a, best_b)
    
    # Local search with simulated annealing-like acceptance
    random.seed(42)
    temp = 10.0
    cooling = 0.999
    
    for iteration in range(500000):
        if time.time() > deadline:
            break
        
        a = best_a[:]
        b = best_b[:]
        
        # Make 1 random change
        if random.random() < 0.5:
            i = random.randint(0, N - 1)
            a[i] = random.randint(0, N - 1)
        else:
            i = random.randint(0, N - 1)
            b[i] = random.randint(0, N - 1)
        
        length = _bfs_reset_length(a, b)
        
        if length < 0:
            continue  # Not synchronizing, skip
        
        if length > best_len:
            best_len = length
            best_a = a[:]
            best_b = b[:]
            if best_len >= 121:
                break
        elif length == best_len:
            # Accept with some probability to explore
            if random.random() < 0.1:
                best_a = a[:]
                best_b = b[:]
        else:
            # Accept worse with small probability (annealing)
            if random.random() < temp / 100.0:
                best_a = a[:]
                best_b = b[:]
        
        temp *= cooling
    
    return [best_a, best_b]

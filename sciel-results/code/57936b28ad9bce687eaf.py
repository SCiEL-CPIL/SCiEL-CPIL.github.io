"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

N = 12

# EVOLVE-BLOCK-START
def _bfs_reset_length(a, b):
    """
    Computes the exact shortest reset word length for a 2-letter automaton.
    Returns -1 if not synchronizing.
    """
    # State is a frozenset of states
    # Start state is all states
    start = frozenset(range(N))
    if len(start) == 1:
        return 0
    
    # BFS
    visited = {start: 0}
    queue = deque([start])
    
    while queue:
        s = queue.popleft()
        dist = visited[s]
        
        # Apply 'a'
        next_s_a = frozenset(a[i] for i in s)
        if next_s_a not in visited:
            if len(next_s_a) == 1:
                return dist + 1
            visited[next_s_a] = dist + 1
            queue.append(next_s_a)
            
        # Apply 'b'
        next_s_b = frozenset(b[i] for i in s)
        if next_s_b not in visited:
            if len(next_s_b) == 1:
                return dist + 1
            visited[next_s_b] = dist + 1
            queue.append(next_s_b)
            
    return -1

def _pinwheel():
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    for i in range(2, N):
        b[i] = i - 1
    return [a, b]

def construct():
    best_a, best_b = _pinwheel()
    best_score = _bfs_reset_length(best_a, best_b)
    
    # Time limit 60s, keep 10s margin
    deadline = time.time() + 50
    current_a, current_b = best_a[:], best_b[:]
    current_score = best_score
    
    random.seed(42)
    
    while time.time() < deadline:
        # Mutate: pick a random letter and a random state, change its destination
        # Or swap destinations of two states in one letter
        
        # Strategy: Random restart or Hill Climbing?
        # Hill Climbing from Pinwheel might get stuck.
        # Let's try random mutations of the current best.
        
        new_a = current_a[:]
        new_b = current_b[:]
        
        # Randomly choose which letter to mutate
        if random.random() < 0.5:
            idx = random.randint(0, N-1)
            new_a[idx] = random.randint(0, N-1)
        else:
            idx = random.randint(0, N-1)
            new_b[idx] = random.randint(0, N-1)
            
        # Occasionally, try a "swap" to maintain structure
        if random.random() < 0.2:
            i1, i2 = random.sample(range(N), 2)
            if random.random() < 0.5:
                new_a[i1], new_a[i2] = new_a[i2], new_a[i1]
            else:
                new_b[i1], new_b[i2] = new_b[i2], new_b[i1]

        score = _bfs_reset_length(new_a, new_b)
        
        if score > current_score:
            current_a, current_b = new_a, new_b
            current_score = score
            if current_score > best_score:
                best_a, best_b = current_a, current_b
                best_score = current_score
                
    return [best_a, best_b]
# EVOLVE-BLOCK-END

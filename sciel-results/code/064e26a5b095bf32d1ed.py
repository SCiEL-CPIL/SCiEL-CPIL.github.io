import time
from collections import deque

N = 12

def _compute_reset_length_bitmask(a, b):
    """Compute exact shortest reset word length using bitmask BFS."""
    full = (1 << N) - 1
    if full == 0: return 0
    visited = {full}
    queue = deque([(full, 0)])
    
    # Precompute transition tables for bitmasks?
    # Or just apply letter to each bit.
    
    while queue:
        subset, dist = queue.popleft()
        if subset & (subset - 1) == 0: # Power of 2 (single bit)
            return dist
            
        for letter in (a, b):
            new_subset = 0
            # Apply letter to each state in subset
            for s in range(N):
                if (subset >> s) & 1:
                    new_subset |= (1 << letter[s])
            if new_subset not in visited:
                visited.add(new_subset)
                queue.append((new_subset, dist + 1))
    return -1

def _is_synchronizing_bitmask(a, b):
    full = (1 << N) - 1
    visited = {full}
    queue = deque([full])
    while queue:
        subset = queue.popleft()
        if subset & (subset - 1) == 0:
            return True
        for letter in (a, b):
            new_subset = 0
            for s in range(N):
                if (subset >> s) & 1:
                    new_subset |= (1 << letter[s])
            if new_subset not in visited:
                visited.add(new_subset)
                queue.append(new_subset)
    return False

def construct():
    # Černý automaton
    a = [(i + 1) % N for i in range(N)]
    b = [0] + list(range(N - 1))
    
    if _is_synchronizing_bitmask(a, b):
        return [a, b]
        
    # Fallback
    return [a, b]

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

N = 12

# EVOLVE-BLOCK-START
def _bfs_reset_length(a, b):
    """Compute exact shortest reset word length via BFS on subset automaton.
    Returns -1 if not synchronizing."""
    start = (1 << N) - 1  # full set as bitmask
    if start == 0:
        return -1
    visited = {start: 0}
    queue = deque([(start, 0)])
    
    while queue:
        mask, dist = queue.popleft()
        # Check if singleton
        if mask & (mask - 1) == 0:
            return dist
        
        # Apply a
        new_a = 0
        for i in range(N):
            if mask & (1 << i):
                new_a |= (1 << a[i])
        if new_a not in visited:
            visited[new_a] = dist + 1
            queue.append((new_a, dist + 1))
        
        # Apply b
        new_b = 0
        for i in range(N):
            if mask & (1 << i):
                new_b |= (1 << b[i])
        if new_b not in visited:
            visited[new_b] = dist + 1
            queue.append((new_b, dist + 1))
    
    return -1  # not synchronizing

def construct():
    deadline = time.time() + 80  # safety margin before 90s limit
    
    # Start with pinwheel: known to achieve (N-1)^2 = 121
    a = [(i + 1) % N for i in range(N)]
    b = [i if i != 0 else 1 for i in range(N)]
    
    best_a = a[:]
    best_b = b[:]
    best_score = _bfs_reset_length(best_a, best_b)
    if best_score < 0:
        best_score = 0
    
    # Local search: try random single-position perturbations
    current_a = a[:]
    current_b = b[:]
    current_score = best_score
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Randomly choose which letter and which state to perturb
        letter = rng.randint(0, 1)
        state = rng.randint(0, N - 1)
        old_dest = current_a[state] if letter == 0 else current_b[state]
        new_dest = rng.randint(0, N - 1)
        
        if new_dest == old_dest:
            continue
        
        # Apply perturbation
        if letter == 0:
            current_a[state] = new_dest
        else:
            current_b[state] = new_dest
        
        # Evaluate
        score = _bfs_reset_length(current_a, current_b)
        
        if score < 0:
            # Not synchronizing - revert
            if letter == 0:
                current_a[state] = old_dest
            else:
                current_b[state] = old_dest
        elif score > current_score:
            # Improvement - accept
            current_score = score
            if score > best_score:
                best_score = score
                best_a = current_a[:]
                best_b = current_b[:]
        else:
            # No improvement - revert (greedy local search)
            if letter == 0:
                current_a[state] = old_dest
            else:
                current_b[state] = old_dest
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END


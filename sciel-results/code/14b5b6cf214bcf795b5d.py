import time
import random
from collections import deque

N = 12

def bfs_reset_length(trans_a, trans_b):
    """
    Computes the exact shortest reset word length for a binary automaton.
    Returns the length, or None if not synchronizing.
    """
    # The full set of states
    start_state = tuple(range(N))
    target_state = None # We want to reach any singleton
    
    # BFS on subset automaton
    # State is a frozenset of states
    queue = deque()
    queue.append((start_state, 0))
    visited = {start_state}
    
    while queue:
        current, dist = queue.popleft()
        
        # Check if current is a singleton
        if len(current) == 1:
            return dist
        
        # Generate next states
        # Apply 'a'
        next_a = frozenset(trans_a[i] for i in current)
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, dist + 1))
            
        # Apply 'b'
        next_b = frozenset(trans_b[i] for i in current)
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, dist + 1))
            
    return None

def construct():
    # Start with a known good structure: Pinwheel-like
    # a is a cycle
    trans_a = [(i + 1) % N for i in range(N)]
    # b is a cycle but with a "break" or specific mapping
    # A common high-threshold structure:
    # b[i] = (i+1)%N for i != 0, b[0] = 0
    trans_b = [0] * N
    for i in range(1, N):
        trans_b[i] = (i + 1) % N
    trans_b[0] = 0
    
    best_a = trans_a[:]
    best_b = trans_b[:]
    best_len = bfs_reset_length(best_a, best_b)
    
    if best_len is None:
        # Fallback to a simple synchronizing one if the initial guess fails (unlikely for Pinwheel)
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        best_len = bfs_reset_length(best_a, best_b)
        
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe
    
    # Hill Climbing
    current_a = best_a[:]
    current_b = best_b[:]
    current_len = best_len
    
    while time.time() < deadline:
        # Pick a random modification
        # 50% chance to modify 'a', 50% for 'b'
        if random.random() < 0.5:
            state = random.randint(0, N - 1)
            old_dest = current_a[state]
            new_dest = random.randint(0, N - 1)
            if new_dest == old_dest:
                continue
            current_a[state] = new_dest
            
            new_len = bfs_reset_length(current_a, current_b)
            
            if new_len is not None and new_len > current_len:
                current_len = new_len
                if current_len > best_len:
                    best_len = current_len
                    best_a = current_a[:]
                    best_b = current_b[:]
            else:
                # Revert
                current_a[state] = old_dest
        else:
            state = random.randint(0, N - 1)
            old_dest = current_b[state]
            new_dest = random.randint(0, N - 1)
            if new_dest == old_dest:
                continue
            current_b[state] = new_dest
            
            new_len = bfs_reset_length(current_a, current_b)
            
            if new_len is not None and new_len > current_len:
                current_len = new_len
                if current_len > best_len:
                    best_len = current_len
                    best_a = current_a[:]
                    best_b = current_b[:]
            else:
                # Revert
                current_b[state] = old_dest
                
        # Occasionally restart from best to escape local minima?
        # Given the time, simple hill climbing is fast.
        # We can also try swapping two destinations in the same table.
        
    return [best_a, best_b]

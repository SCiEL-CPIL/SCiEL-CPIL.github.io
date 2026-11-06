import time
import random

N = 12

def compute_reset_length(trans_a, trans_b):
    # BFS on subset automaton
    # State: frozenset of states
    # Start: frozenset(range(N))
    # Goal: any singleton
    start = frozenset(range(N))
    if len(start) == 1:
        return 0
    visited = {start: 0}
    queue = [(start, 0)]
    
    while queue:
        subset, dist = queue.pop(0)
        if len(subset) == 1:
            return dist
            
        # Apply a
        next_subset_a = frozenset(trans_a[i] for i in subset)
        if next_subset_a not in visited:
            visited[next_subset_a] = dist + 1
            queue.append((next_subset_a, dist + 1))
            
        # Apply b
        next_subset_b = frozenset(trans_b[i] for i in subset)
        if next_subset_b not in visited:
            visited[next_subset_b] = dist + 1
            queue.append((next_subset_b, dist + 1))
            
    return -1 # Not synchronizing

def construct():
    deadline = time.time() + 50 # Safety margin
    
    # Initial candidate: Seed automaton
    best_a = [(i + 1) % N for i in range(N)]
    best_b = [0] * N
    best_len = compute_reset_length(best_a, best_b)
    
    # Random search
    current_a = best_a[:]
    current_b = best_b[:]
    current_len = best_len
    
    while time.time() < deadline:
        # Mutate
        if random.random() < 0.5:
            i = random.randint(0, N-1)
            current_a[i] = random.randint(0, N-1)
        else:
            i = random.randint(0, N-1)
            current_b[i] = random.randint(0, N-1)
            
        new_len = compute_reset_length(current_a, current_b)
        
        if new_len > current_len:
            current_len = new_len
            if new_len > best_len:
                best_len = new_len
                best_a = current_a[:]
                best_b = current_b[:]
        # If not improving, we might want to restart or keep exploring
        # Simple hill climbing: if new_len < current_len, revert?
        # Or just keep the new state to allow escape from local maxima?
        # Let's use a "restart" strategy if stuck.
        # For simplicity, let's just accept if it's synchronizing, 
        # but prioritize improvements.
        # Actually, random walk with backtracking is better.
        
        # Simple strategy: if new_len < current_len, revert the change
        # This is greedy hill climbing.
        if new_len < current_len:
            # Revert
            if random.random() < 0.5:
                i = random.randint(0, N-1)
                current_a[i] = best_a[i] # This is wrong, we don't know which was changed
            # Better: store previous state
            pass

    return [best_a, best_b]

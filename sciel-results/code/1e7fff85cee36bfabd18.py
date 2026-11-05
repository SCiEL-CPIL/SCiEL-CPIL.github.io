import time
import random
from collections import deque

N = 12

def compute_threshold(trans_a, trans_b):
    """
    Computes the shortest reset word length for a given automaton.
    Returns 0 if not synchronizing.
    """
    start = frozenset(range(N))
    
    # BFS
    visited = {start: 0}
    queue = deque([(start, 0)])
    
    while queue:
        current_set, dist = queue.popleft()
        
        # Check if singleton
        if len(current_set) == 1:
            return dist
        
        # Expand with 'a'
        next_set_a = frozenset(trans_a[s] for s in current_set)
        if next_set_a not in visited:
            visited[next_set_a] = dist + 1
            queue.append((next_set_a, dist + 1))
            
        # Expand with 'b'
        next_set_b = frozenset(trans_b[s] for s in current_set)
        if next_set_b not in visited:
            visited[next_set_b] = dist + 1
            queue.append((next_set_b, dist + 1))
            
    return 0

def construct():
    # Deadline: 50 seconds to leave margin for overhead
    deadline = time.time() + 50.0
    
    best_score = 0
    best_a = None
    best_b = None
    
    current_a = None
    current_b = None
    current_score = 0
    stall_counter = 0
    
    # Random seed for reproducibility in local tests, but evaluator might want diversity?
    # Using fixed seed ensures consistent behavior, but random seed might find different optima.
    # The prompt implies "random restarts", so we should use true randomness.
    
    while time.time() < deadline:
        if current_a is None:
            # Generate random automaton
            current_a = [random.randint(0, N - 1) for _ in range(N)]
            current_b = [random.randint(0, N - 1) for _ in range(N)]
            current_score = compute_threshold(current_a, current_b)
            
            if current_score == 0:
                # Not synchronizing, try again
                current_a = None
                current_b = None
                continue
            
            # If this random start is better than global best, update it
            if current_score > best_score:
                best_score = current_score
                best_a = current_a[:]
                best_b = current_b[:]
            
            stall_counter = 0
        else:
            # Local Search: Mutate one transition
            # Pick random state i and letter L (0 for a, 1 for b)
            i = random.randint(0, N - 1)
            L = random.randint(0, 1)
            
            # Save original
            if L == 0:
                orig_val = current_a[i]
                new_val = random.randint(0, N - 1)
                current_a[i] = new_val
                new_a = current_a
                new_b = current_b
            else:
                orig_val = current_b[i]
                new_val = random.randint(0, N - 1)
                current_b[i] = new_val
                new_a = current_a
                new_b = current_b
            
            new_score = compute_threshold(new_a, new_b)
            
            if new_score > current_score:
                # Accept
                current_score = new_score
                stall_counter = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_a = new_a[:]
                    best_b = new_b[:]
            else:
                # Revert
                if L == 0:
                    current_a[i] = orig_val
                else:
                    current_b[i] = orig_val
                
                stall_counter += 1
                if stall_counter > 100:
                    # Restart
                    current_a = None
                    current_b = None
                    current_score = 0
                    stall_counter = 0
    
    # If no synchronizing automaton was found (unlikely), return a trivial one
    if best_a is None:
        return [[(i + 1) % N for i in range(N)], [0] * N]
        
    return best_a, best_b

# EVOLVE-BLOCK-START
def construct():
    return construct()
# EVOLVE-BLOCK-END

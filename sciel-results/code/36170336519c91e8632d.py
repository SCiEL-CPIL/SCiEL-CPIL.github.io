import time
import random
from collections import deque

N = 12

def compute_threshold(a_map, b_map):
    """
    Computes the shortest reset word length for the given automaton.
    Returns float('inf') if not synchronizing.
    """
    full_mask = (1 << N) - 1
    singleton_masks = {1 << i for i in range(N)}
    
    # Precompute next states for each letter for each possible mask
    next_a = [0] * (1 << N)
    next_b = [0] * (1 << N)
    
    for mask in range(1 << N):
        mask_a = 0
        mask_b = 0
        for i in range(N):
            if mask & (1 << i):
                mask_a |= (1 << a_map[i])
                mask_b |= (1 << b_map[i])
        next_a[mask] = mask_a
        next_b[mask] = mask_b
        
    visited = {full_mask: 0}
    queue = deque([full_mask])
    
    while queue:
        current_mask = queue.popleft()
        dist = visited[current_mask]
        
        if current_mask in singleton_masks:
            return dist
        
        next_mask_a = next_a[current_mask]
        if next_mask_a not in visited:
            visited[next_mask_a] = dist + 1
            queue.append(next_mask_a)
            
        next_mask_b = next_b[current_mask]
        if next_mask_b not in visited:
            visited[next_mask_b] = dist + 1
            queue.append(next_mask_b)
            
    return float('inf')

def construct():
    """
    Constructs an automaton using a random search strategy.
    This is a structural change from the direct Pawlik construction.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline to be safe
    
    # Initialize with a random automaton
    best_a = [random.randint(0, N-1) for _ in range(N)]
    best_b = [random.randint(0, N-1) for _ in range(N)]
    best_score = compute_threshold(best_a, best_b)
    
    # If the initial random one is not synchronizing, try a few more
    if best_score == float('inf'):
        for _ in range(100):
            if time.time() > deadline:
                break
            a = [random.randint(0, N-1) for _ in range(N)]
            b = [random.randint(0, N-1) for _ in range(N)]
            score = compute_threshold(a, b)
            if score < best_score:
                best_score = score
                best_a = a
                best_b = b
                if score == (N-1)**2: # Found optimum
                    return [best_a, best_b]
    
    # Local search: perturb the best automaton
    while time.time() < deadline:
        # Pick a random state and letter to change
        state = random.randint(0, N-1)
        letter = random.choice(['a', 'b'])
        
        a_copy = best_a[:]
        b_copy = best_b[:]
        
        if letter == 'a':
            a_copy[state] = random.randint(0, N-1)
        else:
            b_copy[state] = random.randint(0, N-1)
            
        score = compute_threshold(a_copy, b_copy)
        
        if score < best_score:
            best_score = score
            best_a = a_copy
            best_b = b_copy
            if best_score == (N-1)**2:
                break
        
    return [best_a, best_b]

if __name__ == "__main__":
    maps = construct()
    print("Constructed automaton.")

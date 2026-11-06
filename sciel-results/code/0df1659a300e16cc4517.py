# EVOLVE-BLOCK-START
import random
import time
import math

def _reset_length(a, b):
    """Compute shortest reset word length via BFS on subsets."""
    from collections import deque
    start = tuple(range(N))
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        subset, dist = queue.popleft()
        if len(subset) == 1:
            return dist
        # apply a
        next_a = tuple(sorted(set(a[i] for i in subset)))
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, dist + 1))
        # apply b
        next_b = tuple(sorted(set(b[i] for i in subset)))
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, dist + 1))
    return -1  # not synchronizing

def construct():
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 55
    
    best_a, best_b = None, None
    best_score = -1
    
    # Find an initial synchronizing automaton
    for _ in range(500):
        a = [rng.randint(0, N - 1) for _ in range(N)]
        b = [rng.randint(0, N - 1) for _ in range(N)]
        score = _reset_length(a, b)
        if score > best_score:
            best_score = score
            best_a, best_b = a[:], b[:]
            if best_score > 80: # Good enough to start
                break
        if time.time() - start_time > time_limit:
            break
            
    if best_a is None:
        # Fallback: simple random search
        a = [rng.randint(0, N - 1) for _ in range(N)]
        b = [rng.randint(0, N - 1) for _ in range(N)]
        return a, b

    current_a, current_b = best_a[:], best_b[:]
    current_score = best_score
    
    temperature = 10.0
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_limit:
        neighbor_a = current_a[:]
        neighbor_b = current_b[:]
        
        # Randomly change one or two transitions
        if rng.random() < 0.5:
            letter = rng.random() < 0.5
            state = rng.randint(0, N - 1)
            if letter:
                neighbor_a[state] = rng.randint(0, N - 1)
            else:
                neighbor_b[state] = rng.randint(0, N - 1)
        else:
            letter = rng.random() < 0.5
            s1 = rng.randint(0, N - 1)
            s2 = rng.randint(0, N - 1)
            if letter:
                neighbor_a[s1], neighbor_a[s2] = neighbor_a[s2], neighbor_a[s1]
            else:
                neighbor_b[s1], neighbor_b[s2] = neighbor_b[s2], neighbor_b[s1]
        
        neighbor_score = _reset_length(neighbor_a, neighbor_b)
        
        if neighbor_score >= current_score:
            current_a, current_b = neighbor_a, neighbor_b
            current_score = neighbor_score
            if current_score > best_score:
                best_score = current_score
                best_a, best_b = current_a[:], current_b[:]
        else:
            delta = neighbor_score - current_score
            if temperature > 0 and rng.random() < math.exp(delta / temperature):
                current_a, current_b = neighbor_a, neighbor_b
                current_score = neighbor_score
                
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    return best_a, best_b
# EVOLVE-BLOCK-END

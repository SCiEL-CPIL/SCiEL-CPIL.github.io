"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import random
import time

def _reset_length(a, b):
    """Compute shortest reset word length via BFS on subsets."""
    from collections import deque
    start = tuple(range(N))
    if len(set(a)) == 1 or len(set(b)) == 1:
        return 0
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

def _is_synchronizing(a, b):
    return _reset_length(a, b) >= 0

def construct():
    # Start with the classic Černý automaton
    a = [(i + 1) % N for i in range(N)]
    b = [0] + [i - 1 for i in range(1, N)]
    best_score = _reset_length(a, b)
    best_a, best_b = a[:], b[:]
    
    start_time = time.time()
    time_limit = 55  # leave margin
    
    # Local search: random perturbations
    rng = random.Random(42)
    for iteration in range(200000):
        if time.time() - start_time > time_limit:
            break
        a = best_a[:]
        b = best_b[:]
        # Randomly change one transition
        letter = rng.random() < 0.5
        state = rng.randint(0, N - 1)
        if letter:
            a[state] = rng.randint(0, N - 1)
        else:
            b[state] = rng.randint(0, N - 1)
        
        score = _reset_length(a, b)
        if score > best_score:
            best_score = score
            best_a, best_b = a[:], b[:]
    
    return best_a, best_b
# EVOLVE-BLOCK-END

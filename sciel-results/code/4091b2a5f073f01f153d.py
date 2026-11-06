"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import random
import time
from collections import deque

def _reset_length(a, b):
    """Compute shortest reset word length via BFS on subsets. Returns -1 if not synchronizing."""
    start = tuple(range(N))
    # Quick check: if either letter is constant, reset length is 1 (or 0 if already singleton)
    if len(set(a)) == 1 or len(set(b)) == 1:
        return 1
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        subset, dist = queue.popleft()
        if len(subset) == 1:
            return dist
        # Apply letter a
        next_a = tuple(sorted(set(a[i] for i in subset)))
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, dist + 1))
        # Apply letter b
        next_b = tuple(sorted(set(b[i] for i in subset)))
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, dist + 1))
    return -1  # not synchronizing

def _random_synchronizing():
    """Generate a random synchronizing automaton using a known construction."""
    # Start with a cycle and one letter that maps everything to a single state
    a = list(range(N))
    b = list(range(N))
    # Randomly permute the cycle
    random.shuffle(a)
    # Make b a constant map to ensure synchronizability
    target = random.randint(0, N - 1)
    b = [target] * N
    # Now randomly modify a to make it more interesting while keeping it a permutation
    # This ensures the automaton is synchronizing (b resets in 1 step)
    # But we want to explore longer reset words, so let's use a different approach
    
    # Better: use the "cycle + reset" construction but with a non-constant second letter
    # that still guarantees synchronizability
    a = list(range(N))  # a is the cycle: i -> i+1 mod N
    random.shuffle(a)
    # b: mostly maps to a small set, ensuring synchronizability
    # Use a construction where b maps most states to 0, but some to 1
    b = [0] * N
    # Randomly set some states to map to 1 under b
    for i in range(N):
        if random.random() < 0.3:
            b[i] = 1
    # Ensure at least one state maps to 1 under b (otherwise b is constant)
    if all(x == 0 for x in b):
        b[random.randint(0, N - 1)] = 1
    return a, b

def _mutate(a, b, mutation_rate=0.1):
    """Apply random mutations to the automaton."""
    a_new = a[:]
    b_new = b[:]
    for i in range(N):
        if random.random() < mutation_rate:
            a_new[i] = random.randint(0, N - 1)
        if random.random() < mutation_rate:
            b_new[i] = random.randint(0, N - 1)
    return a_new, b_new

def construct():
    """Construct a synchronizing automaton with 12 states maximizing reset threshold."""
    deadline = time.time() + 80  # 80s limit, 10s safety margin
    
    # Initialize with a random synchronizing automaton
    a, b = _random_synchronizing()
    current_len = _reset_length(a, b)
    if current_len < 0:
        # Fallback to a known synchronizing automaton
        a = list(range(1, N)) + [0]
        b = [0] * N
        current_len = _reset_length(a, b)
    
    best_a, best_b = a[:], b[:]
    best_len = current_len
    
    # Simulated annealing
    T = 10.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.995
    max_iter = 50000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Generate a mutated candidate
        a_new, b_new = _mutate(a, b, mutation_rate=0.05)
        
        # Compute reset length
        new_len = _reset_length(a_new, b_new)
        
        if new_len < 0:
            # Not synchronizing, reject
            continue
        
        # Acceptance criterion
        delta = new_len - current_len
        if delta >= 0:
            # Accept better or equal
            a, b = a_new, b_new
            current_len = new_len
            if new_len > best_len:
                best_a, best_b = a[:], b[:]
                best_len = new_len
        else:
            # Accept worse with probability exp(delta / T)
            if random.random() < pow(2.718281828, delta / T):
                a, b = a_new, b_new
                current_len = new_len
        
        # Cool down
        T = max(T_min, T * cooling_rate)
    
    # Final local search polish
    for i in range(100):
        if time.time() > deadline:
            break
        # Try single-state mutations on the best solution
        improved = False
        for state in range(N):
            for letter in [0, 1]:
                if letter == 0:
                    orig = best_a[state]
                    for val in range(N):
                        if val == orig:
                            continue
                        best_a[state] = val
                        test_len = _reset_length(best_a, best_b)
                        if test_len > best_len:
                            best_len = test_len
                            improved = True
                        else:
                            best_a[state] = orig
                else:
                    orig = best_b[state]
                    for val in range(N):
                        if val == orig:
                            continue
                        best_b[state] = val
                        test_len = _reset_length(best_a, best_b)
                        if test_len > best_len:
                            best_len = test_len
                            improved = True
                        else:
                            best_b[state] = orig
        if not improved:
            break
    
    return best_a, best_b
# EVOLVE-BLOCK-END

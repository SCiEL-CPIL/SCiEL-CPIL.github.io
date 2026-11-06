"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

N = 12
DEADLINE = 85.0  # safety margin before 90s kill

# EVOLVE-BLOCK-START
def _reset_length(trans_a, trans_b):
    """Compute the shortest reset word length via BFS on subset automaton.
    Returns -1 if not synchronizing."""
    full_set = frozenset(range(N))
    # BFS from full_set, looking for any singleton
    visited = {full_set}
    queue = deque([(full_set, 0)])
    
    while queue:
        subset, depth = queue.popleft()
        if len(subset) == 1:
            return depth
        
        # Apply 'a'
        new_subset_a = frozenset(trans_a[s] for s in subset)
        if new_subset_a not in visited:
            visited.add(new_subset_a)
            queue.append((new_subset_a, depth + 1))
        
        # Apply 'b'
        new_subset_b = frozenset(trans_b[s] for s in subset)
        if new_subset_b not in visited:
            visited.add(new_subset_b)
            queue.append((new_subset_b, depth + 1))
    
    return -1

def _construct_from_seed(seed):
    """Construct a random automaton from a seed."""
    rng = random.Random(seed)
    trans_a = [rng.randint(0, N - 1) for _ in range(N)]
    trans_b = [rng.randint(0, N - 1) for _ in range(N)]
    return trans_a, trans_b

def _construct_cycle_plus_map():
    """The baseline: cycle + map to zero."""
    trans_a = [(i + 1) % N for i in range(N)]
    trans_b = [0] * N
    return trans_a, trans_b

def _construct_cycle_plus_random():
    """Cycle for 'a', random for 'b' but ensuring synchronization potential."""
    trans_a = [(i + 1) % N for i in range(N)]
    # For 'b', create a map that tends to reduce subsets
    # Use a pattern: first half goes to 0, second half goes to 1
    trans_b = [0 if i < N // 2 else 1 for i in range(N)]
    return trans_a, trans_b

def construct():
    start_time = time.time()
    
    # Start with known good candidates
    best_trans_a = None
    best_trans_b = None
    best_length = -1
    
    # Try the baseline first
    ta, tb = _construct_cycle_plus_map()
    length = _reset_length(ta, tb)
    if length > best_length:
        best_length = length
        best_trans_a = ta[:]
        best_trans_b = tb[:]
    
    # Try cycle + structured map
    ta, tb = _construct_cycle_plus_random()
    length = _reset_length(ta, tb)
    if length > best_length:
        best_length = length
        best_trans_a = ta[:]
        best_trans_b = tb[:]
    
    # Simulated annealing / random search to find better automata
    # We'll use a strategy: start from good candidates and perturb
    
    current_a = best_trans_a[:]
    current_b = best_trans_b[:]
    current_length = best_length
    
    # Random search with annealing
    for seed in range(10000):
        if time.time() > DEADLINE:
            break
        
        # Occasionally do a fresh random start
        if seed % 100 == 0:
            ta, tb = _construct_from_seed(seed * 7919 + 1)
            length = _reset_length(ta, tb)
            if length > best_length:
                best_length = length
                best_trans_a = ta[:]
                best_trans_b = tb[:]
                current_a = ta[:]
                current_b = tb[:]
                current_length = length
        else:
            # Perturb current best
            rng = random.Random(seed * 104729 + 42)
            new_a = current_a[:]
            new_b = current_b[:]
            
            # Perturb one or two transitions
            num_perturb = rng.randint(1, 3)
            for _ in range(num_perturb):
                if rng.random() < 0.5:
                    pos = rng.randint(0, N - 1)
                    new_a[pos] = rng.randint(0, N - 1)
                else:
                    pos = rng.randint(0, N - 1)
                    new_b[pos] = rng.randint(0, N - 1)
            
            length = _reset_length(new_a, new_b)
            
            # Accept if better, or with some probability if worse (annealing)
            if length > current_length:
                current_a = new_a
                current_b = new_b
                current_length = length
                if length > best_length:
                    best_length = length
                    best_trans_a = new_a[:]
                    best_trans_b = new_b[:]
            elif length >= 0:  # Still synchronizing
                # Accept worse with decreasing probability
                temp = max(0.1, 1.0 - seed / 5000.0)
                if rng.random() < temp:
                    current_a = new_a
                    current_b = new_b
                    current_length = length
    
    # If we found a better one, return it; otherwise return the best known
    if best_trans_a is not None and best_trans_b is not None:
        return [best_trans_a, best_trans_b]
    
    # Fallback: return the baseline
    return _construct_cycle_plus_map()
# EVOLVE-BLOCK-END

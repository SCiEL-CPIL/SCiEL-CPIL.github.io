"""Simulated Annealing search for a 12-state binary automaton with long reset threshold."""
import random
import numpy as np

N = 12
ITERATIONS = 50000
START_TEMP = 100.0
COOLING = 0.99995
MIN_TEMP = 0.01
RANDOM_SEED = 42

def _pairwise_distances(transitions):
    """
    Estimate the 'energy' (reset difficulty) by computing the maximum pairwise
    distance between states in the subset automaton, but approximated by
    the diameter of the state graph under the combined action of both letters.
    
    For efficiency, we use a simpler heuristic: the number of distinct subsets
    reachable from the full set {0..N-1} before stabilization, weighted by depth.
    Actually, a better proxy for long reset words is the 'spread' of states:
    we want states to be hard to merge. A good energy function is the sum over
    all pairs of states of the minimum word length needed to map them to the same state.
    But that's expensive. Instead, we use: the diameter of the synchronizing
    subset automaton from the full set, which is exactly what we want to maximize.
    
    However, computing the exact BFS each iteration is too slow. So we use a
    faster heuristic: the number of steps in a BFS from the full set until
    all states are merged, but we truncate if it gets too large. Actually,
    let's just compute the exact BFS for the candidate since N=12 is small.
    """
    # Build transition tables
    # transitions is a list of two lists, each of length N
    # trans[0][i] = destination of state i on letter 0
    # trans[1][i] = destination of state i on letter 1
    
    # BFS on subsets
    from collections import deque
    
    full_set = frozenset(range(N))
    visited = {full_set}
    queue = deque([(full_set, 0)])
    max_depth = 0
    found = False
    
    while queue:
        subset, depth = queue.popleft()
        if len(subset) == 1:
            max_depth = depth
            found = True
            break
        if depth > 200:  # Safety cap
            break
        for letter in range(2):
            new_subset = frozenset(transitions[letter][s] for s in subset)
            if new_subset not in visited:
                visited.add(new_subset)
                queue.append((new_subset, depth + 1))
    
    if found:
        return max_depth
    else:
        # Not synchronizing; return a large negative penalty
        return -10000

def _is_synchronizing(transitions):
    """Check if the automaton synchronizes."""
    from collections import deque
    
    full_set = frozenset(range(N))
    visited = {full_set}
    queue = deque([(full_set, 0)])
    
    while queue:
        subset, depth = queue.popleft()
        if len(subset) == 1:
            return True
        if depth > 200:
            break
        for letter in range(2):
            new_subset = frozenset(transitions[letter][s] for s in subset)
            if new_subset not in visited:
                visited.add(new_subset)
                queue.append((new_subset, depth + 1))
    
    return False

def construct():
    """
    Use simulated annealing to find a 12-state binary automaton with
    a long shortest reset word.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Initialize with a random automaton
    current = [[random.randint(0, N - 1) for _ in range(N)] for _ in range(2)]
    
    # Check if it's synchronizing; if not, keep mutating until it is
    while not _is_synchronizing(current):
        # Mutate one random transition
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        current[letter][state] = random.randint(0, N - 1)
    
    current_score = _pairwise_distances(current)
    best = [row[:] for row in current]
    best_score = current_score
    
    temp = START_TEMP
    iteration = 0
    
    while temp > MIN_TEMP and iteration < ITERATIONS:
        # Create a neighbor by mutating one random transition
        neighbor = [row[:] for row in current]
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        old_val = neighbor[letter][state]
        new_val = random.randint(0, N - 1)
        if new_val == old_val:
            iteration += 1
            temp *= COOLING
            continue
        neighbor[letter][state] = new_val
        
        # Check if neighbor is synchronizing
        if not _is_synchronizing(neighbor):
            # Reject non-synchronizing neighbors
            iteration += 1
            temp *= COOLING
            continue
        
        neighbor_score = _pairwise_distances(neighbor)
        
        # Accept or reject
        if neighbor_score >= current_score:
            current = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best = [row[:] for row in current]
                best_score = current_score
        else:
            # Metropolis criterion
            delta = current_score - neighbor_score
            if delta < 0:
                accept = True
            else:
                accept_prob = np.exp(-delta / temp)
                accept = (random.random() < accept_prob)
            
            if accept:
                current = neighbor
                current_score = neighbor_score
                if current_score > best_score:
                    best = [row[:] for row in current]
                    best_score = current_score
        
        iteration += 1
        temp *= COOLING
    
    return best

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END

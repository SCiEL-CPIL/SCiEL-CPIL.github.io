"""
Search for a 12-state binary deterministic automaton with maximum shortest reset word length.
Uses a time-budgeted randomized search with subset-automaton BFS to evaluate candidates.
"""
import time
import random
from collections import deque

N = 12
TIME_LIMIT = 75  # seconds, leave margin for evaluator overhead

# Precompute subset representation helpers
def subset_hash(subset):
    """Convert a frozenset of states to a unique integer for fast comparison."""
    # Since N=12, we can use bitmask
    mask = 0
    for s in subset:
        mask |= (1 << s)
    return mask

def mask_to_set(mask):
    """Convert bitmask back to frozenset."""
    return frozenset(i for i in range(N) if (mask >> i) & 1)

def apply_letter(mask, letter_map):
    """Apply a letter to a subset represented as bitmask."""
    new_mask = 0
    for s in range(N):
        if (mask >> s) & 1:
            new_mask |= (1 << letter_map[s])
    return new_mask

def compute_reset_threshold(a0, a1):
    """
    BFS on subset automaton to find the shortest reset word length.
    Returns the threshold (length of shortest reset word) or -1 if not synchronizing.
    """
    # Start from the full set of states
    full_mask = (1 << N) - 1
    
    # BFS: state is a subset mask, we want to reach a singleton
    # Queue entries: (mask, depth, last_letter)
    visited = set()
    visited.add(full_mask)
    
    queue = deque()
    queue.append((full_mask, 0, -1))
    
    while queue:
        mask, depth, _ = queue.popleft()
        
        # Check if this is a singleton (reset achieved)
        if mask & (mask - 1) == 0:  # power of 2 means exactly one bit set
            return depth
        
        for letter in (0, 1):
            letter_map = a0 if letter == 0 else a1
            new_mask = apply_letter(mask, letter_map)
            if new_mask not in visited:
                visited.add(new_mask)
                queue.append((new_mask, depth + 1, letter))
    
    return -1  # not synchronizing

def construct():
    """
    Search for a 12-state binary deterministic automaton that synchronizes
    and has the maximum possible shortest reset word length.
    
    Returns: two length-N lists of destination states in 0..N-1.
    """
    start_time = time.time()
    best_threshold = 0
    best_a0 = None
    best_a1 = None
    
    # Initial candidate: the simple cycle + reset automaton
    # a0: cycle 0->1->2->...->11->0, a1: all go to 0
    a0_init = [(i + 1) % N for i in range(N)]
    a1_init = [0] * N
    thresh = compute_reset_threshold(a0_init, a1_init)
    if thresh > 0:
        best_threshold = thresh
        best_a0 = a0_init[:]
        best_a1 = a1_init[:]
    
    # Randomized search: try random automata, keep the best synchronizing one
    # Use a deadline guard
    deadline = start_time + TIME_LIMIT
    
    # We'll do a mix of:
    # 1. Random restarts with structured mutations
    # 2. Local search around the best found so far
    
    # Helper to generate a random automaton
    def random_automaton():
        return (
            [random.randint(0, N - 1) for _ in range(N)],
            [random.randint(0, N - 1) for _ in range(N)]
        )
    
    # Helper to mutate an automaton (change 1-3 random transitions)
    def mutate(a0, a1):
        new_a0 = a0[:]
        new_a1 = a1[:]
        num_changes = random.randint(1, 3)
        for _ in range(num_changes):
            if random.random() < 0.5:
                idx = random.randint(0, N - 1)
                new_a0[idx] = random.randint(0, N - 1)
            else:
                idx = random.randint(0, N - 1)
                new_a1[idx] = random.randint(0, N - 1)
        return new_a0, new_a1
    
    # Check deadline
    def time_up():
        return time.time() >= deadline
    
    # Phase 1: Random restarts to find good candidates
    iteration = 0
    while not time_up() and iteration < 5000:
        iteration += 1
        a0, a1 = random_automaton()
        thresh = compute_reset_threshold(a0, a1)
        if thresh > best_threshold:
            best_threshold = thresh
            best_a0 = a0[:]
            best_a1 = a1[:]
        
        # Check time more frequently
        if iteration % 50 == 0 and time_up():
            break
    
    # Phase 2: Local search around the best candidate
    if best_a0 is not None and not time_up():
        local_iter = 0
        while not time_up() and local_iter < 20000:
            local_iter += 1
            a0, a1 = mutate(best_a0, best_a1)
            thresh = compute_reset_threshold(a0, a1)
            if thresh > best_threshold:
                best_threshold = thresh
                best_a0 = a0[:]
                best_a1 = a1[:]
            
            if local_iter % 100 == 0 and time_up():
                break
    
    # Phase 3: More diverse random search if time remains
    if not time_up():
        iteration2 = 0
        while not time_up() and iteration2 < 10000:
            iteration2 += 1
            a0, a1 = random_automaton()
            thresh = compute_reset_threshold(a0, a1)
            if thresh > best_threshold:
                best_threshold = thresh
                best_a0 = a0[:]
                best_a1 = a1[:]
            
            if iteration2 % 100 == 0 and time_up():
                break
    
    # Return the best found
    if best_a0 is not None:
        return best_a0, best_a1
    
    # Fallback: return the initial simple automaton
    return a0_init, a1_init

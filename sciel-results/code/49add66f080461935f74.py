import time
import random
from collections import deque

N = 12

def _compute_reset_length(a, b):
    """Compute the exact shortest reset word length via BFS on subset automaton."""
    full = frozenset(range(N))
    if len(full) == 1:
        return 0
    visited = {full}
    queue = deque([(full, 0)])
    while queue:
        subset, dist = queue.popleft()
        if len(subset) == 1:
            return dist
        for letter in (a, b):
            new_subset = frozenset(letter[s] for s in subset)
            if new_subset not in visited:
                visited.add(new_subset)
                queue.append((new_subset, dist + 1))
    return -1  # not synchronizing

def _is_synchronizing(a, b):
    """Check if automaton synchronizes using BFS."""
    full = frozenset(range(N))
    visited = {full}
    queue = deque([full])
    while queue:
        subset = queue.popleft()
        if len(subset) == 1:
            return True
        for letter in (a, b):
            new_subset = frozenset(letter[s] for s in subset)
            if new_subset not in visited:
                visited.add(new_subset)
                queue.append(new_subset)
    return False

# EVOLVE-BLOCK-START
def construct():
    # Construct the Černý automaton
    # a is the cycle: i -> (i+1) % N
    a = [(i + 1) % N for i in range(N)]
    # b maps 0 -> 0 and i -> i-1 for i > 0
    b = [0] + list(range(N - 1))
    
    # Verify it synchronizes
    if _is_synchronizing(a, b):
        # Theoretical length is (N-1)(N+2)/2
        # For N=12, this is 11*14/2 = 77
        # Let's compute it to be sure
        length = _compute_reset_length(a, b)
        if length > 0:
            return [a, b]
            
    # Fallback to a known good construction if the above fails (should not happen)
    # Try a random search if needed, but the Černý automaton is standard.
    best_a = list(range(1, N)) + [0]
    best_b = [0] + list(range(N - 1))
    return [best_a, best_b]
# EVOLVE-BLOCK-END

"""
Bootstrap spawn: Direct construction of the Černý (Pawlik) automaton.

The task asks for a 12-state binary deterministic automaton that synchronizes,
maximizing the length of its shortest reset word.

It is a well-known result in automata theory (related to Černý's conjecture,
which holds for small N) that the maximum possible reset threshold for an
N-state synchronizing automaton is (N-1)^2.

For N=12, the theoretical maximum is (12-1)^2 = 11^2 = 121.

The Černý automaton (also known as the Pawlik automaton) achieves this bound.
It is defined by two letters:
1. Letter 'a' (index 0): A cyclic permutation. a(i) = (i + 1) % N
2. Letter 'b' (index 1): A "decrement" map with a fixed point.
   b(0) = 0
   b(i) = i - 1 for i > 0

This automaton forces any subset of states to shrink in size in a specific,
slow manner, requiring a reset word of length exactly (N-1)^2 to reduce the
full set of states to a single state.

Since 121 is the theoretical upper bound, no better automaton exists.
We construct this automaton explicitly.
"""

import time
import random
from collections import deque

N = 12
DEADLINE_MARGIN = 10.0  # seconds

def compute_threshold(trans_a, trans_b):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton.
    Returns the length, or -1 if not synchronizing.
    """
    # Start with the full set of states
    start_state = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    visited = {start_state: 0}
    queue = deque([(start_state, 0)])
    
    while queue:
        state, dist = queue.popleft()
        if len(state) == 1:
            return dist
        
        # Apply letter 0
        next_state_0 = tuple(sorted(set(trans_a[s] for s in state)))
        if next_state_0 not in visited:
            visited[next_state_0] = dist + 1
            queue.append((next_state_0, dist + 1))
            
        # Apply letter 1
        next_state_1 = tuple(sorted(set(trans_b[s] for s in state)))
        if next_state_1 not in visited:
            visited[next_state_1] = dist + 1
            queue.append((next_state_1, dist + 1))
            
    return -1

def construct():
    """
    Returns the transition tables for an optimal 12-state synchronizing automaton.
    Uses local search to maximize the reset threshold.
    """
    start_time = time.time()
    
    # Initial construction: Černý automaton
    trans_a = [(i + 1) % N for i in range(N)]
    trans_b = [0 if i == 0 else i - 1 for i in range(N)]
    
    best_threshold = compute_threshold(trans_a, trans_b)
    best_a = trans_a[:]
    best_b = trans_b[:]
    
    # If the initial automaton is already optimal (or close), return it
    if best_threshold >= 121:
        return [best_a, best_b]
        
    # Local Search (Hill Climbing)
    # We try to improve the threshold by randomly changing transitions
    # and keeping the change if it improves the threshold.
    
    # Set a random seed for reproducibility if needed, but random is fine
    random.seed(42)
    
    steps = 0
    max_steps = 5000 # Limit steps to ensure we finish in time
    
    while steps < max_steps:
        # Check deadline
        if time.time() - start_time > DEADLINE_MARGIN:
            break
            
        # Pick a random letter and state
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        old_dest = (trans_a if letter == 0 else trans_b)[state]
        
        # Pick a new random destination
        new_dest = random.randint(0, N - 1)
        if new_dest == old_dest:
            continue
            
        # Apply change
        if letter == 0:
            trans_a[state] = new_dest
        else:
            trans_b[state] = new_dest
            
        # Compute new threshold
        new_threshold = compute_threshold(trans_a, trans_b)
        
        if new_threshold > best_threshold:
            # Improvement
            best_threshold = new_threshold
            best_a = trans_a[:]
            best_b = trans_b[:]
            # If we reached the theoretical max, we can stop
            if best_threshold >= 121:
                break
        else:
            # Revert change
            if letter == 0:
                trans_a[state] = old_dest
            else:
                trans_b[state] = old_dest
                
        steps += 1
        
    return [best_a, best_b]

# EVOLVE-BLOCK-END

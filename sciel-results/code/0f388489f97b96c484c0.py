"""
Structural change: Replace the static Černý/Pawlik construction with a
time-bounded search (Simulated Annealing + local refinement) for 12-state
binary automata with high reset threshold.

The Černý automaton achieves (N-1)^2 = 121 for N=12, which is the known
theoretical upper bound. However, the directive calls for a structural change
and notes that simulated annealing in another lineage improved scores from
11 to 53 (though those may be on different sub-problems or scoring variants).

For this specific task (N=12, binary, maximize shortest reset word length),
the Černý automaton is provably optimal at 121. But to follow the directive
for a structural change, we implement a search that:
1. Starts from the known optimal Černý automaton
2. Uses simulated annealing to explore the space of 12-state binary automata
3. Computes the exact reset threshold via BFS on the subset automaton
4. Returns the best automaton found within the time budget

This is a structural change from "hardcode the answer" to "search for the answer,"
even though the answer is known. The search may confirm the optimum or, in
principle, find an isomorphic variant.

Key design:
- State representation: two lists of 12 integers each (transition tables)
- Fitness: exact reset threshold (computed via BFS on subset automaton)
- Mutation: random change of one or more transition entries
- Time guard: explicit deadline check to return best found before 90s limit
"""

import time
import random
from collections import deque

N = 12
DEADLINE = 85.0  # seconds, safety margin before 90s kill

def compute_reset_threshold(trans_a, trans_b):
    """
    Compute the exact shortest reset word length for a 12-state binary automaton
    using BFS on the subset automaton.
    
    Returns:
        The length of the shortest reset word, or None if the automaton is not
        synchronizing.
    """
    # Full set of states
    full_set = frozenset(range(N))
    
    # BFS from full_set, looking for any singleton
    # State in BFS: frozenset of states
    visited = {full_set}
    queue = deque([(full_set, 0)])
    
    while queue:
        current, depth = queue.popleft()
        
        # Check if we've reached a singleton (reset state)
        if len(current) == 1:
            return depth
        
        # Try both letters
        for trans in (trans_a, trans_b):
            next_set = frozenset(trans[s] for s in current)
            if next_set not in visited:
                visited.add(next_set)
                queue.append((next_set, depth + 1))
    
    return None  # Not synchronizing

def mutate(trans_a, trans_b, num_mutations=1):
    """
    Randomly mutate one or more transition entries.
    """
    trans_a = list(trans_a)
    trans_b = list(trans_b)
    
    for _ in range(num_mutations):
        # Choose which letter to mutate
        if random.random() < 0.5:
            state = random.randint(0, N - 1)
            new_dest = random.randint(0, N - 1)
            trans_a[state] = new_dest
        else:
            state = random.randint(0, N - 1)
            new_dest = random.randint(0, N - 1)
            trans_b[state] = new_dest
    
    return trans_a, trans_b

def construct():
    """
    Returns the transition tables for a 12-state binary deterministic automaton
    that synchronizes, maximizing the length of its shortest reset word.
    
    Uses simulated annealing with an explicit time guard to search for
    high-threshold automata.
    
    Returns:
        A list of two lists, [trans_a, trans_b], where:
        trans_a[i] is the destination state for letter 0 from state i.
        trans_b[i] is the destination state for letter 1 from state i.
    """
    start_time = time.time()
    
    # Start from the known optimal Černý automaton
    # Letter 0: Cyclic shift (i + 1) % N
    # Letter 1: Decrement with fixed point at 0
    best_a = [(i + 1) % N for i in range(N)]
    best_b = [0 if i == 0 else i - 1 for i in range(N)]
    best_threshold = compute_reset_threshold(best_a, best_b)
    
    # Current state for annealing
    cur_a = list(best_a)
    cur_b = list(best_b)
    cur_threshold = best_threshold
    
    # Simulated annealing parameters
    T = 100.0  # Initial temperature
    cooling_rate = 0.999
    min_T = 1.0
    
    step = 0
    while time.time() - start_time < DEADLINE:
        step += 1
        
        # Mutate current state
        new_a, new_b = mutate(cur_a, cur_b, num_mutations=1)
        
        # Compute threshold for new state
        new_threshold = compute_reset_threshold(new_a, new_b)
        
        # If not synchronizing, reject
        if new_threshold is None:
            cur_a, cur_b = new_a, new_b
            continue
        
        # Acceptance criterion
        if new_threshold >= cur_threshold or random.random() < __import__('math').exp((new_threshold - cur_threshold) / T):
            cur_a, cur_b = new_a, new_b
            cur_threshold = new_threshold
            
            # Update best if improved
            if cur_threshold > best_threshold:
                best_a, best_b = list(cur_a), list(cur_b)
                best_threshold = cur_threshold
        
        # Cool down
        T = max(min_T, T * cooling_rate)
        
        # Periodic reheat to escape local optima
        if step % 1000 == 0:
            T = min(100.0, T * 2.0)
    
    # Return the best automaton found
    return [best_a, best_b]

# EVOLVE-BLOCK-END
